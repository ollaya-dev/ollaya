//! `ollaya run MODEL [STATE]`: pull if needed, load, answer; a REPL when no state is given.

use std::io::{IsTerminal, Read};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use indicatif::ProgressBar;
use ollaya_api::{Client, DecideRequest, DecideResponse, KeepAlive, Questions};
use rustyline::DefaultEditor;
use rustyline::error::ReadlineError;
use serde_json::Value;
use tokio::runtime::Runtime;

use crate::{commands, daemon, presets, render};

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Format {
    Text,
    Json,
}

#[derive(Debug, Clone, clap::Args)]
pub struct RunArgs {
    /// Model to run, e.g. `laya` or `laya:multilingual`.
    pub model: String,
    /// The state to decide about. Without one, reads piped stdin, or starts a REPL on a terminal.
    pub state: Vec<String>,
    /// Question schema (JSON file of question id -> question). Overrides the model's own.
    #[arg(long, value_name = "FILE")]
    pub questions: Option<PathBuf>,
    /// A built-in question set: triage, email, guard, moderation, router, agent.
    #[arg(long, value_parser = clap::builder::PossibleValuesParser::new(presets::NAMES))]
    pub preset: Option<String>,
    /// Output format.
    #[arg(long, value_enum, default_value_t = Format::Text)]
    pub format: Format,
    /// How long to keep the model loaded afterwards (e.g. 5m, 1h, 0, -1).
    #[arg(long, value_name = "DURATION", value_parser = parse_keep_alive, allow_hyphen_values = true)]
    pub keepalive: Option<KeepAlive>,
    /// Show every option's probability, the routing and timings.
    #[arg(long)]
    pub verbose: bool,
    /// Parse the state as JSON (an object or array). JSON input is otherwise detected.
    #[arg(long)]
    pub state_json: bool,
}

fn parse_keep_alive(s: &str) -> Result<KeepAlive, String> {
    KeepAlive::parse(s).map_err(|e| e.to_string())
}

/// The state to send: JSON when asked for (or when the text is a JSON object or array),
/// otherwise the text itself.
pub fn parse_state(text: &str, force_json: bool) -> Result<Value> {
    let trimmed = text.trim();
    if force_json {
        return serde_json::from_str(trimmed).context("--state-json: the state is not valid JSON");
    }
    if trimmed.starts_with(['{', '['])
        && let Ok(v @ (Value::Object(_) | Value::Array(_))) = serde_json::from_str(trimmed)
    {
        return Ok(v);
    }
    Ok(Value::String(
        text.trim_end_matches(['\n', '\r']).to_owned(),
    ))
}

fn load_questions(path: &Path) -> Result<Questions> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&text)
        .with_context(|| format!("{}: not a question schema", path.display()))
}

fn preset(name: &str) -> Result<Questions> {
    let Some(q) = presets::get(name) else {
        bail!(
            "unknown preset {name:?}; choose one of {}",
            presets::NAMES.join(", ")
        );
    };
    Ok(serde_json::from_value(q)?)
}

/// The preset `ollaya run` uses for a model without built-in questions when neither
/// `--questions` nor `--preset` is given, so the first command people try, `ollaya run laya`,
/// answers something instead of stopping.
const DEFAULT_PRESET: &str = "triage";

/// Where the questions come from: explicit flags first, then the model's built-in set, then
/// [`DEFAULT_PRESET`]. `None` means "use the model's own".
fn choose_questions(args: &RunArgs, has_builtin: bool) -> Result<Option<Questions>> {
    if let Some(path) = &args.questions {
        return load_questions(path).map(Some);
    }
    if let Some(name) = &args.preset {
        return preset(name).map(Some);
    }
    if has_builtin {
        return Ok(None);
    }
    preset(DEFAULT_PRESET).map(Some)
}

/// Whether [`choose_questions`] fell back to [`DEFAULT_PRESET`].
fn uses_default_preset(args: &RunArgs, has_builtin: bool) -> bool {
    args.questions.is_none() && args.preset.is_none() && !has_builtin
}

struct Session {
    client: Client,
    model: String,
    questions: Option<Questions>,
    keep_alive: Option<KeepAlive>,
    format: Format,
    verbose: bool,
    state_json: bool,
}

impl Session {
    async fn decide(&self, state: Value) -> Result<DecideResponse> {
        let mut req = DecideRequest::new(&self.model, state, self.questions.clone());
        req.keep_alive = self.keep_alive;
        Ok(self.client.decide(&req).await?)
    }

    fn print(&self, r: &DecideResponse) -> Result<()> {
        if self.format == Format::Json {
            println!("{}", serde_json::to_string_pretty(r)?);
            return Ok(());
        }
        print!(
            "{}",
            render::answers(&serde_json::to_value(&r.answers)?, self.verbose)
        );
        if self.verbose {
            let routed = r
                .routing
                .as_ref()
                .map(|x| format!(" (routed from {}: {})", x.router, x.reason))
                .unwrap_or_default();
            eprintln!("\nmodel:           {}{routed}", r.model);
            eprintln!("total duration:  {}", render::nanos(r.total_duration));
            eprintln!("load duration:   {}", render::nanos(r.load_duration));
            eprintln!("eval duration:   {}", render::nanos(r.eval_duration));
            eprintln!("input tokens:    {}", r.usage.input_tokens);
            if r.state_truncated {
                eprintln!("note:            the state was truncated to fit the model's context");
            }
        }
        Ok(())
    }
}

pub fn run(rt: &Runtime, args: RunArgs) -> Result<()> {
    let session = rt.block_on(prepare(&args))?;
    let piped = !std::io::stdin().is_terminal();
    let state = if !args.state.is_empty() {
        Some(args.state.join(" "))
    } else if piped {
        let mut s = String::new();
        std::io::stdin()
            .read_to_string(&mut s)
            .context("reading the state from stdin")?;
        Some(s)
    } else {
        None
    };
    match state {
        Some(text) => {
            let state = parse_state(&text, args.state_json)?;
            let r = rt.block_on(session.decide(state))?;
            session.print(&r)
        }
        None => repl(rt, session),
    }
}

/// Connect (starting the daemon if needed), pull the model if it is missing, load it.
async fn prepare(args: &RunArgs) -> Result<Session> {
    let client = daemon::client().await?;
    let show = match client.show(&args.model).await {
        Ok(s) => s,
        Err(e) if e.is_model_not_found() => {
            commands::pull(&client, &args.model, false).await?;
            client.show(&args.model).await?
        }
        Err(e) => return Err(e.into()),
    };
    let questions = choose_questions(args, show.questions.is_some())?;
    if uses_default_preset(args, show.questions.is_some()) {
        eprintln!(
            "{} has no built-in questions, so it answers the {DEFAULT_PRESET} preset. \
             Choose with --preset NAME ({}) or --questions FILE.",
            args.model,
            presets::NAMES.join(", ")
        );
    }
    let spinner = if std::io::stderr().is_terminal() {
        let s = ProgressBar::new_spinner();
        s.enable_steady_tick(Duration::from_millis(100));
        Some(s)
    } else {
        None
    };
    let loaded = client.load(&args.model, args.keepalive).await;
    if let Some(s) = spinner {
        s.finish_and_clear();
    }
    loaded?;
    Ok(Session {
        client,
        model: args.model.clone(),
        questions,
        keep_alive: args.keepalive,
        format: args.format,
        verbose: args.verbose,
        state_json: args.state_json,
    })
}

const HELP: &str = "Available commands:
  /set questions <file>   Use the questions in a JSON file
  /preset <name>          Use a built-in question set (triage, email, guard, moderation, router, agent)
  /show                   Show the model and the current questions
  /clear                  Clear the screen
  /bye                    Exit
  /?, /help               Help for a command

Enter a state to decide about. Use \"\"\" to begin and end a multi-line state.
";

enum Command {
    Help,
    Bye,
    Clear,
    Show,
    SetQuestions(PathBuf),
    Preset(String),
    Unknown(String),
}

fn parse_command(line: &str) -> Option<Command> {
    let line = line.trim();
    if !line.starts_with('/') {
        return None;
    }
    let mut words = line.split_whitespace();
    Some(
        match (words.next().unwrap_or_default(), words.next(), words.next()) {
            ("/?" | "/help", _, _) => Command::Help,
            ("/bye" | "/exit", _, _) => Command::Bye,
            ("/clear", _, _) => Command::Clear,
            ("/show", _, _) => Command::Show,
            ("/set", Some("questions"), Some(file)) => Command::SetQuestions(PathBuf::from(file)),
            ("/preset", Some(name), _) => Command::Preset(name.to_owned()),
            (other, _, _) => Command::Unknown(other.to_owned()),
        },
    )
}

fn describe(session: &Session) -> String {
    let mut out = format!("model: {}\n", session.model);
    match &session.questions {
        None => out.push_str("questions: the model's built-in set\n"),
        Some(q) => {
            out.push_str("questions:\n");
            for (id, q) in q {
                out.push_str(&format!("  {id}: {}\n", q.type_name()));
            }
        }
    }
    out
}

fn repl(rt: &Runtime, mut session: Session) -> Result<()> {
    let mut editor = DefaultEditor::new()?;
    let history = daemon::home().join("history");
    if let Some(dir) = history.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let _ = editor.load_history(&history);
    eprintln!("Enter a state to decide about (/? for help, \"\"\" for several lines).");
    let mut block: Option<String> = None;
    loop {
        let prompt = if block.is_some() { "... " } else { ">>> " };
        let line = match editor.readline(prompt) {
            Ok(l) => l,
            Err(ReadlineError::Interrupted) => {
                block = None;
                eprintln!("Use Ctrl + d or /bye to exit.");
                continue;
            }
            Err(ReadlineError::Eof) => break,
            Err(e) => return Err(e.into()),
        };
        let text = match block.take() {
            Some(mut buf) => {
                if let Some(end) = line.find("\"\"\"") {
                    buf.push_str(&line[..end]);
                    buf
                } else {
                    buf.push_str(&line);
                    buf.push('\n');
                    block = Some(buf);
                    continue;
                }
            }
            None => {
                if let Some(rest) = line.trim_start().strip_prefix("\"\"\"") {
                    match rest.find("\"\"\"") {
                        Some(end) => rest[..end].to_owned(),
                        None => {
                            block = Some(format!("{rest}\n"));
                            continue;
                        }
                    }
                } else {
                    line.clone()
                }
            }
        };
        if text.trim().is_empty() {
            continue;
        }
        let _ = editor.add_history_entry(text.as_str());
        match parse_command(&text) {
            Some(Command::Bye) => break,
            Some(Command::Help) => eprint!("{HELP}"),
            Some(Command::Clear) => print!("\x1b[2J\x1b[H"),
            Some(Command::Show) => eprint!("{}", describe(&session)),
            Some(Command::SetQuestions(path)) => match load_questions(&path) {
                Ok(q) => {
                    session.questions = Some(q);
                    eprintln!("Set questions from {}.", path.display());
                }
                Err(e) => eprintln!("error: {e:#}"),
            },
            Some(Command::Preset(name)) => match preset(&name) {
                Ok(q) => {
                    session.questions = Some(q);
                    eprintln!("Set questions to the {name} preset.");
                }
                Err(e) => eprintln!("error: {e:#}"),
            },
            Some(Command::Unknown(c)) => eprintln!("Unknown command '{c}'. Type /? for help"),
            None => {
                let result = parse_state(&text, session.state_json)
                    .and_then(|state| rt.block_on(session.decide(state)))
                    .and_then(|r| session.print(&r));
                if let Err(e) = result {
                    eprintln!("error: {e:#}");
                }
            }
        }
    }
    let _ = editor.save_history(&history);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn states() {
        assert_eq!(parse_state("hello\n", false).unwrap(), json!("hello"));
        assert_eq!(parse_state(r#"{"a": 1}"#, false).unwrap(), json!({"a": 1}));
        assert_eq!(
            parse_state("[urgent] help", false).unwrap(),
            json!("[urgent] help")
        );
        assert_eq!(parse_state("[1, 2]", false).unwrap(), json!([1, 2]));
        assert!(parse_state("not json", true).is_err());
    }

    #[test]
    fn repl_commands() {
        assert!(matches!(parse_command("/bye"), Some(Command::Bye)));
        assert!(matches!(parse_command(" /? "), Some(Command::Help)));
        assert!(
            matches!(parse_command("/set questions q.json"), Some(Command::SetQuestions(p)) if p == Path::new("q.json"))
        );
        assert!(matches!(parse_command("/preset guard"), Some(Command::Preset(n)) if n == "guard"));
        assert!(matches!(parse_command("/nope"), Some(Command::Unknown(_))));
        assert!(parse_command("a state").is_none());
    }

    #[test]
    fn question_sources() {
        let args = |questions: Option<&str>, preset: Option<&str>| RunArgs {
            model: "m".into(),
            state: vec![],
            questions: questions.map(PathBuf::from),
            preset: preset.map(str::to_owned),
            format: Format::Text,
            keepalive: None,
            verbose: false,
            state_json: false,
        };
        assert!(choose_questions(&args(None, None), true).unwrap().is_none());
        assert!(
            choose_questions(&args(None, Some("triage")), true)
                .unwrap()
                .is_some()
        );
        // No flags and no built-in questions: the default preset, and the caller says so.
        let fallback = choose_questions(&args(None, None), false).unwrap().unwrap();
        assert_eq!(
            serde_json::to_value(&fallback).unwrap(),
            serde_json::to_value(preset(DEFAULT_PRESET).unwrap()).unwrap()
        );
        assert!(uses_default_preset(&args(None, None), false));
        assert!(!uses_default_preset(&args(None, None), true));
        assert!(!uses_default_preset(&args(None, Some("guard")), false));
    }
}
