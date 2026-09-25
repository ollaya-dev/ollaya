//! `ollaya`: run open decision models locally.
//!
//! One binary is the CLI, the daemon (`ollaya serve`) and the per-model runner the daemon spawns
//! (`ollaya runner`, hidden). The CLI and the daemon never load ONNX Runtime themselves: only
//! runner processes do, so a GPU provider is never registered in-process here.

mod commands;
mod daemon;
mod mcp;
use ollaya_api::presets;
mod modelfile;
mod render;
mod run;

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};
use ollaya_server::config::VERSION;

#[derive(Parser)]
#[command(
    name = "ollaya",
    about = "Run open decision models locally",
    disable_version_flag = true,
    arg_required_else_help = true
)]
struct Cli {
    /// Show version information
    #[arg(short = 'v', long = "version")]
    version: bool,
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Start ollaya
    Serve,
    /// Run a model: answer questions about a state, or start a REPL
    Run(run::RunArgs),
    /// Pull a model from a registry
    Pull {
        model: String,
        /// Accepted for Ollama compatibility; use an http:// host in the name for dev registries
        #[arg(long)]
        insecure: bool,
    },
    /// List models
    #[command(visible_alias = "ls")]
    List,
    /// List running models
    Ps,
    /// Show information for a model
    Show {
        model: String,
        /// Show the model's built-in questions
        #[arg(long)]
        questions: bool,
        /// Show the license
        #[arg(long)]
        license: bool,
        /// Show a Modelfile that recreates the model
        #[arg(long)]
        modelfile: bool,
        /// Show the parameters
        #[arg(long)]
        parameters: bool,
    },
    /// Remove a model
    Rm {
        #[arg(required = true)]
        models: Vec<String>,
    },
    /// Copy a model
    Cp { source: String, destination: String },
    /// Stop a running model, or without one the server (which unloads every model)
    Stop { model: Option<String> },
    /// Serve the local models to AI agents over the Model Context Protocol
    Mcp {
        /// Serve streamable HTTP at ADDR/mcp instead of stdio (default address 127.0.0.1:11436)
        #[arg(long, value_name = "ADDR", num_args = 0..=1, default_missing_value = "127.0.0.1:11436")]
        http: Option<String>,
    },
    /// Create a model from a Modelfile
    Create {
        name: String,
        /// Name of the Modelfile
        #[arg(short = 'f', long = "file", default_value = "Modelfile")]
        file: PathBuf,
    },
    /// Internal: serve one loaded model for the daemon.
    #[command(hide = true)]
    Runner {
        #[arg(long)]
        graph_fp32: Option<PathBuf>,
        #[arg(long)]
        graph_fp16: Option<PathBuf>,
        #[arg(long)]
        tokenizer: PathBuf,
        #[arg(long)]
        decision: PathBuf,
        /// The arch layer, for the MLX engine
        #[arg(long)]
        arch: Option<PathBuf>,
        /// The weights file, for the MLX engine
        #[arg(long)]
        weights: Option<PathBuf>,
        /// auto, cpu, cuda, cuda:<n> or metal
        #[arg(long, default_value = "auto")]
        device: ollaya_runner::server::DeviceRequest,
        #[arg(long)]
        threads: Option<usize>,
    },
}

fn logging(default: &str) {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env("OLLAYA_LOG")
                .unwrap_or_else(|_| default.into()),
        )
        .with_writer(std::io::stderr)
        .init();
}

/// `ollaya -v`: the server's version, and the client's when they differ (as Ollama prints it).
async fn version() {
    let server = match ollaya_api::Client::from_env() {
        Ok(c) => c.version().await.ok().map(|v| v.version),
        Err(_) => None,
    };
    match server {
        Some(v) => {
            println!("ollaya version is {v}");
            if v != VERSION {
                eprintln!("Warning: client version is {VERSION}");
            }
        }
        None => {
            eprintln!("Warning: could not connect to a running Ollaya instance");
            eprintln!("Warning: client version is {VERSION}");
        }
    }
}

fn main() -> Result<()> {
    // SAFETY: the first statement of `main`; no other thread exists yet.
    unsafe { ollaya_runner::prepare_process() };
    let cli = Cli::parse();
    let rt = tokio::runtime::Runtime::new()?;
    if cli.version {
        rt.block_on(version());
        return Ok(());
    }
    let Some(command) = cli.command else {
        return Ok(());
    };
    match command {
        Command::Runner {
            graph_fp32,
            graph_fp16,
            tokenizer,
            decision,
            arch,
            weights,
            device,
            threads,
        } => {
            logging("info,ort=warn");
            rt.block_on(ollaya_runner::server::run(
                ollaya_runner::server::RunnerConfig {
                    graph_fp32,
                    graph_fp16,
                    tokenizer,
                    decision,
                    arch,
                    weights,
                    device,
                    threads,
                },
            ))?;
        }
        Command::Serve => {
            logging("info");
            let config = ollaya_server::config::ServerConfig::from_env()?;
            rt.block_on(daemon::serve(config))?;
        }
        Command::Mcp { http } => {
            // stdout is the MCP channel over stdio: logs go to stderr, and only warnings.
            logging("warn");
            match http {
                Some(addr) => rt.block_on(mcp::serve_http(&addr))?,
                None => rt.block_on(mcp::serve_stdio())?,
            }
        }
        // Without a model, `stop` must not start a server just to stop it.
        Command::Stop { model: None } => rt.block_on(daemon::stop_server())?,
        Command::Run(args) => run::run(&rt, args)?,
        command => rt.block_on(client_command(command))?,
    }
    Ok(())
}

async fn client_command(command: Command) -> Result<()> {
    let client = daemon::client().await?;
    match command {
        Command::Pull { model, insecure } => commands::pull(&client, &model, insecure).await,
        Command::List => commands::list(&client).await,
        Command::Ps => commands::ps(&client).await,
        Command::Show {
            model,
            questions,
            license,
            modelfile,
            parameters,
        } => {
            let only = commands::ShowOnly {
                questions,
                license,
                modelfile,
                parameters,
            };
            commands::show(&client, &model, only).await
        }
        Command::Rm { models } => commands::rm(&client, &models).await,
        Command::Cp {
            source,
            destination,
        } => commands::cp(&client, &source, &destination).await,
        Command::Stop { model: Some(model) } => commands::stop(&client, &model).await,
        Command::Create { name, file } => commands::create(&client, &name, &file).await,
        Command::Serve
        | Command::Run(_)
        | Command::Runner { .. }
        | Command::Mcp { .. }
        | Command::Stop { model: None } => {
            unreachable!("handled in main")
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    fn parse(args: &[&str]) -> Cli {
        Cli::try_parse_from(std::iter::once("ollaya").chain(args.iter().copied()))
            .unwrap_or_else(|e| panic!("{args:?}: {e}"))
    }

    #[test]
    fn run_arguments() {
        let Some(Command::Run(r)) = parse(&[
            "run",
            "laya",
            "--preset",
            "triage",
            "--format",
            "json",
            "--keepalive",
            "10m",
            "--verbose",
            "I",
            "was",
            "charged",
            "twice",
        ])
        .command
        else {
            panic!()
        };
        assert_eq!(r.model, "laya");
        assert_eq!(r.state.join(" "), "I was charged twice");
        assert_eq!(r.preset.as_deref(), Some("triage"));
        assert_eq!(r.format, run::Format::Json);
        assert_eq!(
            r.keepalive,
            Some(ollaya_api::KeepAlive::For(std::time::Duration::from_secs(
                600
            )))
        );
        assert!(r.verbose && !r.state_json);
        let Some(Command::Run(r)) = parse(&[
            "run",
            "laya",
            "--questions",
            "q.json",
            "--state-json",
            "--keepalive",
            "-1",
        ])
        .command
        else {
            panic!()
        };
        assert_eq!(r.questions, Some(PathBuf::from("q.json")));
        assert_eq!(r.keepalive, Some(ollaya_api::KeepAlive::Forever));
        assert!(r.state.is_empty() && r.state_json);
    }

    #[test]
    fn other_subcommands() {
        assert!(matches!(parse(&["ls"]).command, Some(Command::List)));
        assert!(matches!(parse(&["list"]).command, Some(Command::List)));
        assert!(matches!(parse(&["ps"]).command, Some(Command::Ps)));
        assert!(matches!(parse(&["serve"]).command, Some(Command::Serve)));
        assert!(matches!(
            parse(&["pull", "laya"]).command,
            Some(Command::Pull { .. })
        ));
        assert!(
            matches!(parse(&["rm", "a", "b"]).command, Some(Command::Rm { models }) if models.len() == 2)
        );
        assert!(matches!(
            parse(&["cp", "a", "b"]).command,
            Some(Command::Cp { .. })
        ));
        assert!(matches!(
            parse(&["stop", "laya"]).command,
            Some(Command::Stop { model: Some(m) }) if m == "laya"
        ));
        assert!(matches!(
            parse(&["stop"]).command,
            Some(Command::Stop { model: None })
        ));
        assert!(matches!(
            parse(&["show", "laya", "--license"]).command,
            Some(Command::Show { license: true, .. })
        ));
        assert!(matches!(
            parse(&["create", "triage", "-f", "M"]).command,
            Some(Command::Create { file, .. }) if file == Path::new("M")
        ));
        assert!(matches!(
            parse(&["create", "triage"]).command,
            Some(Command::Create { file, .. }) if file == Path::new("Modelfile")
        ));
        assert!(parse(&["-v"]).version);
    }

    #[test]
    fn rejects_bad_arguments() {
        let bad = |args: &[&str]| {
            Cli::try_parse_from(std::iter::once("ollaya").chain(args.iter().copied())).is_err()
        };
        assert!(bad(&["run"]));
        assert!(bad(&["run", "laya", "--preset", "nope"]));
        assert!(bad(&["run", "laya", "--keepalive", "soon"]));
        assert!(bad(&["run", "laya", "--format", "yaml"]));
        assert!(bad(&["rm"]));
        assert!(bad(&["frobnicate"]));
    }
}
