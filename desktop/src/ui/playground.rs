//! Playground: a model, a preset (or the model's own questions, or custom ones), a text or JSON
//! to decide about, and the answers, one per question with how sure the model is.

use ollaya_api::{Answer, DecideResponse};
use rune::{El, Key, ScrollSpec, Tx, col, el, row, spacer, text, text_input};
use rune_kit::{Placement, Select};

use super::*;
use crate::state::{CUSTOM, Picker};

/// The questions picker's entry for the Custom box.
const CUSTOM_LABEL: &str = "Custom JSON";

/// A pane's header: "State", "Questions", "Answers".
fn pane_head(label: &'static str, trailing: Option<El>) -> El {
    let mut r = row()
        .w_full()
        .h(32.0)
        .px(16.0)
        .items_center()
        .child(text(label).font(S_HEAD).color(MUTED))
        .child(spacer());
    if let Some(t) = trailing {
        r = r.child(t);
    }
    r
}

/// The toolbar (in the window's top band) and the page.
pub fn page(s: &State, layer: &mut Vec<El>) -> (Option<El>, El) {
    if !s.snap.running {
        return (None, stopped(s, "Start it to run a model"));
    }
    let tags = s.installed();
    if tags.is_empty() {
        return (
            None,
            empty(
                "No models yet",
                "Download a model first",
                Some(discover_button()),
                None,
            ),
        );
    }
    (Some(toolbar(s, &tags, layer)), body(s))
}

fn toolbar(s: &State, tags: &[String], layer: &mut Vec<El>) -> El {
    let models = Select::new("model-picker", tags)
        .label("Model")
        .current(Some(&s.run_model))
        .open(s.picker == Some(Picker::Model))
        .placement(Placement::BottomEnd)
        .on_toggle(|s: &mut State| toggle(s, Picker::Model))
        .on_close(|s: &mut State| s.picker = None)
        .on_choose(|s: &mut State, v: &str, _: &mut Tx| s.choose_model(v))
        .build(&s.kit, layer);
    let mut bar = row()
        .gap(8.0)
        .items_center()
        .child(text("Model").font(S_SMALL).color(MUTED))
        .child(models);
    // A model with built-in questions takes no preset or custom questions. Until the server
    // says which kind this version is, neither the picker nor the note shows.
    match s.own_questions() {
        Some(Some(ids)) => {
            bar = bar.child(
                row()
                    .pl(8.0)
                    .gap(4.0)
                    .items_center()
                    .child(text("Asks its own questions:").font(S_SMALL).color(MUTED))
                    .child(text(ids.join(", ")).font(S_MONO).color(BODY)),
            );
        }
        Some(None) => {
            let mut names: Vec<&str> = s.presets.iter().map(|p| p.name.as_str()).collect();
            names.push(CUSTOM_LABEL);
            let details: Vec<Option<&str>> = s
                .presets
                .iter()
                .map(|p| (!p.builtin).then_some("yours"))
                .chain([None])
                .collect();
            let questions = Select::new("questions-picker", &names)
                .details(&details)
                .label("Questions")
                .current(Some(if s.preset == CUSTOM {
                    CUSTOM_LABEL
                } else {
                    &s.preset
                }))
                .open(s.picker == Some(Picker::Questions))
                .placement(Placement::BottomEnd)
                .on_toggle(|s: &mut State| toggle(s, Picker::Questions))
                .on_close(|s: &mut State| s.picker = None)
                .on_choose(|s: &mut State, v: &str, _: &mut Tx| {
                    s.choose_preset(if v == CUSTOM_LABEL { CUSTOM } else { v })
                })
                .build(&s.kit, layer);
            bar = bar
                .child(el().w(8.0))
                .child(text("Questions").font(S_SMALL).color(MUTED))
                .child(questions);
        }
        None => {}
    }
    let can = s.can_run();
    let hint = if cfg!(target_os = "macos") {
        "⌘ Return"
    } else {
        "Ctrl + Enter"
    };
    bar.child(el().w(8.0))
        .child(text(hint).font(S_META).color(FAINT))
        .child(on_press(
            icon_button(
                "run",
                &PLAY,
                if s.running.is_some() {
                    "Running…"
                } else {
                    "Run"
                },
                true,
                can,
            ),
            can,
            |s: &mut State| s.run(),
        ))
}

fn toggle(s: &mut State, p: Picker) {
    s.picker = if s.picker == Some(p) { None } else { Some(p) };
}

/// ⌘ Return (Ctrl + Enter off macOS) runs from either text box.
fn run_key(s: &mut State, k: &rune::KeyEvent, _: &mut Tx) -> bool {
    if k.key == Key::Enter && k.mods.cmd {
        s.run();
        true
    } else {
        false
    }
}

fn body(s: &State) -> El {
    let state_box = text_input(&s.input, S_BODY.lh(20.0))
        .key("state")
        .label("State")
        .placeholder("Paste a message, an email or a JSON object…", MUTED)
        .caret(FG, FILL_STRONG)
        .color(FG)
        .on_edit(|s: &mut State| &mut s.input)
        // Return breaks the line, as in the Tauri app's text area; ⌘ Return runs.
        .on_submit(|s: &mut State, _: &mut Tx| s.input.insert("\n"))
        .on_input_key(run_key)
        .w_full();
    let example = row()
        .key("example")
        .role("button")
        .label("Use an example")
        .focusable(true)
        .on_press(|s: &mut State| s.use_example())
        .child(text("Use an example").font(S_META).color(MUTED));
    let mut input = col()
        .w(0.0)
        .grow(1.0)
        .h_full()
        .child(pane_head("State", Some(example)))
        .child(hairline())
        .child(
            col()
                .key("state-pane")
                .w_full()
                .grow(1.0)
                .min_h(0.0)
                .padding(16.0)
                .scroll(ScrollSpec::default())
                .on_press_tx(|_: &mut State, tx: &mut Tx| tx.focus("state", false))
                .child(state_box),
        );
    if matches!(s.own_questions(), Some(None)) && s.preset == CUSTOM {
        let questions = text_input(&s.custom, S_MONO.lh(18.0))
            .key("questions")
            .label("Questions as JSON")
            .caret(FG, FILL_STRONG)
            .color(FG)
            .on_edit(|s: &mut State| &mut s.custom)
            .on_submit(|s: &mut State, _: &mut Tx| s.custom.insert("\n"))
            .on_input_key(run_key)
            .w_full();
        input = input
            .child(hairline())
            .child(pane_head("Questions", None))
            .child(hairline())
            .child(
                col()
                    .key("questions-pane")
                    .w_full()
                    .h(176.0)
                    .padding(16.0)
                    .scroll(ScrollSpec::default())
                    .on_press_tx(|_: &mut State, tx: &mut Tx| tx.focus("questions", false))
                    .child(questions),
            );
    }
    let answers: El = if let Some(err) = &s.run_error {
        col()
            .w_full()
            .padding(16.0)
            .child(text(err.clone()).font(S_SMALL).color(BAD).wrap())
    } else if let Some(r) = &s.result {
        results(s, r)
    } else {
        col().w_full().grow(1.0).center().px(24.0).child(
            text(if s.running.is_some() {
                "Running…"
            } else {
                "The answers appear here: one per question, with how sure the model is"
            })
            .font(S_BODY)
            .color(MUTED)
            .wrap(),
        )
    };
    let out = col()
        .w(0.0)
        .grow(1.0)
        .h_full()
        .child(pane_head("Answers", None))
        .child(hairline())
        .child(
            col()
                .key("answers")
                .w_full()
                .grow(1.0)
                .min_h(0.0)
                .scroll(ScrollSpec::default())
                .child(answers),
        );
    row()
        .w_full()
        .grow(1.0)
        .min_h(0.0)
        .child(input)
        .child(el().w(1.0).h_full().fill(LINE))
        .child(out)
}

/// One row per answer: the question, the answer, a meter of ten segments and its probability.
fn results(s: &State, r: &DecideResponse) -> El {
    // qwen3guard's `category` is the harm it would name if it judged the text unsafe, whatever
    // the text (docs/families/qwen3guard.md), so it means little while `unsafe` is low.
    let unsafe_p = r.answers.get("unsafe").and_then(|a| match &a.answer {
        Answer::Noul(n) => Some(n.noul),
        _ => None,
    });
    let guard = unsafe_p.is_some()
        && r.answers.contains_key("category")
        && matches!(s.own_questions(), Some(Some(ids)) if ids.iter().any(|i| i == "category"));
    let quiet_category = guard && unsafe_p.unwrap_or(0.0) < 0.5;
    let mut list = col().w_full();
    for (q, a) in &r.answers {
        let (answer, note, p) = describe(&a.answer);
        let lit = (p * 10.0).round() as usize;
        let mut meter = row().w(0.0).grow(1.0).gap(3.0).items_center();
        for i in 0..10 {
            meter = meter.child(el().w(0.0).grow(1.0).h(6.0).radius(1.0).fill(if i < lit {
                OK
            } else {
                FILL_STRONG
            }));
        }
        let mut said = row()
            .w(160.0)
            .gap(6.0)
            .items_center()
            .child(text(answer.clone()).font(S_ANSWER).color(FG));
        if !note.is_empty() {
            said = said.child(
                text(note.clone())
                    .font(S_SMALL)
                    .color(MUTED)
                    .shrink()
                    .truncate(),
            );
        }
        let said = said.help(if note.is_empty() {
            answer
        } else {
            format!("{answer}: {note}")
        });
        list = list.child(
            col()
                .key(format!("answer-{q}"))
                .w_full()
                .px(16.0)
                .py(10.0)
                .gap(4.0)
                .border_sides([0.0, 0.0, 1.0, 0.0], LINE)
                .opacity(if quiet_category && q == "category" {
                    0.5
                } else {
                    1.0
                })
                .child(text(q.clone()).font(S_MONO).color(MUTED))
                .child(
                    row()
                        .w_full()
                        .gap(12.0)
                        .items_center()
                        .child(said)
                        .child(meter)
                        .child(right(
                            36.0,
                            text(format!("{p:.2}")).font(S_SMALL_NUM).color(FG),
                        )),
                ),
        );
    }
    let by = r.routing.as_ref().map_or(&r.model, |r| &r.model);
    list = list.child(
        row()
            .w_full()
            .px(16.0)
            .py(10.0)
            .gap(4.0)
            .items_center()
            .flex_wrap()
            .child(text("Answered by").font(S_META).color(MUTED))
            .child(text(by.clone()).font(S_MONO_META).color(BODY))
            .child(text("in").font(S_META).color(MUTED))
            .child(
                text(format!("{} ms", ms(r.total_duration as f64 / 1e6)))
                    .font(S_META_NUM)
                    .color(FG),
            ),
    );
    if guard {
        list = list.child(
            text(
                "category is the kind of harm the model would name if it judged the text unsafe. \
                 It only means something when unsafe is high.",
            )
            .font(S_META)
            .color(MUTED)
            .px(16.0)
            .pb(12.0)
            .wrap(),
        );
    }
    list
}

/// An answer in words, a note beside it, and how sure the model is (0 to 1).
fn describe(a: &Answer) -> (String, String, f64) {
    match a {
        Answer::Choice(c) => (c.choice.clone(), String::new(), c.confidence),
        Answer::Score(sc) => {
            let levels = sc.legend.len().saturating_sub(1);
            let near = sc
                .legend
                .get(&(sc.score.round() as i64).to_string())
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_owned();
            (format!("{:.2} / {levels}", sc.score), near, sc.confidence)
        }
        Answer::Noul(n) => {
            let yes = n.noul >= 0.5;
            (
                if yes { "yes" } else { "no" }.into(),
                String::new(),
                if yes { n.noul } else { 1.0 - n.noul },
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn answer(json: serde_json::Value) -> Answer {
        serde_json::from_value(json).unwrap()
    }

    #[test]
    fn answers_read_as_the_tauri_app_writes_them() {
        let choice = answer(serde_json::json!({
            "type": "choice", "choice": "refund", "confidence": 0.81, "probabilities": {}
        }));
        assert_eq!(describe(&choice), ("refund".into(), String::new(), 0.81));
        let score = answer(serde_json::json!({
            "type": "score", "score": 1.7, "confidence": 0.6, "probabilities": {},
            "legend": {"0": "calm", "1": "annoyed", "2": "angry"}
        }));
        assert_eq!(describe(&score), ("1.70 / 2".into(), "angry".into(), 0.6));
        let no = answer(serde_json::json!({"type": "noul", "noul": 0.2}));
        assert_eq!(describe(&no), ("no".into(), String::new(), 0.8));
    }
}
