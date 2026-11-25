//! Server: how much the server has done and how fast it answers, where apps reach it, and the
//! requests it answered last.

use rune::{El, Family, TextStyle, Tx, el, row, text};

use super::*;
use crate::backend::Snapshot;

/// The endpoints a local app calls (the Tauri app's server page).
const ENDPOINTS: [(&str, &str, &str); 3] = [
    (
        "POST",
        "/api/decide",
        "Answers questions about a text or JSON, with a preset or your own questions",
    ),
    (
        "POST",
        "/v1/decisions",
        "TypeSafe-compatible: its SDKs work unchanged (also /v1/systemone)",
    ),
    ("GET", "/v1/models", "The models on this computer"),
];

pub fn page(s: &State) -> El {
    let snap = &s.snap;
    if !snap.running {
        return stopped(s, "Start it to run models and answer apps on this computer");
    }
    let (outer, inner) = super::page("server");
    outer.child(
        inner
            .child(el().h(24.0))
            .child(figures(snap))
            .child(section(
                "API",
                Some(link("reference", "Reference", true).on_press_tx(
                    |_: &mut State, tx: &mut Tx| {
                        let _ = tx.open_url("https://ollaya.dev/docs/api");
                    },
                )),
            ))
            .child(api(s))
            .child(section("Recent requests", None))
            .child(recent(snap)),
    )
}

/// Four figures: how much the server has done and how fast it answers.
fn figures(snap: &Snapshot) -> El {
    let sp = snap.speed;
    let cell = |first: bool, value: String, what: &'static str, note: String| {
        let c = rune::col()
            .w(0.0)
            .grow(1.0)
            .px(16.0)
            .py(12.0)
            .gap(2.0)
            .child(text(value).font(S_FIGURE).color(FG))
            .child(text(what).font(S_SMALL).color(BODY))
            .child(text(note).font(S_META).color(MUTED));
        if first {
            c
        } else {
            c.border_sides([0.0, 0.0, 0.0, 1.0], LINE)
        }
    };
    let none = || "–".to_owned();
    row()
        .w_full()
        .radius(8.0)
        .clip_rounded(8.0)
        .fill(PANEL)
        .border(1.0, LINE)
        .child(cell(
            true,
            if snap.activity {
                thousands(snap.requests)
            } else {
                none()
            },
            "Requests",
            "since the server started".into(),
        ))
        .child(cell(
            false,
            if sp.samples == 0 && snap.recent.is_empty() {
                none()
            } else if sp.per_second < 10.0 {
                format!("{:.1}", sp.per_second)
            } else {
                format!("{:.0}", sp.per_second)
            },
            "Per second",
            "over the last minute".into(),
        ))
        .child(cell(
            false,
            sp.median_ms.map_or_else(none, |m| format!("{} ms", ms(m))),
            "Response time",
            if sp.samples > 0 {
                format!("median of the last {}", sp.samples)
            } else {
                "no requests yet".into()
            },
        ))
        .child(cell(
            false,
            sp.median_ms.map_or_else(none, |m| {
                format!("{}/s", thousands((1000.0 / m.max(0.001)).floor() as u64))
            }),
            "Capacity",
            "one request at a time".into(),
        ))
}

/// Where apps reach the server: its address, the endpoints, and a request to try.
fn api(s: &State) -> El {
    let url = &s.snap.url;
    let model = s
        .snap
        .models
        .first()
        .map_or("kev", |m| m.name.trim_end_matches(":latest"));
    let example = format!(
        r#"curl {url}/api/decide -d '{{"model":"{model}","preset":"triage","state":"I was charged twice this month."}}'"#
    );
    let mut p = panel().child(
        panel_row(true)
            .h(40.0)
            .child(text("Address").font(S_SMALL).color(MUTED).w(44.0))
            .child(text(url.clone()).font(S_MONO).color(FG).grow(1.0))
            .child(copy_button(
                s,
                "copy-address".into(),
                url.clone(),
                "Copy the address",
            )),
    );
    for (method, path, what) in ENDPOINTS {
        p = p.child(
            panel_row(false)
                .h(40.0)
                .child(
                    text(method)
                        .font(TextStyle::new(11.0, 500.0).family(Family::SystemMono))
                        .color(MUTED)
                        .w(44.0),
                )
                .child(text(path).font(S_MONO).color(FG).w(128.0))
                .child(
                    text(what)
                        .font(S_SMALL)
                        .color(MUTED)
                        .w(0.0)
                        .grow(1.0)
                        .truncate(),
                )
                .child(copy_button(
                    s,
                    format!("copy-{path}"),
                    format!("{url}{path}"),
                    &format!("Copy {path}"),
                )),
        );
    }
    p.child(
        panel_row(false)
            .py(8.0)
            .items_start()
            .fill(CODE)
            .child(
                text(example.clone())
                    .font(S_MONO.lh(20.0))
                    .color(BODY)
                    .w(0.0)
                    .grow(1.0)
                    .pt(4.0)
                    .wrap_anywhere(),
            )
            .child(copy_button(
                s,
                "copy-example".into(),
                example,
                "Copy the example",
            )),
    )
}

/// The latest requests, newest first.
fn recent(snap: &Snapshot) -> El {
    if !snap.activity {
        return text("This server does not list its requests. Update Ollaya to 0.17.0 or newer")
            .font(S_SMALL)
            .color(MUTED);
    }
    if snap.recent.is_empty() {
        return text("No requests yet. Try a model, or call the API from your code")
            .font(S_SMALL)
            .color(MUTED);
    }
    let head = |label: &'static str| text(label).font(S_HEAD).color(MUTED);
    let mut p = panel().child(
        panel_row(true)
            .h(32.0)
            .child(head("Time").w(72.0))
            .child(head("Model").w(0.0).grow(1.0))
            .child(head("Endpoint").w(120.0))
            .child(right(128.0, head("Questions")))
            .child(right(72.0, head("Took"))),
    );
    for (i, r) in snap.recent.iter().take(20).enumerate() {
        let time = r.at.with_timezone(&chrono::Local).format("%H:%M:%S");
        let answer = match &r.error {
            Some(e) => text(e.clone()).font(S_SMALL).color(BAD),
            None => text(r.questions.to_string()).font(S_SMALL_NUM).color(BODY),
        };
        p = p.child(
            panel_row(false)
                .key(format!("req-{i}"))
                .h(32.0)
                .child(
                    text(time.to_string())
                        .font(S_SMALL_NUM)
                        .color(MUTED)
                        .w(72.0),
                )
                .child(
                    text(r.model.clone())
                        .font(S_MONO)
                        .color(FG)
                        .w(0.0)
                        .grow(1.0)
                        .truncate(),
                )
                .child(text(r.endpoint.clone()).font(S_MONO).color(MUTED).w(120.0))
                .child(right(128.0, answer))
                .child(right(
                    72.0,
                    text(format!("{} ms", ms(r.took_ms)))
                        .font(S_SMALL_NUM)
                        .color(BODY),
                )),
        );
    }
    p
}
