//! Discover: the model library from ollaya.dev, a list to search on the left and the family
//! picked on the right, with its versions to download and its model card.

use rune::{El, ScrollSpec, Tx, col, el, icon, row, spacer, text, text_input};
use rune_kit::{Markdown, MarkdownStyle};

use super::*;
use crate::library::{LibModel, card_link};

/// The list's width.
const LIST: f32 = 288.0;

pub fn page(s: &State) -> El {
    if s.library.is_empty()
        && let Some(err) = &s.library_error
    {
        return empty(
            "The library could not be reached",
            "It comes from ollaya.dev",
            Some(
                button("library-retry", "Try again", false, true)
                    .on_press(|s: &mut State| s.reload_library()),
            ),
            Some(err),
        );
    }
    if s.library.is_empty() {
        return empty(
            if s.library_loading {
                "Loading the library…"
            } else {
                "The library is empty"
            },
            "",
            None,
            None,
        );
    }
    let shown = s.shown_library();
    let picked = s.selected_model();
    row()
        .w_full()
        .grow(1.0)
        .min_h(0.0)
        .child(list(s, &shown, picked.map(|m| m.name.as_str())))
        .child(el().w(1.0).h_full().fill(LINE))
        .child(match picked {
            Some(m) => detail(s, m),
            None => empty("Pick a model", "Search the library on the left", None, None),
        })
}

fn list(s: &State, shown: &[&LibModel], picked: Option<&str>) -> El {
    let search = row()
        .w_full()
        .h(28.0)
        .px(8.0)
        .gap(6.0)
        .items_center()
        .radius(6.0)
        .fill(CANVAS)
        .border(1.0, LINE)
        .child(icon(&SEARCH, 14.0).color(MUTED))
        .child(
            text_input(&s.query, S_SMALL)
                .key("search")
                .label("Search models")
                .placeholder("Search models", MUTED)
                .caret(FG, FILL_STRONG)
                .single_line()
                .on_edit(|s: &mut State| &mut s.query)
                .color(FG)
                .grow(1.0)
                .min_w(0.0),
        );
    let tab = |label: &'static str, only: bool| {
        let on = s.only_installed == only;
        row()
            .key(format!("tab-{label}"))
            .role("button")
            .label(label)
            .focusable(true)
            .px(8.0)
            .py(2.0)
            .radius(4.0)
            .fill(if on {
                FILL_STRONG
            } else {
                rune::Color::TRANSPARENT
            })
            .hover(if on { FILL_STRONG } else { FILL })
            .on_press(move |s: &mut State| s.only_installed = only)
            .child(text(label).font(S_META).color(if on { FG } else { MUTED }))
    };
    let count = if s.library.is_empty() {
        String::new()
    } else {
        format!("{} of {}", shown.len(), s.library.len())
    };
    let head = col().w_full().padding(10.0).gap(8.0).child(search).child(
        row()
            .w_full()
            .gap(4.0)
            .items_center()
            .child(tab("All", false))
            .child(tab("Installed", true))
            .child(spacer())
            .child(text(count).font(S_META_NUM).color(MUTED)),
    );
    let body = if shown.is_empty() {
        let q = s.query.text.trim().is_empty();
        empty(
            if s.only_installed && q {
                "No models on this computer yet"
            } else {
                "No model matches"
            },
            "",
            None,
            None,
        )
    } else {
        let mut rows = col().w_full().padding(6.0).gap(2.0);
        for m in shown {
            rows = rows.child(list_row(s, m, picked == Some(m.name.as_str())));
        }
        col()
            .key("library-list")
            .w_full()
            .grow(1.0)
            .min_h(0.0)
            .scroll(ScrollSpec::default())
            .child(rows)
    };
    col()
        .w(LIST)
        .h_full()
        .child(head)
        .child(hairline())
        .child(body)
}

fn list_row(s: &State, m: &LibModel, on: bool) -> El {
    let have = !s.installed_tags(m).is_empty();
    let size = m
        .tag(&format!("{}:latest", m.name))
        .and_then(|t| t.size)
        .map(bytes)
        .unwrap_or_default();
    let name = m.name.clone();
    let mut side = col()
        .items_end()
        .gap(2.0)
        .child(text(size).font(S_META_NUM).color(MUTED));
    if have {
        side = side.child(icon(&CHECK, 14.0).color(OK));
    }
    row()
        .key(format!("lib-{}", m.name))
        .role("button")
        .label(m.name.clone())
        .focusable(true)
        .w_full()
        .px(10.0)
        .py(8.0)
        .gap(12.0)
        .items_center()
        .radius(6.0)
        .fill(if on { FILL } else { rune::Color::TRANSPARENT })
        .hover(FILL)
        .on_press(move |s: &mut State| s.selected = Some(name.clone()))
        .child(
            col()
                .w(0.0)
                .grow(1.0)
                .gap(2.0)
                .child(text(m.name.clone()).font(S_LIST_NAME).color(FG).truncate())
                .child(
                    text(m.publisher.clone().unwrap_or_else(|| m.title.clone()))
                        .font(S_META)
                        .color(MUTED)
                        .truncate(),
                ),
        )
        .child(side)
}

fn detail(s: &State, m: &LibModel) -> El {
    let running = s.snap.running;
    let have = s.installed_tags(m);
    let byline: Vec<String> = [
        m.publisher.as_ref().map(|p| format!("by {p}")),
        (!m.license.is_empty()).then(|| m.license.clone()),
    ]
    .into_iter()
    .flatten()
    .collect();

    let mut actions = row().gap(8.0).items_center();
    if let Some(first) = have.first().cloned() {
        actions = actions.child(on_press(
            icon_button("try", &PLAY, "Try", true, running),
            running,
            move |s: &mut State| s.try_out(first.clone()),
        ));
    }
    actions = actions.child(pull_state(s, &format!("{}:latest", m.name)));

    let head = row()
        .w_full()
        .pt(24.0)
        .gap(16.0)
        .items_start()
        .child(
            col()
                .w(0.0)
                .grow(1.0)
                .gap(4.0)
                .child(text(m.name.clone()).font(S_NAME).color(FG).truncate())
                .child(text(byline.join(" · ")).font(S_SMALL).color(MUTED)),
        )
        .child(actions);

    let mut facts = row().w_full().pt(12.0).gap(16.0).items_center().flex_wrap();
    let stat = |value: String, what: &str| {
        row()
            .gap(4.0)
            .child(text(value).font(S_STAT).color(FG))
            .child(text(what.to_owned()).font(S_SMALL).color(MUTED))
    };
    if let Some(stats) = &m.stats {
        if let Some(a) = stats.accuracy {
            facts = facts.child(stat(format!("{a:.3}"), "accuracy"));
        }
        if let Some(l) = stats.latency_ms {
            facts = facts.child(stat(format!("{} ms", l.round()), "for five questions"));
        }
    }
    let mut caps = row().gap(6.0).items_center();
    for c in &m.caps {
        caps = caps.child(
            row()
                .px(6.0)
                .py(1.0)
                .radius(4.0)
                .fill(CAP_BG)
                .child(text(c.clone()).font(S_HEAD).color(OK)),
        );
    }
    facts = facts.child(caps);

    let (outer, inner) = super::page("detail");
    let mut inner = inner
        .child(head)
        .child(
            text(m.description.clone())
                .font(S_BODY.lh(20.0))
                .color(BODY)
                .pt(16.0)
                .wrap(),
        )
        .child(facts);
    if let Some(err) = &s.pull_error {
        inner = inner.child(text(err.clone()).font(S_SMALL).color(BAD).pt(12.0).wrap());
    }
    let mut versions = panel();
    for (i, tag) in m.main_tags().iter().enumerate() {
        versions = versions.child(version_row(s, m, tag, i == 0));
    }
    inner = inner.child(section("Versions", None)).child(versions);
    if let Some(card) = &m.card {
        inner = inner
            .child(section("Model card", None))
            .child(model_card(&m.name, card));
    }
    let page_url = format!("https://ollaya.dev/library/{}", m.name);
    inner = inner.child(
        link("on-site", format!("{} on ollaya.dev", m.name), true)
            .pt(32.0)
            .on_press_tx(move |_: &mut State, tx: &mut Tx| {
                let _ = tx.open_url(&page_url);
            }),
    );
    outer.child(inner)
}

fn version_row(s: &State, m: &LibModel, tag: &str, first: bool) -> El {
    let t = m.tag(tag);
    let summary = t
        .and_then(|t| t.summary.clone())
        .filter(|x| !x.is_empty())
        .or_else(|| {
            tag.ends_with(":latest")
                .then(|| format!("The default version: ollaya run {}", m.name))
        });
    let have = s.is_installed(tag) && !s.pulling.contains_key(tag);
    let mut name = row()
        .gap(8.0)
        .items_center()
        .child(text(tag.to_owned()).font(S_MONO).color(FG));
    if let Some(size) = t.and_then(|t| t.size) {
        name = name.child(text(bytes(size)).font(S_META_NUM).color(MUTED));
    }
    let mut what = col().w(0.0).grow(1.0).gap(2.0).child(name);
    if let Some(summary) = summary {
        what = what.child(
            text(summary.clone())
                .font(S_META)
                .color(MUTED)
                .truncate()
                .help(summary),
        );
    }
    let mut side = row().gap(4.0).items_center();
    let running = s.snap.running;
    if have {
        let t = tag.to_owned();
        side = side.child(on_press(
            quiet(format!("try-{tag}"), &PLAY, format!("Try {tag}"), running),
            running,
            move |s: &mut State| s.try_out(t.clone()),
        ));
    }
    side = side.child(pull_state(s, tag));
    if have {
        let t = tag.to_owned();
        let ok = !s.removing.contains(tag);
        side = side.child(on_press(
            quiet(format!("remove-{tag}"), &X, format!("Remove {tag}"), ok),
            ok,
            move |s: &mut State| s.remove(t.clone()),
        ));
    }
    panel_row(first)
        .key(format!("ver-{tag}"))
        .min_h(44.0)
        .py(8.0)
        .child(what)
        .child(side)
}

/// The model card from ollaya.dev, as Markdown. Its links open in the browser, never inside
/// the app.
fn model_card(model: &str, card: &str) -> El {
    let model = model.to_owned();
    Markdown::new(card)
        .on_link_tx(move |_: &mut State, href: &str, tx: &mut Tx| {
            if let Some(url) = card_link(&model, href) {
                let _ = tx.open_url(&url);
            }
        })
        .render(&card_style())
}

fn card_style() -> MarkdownStyle {
    MarkdownStyle::neutral(true)
        .text(BODY)
        .muted(MUTED)
        .link(FG)
        .link_underline(FAINT)
        .link_hover(BTN_HOVER)
        .inline_code_bg(FILL)
        .code_block(CODE, LINE)
        .code_block_text(BODY)
        .rule(LINE)
        .body_font(S_BODY.lh(20.0))
        .heading_font(1, TextStyle::new(16.0, 500.0))
        .heading_font(2, TextStyle::new(14.0, 500.0))
        .heading_font(3, TextStyle::new(13.0, 500.0))
        .code_block_font(S_MONO.lh(18.0))
}
