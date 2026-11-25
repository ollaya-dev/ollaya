//! My Models: the versions on this computer, with their size and whether they are loaded, to
//! load, try or remove.

use rune::{El, el, row, text};

use super::*;

pub fn page(s: &State) -> El {
    let snap = &s.snap;
    if !snap.running {
        return stopped(s, "Start it to see the models on this computer");
    }
    let tags = s.installed();
    if tags.is_empty() {
        return empty(
            "No models yet",
            "Find one in the library, or run ollaya pull kev",
            Some(discover_button()),
            s.pull_error.as_deref(),
        );
    }
    let total: u64 = tags.iter().map(|t| s.size_of(t)).sum();
    let n = tags.len();
    let head = |label: &'static str| text(label).font(S_HEAD).color(MUTED);
    let mut table = panel().child(
        panel_row(true)
            .h(32.0)
            .child(head("Version").w(0.0).grow(1.0))
            .child(right(88.0, head("Size")))
            .child(el().w(16.0))
            .child(head("State").w(96.0))
            .child(el().w(72.0 + 8.0 + 28.0 + 4.0 + 28.0)),
    );
    for tag in &tags {
        let loaded = s.is_loaded(tag);
        let removing = s.removing.contains(tag);
        let state = if removing {
            row()
                .w(96.0)
                .child(text("Removing…").font(S_SMALL).color(MUTED))
        } else if loaded {
            row()
                .w(96.0)
                .gap(8.0)
                .items_center()
                .child(el().size(6.0, 6.0).radius(3.0).fill(OK))
                .child(text("Loaded").font(S_SMALL).color(BODY))
        } else {
            row()
                .w(96.0)
                .child(text("On disk").font(S_SMALL).color(MUTED))
        };
        let (load, try_it, remove) = (tag.clone(), tag.clone(), tag.clone());
        let ok = !removing;
        table = table.child(
            panel_row(false)
                .key(tag.clone())
                .h(44.0)
                .child(
                    text(tag.clone())
                        .font(S_MONO)
                        .color(FG)
                        .w(0.0)
                        .grow(1.0)
                        .truncate(),
                )
                .child(right(
                    88.0,
                    text(bytes(s.size_of(tag))).font(S_SMALL_NUM).color(MUTED),
                ))
                .child(el().w(16.0))
                .child(state)
                .child(
                    row()
                        .gap(4.0)
                        .items_center()
                        .child(
                            on_press(
                                button(
                                    format!("load-{tag}"),
                                    if loaded { "Unload" } else { "Load" },
                                    false,
                                    ok,
                                )
                                .w(72.0)
                                .justify_center(),
                                ok,
                                move |s: &mut State| s.toggle_model(load.clone()),
                            )
                            .mr(4.0),
                        )
                        .child(on_press(
                            quiet(format!("try-{tag}"), &PLAY, format!("Try {tag}"), ok),
                            ok,
                            move |s: &mut State| s.try_out(try_it.clone()),
                        ))
                        .child(on_press(
                            quiet(format!("remove-{tag}"), &X, format!("Remove {tag}"), ok),
                            ok,
                            move |s: &mut State| s.remove(remove.clone()),
                        )),
                ),
        );
    }
    let (outer, inner) = super::page("models");
    let mut inner = inner
        .child(section(
            "On this computer",
            Some(
                text(format!(
                    "{n} {} · {}",
                    if n == 1 { "version" } else { "versions" },
                    bytes(total)
                ))
                .font(S_SMALL_NUM)
                .color(MUTED),
            ),
        ))
        .child(table);
    for err in [&s.pull_error, &s.error].into_iter().flatten() {
        inner = inner.child(text(err.clone()).font(S_SMALL).color(BAD).pt(8.0).wrap());
    }
    outer.child(inner)
}
