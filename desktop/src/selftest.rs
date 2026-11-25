//! `OLLAYA_SELFTEST=1`: drive the real app through the paths its buttons and menu take, printing
//! the state and the menu bar's items at each step: the server switch twice (stop, start, or the
//! other way round), then Quit from the menu bar. A stand-in for clicking through it by hand.

use rune::Tx;

use crate::state::State;

pub fn step(s: &mut State, answered: bool, tx: &mut Tx) {
    let step = s.selftest.unwrap_or(0);
    // Step 0 waits for launch to settle: the first look, and a start if the server was stopped.
    if s.busy || !s.booted || (step > 0 && !answered) {
        return;
    }
    eprintln!(
        "selftest {step}: running={} version={:?} models={} library={} error={:?}",
        s.snap.running,
        s.snap.version,
        s.snap.models.len(),
        s.library.len(),
        s.error
    );
    #[cfg(target_os = "macos")]
    for line in crate::menubar::titles() {
        eprintln!("selftest {step}:   {line}");
    }
    match step {
        0 | 1 => {
            s.selftest = Some(step + 1);
            eprintln!(
                "selftest {step}: pressing {:?}",
                if s.snap.running {
                    "Stop server"
                } else {
                    "Start server"
                }
            );
            s.toggle_server();
        }
        _ => {
            s.selftest = None;
            eprintln!("selftest {step}: Quit Ollaya from the menu bar");
            #[cfg(target_os = "macos")]
            s.menu(crate::menubar::Action::Quit, tx);
            #[cfg(not(target_os = "macos"))]
            s.quit(tx);
        }
    }
}
