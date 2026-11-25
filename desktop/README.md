# Ollaya desktop app

The server, the model library and a playground in one window, drawn with
[Rune](https://github.com/anomalyco/rune): Rust that renders on the GPU itself, with no webview.
It runs `ollaya serve` in the background, as Ollama's app runs `ollama serve`. On macOS it lives
in the menu bar. How it is packaged, signed and released: "Desktop app" in
[`docs/distribution.md`](../docs/distribution.md).

```sh
OLLAYA_BIN=../target/release/ollaya cargo run      # with an engine you built
cargo run -- --snap out.png --scene playground --chrome --scale 2   # a page, headless, made-up data
cargo test
```

## Pages

- **Discover:** the library from `https://ollaya.dev/data/library.json`, with a search. A
  family shows its versions (download with progress, remove), its figures and its model card,
  drawn from the card's Markdown.
- **My Models:** the installed versions: load, unload, try, remove.
- **Playground:** a state, a preset, the model's own questions or custom JSON questions, and the
  answers with their confidence. ⌘ Return (Ctrl + Enter) runs.
- **Server:** requests, requests per second, the median response time and capacity, the address
  and the API endpoints with copy buttons, and the latest requests.

The status bar shows the server, the device it runs on (Windows and Linux), what is loaded, a
newer release, and the switch that starts and stops the server.

## How it is built

One process, three parts. Rune's UI thread owns `State` (`state.rs`) and draws it (`ui/`).
`backend.rs` talks to the server with `ollaya-api`'s client and to the website on a Tokio thread,
and answers through `AppHandle::update`, only when something changed. On macOS `menubar.rs` is an
AppKit status item (Rune has none yet, anomalyco/rune#141). A button calls a Rust method: there is
no IPC and no JavaScript.

| File | What it does |
| --- | --- |
| `main.rs` | The app: window, close and quit, the menu bar's actions |
| `state.rs` | Everything the window shows, and the backend's answers |
| `backend.rs` | The server's snapshot every 2 s, start and stop, load, pull, remove, decide |
| `engine.rs` | Which `ollaya` runs the server and with which libraries (the GPU pack on Windows and Linux) |
| `library.rs` | The website's library and the latest release |
| `menubar.rs`, `login.rs` | macOS: the menu bar and Open at Login (a LaunchAgent) |
| `ui/` | The pages, the rail and the status bar; colours in `theme.rs` |
| `scenes.rs` | Made-up data for headless previews, one scene per page and state |
| `selftest.rs` | `OLLAYA_SELFTEST=1`: drives the real app through the switch and Quit |

## Checking a change

- **Look at it:** `--snap` renders a scene to a PNG without a server (in a debug build: a
  release build ignores Rune's preview flags and starts the real app): `discover`,
  `discover-pull`, `discover-search`, `first-run`, `offline`, `models`, `playground`,
  `playground-custom`, `playground-guard`, `server`, `stopped`.
- **Drive it:** the self-test starts the real app, presses the server switch twice and quits
  from the menu bar, printing the state and the menu at each step. Give it its own port so a
  server you run is left alone:

  ```sh
  PATH=../target/release:$PATH OLLAYA_HOST=127.0.0.1:11535 OLLAYA_SELFTEST=1 OLLAYA_OPEN_WINDOW=1 \
    target/release/ollaya-desktop
  ```

- **Before committing:** `cargo fmt --all` and `cargo clippy --all-targets --locked -- -D warnings`
  here; CI runs both.

## Environment

| Variable | Effect |
| --- | --- |
| `OLLAYA_BIN` | The engine to run, instead of the one next to the app |
| `OLLAYA_HOST` | Where the server listens, as for the command line |
| `OLLAYA_LIBRARY_URL` | Another library index, such as a local build of the site |
| `OLLAYA_NO_AUTOSTART` | Don't start the server at launch |
| `OLLAYA_OPEN_WINDOW` | macOS: open the window at launch |
| `OLLAYA_SELFTEST` | Run the self-test |
| `OLLAYA_TIMING` | Print how long the first frame took |

## Rune

Rune is pre-release (`0.0.1`): `Cargo.toml` pins a rev of its `dev` branch and repeats the
`[patch.crates-io]` entries of Rune's own workspace at the same rev. To update, change the rev in
every entry at once, then check the scenes and run the self-test. Rune has no status item, no
bundler and no native menus on Windows and Linux; the menu bar is ours (`menubar.rs`) and the
installers are cargo-packager's (`Packager.toml`).
