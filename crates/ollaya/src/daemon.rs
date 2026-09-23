//! Reaching the daemon: a client for `OLLAYA_HOST`, starting `ollaya serve` in the background
//! when nothing answers there (as the Ollama CLI starts its server).

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use ollaya_api::Client;
use ollaya_api::host::Host;

/// `~/.ollaya`, where logs and REPL history live.
pub fn home() -> PathBuf {
    dirs::home_dir().unwrap_or_default().join(".ollaya")
}

/// A client for a running daemon, starting one if `OLLAYA_HOST` is local and nothing answers.
pub async fn client() -> Result<Client> {
    let client = Client::from_env()?;
    if client.heartbeat().await.is_ok() {
        return Ok(client);
    }
    let host = Host::from_env()?;
    if !host.is_local() {
        bail!(
            "could not connect to ollaya at {}; is `ollaya serve` running there?",
            client.base_url()
        );
    }
    let log = start_server()?;
    let deadline = Instant::now() + Duration::from_secs(30);
    while Instant::now() < deadline {
        if client.heartbeat().await.is_ok() {
            return Ok(client);
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    bail!(
        "started `ollaya serve`, but it did not answer at {} within 30 s; see {}",
        client.base_url(),
        log.display()
    )
}

/// Spawn `ollaya serve` detached from this terminal, logging to `~/.ollaya/logs/server.log`.
fn start_server() -> Result<PathBuf> {
    let exe = std::env::current_exe().context("locating the ollaya executable")?;
    let dir = home().join("logs");
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    let path = dir.join("server.log");
    let log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .with_context(|| format!("opening {}", path.display()))?;
    let mut cmd = std::process::Command::new(exe);
    cmd.arg("serve")
        .stdin(std::process::Stdio::null())
        .stdout(log.try_clone()?)
        .stderr(log);
    // Its own process group: Ctrl-C in this terminal must not stop the daemon.
    #[cfg(unix)]
    std::os::unix::process::CommandExt::process_group(&mut cmd, 0);
    cmd.spawn().context("starting `ollaya serve`")?;
    Ok(path)
}
