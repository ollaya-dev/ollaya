//! The daemon's lifecycle from the CLI: `ollaya serve`, a client for `OLLAYA_HOST` that starts
//! `ollaya serve` in the background when nothing answers there (as the Ollama CLI starts its
//! server), and `ollaya stop` without a model, which stops that server again.
//!
//! A running server records its process ID in `~/.ollaya/server.<port>.pid`. Only its owner can
//! read that file and signal the process, so `ollaya stop` never stops another user's server,
//! such as the Linux systemd service.

use std::io::ErrorKind;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use ollaya_api::Client;
use ollaya_api::host::Host;
use ollaya_server::config::ServerConfig;
use ollaya_server::http::{Listeners, serve_bound};

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
    // No console window, and not tied to this one.
    #[cfg(windows)]
    {
        std::os::windows::process::CommandExt::creation_flags(
            &mut cmd,
            DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW,
        );
        keep_std_handles_private();
    }
    cmd.spawn().context("starting `ollaya serve`")?;
    Ok(path)
}

/// `ollaya serve`: bind `OLLAYA_HOST`, record the process ID, serve until Ctrl-C or SIGTERM.
pub async fn serve(config: ServerConfig) -> Result<()> {
    let listeners = match Listeners::bind(&config.host).await {
        Ok(l) => l,
        Err(e) if e.kind() == ErrorKind::AddrInUse => bail!(in_use(&config.host).await),
        Err(e) => {
            return Err(e).with_context(|| format!("listening on {}", config.host.bind_addr()));
        }
    };
    let pid_file = pid_path(config.host.port);
    let wrote = std::fs::create_dir_all(home())
        .and_then(|()| std::fs::write(&pid_file, format!("{}\n", std::process::id())));
    if let Err(e) = &wrote {
        tracing::warn!(
            "could not write {}: {e}; `ollaya stop` will not find this server",
            pid_file.display()
        );
    }
    let served = serve_bound(config, listeners).await;
    if wrote.is_ok() {
        let _ = std::fs::remove_file(&pid_file);
    }
    Ok(served?)
}

/// Why `ollaya serve` cannot bind `host`: an Ollaya server already runs there, or another program
/// holds the port.
async fn in_use(host: &Host) -> String {
    let url = host.base_url();
    let running = match Client::new(&url) {
        Ok(c) => c.version().await.ok().map(|v| v.version),
        Err(_) => None,
    };
    let Some(version) = running else {
        return format!(
            "{} is already in use by another program. Run Ollaya on another port, for example:\n  \
             OLLAYA_HOST=127.0.0.1:{} ollaya serve",
            host.bind_addr(),
            host.port.wrapping_add(1)
        );
    };
    let stop = if own_server(host.port).is_some() {
        "stop it first with `ollaya stop`"
    } else {
        "stop it first; it runs as another user, such as the systemd service \
         (sudo systemctl stop ollaya)"
    };
    format!(
        "Ollaya {version} is already running at {url}, so you can use it right away, for \
         example: ollaya run laya\n\
         To run the server in this terminal instead, {stop}."
    )
}

/// `ollaya stop` without a model: stop this user's server at `OLLAYA_HOST`, which unloads every
/// model. Does nothing when no server is running.
pub async fn stop_server() -> Result<()> {
    let host = Host::from_env()?;
    let client = Client::from_env()?;
    let Ok(version) = client.version().await.map(|v| v.version) else {
        println!("Ollaya is not running at {}", client.base_url());
        return Ok(());
    };
    if !host.is_local() {
        bail!(
            "`ollaya stop` stops a server on this machine; {} is not local",
            client.base_url()
        );
    }
    let Some(pid) = own_server(host.port) else {
        bail!(
            "Ollaya {version} at {} was not started by you, so `ollaya stop` cannot stop it. \
             If it is the systemd service: sudo systemctl stop ollaya",
            client.base_url()
        );
    };
    terminate(pid)?;
    // The server finishes open requests (up to 5 s) and stops its runners before it exits.
    let deadline = Instant::now() + Duration::from_secs(15);
    while Instant::now() < deadline {
        if client.heartbeat().await.is_err() {
            println!("Stopped Ollaya {version}");
            return Ok(());
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    bail!("sent Ollaya (process {pid}) the signal to stop, but it is still answering after 15 s")
}

/// `~/.ollaya/server.<port>.pid`.
fn pid_path(port: u16) -> PathBuf {
    home().join(format!("server.{port}.pid"))
}

/// The process ID in this user's PID file for `port`, if that process is still `ollaya serve`.
/// A stale file (the server crashed, or the ID now belongs to another program) is removed.
fn own_server(port: u16) -> Option<u32> {
    let path = pid_path(port);
    let pid = std::fs::read_to_string(&path)
        .ok()?
        .trim()
        .parse::<u32>()
        .ok();
    match pid {
        Some(pid) if is_ollaya_serve(pid) => Some(pid),
        _ => {
            let _ = std::fs::remove_file(&path);
            None
        }
    }
}

/// Whether process `pid` runs `ollaya serve`, from its command line as `ps` prints it.
#[cfg(unix)]
fn is_ollaya_serve(pid: u32) -> bool {
    let Ok(out) = std::process::Command::new("ps")
        .args(["-o", "command=", "-p", &pid.to_string()])
        .output()
    else {
        return false;
    };
    let command = String::from_utf8_lossy(&out.stdout);
    let mut args = command.split_whitespace();
    let exe = args.next().unwrap_or_default();
    exe.rsplit('/').next() == Some("ollaya") && args.next() == Some("serve")
}

/// Whether process `pid` is `ollaya.exe`, from `tasklist` (which has no command lines).
#[cfg(windows)]
fn is_ollaya_serve(pid: u32) -> bool {
    let Ok(out) = std::process::Command::new("tasklist")
        .args(["/FI", &format!("PID eq {pid}"), "/FO", "CSV", "/NH"])
        .output()
    else {
        return false;
    };
    // "ollaya.exe","1234","Console","1","12,345 K"
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .any(|l| l.to_ascii_lowercase().starts_with("\"ollaya.exe\","))
}

#[cfg(not(any(unix, windows)))]
fn is_ollaya_serve(_pid: u32) -> bool {
    false
}

/// Ask `pid` to shut down gracefully (SIGTERM), as systemd and Ctrl-C do.
#[cfg(unix)]
fn terminate(pid: u32) -> Result<()> {
    let status = std::process::Command::new("kill")
        .args(["-TERM", &pid.to_string()])
        .status()
        .context("running kill")?;
    if !status.success() {
        bail!("could not signal Ollaya (process {pid}) to stop");
    }
    Ok(())
}

/// End `pid` and its runners. A detached process on Windows has no console to receive Ctrl-C,
/// so this is not graceful: requests in flight fail.
#[cfg(windows)]
fn terminate(pid: u32) -> Result<()> {
    let status = std::process::Command::new("taskkill")
        .args(["/PID", &pid.to_string(), "/T", "/F"])
        .stdout(std::process::Stdio::null())
        .status()
        .context("running taskkill")?;
    if !status.success() {
        bail!("could not stop Ollaya (process {pid})");
    }
    Ok(())
}

#[cfg(not(any(unix, windows)))]
fn terminate(pid: u32) -> Result<()> {
    bail!("cannot stop Ollaya (process {pid}) on this platform")
}

/// Windows hands a new process every inheritable handle of its parent, not only the ones given as
/// its stdio. The server outlives this command, so if it inherited this command's stdout (a pipe
/// when a script or app captures `ollaya pull`), the reader would wait for the server to exit
/// before it saw end-of-file. Our own standard handles are therefore made non-inheritable before
/// the spawn; the server still gets its log file, which std duplicates for it explicitly.
#[cfg(windows)]
fn keep_std_handles_private() {
    type Handle = *mut std::ffi::c_void;
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetStdHandle(which: u32) -> Handle;
        fn SetHandleInformation(handle: Handle, mask: u32, flags: u32) -> i32;
    }
    const STD_HANDLES: [u32; 3] = [-10i32 as u32, -11i32 as u32, -12i32 as u32]; // input, output, error
    const HANDLE_FLAG_INHERIT: u32 = 0x1;
    for which in STD_HANDLES {
        // SAFETY: plain Win32 calls on this process's own standard handles; a null or invalid
        // handle (no console) just makes SetHandleInformation fail, which is fine.
        unsafe {
            let handle = GetStdHandle(which);
            if !handle.is_null() && handle as isize != -1 {
                SetHandleInformation(handle, HANDLE_FLAG_INHERIT, 0);
            }
        }
    }
}

// Process creation flags (winbase.h).
#[cfg(windows)]
const DETACHED_PROCESS: u32 = 0x0000_0008;
#[cfg(windows)]
const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[test]
    fn recognises_only_ollaya_serve() {
        // This test binary is not `ollaya serve`, and neither is a process that does not exist.
        assert!(!is_ollaya_serve(std::process::id()));
        assert!(!is_ollaya_serve(u32::MAX - 1));
    }
}
