//! Open at Login on macOS: a LaunchAgent that opens the app when the user logs in, the same file
//! the Tauri app wrote, so the setting carries over.

use std::path::PathBuf;

const LABEL: &str = "dev.ollaya.app";

fn agent() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_default()
        .join("Library/LaunchAgents")
        .join(format!("{LABEL}.plist"))
}

pub fn is_on() -> bool {
    agent().exists()
}

pub fn set(on: bool) -> Result<(), String> {
    let path = agent();
    if !on {
        return match std::fs::remove_file(&path) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(e.to_string()),
            _ => Ok(()),
        };
    }
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    std::fs::write(&path, plist(&exe.display().to_string())).map_err(|e| e.to_string())
}

fn plist(exe: &str) -> String {
    let exe = exe
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;");
    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>{LABEL}</string>
  <key>ProgramArguments</key><array><string>{exe}</string></array>
  <key>RunAtLoad</key><true/>
  <key>ProcessType</key><string>Interactive</string>
</dict>
</plist>
"#
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_agent_runs_this_executable_at_load() {
        let p = plist("/Applications/Ollaya & Co.app/Contents/MacOS/ollaya-desktop");
        assert!(p.contains("<string>dev.ollaya.app</string>"));
        assert!(p.contains("Ollaya &amp; Co.app/Contents/MacOS/ollaya-desktop</string>"));
        assert!(p.contains("<key>RunAtLoad</key><true/>"));
    }
}
