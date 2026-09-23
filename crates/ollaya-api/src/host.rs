//! `OLLAYA_HOST`: where the daemon listens and where the client connects (`docs/api.md` §15).
//!
//! Accepted forms, as with `OLLAMA_HOST`: `127.0.0.1:11435`, `0.0.0.0`, `:8080`, `localhost`,
//! `[::1]:11435`, `http://host:port`, `https://example.com/ollaya`. A missing scheme means
//! `http`; a missing port means 11435 for `http` and 443 for `https`; an empty host means
//! `127.0.0.1`; a path is kept as a prefix.

use crate::DEFAULT_PORT;

pub const HOST_ENV: &str = "OLLAYA_HOST";

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("invalid OLLAYA_HOST {input:?}: {reason}")]
pub struct InvalidHost {
    pub input: String,
    pub reason: &'static str,
}

/// A parsed host: scheme, host (IPv6 in brackets), port and path prefix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Host {
    pub scheme: String,
    pub host: String,
    pub port: u16,
    pub path: String,
}

impl Host {
    pub fn parse(input: &str) -> Result<Host, InvalidHost> {
        let bad = |reason| InvalidHost {
            input: input.to_owned(),
            reason,
        };
        let s = input.trim();
        let (scheme, rest) = match s.split_once("://") {
            Some((scheme, rest)) => match scheme.to_ascii_lowercase().as_str() {
                "http" => ("http", rest),
                "https" => ("https", rest),
                _ => return Err(bad("the scheme must be http or https")),
            },
            None => ("http", s),
        };
        let (hostport, path) = match rest.find('/') {
            Some(i) => (&rest[..i], rest[i..].trim_end_matches('/')),
            None => (rest, ""),
        };
        let default_port = if scheme == "https" { 443 } else { DEFAULT_PORT };
        let parse_port = |p: &str| {
            p.parse::<u16>()
                .map_err(|_| bad("the port must be 0-65535"))
        };
        let (host, port) = if let Some(v6) = hostport.strip_prefix('[') {
            let (addr, after) = v6.split_once(']').ok_or_else(|| bad("unclosed '['"))?;
            let port = match after {
                "" => default_port,
                p => parse_port(p.strip_prefix(':').ok_or_else(|| bad("junk after ']'"))?)?,
            };
            (format!("[{addr}]"), port)
        } else if hostport.matches(':').count() > 1 {
            // A bare IPv6 address without a port.
            (format!("[{hostport}]"), default_port)
        } else if let Some((h, p)) = hostport.rsplit_once(':') {
            (h.to_owned(), parse_port(p)?)
        } else {
            (hostport.to_owned(), default_port)
        };
        if host
            .chars()
            .any(|c| c.is_whitespace() || matches!(c, '@' | '?' | '#'))
        {
            return Err(bad("the host contains invalid characters"));
        }
        Ok(Host {
            scheme: scheme.to_owned(),
            host: if host.is_empty() {
                "127.0.0.1".to_owned()
            } else {
                host
            },
            port,
            path: path.to_owned(),
        })
    }

    /// `OLLAYA_HOST`, or `127.0.0.1:11435` when unset or empty.
    pub fn from_env() -> Result<Host, InvalidHost> {
        Host::parse(&std::env::var(HOST_ENV).unwrap_or_default())
    }

    /// Base URL without a trailing slash, e.g. `http://127.0.0.1:11435`.
    pub fn base_url(&self) -> String {
        format!("{}://{}:{}{}", self.scheme, self.host, self.port, self.path)
    }

    /// `host:port`, the socket address the daemon binds.
    pub fn bind_addr(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }

    /// Loopback, unspecified or `localhost`: no proxy, no Host-header exposure.
    pub fn is_local(&self) -> bool {
        let h = self.host.trim_start_matches('[').trim_end_matches(']');
        h.eq_ignore_ascii_case("localhost")
            || h.to_ascii_lowercase().ends_with(".localhost")
            || h.parse::<std::net::IpAddr>()
                .is_ok_and(|ip| ip.is_loopback() || ip.is_unspecified())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_ollama_style_hosts() {
        let cases = [
            ("", "http://127.0.0.1:11435"),
            ("127.0.0.1:11435", "http://127.0.0.1:11435"),
            ("0.0.0.0", "http://0.0.0.0:11435"),
            (":8080", "http://127.0.0.1:8080"),
            ("localhost", "http://localhost:11435"),
            ("example.com:80", "http://example.com:80"),
            ("http://example.com", "http://example.com:11435"),
            ("https://example.com", "https://example.com:443"),
            (
                "HTTPS://example.com:8443/ollaya/",
                "https://example.com:8443/ollaya",
            ),
            ("[::1]:9000", "http://[::1]:9000"),
            ("[::1]", "http://[::1]:11435"),
            ("::1", "http://[::1]:11435"),
            ("  10.0.0.5:1234  ", "http://10.0.0.5:1234"),
        ];
        for (input, want) in cases {
            assert_eq!(Host::parse(input).unwrap().base_url(), want, "{input:?}");
        }
        assert_eq!(Host::parse("0.0.0.0").unwrap().bind_addr(), "0.0.0.0:11435");
    }

    #[test]
    fn rejects_bad_hosts() {
        for bad in [
            "ftp://x",
            "host:99999",
            "host:",
            "[::1",
            "[::1]x",
            "a b:1",
            "user@host",
        ] {
            assert!(Host::parse(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn knows_local_hosts() {
        for local in [
            "",
            "localhost",
            "127.0.0.2",
            "[::1]",
            "0.0.0.0",
            "app.localhost",
        ] {
            assert!(Host::parse(local).unwrap().is_local(), "{local:?}");
        }
        for remote in ["10.0.0.5", "example.com", "choso-wsl"] {
            assert!(!Host::parse(remote).unwrap().is_local(), "{remote:?}");
        }
    }
}
