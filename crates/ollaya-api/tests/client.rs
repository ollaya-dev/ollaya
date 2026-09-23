//! The client against a scripted HTTP server: streaming, errors, auth and base URLs.

use std::time::Duration;

use ollaya_api::{
    Client, ClientError, DecideRequest, ErrorCode, KeepAlive, ProgressResponse, PullRequest,
};
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::task::JoinHandle;

/// Serve one connection: capture the request, then write `chunks` with a pause between them so
/// the client sees them as separate reads. Returns the base URL and the captured request.
async fn serve_once(chunks: Vec<Vec<u8>>) -> (String, JoinHandle<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("127.0.0.1:{}", listener.local_addr().unwrap().port());
    let handle = tokio::spawn(async move {
        let (mut sock, _) = listener.accept().await.unwrap();
        let mut buf = Vec::new();
        let mut tmp = [0u8; 4096];
        let head_end = loop {
            let n = sock.read(&mut tmp).await.unwrap();
            buf.extend_from_slice(&tmp[..n]);
            if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                break i + 4;
            }
        };
        let head = String::from_utf8_lossy(&buf[..head_end]).to_string();
        let length = head
            .lines()
            .find_map(|l| {
                let (k, v) = l.split_once(':')?;
                k.eq_ignore_ascii_case("content-length")
                    .then(|| v.trim().parse::<usize>().ok())?
            })
            .unwrap_or(0);
        while buf.len() < head_end + length {
            let n = sock.read(&mut tmp).await.unwrap();
            buf.extend_from_slice(&tmp[..n]);
        }
        for chunk in chunks {
            sock.write_all(&chunk).await.unwrap();
            sock.flush().await.unwrap();
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        sock.shutdown().await.unwrap();
        String::from_utf8_lossy(&buf).to_string()
    });
    (base, handle)
}

fn ndjson_head() -> Vec<u8> {
    b"HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nConnection: close\r\n\r\n".to_vec()
}

fn json_response(status: &str, body: &str) -> Vec<u8> {
    format!(
        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
    .into_bytes()
}

const DIGEST: &str = "sha256:8d32a80bb199bcd4ff10abc28d651fe576fb59f86b24039402e49be9e01578c2";

#[tokio::test]
async fn pull_streams_lines_split_across_reads() {
    let layer = serde_json::to_string(&ProgressResponse::layer(DIGEST, 10, 5)).unwrap();
    let (split_a, split_b) = layer.split_at(20);
    let (base, server) = serve_once(vec![
        ndjson_head(),
        format!("{{\"status\":\"pulling manifest\"}}\n{split_a}").into_bytes(),
        format!("{split_b}\n\n").into_bytes(),
        b"{\"status\":\"success\"}\n".to_vec(),
    ])
    .await;
    let client = Client::new(&base).unwrap().with_api_key("s3cret");
    let mut seen = Vec::new();
    client
        .pull(&PullRequest::new("laya:en"), |p| seen.push(p.clone()))
        .await
        .unwrap();
    assert_eq!(
        seen.iter().map(|p| p.status.as_str()).collect::<Vec<_>>(),
        ["pulling manifest", "pulling 8d32a80bb199", "success"]
    );
    assert_eq!(seen[1].completed, Some(5));
    let request = server.await.unwrap();
    assert!(request.starts_with("POST /api/pull HTTP/1.1"), "{request}");
    assert!(
        request.contains("authorization: Bearer s3cret"),
        "{request}"
    );
    let body = &request[request.find("\r\n\r\n").unwrap() + 4..];
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(body).unwrap(),
        json!({"model": "laya:en", "stream": true})
    );
}

#[tokio::test]
async fn an_error_line_ends_the_stream_as_an_api_error() {
    let (base, _server) = serve_once(vec![
        ndjson_head(),
        b"{\"status\":\"pulling manifest\"}\n".to_vec(),
        format!("{{\"error\":\"blob {DIGEST} does not match its digest\",\"code\":\"DIGEST_MISMATCH\"}}\n")
            .into_bytes(),
    ])
    .await;
    let client = Client::new(&base).unwrap();
    let mut seen = 0;
    let err = client
        .pull(&PullRequest::new("laya:en"), |_| seen += 1)
        .await
        .unwrap_err();
    assert_eq!(seen, 1);
    match err {
        ClientError::Api { status, body } => {
            assert_eq!((status, body.code), (502, ErrorCode::DigestMismatch));
        }
        other => panic!("{other:?}"),
    }
}

#[tokio::test]
async fn a_cut_off_stream_is_incomplete() {
    let (base, _server) = serve_once(vec![
        ndjson_head(),
        b"{\"status\":\"pulling manifest\"}\n".to_vec(),
    ])
    .await;
    let err = Client::new(&base)
        .unwrap()
        .pull(&PullRequest::new("laya"), |_| {})
        .await
        .unwrap_err();
    assert!(matches!(err, ClientError::Incomplete { .. }), "{err:?}");
}

#[tokio::test]
async fn a_single_success_body_without_newline_completes() {
    let (base, _server) =
        serve_once(vec![json_response("200 OK", r#"{"status":"success"}"#)]).await;
    Client::new(&base)
        .unwrap()
        .pull(&PullRequest::new("laya"), |_| {})
        .await
        .unwrap();
}

#[tokio::test]
async fn error_responses_carry_the_body() {
    let body =
        r#"{"error":"model \"laya:xl\" not found, try pulling it first","code":"MODEL_NOT_FOUND"}"#;
    let (base, server) = serve_once(vec![json_response("404 Not Found", body)]).await;
    let req = DecideRequest::new("laya:xl", json!("hi"), None);
    let err = Client::new(&base).unwrap().decide(&req).await.unwrap_err();
    assert!(err.is_model_not_found());
    assert_eq!(
        err.to_string(),
        "model \"laya:xl\" not found, try pulling it first"
    );
    let request = server.await.unwrap();
    assert!(request.starts_with("POST /api/decide HTTP/1.1"));
    assert!(!request.contains("authorization"), "no key, no header");
}

#[tokio::test]
async fn unload_sends_keep_alive_zero_and_parses_the_response() {
    let response = json!({
        "model": "laya:en", "answers": {}, "usage": {"input_tokens": 0, "output_tokens": 0},
        "routing": null, "state_truncated": false, "done_reason": "unload",
        "created_at": "2026-09-24T09:45:01.531Z",
        "total_duration": 48210, "load_duration": 0, "eval_duration": 0,
    });
    let (base, server) = serve_once(vec![json_response("200 OK", &response.to_string())]).await;
    let r = Client::new(&base).unwrap().unload("laya:en").await.unwrap();
    assert_eq!(r.done_reason, ollaya_api::DoneReason::Unload);
    let request = server.await.unwrap();
    let body = &request[request.find("\r\n\r\n").unwrap() + 4..];
    let sent: serde_json::Value = serde_json::from_str(body).unwrap();
    assert_eq!(sent, json!({"model": "laya:en", "keep_alive": "0s"}));
    assert_eq!(
        KeepAlive::from_json(&sent["keep_alive"]).unwrap(),
        Some(KeepAlive::UNLOAD)
    );
}

#[tokio::test]
async fn base_path_prefix_and_heartbeat() {
    let (base, server) = serve_once(vec![json_response("200 OK", r#"{"version":"0.1.0"}"#)]).await;
    let client = Client::new(&format!("http://{base}/ollaya/")).unwrap();
    assert_eq!(client.version().await.unwrap().version, "0.1.0");
    assert!(
        server
            .await
            .unwrap()
            .starts_with("GET /ollaya/api/version HTTP/1.1")
    );

    let (base, server) = serve_once(vec![
        b"HTTP/1.1 200 OK\r\nContent-Length: 17\r\nConnection: close\r\n\r\n".to_vec(),
    ])
    .await;
    Client::new(&base).unwrap().heartbeat().await.unwrap();
    assert!(server.await.unwrap().starts_with("HEAD / HTTP/1.1"));
}

#[tokio::test]
async fn connection_refused_is_a_connect_error() {
    // Bind then drop a listener to get a port nothing listens on.
    let port = TcpListener::bind("127.0.0.1:0")
        .await
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let err = Client::new(&format!("127.0.0.1:{port}"))
        .unwrap()
        .heartbeat()
        .await
        .unwrap_err();
    assert!(matches!(err, ClientError::Connect { .. }), "{err:?}");
}
