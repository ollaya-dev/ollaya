//! `ollaya mcp` over stdio, driven by the official MCP SDK's client: the handshake, the tool and
//! resource listings, a preset resource, and a tool error. None of it needs a running daemon.

use rmcp::ServiceExt;
use rmcp::model::{CallToolRequestParams, ReadResourceRequestParams, ResourceContents};
use rmcp::transport::{ConfigureCommandExt, TokioChildProcess};
use serde_json::json;

async fn client() -> rmcp::service::RunningService<rmcp::RoleClient, ()> {
    let home = tempfile::tempdir().unwrap();
    let transport = TokioChildProcess::new(
        tokio::process::Command::new(env!("CARGO_BIN_EXE_ollaya")).configure(|cmd| {
            cmd.arg("mcp")
                // Nothing in these tests may reach (or start) a real daemon.
                .env("OLLAYA_HOST", "127.0.0.1:9")
                .env("HOME", home.path());
        }),
    )
    .unwrap();
    ().serve(transport).await.unwrap()
}

#[tokio::test]
async fn handshake_tools_and_resources() {
    let client = client().await;
    let info = client.peer_info().unwrap();
    assert_eq!(info.server_info.as_ref().unwrap().name, "ollaya");
    assert!(
        info.instructions
            .as_deref()
            .unwrap_or_default()
            .contains("decide")
    );

    let mut tools: Vec<_> = client
        .list_all_tools()
        .await
        .unwrap()
        .into_iter()
        .map(|t| t.name.to_string())
        .collect();
    tools.sort();
    assert_eq!(tools, ["decide", "list_models", "pull_model", "show_model"]);
    let decide = client
        .list_all_tools()
        .await
        .unwrap()
        .into_iter()
        .find(|t| t.name == "decide")
        .unwrap();
    let schema = serde_json::to_value(&decide.input_schema).unwrap();
    assert_eq!(schema["required"], json!(["state"]));
    for field in ["model", "state", "questions", "preset"] {
        assert!(
            schema["properties"][field].is_object(),
            "decide has no {field}: {schema}"
        );
    }

    let uris: Vec<_> = client
        .list_all_resources()
        .await
        .unwrap()
        .into_iter()
        .map(|r| r.uri.clone())
        .collect();
    assert!(uris.contains(&"ollaya://models".to_owned()));
    assert!(uris.contains(&"ollaya://presets/triage".to_owned()));

    let triage = client
        .read_resource(ReadResourceRequestParams::new("ollaya://presets/triage"))
        .await
        .unwrap();
    let ResourceContents::TextResourceContents { text, .. } = &triage.contents[0] else {
        panic!("not text: {triage:?}");
    };
    let questions: serde_json::Value = serde_json::from_str(text).unwrap();
    assert_eq!(questions["intent"]["type"], "choice");

    client.cancel().await.unwrap();
}

#[tokio::test]
async fn bad_arguments_are_tool_errors() {
    let client = client().await;
    let call = |args: serde_json::Value| {
        CallToolRequestParams::new("decide").with_arguments(args.as_object().unwrap().clone())
    };
    let r = client
        .call_tool(call(json!({"state": "hi", "preset": "nope"})))
        .await
        .unwrap();
    assert_eq!(r.is_error, Some(true));
    let r = client
        .call_tool(call(
            json!({"state": "hi", "preset": "triage", "questions": {}}),
        ))
        .await
        .unwrap();
    assert_eq!(r.is_error, Some(true));
    client.cancel().await.unwrap();
}
