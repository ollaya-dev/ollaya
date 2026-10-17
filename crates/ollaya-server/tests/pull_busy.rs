use ollaya_registry::ModelName;
use ollaya_server::Error;
use ollaya_server::config::ServerConfig;
use ollaya_server::http::{RunnerLaunch, build, error_body};

#[tokio::test(flavor = "current_thread")]
async fn delete_respects_a_pull_started_with_an_explicit_scheme() {
    for (pulling, deleting) in [
        (
            "https://ollaya.dev/library/laya:en",
            "ollaya.dev/library/laya:en",
        ),
        (
            "http://localhost:8080/library/laya:en",
            "localhost:8080/library/laya:en",
        ),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let models = dir.path().display().to_string();
        let config =
            ServerConfig::from_lookup(|k| (k == "OLLAYA_MODELS").then(|| models.clone())).unwrap();
        let state = build(
            config,
            RunnerLaunch {
                exe: std::env::current_exe().unwrap(),
                arg0: None,
                env: vec![],
                llama_dir: None,
                cpu_exe: None,
            },
        )
        .unwrap();
        assert_eq!(
            state
                .ollaya
                .store
                .manifest_path(&ModelName::parse(pulling).unwrap()),
            state
                .ollaya
                .store
                .manifest_path(&ModelName::parse(deleting).unwrap()),
        );
        // Claim the pull without yielding to its download task. Deleting an absent name also
        // completes without yielding, so this test needs no registry or listener.
        let _rx = state.ollaya.pull(pulling).unwrap();
        let result = state.ollaya.delete(deleting).await;
        assert!(matches!(result, Err(Error::Busy(_))), "{result:?}");
        assert_eq!(error_body(&result.unwrap_err(), false).status(), 409);
    }
}
