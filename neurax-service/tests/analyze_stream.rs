//! A streaming analysis must actually stream its result, even when the
//! analysis finishes before anyone is listening for it.
//!
//! `tokio::sync::broadcast` does not replay past sends to a late subscriber
//! — and NEURAX's whole point is that analysis is fast, comfortably under
//! 50ms for most models. `/analyze/stream` starts a job and hands back a
//! `job_id`; the caller then opens a *second* connection, `GET
//! /analyze/stream/{job_id}`, to listen for its events. For a model this
//! small, the job routinely finishes and broadcasts Completed/Result before
//! that second connection is even open, let alone subscribed — so the
//! subscriber sat on an empty channel forever: the SSE response never
//! closed, the client's "Analyzing…" state had nothing to end it, and the
//! studio's Run Analysis button never came back, even though the result had
//! been sitting in `state.results` the whole time.
//!
//! This test does not wait between starting the job and connecting to its
//! stream, on purpose — that gap is the bug.

use neurax_service::{serve_on_listener, AppState, DESKTOP_ORIGINS};
use serde_json::{json, Value};
use std::net::{Ipv4Addr, SocketAddrV4, TcpListener};
use std::time::Duration;

fn desktop_api() -> String {
    std::env::set_var("NEURAX_DEBUG_NOAUTH", "true");

    let listener =
        TcpListener::bind(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0)).expect("loopback binds");
    let addr = listener
        .local_addr()
        .expect("a bound listener has an address");
    let origins: Vec<String> = DESKTOP_ORIGINS.iter().map(|s| s.to_string()).collect();

    std::thread::spawn(move || {
        actix_web::rt::System::new().block_on(async move {
            serve_on_listener(listener, origins, AppState::new())
                .expect("the listener is already bound")
                .await
        })
    });

    format!("http://{addr}")
}

fn client() -> reqwest::blocking::Client {
    // A hung SSE response (the exact regression this file guards against)
    // must fail this test loudly rather than block the suite forever.
    reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .expect("client builds")
}

/// The same small transformer `desktop_parity.rs` uses — small enough that
/// its analysis reliably finishes before a second HTTP round trip can land.
fn topology() -> Value {
    json!({
        "schema_version": "1.0.0",
        "model": {
            "name": "stream-race-check",
            "type": "transformer",
            "global_params": {
                "num_layers": 2,
                "hidden_size": 256,
                "num_heads": 4,
                "intermediate_size": 1024,
                "vocab_size": 32000,
                "sequence_length": 512
            },
            "layers": [
                {"id": "embed", "layer_type": "embedding",
                 "params": {"vocab_size": 32000, "hidden_size": 256}},
                {"id": "layer_0", "layer_type": "attention",
                 "params": {"hidden_size": 256, "num_attention_heads": 4, "intermediate_size": 1024}},
                {"id": "layer_1", "layer_type": "feed_forward",
                 "params": {"hidden_size": 256, "intermediate_size": 1024}}
            ]
        },
        "training": {
            "batch_size": 8,
            "sequence_length": 512,
            "precision": "fp16",
            "learning_rate": 0.0003,
            "num_epochs": 1
        },
        "hardware": {"gpus": [{"name": "RTX4090", "memory_gb": 24, "count": 1}]},
        "data": {"dataset_size": 1000000, "vocab_size": 32000, "num_classes": 0}
    })
}

#[test]
fn a_stream_opened_after_the_job_already_finished_still_delivers_the_result() {
    let base = desktop_api();

    let start = client()
        .post(format!("{base}/analyze/stream"))
        .header("Authorization", "Bearer dev-token")
        .json(&json!({ "topology": topology() }))
        .send()
        .expect("starting the job should succeed");
    assert_eq!(start.status().as_u16(), 202, "job did not start");
    let started: Value = start.json().expect("start response is JSON");
    let job_id = started["job_id"]
        .as_str()
        .expect("job_id present")
        .to_string();
    let token = started["view_token"]
        .as_str()
        .expect("view_token present")
        .to_string();

    // No sleep here — connecting immediately is what used to lose the race.
    let stream = client()
        .get(format!("{base}/analyze/stream/{job_id}?token={token}"))
        .header("Authorization", "Bearer dev-token")
        .send()
        .expect("the stream must answer, not hang until the client timeout");
    assert_eq!(
        stream.status().as_u16(),
        200,
        "stream endpoint rejected the request"
    );

    let body = stream.text().expect("stream body reads");
    assert!(
        body.contains("\"type\":\"Result\"") || body.contains("\"type\": \"Result\""),
        "a late subscriber must still get the Result event; got: {body}"
    );
}
