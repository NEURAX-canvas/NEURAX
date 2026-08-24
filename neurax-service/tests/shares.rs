//! Public share links: create, view, and delete — the growth-loop mechanic
//! that turns an analysis into a link someone else can open with no account
//! and no install. Anonymous by design on both ends, so these tests hold the
//! `edit_token` ownership model to the same standard the routing table
//! itself is held to in `embedded_server.rs`: real requests against a real
//! bound listener, not a call into the handler function directly.

use neurax_service::{serve_on_listener, AppState, DESKTOP_ORIGINS};
use std::net::{Ipv4Addr, SocketAddrV4, TcpListener};

fn spawn_api() -> String {
    let listener = TcpListener::bind(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0))
        .expect("loopback should be bindable");
    let addr = listener
        .local_addr()
        .expect("a bound listener has an address");

    let origins: Vec<String> = DESKTOP_ORIGINS.iter().map(|s| s.to_string()).collect();

    std::thread::spawn(move || {
        actix_web::rt::System::new().block_on(async move {
            let server = serve_on_listener(listener, origins, AppState::new())
                .expect("the listener is already bound");
            server.await
        })
    });

    format!("http://{addr}")
}

fn post_json(url: &str, body: serde_json::Value) -> (u16, serde_json::Value) {
    let response = reqwest::blocking::Client::new()
        .post(url)
        .json(&body)
        .send()
        .expect("the embedded server should answer");
    let status = response.status().as_u16();
    let json = response.json().unwrap_or(serde_json::Value::Null);
    (status, json)
}

fn get(url: &str) -> (u16, serde_json::Value) {
    let response = reqwest::blocking::Client::new()
        .get(url)
        .send()
        .expect("the embedded server should answer");
    let status = response.status().as_u16();
    let json = response.json().unwrap_or(serde_json::Value::Null);
    (status, json)
}

fn get_raw(url: &str) -> reqwest::blocking::Response {
    reqwest::blocking::Client::new()
        .get(url)
        .send()
        .expect("the embedded server should answer")
}

fn delete_with_token(url: &str, token: &str) -> u16 {
    reqwest::blocking::Client::new()
        .delete(url)
        .header("X-Edit-Token", token)
        .send()
        .expect("the embedded server should answer")
        .status()
        .as_u16()
}

fn card_payload() -> serde_json::Value {
    serde_json::json!({
        "mode": "card",
        "display_name": "LLaMA-2 70B, dense",
        "family": "transformer",
        "report": { "totalParams": 70_000_000_000_u64, "trainingCostUsd": 2_000_000.0 },
        "design": { "nodes": [{"type": "attention", "name": "my_internal_codename"}] },
    })
}

#[test]
fn creating_a_share_needs_no_account_and_returns_an_id_and_edit_token() {
    let base = spawn_api();
    let (status, body) = post_json(&format!("{base}/shares"), card_payload());

    assert_eq!(status, 201, "body was: {body}");
    assert!(body["id"].as_str().is_some_and(|s| s.len() == 10));
    assert!(body["edit_token"].as_str().is_some_and(|s| !s.is_empty()));
}

#[test]
fn a_card_share_never_carries_a_topology_even_if_the_client_sent_one() {
    let base = spawn_api();
    let (_, created) = post_json(&format!("{base}/shares"), card_payload());
    let id = created["id"].as_str().unwrap();

    let (status, body) = get(&format!("{base}/shares/{id}"));
    assert_eq!(status, 200);
    assert!(
        body["share"]["design"].is_null(),
        "a Card share leaked a topology: {body}"
    );
}

#[test]
fn viewing_a_share_never_exposes_its_edit_token() {
    let base = spawn_api();
    let (_, created) = post_json(&format!("{base}/shares"), card_payload());
    let id = created["id"].as_str().unwrap();

    let (_, body) = get(&format!("{base}/shares/{id}"));
    assert!(
        body["share"].get("edit_token").is_none(),
        "edit_token must never appear in a public read: {body}"
    );
}

#[test]
fn a_full_share_keeps_the_topology_but_scrubs_the_node_name() {
    let base = spawn_api();
    let mut payload = card_payload();
    payload["mode"] = serde_json::json!("full");
    let (_, created) = post_json(&format!("{base}/shares"), payload);
    let id = created["id"].as_str().unwrap();

    let (_, body) = get(&format!("{base}/shares/{id}"));
    let nodes = body["share"]["design"]["nodes"].as_array().unwrap();
    assert_eq!(nodes.len(), 1);
    let name = nodes[0]["name"].as_str().unwrap();
    assert_ne!(
        name, "my_internal_codename",
        "scrub_design should have replaced the free-text node name"
    );
    assert!(
        name.contains("attention"),
        "the replacement name should still say what kind of block this is, got: {name}"
    );
}

#[test]
fn viewing_a_share_increments_its_view_count() {
    let base = spawn_api();
    let (_, created) = post_json(&format!("{base}/shares"), card_payload());
    let id = created["id"].as_str().unwrap();

    get(&format!("{base}/shares/{id}"));
    let (_, second) = get(&format!("{base}/shares/{id}"));

    assert_eq!(second["share"]["view_count"].as_u64(), Some(2));
}

#[test]
fn an_unknown_share_id_is_a_404_not_a_hang() {
    let base = spawn_api();
    let (status, _) = get(&format!("{base}/shares/doesnotexist"));
    assert_eq!(status, 404);
}

#[test]
fn deleting_with_the_right_edit_token_removes_the_share() {
    let base = spawn_api();
    let (_, created) = post_json(&format!("{base}/shares"), card_payload());
    let id = created["id"].as_str().unwrap();
    let token = created["edit_token"].as_str().unwrap();

    let status = delete_with_token(&format!("{base}/shares/{id}"), token);
    assert_eq!(status, 204);

    let (status, _) = get(&format!("{base}/shares/{id}"));
    assert_eq!(status, 404, "the share should be gone after deletion");
}

#[test]
fn deleting_with_the_wrong_edit_token_is_rejected_and_keeps_the_share() {
    let base = spawn_api();
    let (_, created) = post_json(&format!("{base}/shares"), card_payload());
    let id = created["id"].as_str().unwrap();

    let status = delete_with_token(&format!("{base}/shares/{id}"), "not-the-real-token");
    assert_eq!(status, 403);

    let (status, _) = get(&format!("{base}/shares/{id}"));
    assert_eq!(status, 200, "a rejected delete must not remove the share");
}

#[test]
fn downloading_a_card_share_returns_the_raw_report_with_download_headers() {
    let base = spawn_api();
    let (_, created) = post_json(&format!("{base}/shares"), card_payload());
    let id = created["id"].as_str().unwrap();

    let response = get_raw(&format!("{base}/shares/{id}/download"));
    assert_eq!(response.status().as_u16(), 200);
    let content_disposition = response
        .headers()
        .get(reqwest::header::CONTENT_DISPOSITION)
        .expect("a download must set Content-Disposition")
        .to_str()
        .unwrap()
        .to_string();
    assert!(content_disposition.starts_with("attachment;"));
    assert!(content_disposition.contains(".json"));

    let body: serde_json::Value = response.json().unwrap();
    // Unwrapped — a downloaded file is the content itself, not an API
    // envelope like `{ "share": ... }`.
    assert!(body.get("share").is_none());
    assert_eq!(body["report"]["totalParams"].as_u64(), Some(70_000_000_000));
    assert!(
        body.get("design").is_none(),
        "a Card download must not carry a topology"
    );
}

#[test]
fn downloading_a_full_share_bundles_the_scrubbed_design() {
    let base = spawn_api();
    let mut payload = card_payload();
    payload["mode"] = serde_json::json!("full");
    let (_, created) = post_json(&format!("{base}/shares"), payload);
    let id = created["id"].as_str().unwrap();

    let response = get_raw(&format!("{base}/shares/{id}/download"));
    let body: serde_json::Value = response.json().unwrap();
    let nodes = body["design"]["nodes"].as_array().expect("design.nodes");
    assert_eq!(
        nodes[0]["name"].as_str().unwrap(),
        "attention block",
        "the download must carry the scrubbed name, not the raw client input"
    );
}

#[test]
fn a_share_can_be_viewed_many_times_by_many_people_with_no_cap() {
    // The whole point of a link is that it is not single-use: forwarding it
    // to a team, or posting it publicly, means many different people load
    // the same id. Nothing about creation, viewing, or downloading should
    // restrict who or how many can do that.
    let base = spawn_api();
    let (_, created) = post_json(&format!("{base}/shares"), card_payload());
    let id = created["id"].as_str().unwrap();

    for _ in 0..25 {
        let (status, _) = get(&format!("{base}/shares/{id}"));
        assert_eq!(status, 200);
    }
    let response = get_raw(&format!("{base}/shares/{id}/download"));
    assert_eq!(response.status().as_u16(), 200);

    let (_, final_view) = get(&format!("{base}/shares/{id}"));
    assert_eq!(
        final_view["share"]["view_count"].as_u64(),
        Some(27),
        "25 GETs + 1 download + 1 final GET, all unrestricted"
    );
}

#[test]
fn an_empty_display_name_is_rejected() {
    let base = spawn_api();
    let mut payload = card_payload();
    payload["display_name"] = serde_json::json!("   ");

    let (status, _) = post_json(&format!("{base}/shares"), payload);
    assert_eq!(status, 400);
}
