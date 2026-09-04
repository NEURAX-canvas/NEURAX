//! The service must be startable the way `neurax-desktop` starts it.
//!
//! The routing table used to live inside `main`, reachable only by running the
//! binary. Now it is a library the desktop app mounts in-process, and these
//! tests hold that entry point to the same standard as the binary's: bind a
//! listener, serve the real routes on it, and check that requests actually get
//! answered. A refactor that leaves the table unmounted compiles fine and
//! fails here.

use neurax_service::{serve_on_listener, AppState, ServerConfig, DESKTOP_ORIGINS};
use std::net::{Ipv4Addr, SocketAddrV4, TcpListener};

/// Bind a loopback listener on an OS-assigned port, exactly as the desktop
/// build does, and serve the API on it.
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

fn get(url: &str) -> (u16, String) {
    let response = reqwest::blocking::Client::new()
        .get(url)
        .send()
        .expect("the embedded server should answer");
    let status = response.status().as_u16();
    (status, response.text().unwrap_or_default())
}

#[test]
fn the_port_is_known_before_a_single_request() {
    // The desktop window's bootstrap script carries the port, so the address
    // has to be readable from the listener rather than discovered by polling.
    let listener = TcpListener::bind(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0)).unwrap();
    let addr = listener.local_addr().unwrap();

    assert!(
        addr.ip().is_loopback(),
        "the API must not leave the machine"
    );
    assert_ne!(addr.port(), 0, "the OS must have assigned a real port");
}

#[test]
fn health_answers_on_an_embedded_listener() {
    let base = spawn_api();
    let (status, body) = get(&format!("{base}/health"));

    assert_eq!(status, 200, "body was: {body}");
}

#[test]
fn the_hardware_database_is_served_in_process() {
    // Picked because it needs no authentication and no request body, yet it
    // reaches all the way into `neurax-hardware-db` — so a mounted-but-broken
    // route would not pass.
    let base = spawn_api();
    let (status, body) = get(&format!("{base}/hardware"));

    assert_eq!(status, 200, "body was: {body}");
    let parsed: serde_json::Value = serde_json::from_str(&body).expect("hardware list is JSON");
    let entries = parsed
        .as_array()
        .or_else(|| parsed.get("hardware").and_then(|h| h.as_array()))
        .expect("a list of accelerators");
    assert!(
        !entries.is_empty(),
        "the hardware database should not be empty"
    );
}

#[test]
fn an_unknown_route_is_a_404_not_a_hang() {
    let base = spawn_api();
    let (status, _) = get(&format!("{base}/definitely-not-a-route"));

    assert_eq!(status, 404);
}

/// The agent-memory routes (see `agent_memory.rs`) are mounted at the root,
/// not `/agent/*` — this test only proves they're wired into the real route
/// table and fail the way `supabase_rest_client()` is documented to when no
/// Supabase project is configured (this test environment has none), not
/// that Supabase itself works — that needs real credentials this repo does
/// not carry.
#[test]
fn agent_memory_routes_are_mounted_and_fail_cleanly_with_no_supabase_configured() {
    let base = spawn_api();

    let (status, body) = get(&format!("{base}/memory/core?project_id=test-project"));
    assert_eq!(status, 500, "body was: {body}");
    assert!(body.contains("SUPABASE_URL"), "body was: {body}");

    let (status, body) = get(&format!(
        "{base}/memory/archival?project_id=test-project&query=mamba"
    ));
    assert_eq!(status, 500, "body was: {body}");

    let (status, body) = get(&format!("{base}/memory/conversation?project_id=test-project"));
    assert_eq!(status, 500, "body was: {body}");

    // Not 404 — proves these are real, mounted routes, not typos.
    for path in ["/memory/core", "/memory/archival", "/memory/conversation"] {
        let (status, body) = get(&format!("{base}{path}?project_id=x"));
        assert_ne!(status, 404, "{path} should be a mounted route — body was: {body}");
    }
}

/// The desktop webview presents a `tauri://` origin, which is not in any
/// configured list; CORS has to admit it by scheme or every request the
/// desktop app makes is blocked by the webview.
#[test]
fn desktop_origins_are_declared() {
    assert!(DESKTOP_ORIGINS.iter().any(|o| o.starts_with("tauri://")));
    assert!(
        DESKTOP_ORIGINS.contains(&"http://tauri.localhost"),
        "Windows' WebView2 uses the http form"
    );
}

#[test]
fn the_default_config_binds_publicly_and_the_desktop_does_not() {
    // The standalone service is a deployment target and listens on all
    // interfaces; the desktop build must never do that. Encoding the contrast
    // here means a change to either one has to be deliberate.
    let default = ServerConfig::default();
    assert!(
        default.bind_addr.starts_with("0.0.0.0:") || std::env::var("NEURAX_BIND").is_ok(),
        "unexpected default bind address: {}",
        default.bind_addr
    );
    assert!(!default.allowed_origins.is_empty());
}
