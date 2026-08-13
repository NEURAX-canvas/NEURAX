//! NEURAX desktop.
//!
//! The same interface as the web application and the same compiler behind it,
//! with the network removed from the middle: the UI is served from the bundle
//! and the API runs inside this process on a loopback socket. Once installed,
//! nothing here needs an internet connection to analyse a model.

// On Windows a `bin` target opens a console window alongside the app unless the
// subsystem says otherwise. Only in release — a console is wanted for `cargo
// run`.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;
mod server;

use commands::ApiBase;
use tauri::{Manager, WebviewUrl, WebviewWindowBuilder};
use tracing_subscriber::EnvFilter;

/// Script evaluated before any page code runs.
///
/// The frontend is one codebase serving web and desktop. On the web it reads
/// its API base from a build-time environment variable; here there is no such
/// value to bake in, because the port is assigned at launch. This hands it over
/// at runtime, and its presence is also how the UI knows it is running natively
/// and may offer the file dialogs.
fn bootstrap_script(api_base: &str) -> String {
    format!(
        "window.__NEURAX_DESKTOP__ = Object.freeze({{ apiBase: {} }});",
        serde_json::to_string(api_base).expect("a URL string always serialises")
    )
}

fn main() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt().with_env_filter(filter).init();

    // Desktop NEURAX is single-user and its API is bound to loopback, so there
    // is no session to authenticate and no second party to authenticate
    // against. The service requires this to be explicit; see `noauth_enabled`.
    std::env::set_var("NEURAX_DEBUG_NOAUTH", "true");

    let api = match server::start() {
        Ok(api) => api,
        Err(err) => {
            // Without the API there is no application, so fail loudly here
            // rather than opening a window that cannot do anything.
            eprintln!("NEURAX could not start its local engine: {err}");
            std::process::exit(1);
        }
    };
    let api_base = api.base_url();

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .manage(ApiBase(api_base.clone()))
        .invoke_handler(tauri::generate_handler![
            commands::desktop_info,
            commands::save_text_file,
            commands::open_text_file,
        ])
        .setup(move |app| {
            // The path is deliberately empty rather than "index.html". The UI
            // routes on `location.pathname`, and its route table ends in a
            // catch-all that renders "not found" — so loading
            // `tauri://localhost/index.html` would open the app on the 404
            // page. `tauri://localhost/` serves the same file and leaves the
            // path at `/`, which is the route the UI expects.
            WebviewWindowBuilder::new(app, "main", WebviewUrl::App("".into()))
                .title("NEURAX")
                .inner_size(1440.0, 900.0)
                // The studio's toolbar needs room; below this the canvas and the
                // analysis panel start overlapping rather than reflowing.
                .min_inner_size(1100.0, 700.0)
                .resizable(true)
                .initialization_script(&bootstrap_script(&api_base))
                .build()?;
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running NEURAX");
}

#[cfg(test)]
mod tests {
    use super::bootstrap_script;

    #[test]
    fn bootstrap_script_embeds_the_api_base() {
        let script = bootstrap_script("http://127.0.0.1:41234");
        assert!(script.contains("\"http://127.0.0.1:41234\""));
        assert!(script.contains("window.__NEURAX_DESKTOP__"));
    }

    /// The base URL is interpolated into JavaScript source, so it goes through
    /// a JSON encoder rather than string formatting.
    #[test]
    fn bootstrap_script_escapes_its_input() {
        let script = bootstrap_script("http://x\";alert(1);//");
        assert!(!script.contains("\";alert(1)"));
        assert!(script.contains("\\\""));
    }
}
