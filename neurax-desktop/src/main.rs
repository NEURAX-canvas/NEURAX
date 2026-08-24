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

use commands::{ApiBase, AuthorizedPaths};
use tauri::{WebviewUrl, WebviewWindowBuilder};
use tracing_subscriber::EnvFilter;

/// Whether the application draws its own title bar rather than using the
/// system's.
///
/// macOS and Windows always decorate a window: the traffic lights and the
/// minimise/maximise/close buttons are part of the platform, and replacing
/// them would make NEURAX the one application on the machine whose window
/// behaves differently.
///
/// Linux does not guarantee it. Under the desktop this was tested on, the
/// window came back with `_NET_FRAME_EXTENTS = 0, 0, 0, 0` — no frame at all —
/// while `_NET_WM_ALLOWED_ACTIONS` still listed minimise, maximise and close.
/// The window could be controlled; there was simply nothing to click. So on
/// Linux the buttons are drawn by the application, which is what every
/// webview-based desktop application ends up doing there.
const OWN_TITLE_BAR: bool = cfg!(target_os = "linux");

/// Script evaluated before any page code runs.
///
/// The frontend is one codebase serving web and desktop. On the web it reads
/// its API base from a build-time environment variable; here there is no such
/// value to bake in, because the port is assigned at launch. This hands it over
/// at runtime, and its presence is also how the UI knows it is running natively
/// and may offer the file dialogs.
fn bootstrap_script(api_base: &str) -> String {
    format!(
        "window.__NEURAX_DESKTOP__ = Object.freeze({{ apiBase: {}, platform: {}, ownTitleBar: {} }});",
        serde_json::to_string(api_base).expect("a URL string always serialises"),
        serde_json::to_string(std::env::consts::OS).expect("a platform name always serialises"),
        OWN_TITLE_BAR,
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
        .manage(AuthorizedPaths::default())
        .invoke_handler(tauri::generate_handler![
            commands::desktop_info,
            commands::save_text_file,
            commands::save_binary_file,
            commands::write_text_file,
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
                // Window controls. These are the platform's own — the title bar
                // sits where every other application on the system puts it, with
                // the buttons in the order and position the user already knows.
                // Stated explicitly rather than left to the defaults, so that a
                // window without them would have to be a deliberate change.
                // See OWN_TITLE_BAR: the system's chrome where the platform
                // reliably provides it, ours where it does not.
                .decorations(!OWN_TITLE_BAR)
                .resizable(true)
                .minimizable(true)
                .maximizable(true)
                .closable(true)
                .center()
                .initialization_script(&bootstrap_script(&api_base))
                .build()?;
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building NEURAX")
        .run(move |_app, event| {
            // Projects autosave every few seconds; this catches the work done
            // since the last one. `ExitRequested` fires before the process is
            // torn down, which is the last point at which writing is still
            // reliable.
            if matches!(
                event,
                tauri::RunEvent::ExitRequested { .. } | tauri::RunEvent::Exit
            ) {
                api.save_now();
            }
        });
}

#[cfg(test)]
mod tests {
    use super::{bootstrap_script, OWN_TITLE_BAR};

    #[test]
    fn bootstrap_script_embeds_the_api_base() {
        let script = bootstrap_script("http://127.0.0.1:41234");
        assert!(script.contains("\"http://127.0.0.1:41234\""));
        assert!(script.contains("window.__NEURAX_DESKTOP__"));
    }

    /// The base URL is interpolated into JavaScript source, so it goes through
    /// a JSON encoder rather than string formatting.
    ///
    /// Checked by round-tripping rather than by looking for escape sequences:
    /// what matters is not that a backslash appears somewhere, but that the
    /// literal in the emitted script parses back to exactly the input — which
    /// is precisely the property that stops a value from ending the string
    /// early and running as code.
    #[test]
    fn bootstrap_script_escapes_its_input() {
        let hostile = "http://x\";alert(1);//";
        let script = bootstrap_script(hostile);

        // Read exactly one JSON value starting at the field, rather than
        // slicing between quotes. The slicing version passed until a second
        // field was added and the closing quote it looked for stopped
        // belonging to this value.
        let after = script
            .split_once("apiBase:")
            .expect("the bridge carries an apiBase")
            .1
            .trim_start();

        let decoded: String = serde_json::Deserializer::from_str(after)
            .into_iter::<String>()
            .next()
            .expect("a value follows the field")
            .expect("that value is a JSON string");

        assert_eq!(
            decoded, hostile,
            "the value did not survive encoding intact"
        );
    }

    /// The platform and the title-bar decision travel with the API base, so the
    /// frontend can size its own window chrome without asking.
    #[test]
    fn bootstrap_script_declares_the_window_chrome() {
        let script = bootstrap_script("http://127.0.0.1:1");
        assert!(script.contains("platform:"), "{script}");
        assert!(
            script.contains(&format!("ownTitleBar: {OWN_TITLE_BAR}")),
            "{script}"
        );
    }
}
