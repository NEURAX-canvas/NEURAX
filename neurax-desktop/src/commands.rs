//! Native capabilities the browser cannot offer.
//!
//! Everything NEURAX *analyses* goes through the embedded HTTP API, unchanged
//! from the web build. These commands cover only the things a web page is
//! structurally unable to do: write to a path the user chose, read a file the
//! user picked, and hand a URL to the real browser.

use serde::Serialize;
use tauri::{AppHandle, Manager};
use tauri_plugin_dialog::DialogExt;

/// What the UI needs to know about the process hosting it.
#[derive(Debug, Clone, Serialize)]
pub struct DesktopInfo {
    /// Base URL of the in-process API.
    pub api_base: String,
    /// Application version, from `Cargo.toml`.
    pub version: String,
    /// `linux`, `macos` or `windows`.
    pub platform: String,
}

/// Handle stored in Tauri's state so commands can read the API address.
pub struct ApiBase(pub String);

#[tauri::command]
pub fn desktop_info(app: AppHandle) -> DesktopInfo {
    DesktopInfo {
        api_base: app.state::<ApiBase>().0.clone(),
        version: app.package_info().version.to_string(),
        platform: std::env::consts::OS.to_string(),
    }
}

/// Save text to a path the user picks.
///
/// The web build can only push a file into the browser's download directory
/// under a name the browser decides. Here the user chooses the directory and
/// the name, and gets the real path back so the UI can say where it went.
#[tauri::command]
pub async fn save_text_file(
    app: AppHandle,
    default_name: String,
    contents: String,
) -> Result<Option<String>, String> {
    let (tx, mut rx) = tauri::async_runtime::channel(1);

    app.dialog()
        .file()
        .set_file_name(&default_name)
        .save_file(move |path| {
            let _ = tx.blocking_send(path);
        });

    let Some(Some(path)) = rx.recv().await else {
        // The user dismissed the dialog. Not an error — report "nothing saved"
        // so the UI can stay quiet rather than showing a failure toast.
        return Ok(None);
    };

    let path = path.into_path().map_err(|e| e.to_string())?;
    std::fs::write(&path, contents).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(Some(path.display().to_string()))
}

/// Read a file the user picks, returning its name and contents.
#[tauri::command]
pub async fn open_text_file(
    app: AppHandle,
    extensions: Vec<String>,
) -> Result<Option<(String, String)>, String> {
    let (tx, mut rx) = tauri::async_runtime::channel(1);

    let exts: Vec<&str> = extensions.iter().map(|s| s.as_str()).collect();
    app.dialog()
        .file()
        .add_filter("NEURAX", &exts)
        .pick_file(move |path| {
            let _ = tx.blocking_send(path);
        });

    let Some(Some(path)) = rx.recv().await else {
        return Ok(None);
    };

    let path = path.into_path().map_err(|e| e.to_string())?;
    let contents =
        std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string());
    Ok(Some((name, contents)))
}
