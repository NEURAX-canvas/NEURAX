//! Native capabilities the browser cannot offer.
//!
//! Everything NEURAX *analyses* goes through the embedded HTTP API, unchanged
//! from the web build. These commands cover only the things a web page is
//! structurally unable to do: write to a path the user chose, read a file the
//! user picked, and hand a URL to the real browser.

use serde::Serialize;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
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

/// Paths the user has chosen through a dialog *and* meant to write to.
///
/// `write_text_file` exists so that Ctrl+S can overwrite the file already open
/// instead of asking for its name again — but a command that writes to any path
/// the page names is a much larger capability than the dialogs it sits beside,
/// and the page is a webview rendering content this application does not fully
/// control. So writing is limited to paths the *user* has already pointed at.
///
/// Pointing at a file is not by itself consent to overwrite it, and the two
/// intents must not be merged. Opening is used for more than documents: the
/// import dialog reads a HuggingFace `config.json` straight out of a cloned
/// model directory, and that file must not become writable because someone
/// looked at it. So authorisation is granted by the caller that knows the
/// intent — always by Save As, and by Open only for a NEURAX document, which
/// is the one case where a later Ctrl+S is meant to land back on the same file.
///
/// The set is per-run and lives only in memory: authorisation does not outlive
/// the session that granted it.
#[derive(Default)]
pub struct AuthorizedPaths(Mutex<HashSet<PathBuf>>);

impl AuthorizedPaths {
    fn authorize(&self, path: &Path) {
        if let Ok(mut paths) = self.0.lock() {
            paths.insert(path.to_path_buf());
        }
    }

    fn is_authorized(&self, path: &Path) -> bool {
        self.0
            .lock()
            .map(|paths| paths.contains(path))
            .unwrap_or(false)
    }
}

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
    app.state::<AuthorizedPaths>().authorize(&path);
    Ok(Some(path.display().to_string()))
}

/// Overwrite a file the user has already chosen, without asking again.
///
/// This is the difference between "Save" and "Save As". A design that lives in
/// a file the user named should be re-saved by pressing Ctrl+S, not by walking
/// through the save dialog every time — that is the whole point of a document
/// having a path.
///
/// The path must be one the user selected earlier in this session; see
/// [`AuthorizedPaths`]. A rejection here is a bug in the caller, not something
/// the user can fix, so it reports as an error rather than a silent no-op.
#[tauri::command]
pub fn write_text_file(app: AppHandle, path: String, contents: String) -> Result<String, String> {
    let target = PathBuf::from(&path);

    if !app.state::<AuthorizedPaths>().is_authorized(&target) {
        return Err(format!(
            "{path}: not a file this session opened or saved. Use Save As to choose it."
        ));
    }

    std::fs::write(&target, contents).map_err(|e| format!("{path}: {e}"))?;
    Ok(target.display().to_string())
}

/// A file the user picked, and where it came from.
#[derive(Debug, Clone, Serialize)]
pub struct PickedFile {
    /// Base name, for display.
    pub name: String,
    /// Full path, so a later Save can write back to the same file.
    pub path: String,
    pub contents: String,
}

/// Read a file the user picks, returning its name, path and contents.
///
/// `writable` says whether this open is the start of an editing session on that
/// file — true when opening a NEURAX document, false when merely reading one
/// in, such as a `config.json` being imported. Only the former makes the path
/// eligible for a later in-place save; see [`AuthorizedPaths`].
#[tauri::command]
pub async fn open_text_file(
    app: AppHandle,
    extensions: Vec<String>,
    writable: Option<bool>,
) -> Result<Option<PickedFile>, String> {
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

    // Only an open that begins an editing session grants write access. The
    // default is no: a caller that does not say must not silently get it.
    if writable.unwrap_or(false) {
        app.state::<AuthorizedPaths>().authorize(&path);
    }

    Ok(Some(PickedFile {
        name,
        path: path.display().to_string(),
        contents,
    }))
}

#[cfg(test)]
mod tests {
    use super::AuthorizedPaths;
    use std::path::PathBuf;

    #[test]
    fn a_path_is_not_writable_until_the_user_has_chosen_it() {
        let paths = AuthorizedPaths::default();
        let target = PathBuf::from("/tmp/neurax-design.neurax");
        assert!(
            !paths.is_authorized(&target),
            "an arbitrary path must not be writable"
        );

        paths.authorize(&target);
        assert!(paths.is_authorized(&target));
    }

    /// Reading a file is not consent to overwrite it.
    ///
    /// The import dialog opens a `config.json` out of a model directory. If
    /// that open authorised writing, the page could then overwrite a file the
    /// user only asked it to read.
    #[test]
    fn reading_a_file_does_not_make_it_writable() {
        let paths = AuthorizedPaths::default();
        let config = PathBuf::from("/home/someone/models/llama/config.json");

        // What `open_text_file` does when `writable` is absent or false.
        for writable in [None, Some(false)] {
            if writable.unwrap_or(false) {
                paths.authorize(&config);
            }
        }

        assert!(
            !paths.is_authorized(&config),
            "a file opened for reading must not become writable"
        );
    }

    /// Authorising one file must not authorise its neighbours or its directory.
    #[test]
    fn authorisation_does_not_spread_to_other_paths() {
        let paths = AuthorizedPaths::default();
        paths.authorize(&PathBuf::from("/tmp/a.neurax"));

        assert!(!paths.is_authorized(&PathBuf::from("/tmp/b.neurax")));
        assert!(!paths.is_authorized(&PathBuf::from("/tmp")));
        assert!(!paths.is_authorized(&PathBuf::from("/tmp/a.neurax.bak")));
    }
}
