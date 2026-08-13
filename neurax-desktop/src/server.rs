//! The NEURAX API, running inside the desktop process.
//!
//! The desktop build does not shell out to `neurax-service`, and does not
//! reimplement it either: it mounts the very same routing table from
//! `neurax_service::configure_routes` on a loopback socket. Two consequences
//! are worth stating, because they are the reason for this design:
//!
//!   * There is no second process. Nothing to leave running after the window
//!     closes, nothing to fail to start, nothing for a user to have to kill.
//!   * There is no API drift. An endpoint added for the web app is present in
//!     the desktop app the moment it is written, because it is the same code.
//!
//! The one thing this adds on top of the service is persistence. The hosted
//! service keeps projects in memory and always meant to put a database behind
//! it; here the process *is* the product, so projects are read from disk at
//! start and written back as they change.

use neurax_service::persistence;
use neurax_service::{AppState, DESKTOP_ORIGINS};
use std::io;
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4, TcpListener};
use std::path::PathBuf;

/// Browser origins the embedded API accepts.
///
/// A release build serves the UI from the bundle, so the only origin that ever
/// appears is `tauri://localhost`. A debug build is different: Tauri loads
/// `devUrl` instead, which means the page comes from the Vite dev server and
/// presents `http://localhost:8081` as its origin.
///
/// This cost an afternoon to find. The app opened, the window rendered, the
/// studio drew — and every request failed with "Load failed", because CORS was
/// refusing an origin the desktop build had no reason to expect. The API
/// logged nothing, since a rejected response is still a served request as far
/// as the browser's error is concerned. Allowing the dev origins in debug
/// builds only is what makes `cargo tauri dev` work at all; a release build
/// still accepts nothing but the bundle's own origin.
fn allowed_origins() -> Vec<String> {
    let mut origins: Vec<String> = DESKTOP_ORIGINS.iter().map(|s| s.to_string()).collect();
    if cfg!(debug_assertions) {
        origins.extend(neurax_service::default_web_origins());
    }
    origins
}

/// A running in-process API.
pub struct EmbeddedApi {
    /// The address the OS actually assigned.
    pub addr: SocketAddr,
    /// Shared state, so the caller can force a save before quitting.
    pub state: AppState,
    /// Where projects are kept, or `None` when the existing file could not be
    /// read and must not be overwritten.
    pub projects_path: Option<PathBuf>,
}

impl EmbeddedApi {
    /// Base URL the webview should call, e.g. `http://127.0.0.1:41234`.
    pub fn base_url(&self) -> String {
        format!("http://{}", self.addr)
    }

    /// Write projects out now.
    ///
    /// Called when the application is quitting, so the few seconds since the
    /// last autosave are not the ones the user loses.
    pub fn save_now(&self) {
        let Some(path) = &self.projects_path else {
            return;
        };
        let projects = self.state.snapshot_projects();
        if let Err(err) = persistence::save_projects(path, &projects) {
            tracing::error!("could not save projects to {path:?}: {err}");
        }
    }
}

/// Bind the API to an OS-assigned loopback port and serve it on a dedicated
/// thread.
///
/// The socket is bound here, synchronously, rather than inside the server
/// thread: the window's bootstrap script has to carry the port number, so the
/// port must be known before the window is created. Binding first and handing
/// the listener over removes the race entirely — there is no polling, no
/// retry loop, and no "wait for the backend" state in the UI.
///
/// `127.0.0.1` rather than `0.0.0.0` is load-bearing. The API is reachable only
/// from this machine; nothing NEURAX analyses is exposed to the network.
pub fn start() -> io::Result<EmbeddedApi> {
    let listener = TcpListener::bind(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0))?;
    let addr = listener.local_addr()?;

    let state = AppState::new();

    // The same call the standalone service makes when it is configured to
    // persist, so the two cannot drift in how they load, save or recover.
    let path = persistence::projects_path();
    let projects_path = persistence::attach(&state, &path).then_some(path);

    let origins = allowed_origins();
    let served = state.clone();

    std::thread::Builder::new()
        .name("neurax-api".into())
        .spawn(move || {
            let system = actix_web::rt::System::new();
            let result = system.block_on(async move {
                let server = neurax_service::serve_on_listener(listener, origins, served)?;
                server.await
            });
            if let Err(err) = result {
                tracing::error!("embedded NEURAX API stopped: {err}");
            }
        })?;

    tracing::info!("embedded NEURAX API listening on http://{addr}");

    Ok(EmbeddedApi {
        addr,
        state,
        projects_path,
    })
}
