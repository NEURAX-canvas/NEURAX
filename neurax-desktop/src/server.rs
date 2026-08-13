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

use neurax_service::{AppState, DESKTOP_ORIGINS};
use std::io;
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4, TcpListener};

/// A running in-process API.
pub struct EmbeddedApi {
    /// The address the OS actually assigned.
    pub addr: SocketAddr,
}

impl EmbeddedApi {
    /// Base URL the webview should call, e.g. `http://127.0.0.1:41234`.
    pub fn base_url(&self) -> String {
        format!("http://{}", self.addr)
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

    let origins: Vec<String> = DESKTOP_ORIGINS.iter().map(|s| s.to_string()).collect();

    std::thread::Builder::new()
        .name("neurax-api".into())
        .spawn(move || {
            let system = actix_web::rt::System::new();
            let result = system.block_on(async move {
                let server = neurax_service::serve_on_listener(listener, origins, AppState::new())?;
                server.await
            });
            if let Err(err) = result {
                tracing::error!("embedded NEURAX API stopped: {err}");
            }
        })?;

    tracing::info!("embedded NEURAX API listening on http://{addr}");
    Ok(EmbeddedApi { addr })
}
