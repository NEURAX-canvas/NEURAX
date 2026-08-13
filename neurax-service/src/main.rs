//! Standalone NEURAX HTTP service.
//!
//! Everything of substance lives in the library half of this crate, which the
//! desktop build mounts in-process. This binary only decides where to listen.

use neurax_service::{run_server, ServerConfig};

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    neurax_service::init_runtime();
    tracing::info!("[STARTUP] Neurax service starting...");
    run_server(ServerConfig::default()).await
}
