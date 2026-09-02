# neurax-service

**HTTP API service for the NEURAX analytical compiler.**

Part of [NEURAX](https://github.com/rustnew/NEURAX), the analytical compiler for neural architectures. An Actix-Web service exposing the NEURAX analytical compilation pipeline over REST with SSE streaming, authentication and billing.

## Features

- 38 REST endpoints: analysis, models, projects, auth, billing
- Server-Sent Events (SSE) streaming for long analyses
- JWT auth + API-key billing
- CORS-enabled for the web UI

## Run

```bash
cargo run -p neurax-service
```

## License

Proprietary — closed-source, commercial software. All rights reserved.