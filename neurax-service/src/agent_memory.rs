//! Agentic memory for `neurax-agent`'s step-by-step loop — real state that
//! survives one run, instead of what the loop had before this: a caller
//! resending its whole conversation history every request, and every
//! finished run's own reasoning (why a design turned out the way it did)
//! discarded the moment the SSE stream closed.
//!
//! ## Scoped by `project_id` only — not `user_id`, and this is a deliberate
//! ## correction from the original design
//!
//! The plan this module implements originally called for memory scoped by
//! project *and* user, with a user-level fallback. Verified before writing
//! any of this: `neurax-ui` has no real Supabase auth integration anywhere
//! — `AuthContext.tsx` is its own module comment's word for it, "a local
//! profile," never validated against this service's Supabase project, and
//! `AIChatDrawer.tsx`'s `/runs` request carries no user identity of any
//! kind today. Building a `user_id`-keyed table against an identity that
//! does not exist yet would be memory nothing could ever actually read back
//! correctly. `project_id` is real: `Index.tsx` already tracks
//! `currentProjectId` as live state, one prop away from reaching the
//! request that needs it. Scoping to that alone, and only that, is what
//! this module actually does — a real per-user tier is future work gated
//! on `neurax-ui` getting a real Supabase session, not on anything here.
//!
//! ## Root-level endpoints, not `/agent/*`
//!
//! The same correction already made once this session for `neurax-agent`'s
//! analysis tools applies again here: `/agent/*` requires an API key with
//! the `agent` scope (`check_api_key_scope`), and `neurax-agent` has no
//! service-level credential anywhere in its own codebase to present one
//! with. These endpoints sit at the root, reachable the same
//! no-credential way `neurax-agent` already reaches `/analyze` and
//! `/sweep`, rather than gated behind an auth story nothing can satisfy.
//!
//! ## Required Supabase schema
//!
//! No migrations directory exists in this repo — the Supabase schema is
//! managed outside it. These three tables need to exist before this module
//! can do anything but return empty results / fail writes:
//!
//! ```sql
//! create table agent_core_memory (
//!     project_id text primary key,
//!     preferences jsonb not null default '[]'::jsonb,
//!     updated_at timestamptz not null default now()
//! );
//!
//! create table agent_archival_memory (
//!     id uuid primary key default gen_random_uuid(),
//!     project_id text not null,
//!     content text not null,
//!     created_at timestamptz not null default now()
//! );
//! create index on agent_archival_memory (project_id, created_at desc);
//!
//! create table agent_conversation_log (
//!     id uuid primary key default gen_random_uuid(),
//!     project_id text not null,
//!     role text not null,
//!     content text not null,
//!     created_at timestamptz not null default now()
//! );
//! create index on agent_conversation_log (project_id, created_at desc);
//! ```
//!
//! Archival search is keyword matching (PostgREST `ilike`), not semantic —
//! the plan's own stated v1 fallback for a Supabase instance with no
//! `pgvector` extension confirmed available. Upgrading to real vector
//! search is a real, separate follow-up, not simulated here.

use actix_web::HttpResponse;
use serde::{Deserialize, Serialize};

use crate::supabase_rest_client;

#[derive(Debug, Serialize, Deserialize)]
struct CoreMemoryRow {
    project_id: String,
    #[serde(default)]
    preferences: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ArchivalRow {
    #[serde(default)]
    id: Option<String>,
    project_id: String,
    content: String,
    #[serde(default)]
    created_at: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ConversationRow {
    #[serde(default)]
    id: Option<String>,
    project_id: String,
    role: String,
    content: String,
    #[serde(default)]
    created_at: Option<String>,
}

/// The preferences list for a project, or empty if none has ever been
/// recorded — a project with no memory yet is a normal, expected state,
/// not an error.
pub async fn get_core_preferences(project_id: &str) -> Result<Vec<String>, HttpResponse> {
    let (supabase_url, service_role_key, client) = supabase_rest_client().await?;
    let url = format!("{}/rest/v1/agent_core_memory", supabase_url.trim_end_matches('/'));

    let res = client
        .get(url)
        .query(&[
            ("project_id", format!("eq.{project_id}")),
            ("select", "preferences".to_string()),
        ])
        .header("apikey", &service_role_key)
        .header(reqwest::header::AUTHORIZATION, format!("Bearer {service_role_key}"))
        .header(reqwest::header::ACCEPT, "application/json")
        .send()
        .await
        .map_err(|_| HttpResponse::BadGateway().body("Failed to reach Supabase REST"))?;

    if !res.status().is_success() {
        return Err(HttpResponse::BadGateway().body("Supabase REST returned an error"));
    }

    let rows: Vec<CoreMemoryRow> = res
        .json()
        .await
        .map_err(|_| HttpResponse::BadGateway().body("Supabase REST returned invalid JSON"))?;
    Ok(rows.into_iter().next().map(|r| r.preferences).unwrap_or_default())
}

/// Append one preference, read-modify-write. Not race-safe under
/// concurrent writers to the same project (two simultaneous runs could
/// both read the same starting list and one append could clobber the
/// other) — a real, known limitation, not an oversight; a project's own
/// runs are already serialized in practice by the loop's own step-by-step
/// nature, and this is meant for "remember one stated preference," not a
/// high-throughput write path.
pub async fn add_core_preference(project_id: &str, preference: &str) -> Result<(), HttpResponse> {
    let mut current = get_core_preferences(project_id).await.unwrap_or_default();
    if current.iter().any(|p| p == preference) {
        return Ok(()); // already remembered, nothing to do
    }
    current.push(preference.to_string());

    let (supabase_url, service_role_key, client) = supabase_rest_client().await?;
    let url = format!("{}/rest/v1/agent_core_memory", supabase_url.trim_end_matches('/'));

    let body = serde_json::json!({
        "project_id": project_id,
        "preferences": current,
        "updated_at": chrono::Utc::now().to_rfc3339(),
    });

    let res = client
        .post(url)
        .header("apikey", &service_role_key)
        .header(reqwest::header::AUTHORIZATION, format!("Bearer {service_role_key}"))
        // Upsert on the primary key rather than a separate get-or-insert
        // round trip — one project, one core-memory row.
        .header("Prefer", "resolution=merge-duplicates,return=minimal")
        .json(&body)
        .send()
        .await
        .map_err(|_| HttpResponse::BadGateway().body("Failed to reach Supabase REST"))?;

    if !res.status().is_success() {
        return Err(HttpResponse::BadGateway().body("Supabase core-memory write failed"));
    }
    Ok(())
}

/// Keyword search over a project's archival memory — real matches, not a
/// ranked/semantic result set (see the module doc for why). Empty `query`
/// returns the most recent entries instead of matching nothing.
pub async fn search_archival(
    project_id: &str,
    query: &str,
    limit: u32,
) -> Result<Vec<String>, HttpResponse> {
    let (supabase_url, service_role_key, client) = supabase_rest_client().await?;
    let url = format!("{}/rest/v1/agent_archival_memory", supabase_url.trim_end_matches('/'));

    let mut params = vec![
        ("project_id".to_string(), format!("eq.{project_id}")),
        ("select".to_string(), "content".to_string()),
        ("order".to_string(), "created_at.desc".to_string()),
        ("limit".to_string(), limit.to_string()),
    ];
    if !query.trim().is_empty() {
        params.push(("content".to_string(), format!("ilike.*{}*", query.trim())));
    }

    let res = client
        .get(url)
        .query(&params)
        .header("apikey", &service_role_key)
        .header(reqwest::header::AUTHORIZATION, format!("Bearer {service_role_key}"))
        .header(reqwest::header::ACCEPT, "application/json")
        .send()
        .await
        .map_err(|_| HttpResponse::BadGateway().body("Failed to reach Supabase REST"))?;

    if !res.status().is_success() {
        return Err(HttpResponse::BadGateway().body("Supabase REST returned an error"));
    }

    let rows: Vec<ArchivalRow> = res
        .json()
        .await
        .map_err(|_| HttpResponse::BadGateway().body("Supabase REST returned invalid JSON"))?;
    Ok(rows.into_iter().map(|r| r.content).collect())
}

pub async fn add_archival_entry(project_id: &str, content: &str) -> Result<(), HttpResponse> {
    let (supabase_url, service_role_key, client) = supabase_rest_client().await?;
    let url = format!("{}/rest/v1/agent_archival_memory", supabase_url.trim_end_matches('/'));

    let body = serde_json::json!({ "project_id": project_id, "content": content });
    let res = client
        .post(url)
        .header("apikey", &service_role_key)
        .header(reqwest::header::AUTHORIZATION, format!("Bearer {service_role_key}"))
        .header("Prefer", "return=minimal")
        .json(&body)
        .send()
        .await
        .map_err(|_| HttpResponse::BadGateway().body("Failed to reach Supabase REST"))?;

    if !res.status().is_success() {
        return Err(HttpResponse::BadGateway().body("Supabase archival-memory write failed"));
    }
    Ok(())
}

/// Most recent conversation turns for a project, oldest first — the shape
/// `neurax-agent`'s `ConversationTurn` already expects, so this can be
/// prepended directly to whatever the caller sends without reshaping.
pub async fn get_recent_conversation(
    project_id: &str,
    limit: u32,
) -> Result<Vec<(String, String)>, HttpResponse> {
    let (supabase_url, service_role_key, client) = supabase_rest_client().await?;
    let url = format!("{}/rest/v1/agent_conversation_log", supabase_url.trim_end_matches('/'));

    let res = client
        .get(url)
        .query(&[
            ("project_id", format!("eq.{project_id}")),
            ("select", "role,content".to_string()),
            ("order", "created_at.desc".to_string()),
            ("limit", limit.to_string()),
        ])
        .header("apikey", &service_role_key)
        .header(reqwest::header::AUTHORIZATION, format!("Bearer {service_role_key}"))
        .header(reqwest::header::ACCEPT, "application/json")
        .send()
        .await
        .map_err(|_| HttpResponse::BadGateway().body("Failed to reach Supabase REST"))?;

    if !res.status().is_success() {
        return Err(HttpResponse::BadGateway().body("Supabase REST returned an error"));
    }

    let mut rows: Vec<ConversationRow> = res
        .json()
        .await
        .map_err(|_| HttpResponse::BadGateway().body("Supabase REST returned invalid JSON"))?;
    // Fetched newest-first (so `limit` keeps the *most recent* N turns, not
    // the oldest N) — reversed here so the caller gets chronological order.
    rows.reverse();
    Ok(rows.into_iter().map(|r| (r.role, r.content)).collect())
}

pub async fn append_conversation_turns(
    project_id: &str,
    turns: &[(String, String)],
) -> Result<(), HttpResponse> {
    if turns.is_empty() {
        return Ok(());
    }
    let (supabase_url, service_role_key, client) = supabase_rest_client().await?;
    let url = format!("{}/rest/v1/agent_conversation_log", supabase_url.trim_end_matches('/'));

    let body: Vec<serde_json::Value> = turns
        .iter()
        .map(|(role, content)| serde_json::json!({
            "project_id": project_id,
            "role": role,
            "content": content,
        }))
        .collect();

    let res = client
        .post(url)
        .header("apikey", &service_role_key)
        .header(reqwest::header::AUTHORIZATION, format!("Bearer {service_role_key}"))
        .header("Prefer", "return=minimal")
        .json(&body)
        .send()
        .await
        .map_err(|_| HttpResponse::BadGateway().body("Failed to reach Supabase REST"))?;

    if !res.status().is_success() {
        return Err(HttpResponse::BadGateway().body("Supabase conversation-log write failed"));
    }
    Ok(())
}
