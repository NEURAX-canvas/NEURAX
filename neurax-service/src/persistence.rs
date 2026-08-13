//! Keeping projects across restarts.
//!
//! `AppState` holds projects in a `DashMap`, which means that until now every
//! saved project was lost the moment the process stopped. On the hosted service
//! that is survivable — it is one process among several and the intent was
//! always that a database would sit behind it. In the desktop application it is
//! not: the process *is* the product, quitting the app is normal, and a design
//! someone spent an hour on has to still be there tomorrow.
//!
//! This module is the smallest thing that fixes it: read a JSON file at start,
//! write it back when the projects change. It deliberately does not touch the
//! request handlers — the state is an `Arc`, so whoever owns the process can
//! snapshot it without the API knowing persistence exists.
//!
//! It is a file, not a database, and that is a real limit: two NEURAX processes
//! pointed at the same file will overwrite each other's projects. One desktop
//! application on one machine is exactly the case where that cannot happen.

use crate::{AppState, Project, ProjectKey};
use std::io;
use std::path::{Path, PathBuf};

/// Reverse-DNS identifier, matching `tauri.conf.json`, used as the directory
/// name so the desktop app and this module agree on where state lives.
pub const APP_ID: &str = "dev.neurax.desktop";

/// The file projects are kept in, inside [`data_dir`].
pub const PROJECTS_FILE: &str = "projects.json";

/// Per-user data directory for this application, following each platform's
/// convention.
///
/// Resolved from environment variables rather than through Tauri's path API so
/// that it can be computed before any window exists — the API server is
/// started, and its projects restored, before the UI is created.
pub fn data_dir() -> PathBuf {
    let base = if cfg!(target_os = "macos") {
        std::env::var_os("HOME").map(|home| PathBuf::from(home).join("Library/Application Support"))
    } else if cfg!(target_os = "windows") {
        std::env::var_os("APPDATA").map(PathBuf::from)
    } else {
        // XDG: $XDG_DATA_HOME, falling back to the value the spec defines.
        std::env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .filter(|path| !path.as_os_str().is_empty())
            .or_else(|| {
                std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".local/share"))
            })
    };

    // A home directory that cannot be determined is not a reason to lose the
    // user's work; fall back to the current directory so it lands somewhere.
    base.unwrap_or_else(|| PathBuf::from(".")).join(APP_ID)
}

/// Full path to the projects file.
pub fn projects_path() -> PathBuf {
    data_dir().join(PROJECTS_FILE)
}

impl AppState {
    /// Every project currently held, in a form that can be written out.
    ///
    /// Returns a `Vec` rather than the map because `ProjectKey` is built from
    /// fields the project already carries, so storing both would let the two
    /// disagree.
    pub fn snapshot_projects(&self) -> Vec<Project> {
        self.projects
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Replace the held projects with the given ones.
    pub fn restore_projects(&self, projects: Vec<Project>) {
        self.projects.clear();
        for project in projects {
            self.projects.insert(ProjectKey::of(&project), project);
        }
    }
}

impl ProjectKey {
    /// The key under which a project is stored.
    pub fn of(project: &Project) -> Self {
        Self {
            user_id: project.user_id.clone(),
            id: project.id.clone(),
        }
    }
}

/// Read projects from `path`.
///
/// A missing file is an empty list, not an error — that is simply the first
/// launch. A file that exists but cannot be parsed *is* an error, and the
/// caller should say so rather than silently starting empty and then
/// overwriting it.
pub fn load_projects(path: &Path) -> io::Result<Vec<Project>> {
    let contents = match std::fs::read_to_string(path) {
        Ok(contents) => contents,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(err) => return Err(err),
    };

    if contents.trim().is_empty() {
        return Ok(Vec::new());
    }

    serde_json::from_str(&contents).map_err(|err| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "{} is not a valid NEURAX project file: {err}",
                path.display()
            ),
        )
    })
}

/// Write projects to `path`, creating the directory if needed.
///
/// Written to a temporary file in the same directory and then renamed, because
/// `rename` within a filesystem is atomic: a crash or a full disk partway
/// through leaves the previous file intact rather than a truncated one. Writing
/// in place would make every save a window in which all the user's projects
/// could be lost.
pub fn save_projects(path: &Path, projects: &[Project]) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let json = serde_json::to_string_pretty(projects)
        .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;

    let temp = path.with_extension("json.tmp");
    std::fs::write(&temp, json)?;
    std::fs::rename(&temp, path)
}

/// How often the autosave thread looks for changes.
///
/// Short enough that a crash costs seconds of work, long enough that a burst of
/// edits becomes one write rather than dozens. Nothing is written when nothing
/// changed, so an idle process does not touch the disk at all.
pub const AUTOSAVE_INTERVAL: std::time::Duration = std::time::Duration::from_secs(5);

/// Load projects into `state`, then keep the file in step with them.
///
/// Shared by the desktop application and the standalone service so that the
/// two behave identically: a project saved in one and a project saved in the
/// other survive a restart the same way. The desktop always calls this; the
/// service calls it when `NEURAX_PROJECTS_FILE` is set, since a deployment
/// running several replicas needs a database rather than a shared file and
/// must opt in deliberately.
///
/// Returns `false` when the existing file could not be read. The caller should
/// carry on — the application is perfectly usable for new work — but nothing
/// further is written, because overwriting a file we failed to parse would
/// destroy whatever it was holding.
pub fn attach(state: &AppState, path: &Path) -> bool {
    match load_projects(path) {
        Ok(projects) => {
            if !projects.is_empty() {
                tracing::info!("restored {} project(s) from {:?}", projects.len(), path);
            }
            state.restore_projects(projects);
        }
        Err(err) => {
            tracing::error!("could not read {:?}: {err}", path);
            tracing::error!("continuing with no projects; this file will not be overwritten");
            return false;
        }
    }

    spawn_autosave(state.clone(), path.to_path_buf());
    true
}

/// Write projects to disk whenever they change.
///
/// Comparing the serialised form rather than tracking mutations means the
/// request handlers need to know nothing about persistence — and it is what
/// makes an idle tick genuinely free, since an unchanged snapshot never
/// reaches the filesystem.
fn spawn_autosave(state: AppState, path: PathBuf) {
    std::thread::Builder::new()
        .name("neurax-autosave".into())
        .spawn(move || {
            let mut last = String::new();
            loop {
                std::thread::sleep(AUTOSAVE_INTERVAL);

                let projects = state.snapshot_projects();
                let Ok(current) = serde_json::to_string(&projects) else {
                    continue;
                };
                if current == last {
                    continue;
                }

                match save_projects(&path, &projects) {
                    Ok(()) => {
                        tracing::debug!("saved {} project(s)", projects.len());
                        last = current;
                    }
                    Err(err) => tracing::error!("could not save projects to {path:?}: {err}"),
                }
            }
        })
        .ok();
}

/// The projects file the standalone service should use, if any.
///
/// Unset means the previous behaviour: projects live in memory for the life of
/// the process. Set it to give a self-hosted deployment the same persistence
/// the desktop application has.
pub fn configured_path() -> Option<PathBuf> {
    std::env::var_os("NEURAX_PROJECTS_FILE")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn project(user: &str, id: &str, name: &str) -> Project {
        Project {
            id: id.to_string(),
            user_id: user.to_string(),
            name: name.to_string(),
            description: None,
            architecture: Some("transformer".to_string()),
            canvas: json!({ "nodes": [], "connections": [] }),
            hardware_config: None,
            last_analysis: None,
            created_at: "2026-01-01T00:00:00Z".to_string(),
            updated_at: "2026-01-01T00:00:00Z".to_string(),
        }
    }

    fn temp_path(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("neurax-persistence-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    #[test]
    fn a_first_launch_starts_empty_rather_than_failing() {
        let path = temp_path("does-not-exist.json");
        let _ = std::fs::remove_file(&path);
        assert_eq!(load_projects(&path).unwrap().len(), 0);
    }

    #[test]
    fn projects_survive_a_write_and_read() {
        let path = temp_path("roundtrip.json");
        let projects = vec![
            project("dev-user", "p1", "GPT-2 small"),
            project("dev-user", "p2", "LLaMA 7B"),
        ];

        save_projects(&path, &projects).unwrap();
        let loaded = load_projects(&path).unwrap();

        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].name, "GPT-2 small");
        assert_eq!(loaded[1].canvas, json!({ "nodes": [], "connections": [] }));
    }

    #[test]
    fn a_corrupt_file_is_reported_not_swallowed() {
        // Silently starting empty would be worse than failing: the next save
        // would overwrite whatever the file actually held.
        let path = temp_path("corrupt.json");
        std::fs::write(&path, "{ this is not json").unwrap();

        let err = load_projects(&path).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("not a valid NEURAX project file"));
    }

    #[test]
    fn an_empty_file_is_treated_as_no_projects() {
        let path = temp_path("empty.json");
        std::fs::write(&path, "   \n").unwrap();
        assert_eq!(load_projects(&path).unwrap().len(), 0);
    }

    #[test]
    fn saving_leaves_no_temporary_file_behind() {
        let path = temp_path("clean.json");
        save_projects(&path, &[project("dev-user", "p1", "One")]).unwrap();
        assert!(!path.with_extension("json.tmp").exists());
    }

    #[test]
    fn state_round_trips_through_a_snapshot() {
        let state = AppState::new();
        state.restore_projects(vec![
            project("dev-user", "p1", "One"),
            project("dev-user", "p2", "Two"),
        ]);

        let mut names: Vec<String> = state
            .snapshot_projects()
            .into_iter()
            .map(|p| p.name)
            .collect();
        names.sort();
        assert_eq!(names, vec!["One", "Two"]);
    }

    #[test]
    fn restoring_replaces_rather_than_merges() {
        let state = AppState::new();
        state.restore_projects(vec![project("dev-user", "p1", "Old")]);
        state.restore_projects(vec![project("dev-user", "p2", "New")]);

        let snapshot = state.snapshot_projects();
        assert_eq!(snapshot.len(), 1);
        assert_eq!(snapshot[0].name, "New");
    }

    /// Two users' projects must not collide: the key is the pair, not the id.
    #[test]
    fn projects_are_keyed_by_user_and_id() {
        let state = AppState::new();
        state.restore_projects(vec![
            project("alice", "p1", "Alice's"),
            project("bob", "p1", "Bob's"),
        ]);
        assert_eq!(state.snapshot_projects().len(), 2);
    }

    #[test]
    fn the_data_directory_is_under_the_user_profile() {
        let dir = data_dir();
        assert!(
            dir.ends_with(APP_ID),
            "unexpected data dir: {}",
            dir.display()
        );
    }
}
