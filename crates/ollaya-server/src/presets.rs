//! Custom presets: named question sets kept in the model store (`<models>/presets/<name>.json`),
//! next to the six built into Ollaya (`ollaya_api::presets`). A file holds
//! `{"description": ..., "questions": {...}}` in the API's question schema.

use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use ollaya_api::presets::{self as builtin, PresetInfo, PresetResponse};
use ollaya_api::{CreatePresetRequest, Questions};
use serde::{Deserialize, Serialize};

use crate::Error;

#[derive(Debug, Serialize, Deserialize)]
struct PresetFile {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    questions: Questions,
}

#[derive(Debug, Clone)]
pub struct PresetStore {
    dir: PathBuf,
}

fn io(e: std::io::Error) -> Error {
    Error::Registry(ollaya_registry::Error::Io(e))
}

impl PresetStore {
    /// The presets of the model store at `root`.
    pub fn new(root: &Path) -> Self {
        PresetStore {
            dir: root.join("presets"),
        }
    }

    fn path(&self, name: &str) -> PathBuf {
        self.dir.join(format!("{name}.json"))
    }

    fn read(&self, name: &str) -> Result<Option<(PresetFile, DateTime<Utc>)>, Error> {
        if !builtin::valid_name(name) || builtin::is_builtin(name) {
            return Ok(None);
        }
        let path = self.path(name);
        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(io(e)),
        };
        let file: PresetFile = serde_json::from_str(&text)
            .map_err(|e| Error::Corrupt(format!("preset {}: {e}", path.display())))?;
        let modified = std::fs::metadata(&path)
            .and_then(|m| m.modified())
            .map(DateTime::<Utc>::from)
            .map_err(io)?;
        Ok(Some((file, modified)))
    }

    /// A preset, built-in or custom.
    pub fn get(&self, name: &str) -> Result<Option<PresetResponse>, Error> {
        if let Some(q) = builtin::get(name) {
            let questions = serde_json::from_value(q).expect("built-in presets are valid");
            return Ok(Some(PresetResponse {
                name: name.to_owned(),
                builtin: true,
                description: builtin::describe(name).map(str::to_owned),
                questions,
                modified_at: None,
            }));
        }
        Ok(self.read(name)?.map(|(file, modified)| PresetResponse {
            name: name.to_owned(),
            builtin: false,
            description: file.description,
            questions: file.questions,
            modified_at: Some(modified),
        }))
    }

    /// Built-in presets in their fixed order, then custom ones by name. A custom file that no
    /// longer parses is skipped, so one bad file does not hide the others.
    pub fn list(&self) -> Result<Vec<PresetInfo>, Error> {
        let info = |p: PresetResponse| PresetInfo {
            name: p.name,
            builtin: p.builtin,
            description: p.description,
            questions: p.questions.keys().cloned().collect(),
            modified_at: p.modified_at,
        };
        let mut out: Vec<PresetInfo> = builtin::NAMES
            .iter()
            .filter_map(|n| self.get(n).ok().flatten().map(info))
            .collect();
        let entries = match std::fs::read_dir(&self.dir) {
            Ok(e) => e,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(out),
            Err(e) => return Err(io(e)),
        };
        let mut names: Vec<String> = entries
            .filter_map(Result::ok)
            .filter_map(|e| {
                e.file_name()
                    .to_str()
                    .and_then(|f| f.strip_suffix(".json"))
                    .map(str::to_owned)
            })
            .filter(|n| builtin::valid_name(n) && !builtin::is_builtin(n))
            .collect();
        names.sort();
        for name in names {
            match self.get(&name) {
                Ok(Some(p)) => out.push(info(p)),
                Ok(None) => {}
                Err(e) => tracing::warn!("skipping preset {name}: {e}"),
            }
        }
        Ok(out)
    }

    /// Create or replace a custom preset. Validation (name, built-in names, questions) happened
    /// at the API boundary; this re-checks the name, because it becomes a file name.
    pub fn put(&self, req: &CreatePresetRequest) -> Result<(), Error> {
        if !builtin::valid_name(&req.name) || builtin::is_builtin(&req.name) {
            return Err(Error::InvalidRequest(format!(
                "{:?} cannot be a custom preset name",
                req.name
            )));
        }
        std::fs::create_dir_all(&self.dir).map_err(io)?;
        let file = PresetFile {
            description: req.description.clone(),
            questions: req.questions.clone(),
        };
        let bytes = serde_json::to_vec_pretty(&file).expect("presets serialize");
        // Write, then rename: a reader never sees half a file.
        let tmp = self.dir.join(format!(".{}.json.tmp", req.name));
        std::fs::write(&tmp, bytes).map_err(io)?;
        std::fs::rename(&tmp, self.path(&req.name)).map_err(io)
    }

    /// Delete a custom preset: `Ok(false)` if there is none by that name. Built-in presets
    /// cannot be deleted ([`Error::BuiltinPreset`]).
    pub fn delete(&self, name: &str) -> Result<bool, Error> {
        if builtin::is_builtin(name) {
            return Err(Error::BuiltinPreset(name.to_owned()));
        }
        if !builtin::valid_name(name) {
            return Ok(false);
        }
        match std::fs::remove_file(self.path(name)) {
            Ok(()) => Ok(true),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(e) => Err(io(e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn request(name: &str) -> CreatePresetRequest {
        CreatePresetRequest {
            name: name.into(),
            questions: serde_json::from_value(json!({
                "billing": {"type": "noul", "instructions": "Is it about billing?"},
                "tone": {"type": "choice", "criteria": {"calm": null, "angry": null}},
            }))
            .unwrap(),
            description: Some("Billing check".into()),
        }
    }

    #[test]
    fn create_list_show_delete() {
        let dir = tempfile::tempdir().unwrap();
        let store = PresetStore::new(dir.path());
        let builtins = store.list().unwrap();
        assert_eq!(builtins.len(), builtin::NAMES.len());
        assert!(
            builtins
                .iter()
                .all(|p| p.builtin && p.modified_at.is_none())
        );

        store.put(&request("billing")).unwrap();
        let list = store.list().unwrap();
        let mine = list.last().unwrap();
        assert_eq!((mine.name.as_str(), mine.builtin), ("billing", false));
        assert_eq!(mine.questions, ["billing", "tone"]);
        let shown = store.get("billing").unwrap().unwrap();
        assert_eq!(shown.description.as_deref(), Some("Billing check"));
        // Key order is kept, as the caller gave it.
        assert_eq!(
            shown.questions.keys().collect::<Vec<_>>(),
            ["billing", "tone"]
        );

        // Replacing keeps one file.
        store.put(&request("billing")).unwrap();
        assert_eq!(store.list().unwrap().len(), builtin::NAMES.len() + 1);

        assert!(store.delete("billing").unwrap());
        assert!(!store.delete("billing").unwrap());
        assert!(store.get("billing").unwrap().is_none());
    }

    #[test]
    fn built_in_presets_are_protected() {
        let dir = tempfile::tempdir().unwrap();
        let store = PresetStore::new(dir.path());
        assert!(store.get("triage").unwrap().unwrap().builtin);
        assert!(matches!(
            store.delete("triage"),
            Err(Error::BuiltinPreset(_))
        ));
        assert!(store.put(&request("triage")).is_err());
        assert!(store.put(&request("../escape")).is_err());
        // A file named like a built-in preset never shadows it.
        std::fs::create_dir_all(dir.path().join("presets")).unwrap();
        std::fs::write(
            dir.path().join("presets/triage.json"),
            r#"{"questions": {}}"#,
        )
        .unwrap();
        assert!(store.get("triage").unwrap().unwrap().builtin);
        assert_eq!(store.list().unwrap().len(), builtin::NAMES.len());
    }
}
