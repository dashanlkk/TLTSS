use hermes_cfg::error::SkillError;
use crate::manifest::{SkillManifest, SkillStatus};
use std::path::PathBuf;
use tracing::info;
use walkdir::WalkDir;

/// 技能存储：管理 skills 目录
pub struct SkillStore {
    skills_dir: PathBuf,
}

impl SkillStore {
    pub fn new(skills_dir: impl Into<PathBuf>) -> Self {
        Self { skills_dir: skills_dir.into() }
    }

    /// 扫描并加载所有 skills
    pub fn load_all(&self) -> Vec<SkillManifest> {
        let mut skills = Vec::new();
        if !self.skills_dir.exists() {
            return skills;
        }

        for entry in WalkDir::new(&self.skills_dir)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let path = entry.path();
            if path.extension().map(|e| e == "yaml" || e == "yml").unwrap_or(false) {
                match SkillManifest::from_file(path) {
                    Ok(manifest) => {
                        info!("Loaded skill: {}", manifest.name);
                        skills.push(manifest);
                    }
                    Err(e) => {
                        tracing::warn!("Failed to load skill {:?}: {}", path, e);
                    }
                }
            }
        }
        skills
    }

    /// 保存 skill
    pub fn save(&self, manifest: &SkillManifest) -> Result<(), SkillError> {
        if !self.skills_dir.exists() {
            std::fs::create_dir_all(&self.skills_dir)
                .map_err(|e| SkillError::IoError(e.to_string()))?;
        }
        let path = self.skills_dir.join(format!("{}.yaml", manifest.name));
        let yaml = manifest.to_yaml()?;
        std::fs::write(&path, yaml)
            .map_err(|e| SkillError::IoError(e.to_string()))?;
        info!("Saved skill: {}", manifest.name);
        Ok(())
    }

    /// 删除 skill
    pub fn delete(&self, name: &str) -> Result<(), SkillError> {
        let path = self.skills_dir.join(format!("{}.yaml", name));
        if path.exists() {
            std::fs::remove_file(&path)
                .map_err(|e| SkillError::IoError(e.to_string()))?;
            info!("Deleted skill: {}", name);
        }
        Ok(())
    }

    /// 查找 skill
    pub fn find(&self, name: &str) -> Option<SkillManifest> {
        let path = self.skills_dir.join(format!("{}.yaml", name));
        if path.exists() {
            SkillManifest::from_file(&path).ok()
        } else {
            None
        }
    }

    /// 发布 skill（draft → published）
    pub fn publish(&self, name: &str) -> Result<SkillManifest, SkillError> {
        let mut manifest = self.find(name)
            .ok_or_else(|| SkillError::NotFound(name.to_string()))?;
        manifest.status = SkillStatus::Published;
        self.save(&manifest)?;
        Ok(manifest)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_manifest(name: &str) -> SkillManifest {
        SkillManifest {
            name: name.to_string(),
            version: "1.0".to_string(),
            description: format!("Test skill {}", name),
            trigger_patterns: vec!["test".to_string()],
            steps: vec![],
            status: SkillStatus::Draft,
        }
    }

    #[test]
    fn new_creates_store_with_given_dir() {
        let dir = tempfile::tempdir().unwrap();
        let store = SkillStore::new(dir.path());
        // Store should be usable even if directory does not exist yet
        let skills = store.load_all();
        assert!(skills.is_empty());
    }

    #[test]
    fn load_all_returns_empty_for_nonexistent_dir() {
        let dir = tempfile::tempdir().unwrap();
        let nonexistent = dir.path().join("does_not_exist");
        let store = SkillStore::new(&nonexistent);
        assert!(store.load_all().is_empty());
    }

    #[test]
    fn save_and_find_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let store = SkillStore::new(dir.path());

        let manifest = sample_manifest("my_skill");
        store.save(&manifest).unwrap();

        let found = store.find("my_skill").unwrap();
        assert_eq!(found.name, "my_skill");
        assert_eq!(found.version, "1.0");
        assert_eq!(found.description, "Test skill my_skill");
    }

    #[test]
    fn find_returns_none_for_missing_skill() {
        let dir = tempfile::tempdir().unwrap();
        let store = SkillStore::new(dir.path());
        assert!(store.find("nonexistent").is_none());
    }

    #[test]
    fn load_all_returns_saved_skills() {
        let dir = tempfile::tempdir().unwrap();
        let store = SkillStore::new(dir.path());

        store.save(&sample_manifest("skill_a")).unwrap();
        store.save(&sample_manifest("skill_b")).unwrap();

        let skills = store.load_all();
        assert_eq!(skills.len(), 2);
        let names: Vec<&str> = skills.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"skill_a"));
        assert!(names.contains(&"skill_b"));
    }

    #[test]
    fn delete_removes_skill() {
        let dir = tempfile::tempdir().unwrap();
        let store = SkillStore::new(dir.path());

        store.save(&sample_manifest("to_delete")).unwrap();
        assert!(store.find("to_delete").is_some());

        store.delete("to_delete").unwrap();
        assert!(store.find("to_delete").is_none());
    }

    #[test]
    fn delete_nonexistent_is_ok() {
        let dir = tempfile::tempdir().unwrap();
        let store = SkillStore::new(dir.path());
        assert!(store.delete("ghost").is_ok());
    }

    #[test]
    fn publish_changes_status_to_published() {
        let dir = tempfile::tempdir().unwrap();
        let store = SkillStore::new(dir.path());

        store.save(&sample_manifest("pub_skill")).unwrap();
        assert_eq!(store.find("pub_skill").unwrap().status, SkillStatus::Draft);

        let published = store.publish("pub_skill").unwrap();
        assert_eq!(published.status, SkillStatus::Published);

        // Verify persistence
        assert_eq!(store.find("pub_skill").unwrap().status, SkillStatus::Published);
    }

    #[test]
    fn publish_returns_error_for_missing_skill() {
        let dir = tempfile::tempdir().unwrap();
        let store = SkillStore::new(dir.path());
        let result = store.publish("missing");
        assert!(result.is_err());
    }

    #[test]
    fn save_creates_directory_if_missing() {
        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("a").join("b").join("c");
        let store = SkillStore::new(&nested);

        assert!(!nested.exists());
        store.save(&sample_manifest("nested_skill")).unwrap();
        assert!(nested.exists());
        assert!(store.find("nested_skill").is_some());
    }

    #[test]
    fn load_all_ignores_non_yaml_files() {
        let dir = tempfile::tempdir().unwrap();
        // Write a non-YAML file into the skills directory
        std::fs::write(dir.path().join("readme.txt"), "not a skill").unwrap();

        let store = SkillStore::new(dir.path());
        assert!(store.load_all().is_empty());
    }
}
