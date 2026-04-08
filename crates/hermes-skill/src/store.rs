use hermes_cfg::error::SkillError;
use crate::manifest::{SkillManifest, SkillStatus};
use std::path::{Path, PathBuf};
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
