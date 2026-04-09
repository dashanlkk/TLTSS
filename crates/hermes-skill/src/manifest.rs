use serde::{Deserialize, Serialize};
use hermes_cfg::error::SkillError;

/// 技能清单
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillManifest {
    pub name: String,
    pub version: String,
    pub description: String,
    pub trigger_patterns: Vec<String>,
    #[serde(default)]
    pub steps: Vec<SkillStep>,
    #[serde(default)]
    pub status: SkillStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SkillStatus {
    Draft,
    Published,
}

impl Default for SkillStatus {
    fn default() -> Self {
        Self::Draft
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillStep {
    pub action: String,
    #[serde(default)]
    pub params: serde_json::Value,
}

impl SkillManifest {
    /// 从 YAML 解析
    pub fn from_yaml(yaml: &str) -> Result<Self, SkillError> {
        serde_yaml::from_str(yaml)
            .map_err(|e| SkillError::InvalidManifest(e.to_string()))
    }

    /// 从文件加载
    pub fn from_file(path: &std::path::Path) -> Result<Self, SkillError> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| SkillError::IoError(e.to_string()))?;
        Self::from_yaml(&content)
    }

    /// 序列化为 YAML
    pub fn to_yaml(&self) -> Result<String, SkillError> {
        serde_yaml::to_string(self)
            .map_err(|e| SkillError::InvalidManifest(e.to_string()))
    }

    /// 检查用户输入是否匹配触发模式（大小写不敏感）
    pub fn matches(&self, input: &str) -> bool {
        let input_lower = input.to_lowercase();
        self.trigger_patterns.iter().any(|pattern| {
            input_lower.contains(&pattern.to_lowercase())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_manifest() {
        let yaml = r#"
name: backup_task
version: "1.0"
description: Backup files to archive
trigger_patterns:
  - "backup"
  - "archive files"
steps:
  - action: execute_command
    params:
      command: "tar -czf backup.tar.gz ./data"
"#;
        let manifest = SkillManifest::from_yaml(yaml).unwrap();
        assert_eq!(manifest.name, "backup_task");
        assert_eq!(manifest.trigger_patterns.len(), 2);
        assert_eq!(manifest.steps.len(), 1);
    }

    #[test]
    fn test_invalid_manifest() {
        let yaml = "not: a\nvalid: manifest";
        let result = SkillManifest::from_yaml(yaml);
        // 缺少 name 字段应导致解析失败
        assert!(result.is_err());
    }

    #[test]
    fn test_trigger_match() {
        let yaml = r#"
name: backup_task
version: "1.0"
description: Backup files
trigger_patterns:
  - "backup"
  - "archive"
steps:
  - action: execute_command
    params: {}
"#;
        let manifest = SkillManifest::from_yaml(yaml).unwrap();
        assert!(manifest.matches("please backup my files"));
        assert!(manifest.matches("I want to archive data"));
        assert!(!manifest.matches("hello world"));
    }

    #[test]
    fn test_trigger_case_insensitive() {
        let yaml = r#"
name: test
version: "1.0"
description: Test
trigger_patterns:
  - "Hello"
steps: []
"#;
        let manifest = SkillManifest::from_yaml(yaml).unwrap();
        assert!(manifest.matches("say HELLO"));
    }
}
