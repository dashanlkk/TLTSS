use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// 应用顶层配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    #[serde(default)]
    pub model: ModelConfig,
    #[serde(default)]
    pub terminal: TerminalConfig,
    #[serde(default)]
    pub tools: ToolsConfig,
    #[serde(default)]
    pub gateway: GatewayConfig,
    #[serde(default)]
    pub mcp: McpConfig,
    #[serde(default)]
    pub skill: SkillConfig,
    #[serde(default)]
    pub cron: CronConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    #[serde(default = "default_model")]
    pub default: String,
    #[serde(default)]
    pub max_tokens: u32,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default)]
    pub fallback_model: Option<String>,
    #[serde(default)]
    pub base_url: Option<String>,
}

fn default_model() -> String {
    "gpt-4".to_string()
}

fn default_temperature() -> f32 {
    0.7
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            default: default_model(),
            max_tokens: 4096,
            temperature: default_temperature(),
            fallback_model: None,
            base_url: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerminalConfig {
    #[serde(default = "default_backend")]
    pub backend: String,
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
    #[serde(default)]
    pub docker_image: Option<String>,
}

fn default_backend() -> String {
    "local".to_string()
}

fn default_timeout() -> u64 {
    30
}

impl Default for TerminalConfig {
    fn default() -> Self {
        Self {
            backend: default_backend(),
            timeout_secs: default_timeout(),
            docker_image: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ToolsConfig {
    #[serde(default)]
    pub approval: std::collections::HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GatewayConfig {
    #[serde(default)]
    pub enabled: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct McpConfig {
    #[serde(default)]
    pub servers: Vec<McpServerConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerConfig {
    pub name: String,
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: std::collections::HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SkillConfig {
    #[serde(default)]
    pub auto_publish: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CronConfig {
    #[serde(default)]
    pub max_concurrent_jobs: u32,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            model: ModelConfig::default(),
            terminal: TerminalConfig::default(),
            tools: ToolsConfig::default(),
            gateway: GatewayConfig::default(),
            mcp: McpConfig::default(),
            skill: SkillConfig::default(),
            cron: CronConfig::default(),
        }
    }
}

impl AppConfig {
    /// 从 YAML 字符串解析
    pub fn from_yaml(yaml: &str) -> Result<Self, hermes_cfg::error::ConfigError> {
        serde_yaml::from_str(yaml)
            .map_err(|e| hermes_cfg::error::ConfigError::InvalidYaml(e.to_string()))
    }

    /// 从文件加载
    pub fn from_file(path: &std::path::Path) -> Result<Self, hermes_cfg::error::ConfigError> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| hermes_cfg::error::ConfigError::FileNotFound(e.to_string()))?;
        Self::from_yaml(&content)
    }

    /// 合并配置（other 覆盖 self 的非空字段）
    pub fn merge(&self, other: &Self) -> Self {
        let mut merged = self.clone();
        if other.model.default != default_model() {
            merged.model.default = other.model.default.clone();
        }
        if other.model.base_url.is_some() {
            merged.model.base_url = other.model.base_url.clone();
        }
        if other.terminal.backend != default_backend() {
            merged.terminal.backend = other.terminal.backend.clone();
        }
        merged
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_minimal_yaml() {
        let yaml = "model:\n  default: gpt-3.5-turbo";
        let cfg = AppConfig::from_yaml(yaml).unwrap();
        assert_eq!(cfg.model.default, "gpt-3.5-turbo");
        assert_eq!(cfg.terminal.backend, "local"); // default
    }

    #[test]
    fn test_empty_yaml_uses_defaults() {
        let yaml = "{}";
        let cfg = AppConfig::from_yaml(yaml).unwrap();
        assert_eq!(cfg.model.default, "gpt-4");
        assert_eq!(cfg.terminal.timeout_secs, 30);
    }

    #[test]
    fn test_config_merge() {
        let global = AppConfig::from_yaml("model:\n  default: gpt-4").unwrap();
        let local = AppConfig::from_yaml("model:\n  default: claude-3").unwrap();
        let merged = global.merge(&local);
        assert_eq!(merged.model.default, "claude-3");
    }
}
