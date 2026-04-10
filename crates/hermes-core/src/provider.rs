//! Provider configuration registry.
//!
//! Supports multiple named LLM provider configurations with
//! structured settings for OpenAI and Anthropic endpoints.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A single provider configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// Provider type: "openai" or "anthropic"
    #[serde(rename = "type")]
    pub provider_type: ProviderType,
    /// API key (can reference env var via "${VAR_NAME}")
    #[serde(default)]
    pub api_key: Option<String>,
    /// Base URL override
    #[serde(default)]
    pub base_url: Option<String>,
    /// Model name
    pub model: String,
    /// Max tokens for this provider
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    /// Temperature
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    /// Whether this is the default provider
    #[serde(default)]
    pub default: bool,
}

/// Provider type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum ProviderType {
    Openai,
    Anthropic,
}

fn default_max_tokens() -> u32 {
    4096
}

fn default_temperature() -> f32 {
    0.7
}

impl Default for ProviderConfig {
    fn default() -> Self {
        Self {
            provider_type: ProviderType::Openai,
            api_key: None,
            base_url: None,
            model: "gpt-4".to_string(),
            max_tokens: default_max_tokens(),
            temperature: default_temperature(),
            default: false,
        }
    }
}

/// Registry of named providers
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProviderRegistry {
    #[serde(default)]
    providers: HashMap<String, ProviderConfig>,
}

impl ProviderRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a named provider
    pub fn add(&mut self, name: impl Into<String>, config: ProviderConfig) {
        self.providers.insert(name.into(), config);
    }

    /// Get a provider by name
    pub fn get(&self, name: &str) -> Option<&ProviderConfig> {
        self.providers.get(name)
    }

    /// Get the default provider (first one marked as default, or the only one)
    pub fn default_provider(&self) -> Option<(&String, &ProviderConfig)> {
        // First try explicit default
        let explicit = self.providers.iter().find(|(_, c)| c.default);
        if explicit.is_some() {
            return explicit;
        }
        // Fallback: first provider
        self.providers.iter().next()
    }

    /// List all provider names
    pub fn names(&self) -> Vec<String> {
        self.providers.keys().cloned().collect()
    }

    /// Resolve an API key: if the value is "${ENV_VAR}", read from env
    pub fn resolve_api_key(key: &str) -> String {
        if let Some(var) = key.strip_prefix("${").and_then(|s| s.strip_suffix('}')) {
            std::env::var(var).unwrap_or_default()
        } else {
            key.to_string()
        }
    }

    /// Build a registry from AppConfig's provider-related fields
    pub fn from_app_config(config: &super::config::AppConfig) -> Self {
        let mut registry = Self::new();

        // Check environment for auto-detected providers
        // Support both ANTHROPIC_API_KEY and ANTHROPIC_AUTH_TOKEN
        let anthropic_key = std::env::var("ANTHROPIC_API_KEY")
            .or_else(|_| std::env::var("ANTHROPIC_AUTH_TOKEN"))
            .unwrap_or_default();
        let anthropic_base = std::env::var("ANTHROPIC_BASE_URL")
            .ok()
            .or_else(|| config.model.anthropic_base_url.clone());

        let openai_key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
        let openai_base = std::env::var("OPENAI_BASE_URL")
            .ok()
            .or_else(|| config.model.base_url.clone());

        // Model priority: HERMES_MODEL > ANTHROPIC_MODEL > config default
        let model = std::env::var("HERMES_MODEL")
            .or_else(|_| std::env::var("ANTHROPIC_MODEL"))
            .unwrap_or_else(|_| config.model.default.clone());

        if !anthropic_key.is_empty() {
            registry.add(
                "anthropic",
                ProviderConfig {
                    provider_type: ProviderType::Anthropic,
                    api_key: Some(anthropic_key.clone()),
                    base_url: anthropic_base,
                    model: model.clone(),
                    max_tokens: config.model.max_tokens,
                    temperature: config.model.temperature,
                    default: true,
                },
            );
        }

        if !openai_key.is_empty() {
            registry.add(
                "openai",
                ProviderConfig {
                    provider_type: ProviderType::Openai,
                    api_key: Some(openai_key),
                    base_url: openai_base,
                    model,
                    max_tokens: config.model.max_tokens,
                    temperature: config.model.temperature,
                    default: anthropic_key.is_empty(),
                },
            );
        }

        registry
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_config_default() {
        let cfg = ProviderConfig::default();
        assert_eq!(cfg.provider_type, ProviderType::Openai);
        assert_eq!(cfg.model, "gpt-4");
        assert_eq!(cfg.max_tokens, 4096);
    }

    #[test]
    fn test_registry_add_get() {
        let mut registry = ProviderRegistry::new();
        registry.add("primary", ProviderConfig {
            provider_type: ProviderType::Anthropic,
            model: "claude-sonnet-4-5".into(),
            api_key: Some("test-key".into()),
            default: true,
            ..Default::default()
        });

        assert!(registry.get("primary").is_some());
        assert!(registry.get("missing").is_none());
        assert_eq!(registry.names(), vec!["primary"]);
    }

    #[test]
    fn test_registry_default_provider() {
        let mut registry = ProviderRegistry::new();
        registry.add("secondary", ProviderConfig {
            provider_type: ProviderType::Openai,
            model: "gpt-4".into(),
            ..Default::default()
        });
        registry.add("primary", ProviderConfig {
            provider_type: ProviderType::Anthropic,
            model: "claude-sonnet-4-5".into(),
            default: true,
            ..Default::default()
        });

        let (name, cfg) = registry.default_provider().unwrap();
        assert_eq!(name, "primary");
        assert_eq!(cfg.model, "claude-sonnet-4-5");
    }

    #[test]
    fn test_registry_default_fallback_first() {
        let mut registry = ProviderRegistry::new();
        registry.add("only", ProviderConfig {
            provider_type: ProviderType::Openai,
            model: "gpt-4".into(),
            ..Default::default()
        });

        let (name, _) = registry.default_provider().unwrap();
        assert_eq!(name, "only");
    }

    #[test]
    fn test_resolve_api_key_literal() {
        assert_eq!(ProviderRegistry::resolve_api_key("sk-123"), "sk-123");
    }

    #[test]
    fn test_provider_type_serialization() {
        let json = serde_json::to_string(&ProviderType::Anthropic).unwrap();
        assert_eq!(json, "\"anthropic\"");
        let json = serde_json::to_string(&ProviderType::Openai).unwrap();
        assert_eq!(json, "\"openai\"");
    }

    #[test]
    fn test_registry_empty() {
        let registry = ProviderRegistry::new();
        assert!(registry.default_provider().is_none());
        assert!(registry.names().is_empty());
    }
}
