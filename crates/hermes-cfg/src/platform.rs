use serde::{Deserialize, Serialize};

/// 消息来源平台枚举
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum Platform {
    Cli,
    Telegram,
    Discord,
    Slack,
    Api,
    Cron,
}

/// 会话来源标识
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionSource {
    pub platform: Platform,
    pub chat_id: String,
}

impl Platform {
    /// Get platform name as lowercase string for routing.
    pub fn as_str(&self) -> &'static str {
        match self {
            Platform::Cli => "cli",
            Platform::Telegram => "telegram",
            Platform::Discord => "discord",
            Platform::Slack => "slack",
            Platform::Api => "api",
            Platform::Cron => "cron",
        }
    }
}

impl std::fmt::Display for Platform {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl SessionSource {
    pub fn cli() -> Self {
        Self {
            platform: Platform::Cli,
            chat_id: "cli".to_string(),
        }
    }

    pub fn api() -> Self {
        Self {
            platform: Platform::Api,
            chat_id: "api".to_string(),
        }
    }

    pub fn telegram(chat_id: impl Into<String>) -> Self {
        Self {
            platform: Platform::Telegram,
            chat_id: chat_id.into(),
        }
    }

    pub fn cron(job_id: impl Into<String>) -> Self {
        Self {
            platform: Platform::Cron,
            chat_id: job_id.into(),
        }
    }

    pub fn discord(channel_id: impl Into<String>) -> Self {
        Self {
            platform: Platform::Discord,
            chat_id: channel_id.into(),
        }
    }

    pub fn slack(channel_id: impl Into<String>) -> Self {
        Self {
            platform: Platform::Slack,
            chat_id: channel_id.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_platform_serde() {
        let variants = vec![
            Platform::Cli,
            Platform::Telegram,
            Platform::Discord,
            Platform::Slack,
            Platform::Api,
            Platform::Cron,
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let de: Platform = serde_json::from_str(&json).unwrap();
            assert_eq!(v, de);
        }
    }

    #[test]
    fn test_session_source_json() {
        let src = SessionSource::telegram("chat_42");
        let json = serde_json::to_string(&src).unwrap();
        assert!(json.contains("\"platform\":\"telegram\""));
        assert!(json.contains("\"chat_id\":\"chat_42\""));
    }
}
