use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// 对话角色枚举
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// 对话消息结构体
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub id: String,
    pub role: Role,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<super::tool::ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    pub timestamp: DateTime<Utc>,
}

impl Message {
    pub fn new_user(content: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            role: Role::User,
            content: content.into(),
            tool_calls: None,
            tool_call_id: None,
            timestamp: Utc::now(),
        }
    }

    pub fn new_assistant(content: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            role: Role::Assistant,
            content: content.into(),
            tool_calls: None,
            tool_call_id: None,
            timestamp: Utc::now(),
        }
    }

    pub fn new_system(content: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            role: Role::System,
            content: content.into(),
            tool_calls: None,
            tool_call_id: None,
            timestamp: Utc::now(),
        }
    }

    pub fn new_tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            role: Role::Tool,
            content: content.into(),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
            timestamp: Utc::now(),
        }
    }

    pub fn with_tool_calls(mut self, calls: Vec<super::tool::ToolCall>) -> Self {
        self.tool_calls = Some(calls);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_serde_roundtrip() {
        let msg = Message::new_user("Hello, Hermes!");
        let json = serde_json::to_string(&msg).unwrap();
        let de: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(de.role, Role::User);
        assert_eq!(de.content, "Hello, Hermes!");
        assert!(json.contains("\"role\""));
        assert!(json.contains("\"content\""));
        assert!(json.contains("\"timestamp\""));
    }

    #[test]
    fn test_message_roles() {
        let sys = Message::new_system("You are helpful");
        assert_eq!(sys.role, Role::System);

        let assistant = Message::new_assistant("Hi there");
        assert_eq!(assistant.role, Role::Assistant);
    }

    #[test]
    fn test_tool_result_message() {
        let msg = Message::new_tool_result("call_123", "file contents");
        assert_eq!(msg.role, Role::Tool);
        assert_eq!(msg.tool_call_id.unwrap(), "call_123");
    }
}
