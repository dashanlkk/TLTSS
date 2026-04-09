use thiserror::Error;

/// LLM 调用错误
#[derive(Debug, Error)]
pub enum LlmError {
    #[error("Authentication failed: {0}")]
    AuthenticationFailed(String),
    #[error("Rate limited: retry after {0}s")]
    RateLimited(u64),
    #[error("Request timeout")]
    Timeout,
    #[error("Provider error: {0}")]
    ProviderError(String),
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),
    #[error("Stream error: {0}")]
    StreamError(String),
    #[error("Context length exceeded")]
    ContextLengthExceeded,
}

/// 工具执行错误
#[derive(Debug, Error)]
pub enum ToolError {
    #[error("Tool not found: {0}")]
    NotFound(String),
    #[error("Execution failed: {0}")]
    ExecutionFailed(String),
    #[error("Path traversal detected")]
    PathTraversal,
    #[error("Timeout after {0}s")]
    Timeout(u64),
    #[error("Rejected by user")]
    Rejected,
    #[error("Invalid arguments: {0}")]
    InvalidArguments(String),
    #[error("Sensitive file access denied: {0}")]
    SensitiveFileAccess(String),
    #[error("Permission denied: {0}")]
    PermissionDenied(String),
}

/// 终端执行错误
#[derive(Debug, Error)]
pub enum TerminalError {
    #[error("Command execution failed: {0}")]
    ExecutionFailed(String),
    #[error("Timeout after {0}s")]
    Timeout(u64),
    #[error("Container not found: {0}")]
    ContainerNotFound(String),
    #[error("Process exited with code {0}: {1}")]
    ExitCode(i32, String),
}

/// 配置错误
#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("Config file not found: {0}")]
    FileNotFound(String),
    #[error("Invalid YAML: {0}")]
    InvalidYaml(String),
    #[error("Missing required field: {0}")]
    MissingField(String),
    #[error("Environment variable not set: {0}")]
    EnvNotSet(String),
}

/// MCP 协议错误
#[derive(Debug, Error)]
pub enum McpError {
    #[error("JSON-RPC error {code}: {message}")]
    JsonRpc { code: i32, message: String },
    #[error("Process exited unexpectedly")]
    ProcessExited,
    #[error("Transport error: {0}")]
    Transport(String),
    #[error("Tool not found on MCP server: {0}")]
    ToolNotFound(String),
}

/// 技能系统错误
#[derive(Debug, Error)]
pub enum SkillError {
    #[error("Invalid manifest: {0}")]
    InvalidManifest(String),
    #[error("Skill not found: {0}")]
    NotFound(String),
    #[error("Execution failed: {0}")]
    ExecutionFailed(String),
    #[error("IO error: {0}")]
    IoError(String),
}

/// Cron 调度错误
#[derive(Debug, Error)]
pub enum CronError {
    #[error("Invalid cron expression: {0}")]
    InvalidExpression(String),
    #[error("Job not found: {0}")]
    NotFound(String),
    #[error("Execution failed: {0}")]
    ExecutionFailed(String),
    #[error("Recursive cron creation blocked")]
    RecursiveBlocked,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        assert_eq!(
            LlmError::AuthenticationFailed("bad key".to_string()).to_string(),
            "Authentication failed: bad key"
        );
        assert_eq!(
            ToolError::NotFound("xyz".to_string()).to_string(),
            "Tool not found: xyz"
        );
        assert_eq!(
            TerminalError::Timeout(30).to_string(),
            "Timeout after 30s"
        );
        assert_eq!(
            ConfigError::MissingField("model".to_string()).to_string(),
            "Missing required field: model"
        );
    }
}
