use async_trait::async_trait;
use hermes_cfg::prelude::*;
use hermes_cfg::traits::{TerminalBackend, ToolContext, ToolHandler};
use hermes_security::{validate_path, is_sensitive_file, filter_sensitive};
use std::path::PathBuf;
use std::sync::Arc;

/// 内置工具：读取文件
pub struct ReadFileTool {
    base_dir: PathBuf,
}

impl ReadFileTool {
    pub fn new(base_dir: impl Into<PathBuf>) -> Self {
        Self { base_dir: base_dir.into() }
    }
}

#[async_trait]
impl ToolHandler for ReadFileTool {
    fn name(&self) -> &str { "read_file" }
    fn description(&self) -> &str { "Read the contents of a file" }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Path to the file to read" }
            },
            "required": ["path"]
        })
    }

    async fn execute(&self, arguments: &str, _ctx: &ToolContext) -> Result<ToolResult, ToolError> {
        let args: serde_json::Value = serde_json::from_str(arguments)
            .map_err(|e| ToolError::InvalidArguments(e.to_string()))?;
        let path = args["path"].as_str()
            .ok_or_else(|| ToolError::InvalidArguments("missing path".into()))?;

        // 安全检查
        let validated = validate_path(&self.base_dir, path)
            .map_err(|_| ToolError::PathTraversal)?;

        if is_sensitive_file(path) {
            return Err(ToolError::SensitiveFileAccess(path.to_string()));
        }

        let content = tokio::fs::read_to_string(&validated).await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

        Ok(ToolResult::success("read_file", filter_sensitive(&content)))
    }
}

/// 内置工具：执行命令
pub struct ExecuteCommandTool {
    terminal: Arc<dyn TerminalBackend>,
}

impl ExecuteCommandTool {
    pub fn new(terminal: Arc<dyn TerminalBackend>) -> Self {
        Self { terminal }
    }
}

#[async_trait]
impl ToolHandler for ExecuteCommandTool {
    fn name(&self) -> &str { "execute_command" }
    fn description(&self) -> &str { "Execute a shell command" }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": { "type": "string", "description": "The command to execute" }
            },
            "required": ["command"]
        })
    }

    async fn execute(&self, arguments: &str, _ctx: &ToolContext) -> Result<ToolResult, ToolError> {
        let args: serde_json::Value = serde_json::from_str(arguments)
            .map_err(|e| ToolError::InvalidArguments(e.to_string()))?;
        let command = args["command"].as_str()
            .ok_or_else(|| ToolError::InvalidArguments("missing command".into()))?;

        let output = self.terminal.execute(command, None).await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

        let result = if output.is_success() {
            ToolResult::success("execute_command", filter_sensitive(&output.stdout))
        } else {
            ToolResult::error("execute_command", format!("exit {}: {}", output.exit_code, output.stderr))
        };
        Ok(result)
    }
}

/// 内置工具：写入文件
pub struct WriteFileTool {
    base_dir: PathBuf,
}

impl WriteFileTool {
    pub fn new(base_dir: impl Into<PathBuf>) -> Self {
        Self { base_dir: base_dir.into() }
    }
}

#[async_trait]
impl ToolHandler for WriteFileTool {
    fn name(&self) -> &str { "write_file" }
    fn description(&self) -> &str { "Write content to a file" }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Path to write to" },
                "content": { "type": "string", "description": "Content to write" }
            },
            "required": ["path", "content"]
        })
    }

    async fn execute(&self, arguments: &str, _ctx: &ToolContext) -> Result<ToolResult, ToolError> {
        let args: serde_json::Value = serde_json::from_str(arguments)
            .map_err(|e| ToolError::InvalidArguments(e.to_string()))?;
        let path = args["path"].as_str()
            .ok_or_else(|| ToolError::InvalidArguments("missing path".into()))?;
        let content = args["content"].as_str()
            .ok_or_else(|| ToolError::InvalidArguments("missing content".into()))?;

        let validated = validate_path(&self.base_dir, path)
            .map_err(|_| ToolError::PathTraversal)?;

        if let Some(parent) = validated.parent() {
            tokio::fs::create_dir_all(parent).await
                .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;
        }

        tokio::fs::write(&validated, content).await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

        Ok(ToolResult::success("write_file", format!("Written {} bytes", content.len())))
    }
}

/// 内置工具：列出目录
pub struct ListDirTool {
    base_dir: PathBuf,
}

impl ListDirTool {
    pub fn new(base_dir: impl Into<PathBuf>) -> Self {
        Self { base_dir: base_dir.into() }
    }
}

#[async_trait]
impl ToolHandler for ListDirTool {
    fn name(&self) -> &str { "list_dir" }
    fn description(&self) -> &str { "List directory contents" }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Directory path to list" }
            },
            "required": ["path"]
        })
    }

    async fn execute(&self, arguments: &str, _ctx: &ToolContext) -> Result<ToolResult, ToolError> {
        let args: serde_json::Value = serde_json::from_str(arguments)
            .map_err(|e| ToolError::InvalidArguments(e.to_string()))?;
        let path = args["path"].as_str()
            .unwrap_or(".");

        let validated = validate_path(&self.base_dir, path)
            .map_err(|_| ToolError::PathTraversal)?;

        let mut entries = Vec::new();
        let mut dir = tokio::fs::read_dir(&validated).await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

        while let Some(entry) = dir.next_entry().await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?
        {
            entries.push(entry.file_name().to_string_lossy().to_string());
        }

        entries.sort();
        Ok(ToolResult::success("list_dir", entries.join("\n")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hermes_cfg::platform::SessionSource;

    fn test_ctx() -> ToolContext {
        ToolContext::new("test-session", SessionSource::cli())
    }

    #[tokio::test]
    async fn test_read_file_tool() {
        let dir = std::env::temp_dir();
        let file_path = dir.join("hermes_test_read.txt");
        std::fs::write(&file_path, "hello world").unwrap();

        let tool = ReadFileTool::new(&dir);
        let args = serde_json::json!({"path": "hermes_test_read.txt"}).to_string();
        let result = tool.execute(&args, &test_ctx()).await.unwrap();
        assert!(result.content.contains("hello world"));
        assert!(!result.is_error);

        std::fs::remove_file(&file_path).ok();
    }

    #[tokio::test]
    async fn test_read_file_traversal_blocked() {
        let dir = std::env::temp_dir();
        let tool = ReadFileTool::new(&dir);
        let args = serde_json::json!({"path": "../../../etc/passwd"}).to_string();
        let result = tool.execute(&args, &test_ctx()).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_write_file_tool() {
        let dir = std::env::temp_dir().join("hermes_test_write");
        std::fs::create_dir_all(&dir).ok();

        let tool = WriteFileTool::new(&dir);
        let args = serde_json::json!({"path": "test_out.txt", "content": "written content"}).to_string();
        let result = tool.execute(&args, &test_ctx()).await.unwrap();
        assert!(!result.is_error);

        let written = std::fs::read_to_string(dir.join("test_out.txt")).unwrap();
        assert_eq!(written, "written content");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn test_list_dir_tool() {
        let dir = std::env::temp_dir().join("hermes_test_list");
        std::fs::create_dir_all(&dir).ok();
        std::fs::write(dir.join("a.txt"), "a").ok();
        std::fs::write(dir.join("b.txt"), "b").ok();

        let tool = ListDirTool::new(&dir);
        let args = serde_json::json!({"path": "."}).to_string();
        let result = tool.execute(&args, &test_ctx()).await.unwrap();
        assert!(result.content.contains("a.txt"));
        assert!(result.content.contains("b.txt"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn test_execute_command_tool() {
        let terminal = Arc::new(
            hermes_terminal::backend::LocalBackend::new(std::env::temp_dir())
        );
        let tool = ExecuteCommandTool::new(terminal);
        let cmd = if cfg!(target_os = "windows") { "echo hello" } else { "echo hello" };
        let args = serde_json::json!({"command": cmd}).to_string();
        let result = tool.execute(&args, &test_ctx()).await.unwrap();
        assert!(result.content.contains("hello"));
    }
}
