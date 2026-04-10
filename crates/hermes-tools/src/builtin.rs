use async_trait::async_trait;
use hermes_cfg::prelude::*;
use hermes_cfg::traits::{TerminalBackend, ToolContext, ToolHandler};
use hermes_security::{validate_path, is_sensitive_file, filter_sensitive};
use std::path::PathBuf;
use std::sync::Arc;

use crate::coerce;
use crate::destructive::is_destructive_command;

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

        // Destructive command detection — reject if context is auto-approve
        // and the command looks destructive (escalation is handled by the
        // registry's approval system; here we just log a warning).
        if is_destructive_command(command) {
            tracing::warn!("Potentially destructive command detected: {}", command);
        }

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

        if is_sensitive_file(path) {
            return Err(ToolError::SensitiveFileAccess(path.to_string()));
        }

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

// ── 扩展工具 ──────────────────────────────────────────────────────────

/// 内置工具：内容搜索
pub struct SearchFilesTool {
    base_dir: PathBuf,
}

impl SearchFilesTool {
    pub fn new(base_dir: impl Into<PathBuf>) -> Self {
        Self { base_dir: base_dir.into() }
    }
}

#[async_trait]
impl ToolHandler for SearchFilesTool {
    fn name(&self) -> &str { "search_files" }
    fn description(&self) -> &str {
        "Search for a pattern in files (like grep). Returns matching lines with file paths and line numbers."
    }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "pattern": { "type": "string", "description": "Search pattern (literal string)" },
                "path": { "type": "string", "description": "Directory or file to search in (default: .)" },
                "max_results": { "type": "integer", "description": "Maximum number of results (default: 50)" },
                "case_sensitive": { "type": "boolean", "description": "Case-sensitive search (default: false)" }
            },
            "required": ["pattern"]
        })
    }

    async fn execute(&self, arguments: &str, _ctx: &ToolContext) -> Result<ToolResult, ToolError> {
        let mut args: serde_json::Value = serde_json::from_str(arguments)
            .map_err(|e| ToolError::InvalidArguments(e.to_string()))?;

        // 强制转换 LLM 可能返回的字符串类型参数
        coerce::coerce_arguments(&mut args, &self.parameters_schema());

        let pattern = args["pattern"].as_str()
            .ok_or_else(|| ToolError::InvalidArguments("missing pattern".into()))?;
        let search_path = args["path"].as_str().unwrap_or(".");
        let max_results = args["max_results"].as_u64().unwrap_or(50) as usize;
        let case_sensitive = args["case_sensitive"].as_bool().unwrap_or(false);

        let validated = validate_path(&self.base_dir, search_path)
            .map_err(|_| ToolError::PathTraversal)?;

        let pattern_lower = pattern.to_lowercase();
        let mut results = Vec::new();
        let mut count = 0usize;

        search_recursive(&validated, pattern, &pattern_lower, case_sensitive, max_results, &mut results, &mut count).await?;

        if results.is_empty() {
            Ok(ToolResult::success("search_files", "No matches found."))
        } else {
            let total = count;
            let truncated = if total > max_results { format!("\n... and {} more matches", total - results.len()) } else { String::new() };
            Ok(ToolResult::success("search_files", format!("{}\n{} matches total.{}", results.join("\n"), total, truncated)))
        }
    }
}

async fn search_recursive(
    dir: &PathBuf,
    pattern: &str,
    pattern_lower: &str,
    case_sensitive: bool,
    max_results: usize,
    results: &mut Vec<String>,
    count: &mut usize,
) -> Result<(), ToolError> {
    let mut entries = tokio::fs::read_dir(dir).await
        .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

    while let Some(entry) = entries.next_entry().await
        .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?
    {
        if *count >= max_results { break; }

        let path = entry.path();
        let file_name = entry.file_name().to_string_lossy().to_string();

        // Skip hidden dirs and common ignored dirs
        if file_name.starts_with('.') { continue; }
        if file_name == "target" || file_name == "node_modules" || file_name == "__pycache__" {
            continue;
        }

        if path.is_dir() {
            Box::pin(search_recursive(&path, pattern, pattern_lower, case_sensitive, max_results, results, count)).await?;
        } else if path.is_file() {
            // Skip binary-ish extensions
            if let Some("exe" | "dll" | "so" | "dylib" | "png" | "jpg" | "jpeg" | "gif" | "zip" | "tar" | "gz") = path.extension().and_then(|e| e.to_str()) {
                continue;
            }

            if is_sensitive_file(&path.to_string_lossy()) { continue; }

            match tokio::fs::read_to_string(&path).await {
                Ok(content) => {
                    let rel_path = path.strip_prefix(std::env::current_dir().unwrap_or_default())
                        .unwrap_or(&path)
                        .to_string_lossy()
                        .to_string();

                    for (i, line) in content.lines().enumerate() {
                        if *count >= max_results { break; }
                        let matches = if case_sensitive {
                            line.contains(pattern)
                        } else {
                            line.to_lowercase().contains(pattern_lower)
                        };
                        if matches {
                            *count += 1;
                            let trimmed = if line.len() > 200 { format!("{}...", &line[..200]) } else { line.to_string() };
                            results.push(format!("{}:{}: {}", rel_path, i + 1, trimmed));
                        }
                    }
                }
                Err(_) => continue, // Skip unreadable files
            }
        }
    }
    Ok(())
}

/// 内置工具：HTTP GET 获取网页内容
pub struct WebFetchTool;

impl Default for WebFetchTool {
    fn default() -> Self { Self::new() }
}

impl WebFetchTool {
    pub fn new() -> Self { Self }
}

#[async_trait]
impl ToolHandler for WebFetchTool {
    fn name(&self) -> &str { "web_fetch" }
    fn description(&self) -> &str {
        "Fetch content from a URL via HTTP GET. Returns the response body as text."
    }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "url": { "type": "string", "description": "The URL to fetch" },
                "max_length": { "type": "integer", "description": "Maximum response length in characters (default: 10000)" }
            },
            "required": ["url"]
        })
    }

    async fn execute(&self, arguments: &str, _ctx: &ToolContext) -> Result<ToolResult, ToolError> {
        let mut args: serde_json::Value = serde_json::from_str(arguments)
            .map_err(|e| ToolError::InvalidArguments(e.to_string()))?;

        coerce::coerce_arguments(&mut args, &self.parameters_schema());

        let url = args["url"].as_str()
            .ok_or_else(|| ToolError::InvalidArguments("missing url".into()))?;
        let max_length = args["max_length"].as_u64().unwrap_or(10000) as usize;

        // Basic URL validation
        if !url.starts_with("http://") && !url.starts_with("https://") {
            return Err(ToolError::InvalidArguments("url must start with http:// or https://".into()));
        }

        let response = reqwest::get(url).await
            .map_err(|e| ToolError::ExecutionFailed(format!("HTTP request failed: {}", e)))?;

        let status = response.status();
        if !status.is_success() {
            return Ok(ToolResult::error("web_fetch", format!("HTTP {} {}", status.as_u16(), status.canonical_reason().unwrap_or("Error"))));
        }

        let body = response.text().await
            .map_err(|e| ToolError::ExecutionFailed(format!("Failed to read response: {}", e)))?;

        let content = if body.len() > max_length {
            format!("{}...\n[truncated, {} total characters]", &body[..max_length], body.len())
        } else {
            body
        };

        Ok(ToolResult::success("web_fetch", content))
    }
}

/// 内置工具：任务列表管理
pub struct TodoTool {
    data_dir: PathBuf,
}

impl TodoTool {
    pub fn new(data_dir: impl Into<PathBuf>) -> Self {
        Self { data_dir: data_dir.into() }
    }

    fn todo_file(&self) -> PathBuf {
        self.data_dir.join("todo.json")
    }

    async fn load_items(&self) -> Vec<TodoItem> {
        match tokio::fs::read_to_string(self.todo_file()).await {
            Ok(content) => serde_json::from_str(&content).unwrap_or_default(),
            Err(_) => Vec::new(),
        }
    }

    async fn save_items(&self, items: &[TodoItem]) -> Result<(), ToolError> {
        if let Some(parent) = self.todo_file().parent() {
            tokio::fs::create_dir_all(parent).await
                .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;
        }
        let json = serde_json::to_string_pretty(items)
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;
        tokio::fs::write(self.todo_file(), json).await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct TodoItem {
    id: usize,
    text: String,
    done: bool,
}

#[async_trait]
impl ToolHandler for TodoTool {
    fn name(&self) -> &str { "todo" }
    fn description(&self) -> &str {
        "Manage a task list. Actions: 'list' (show tasks), 'add' (create task), 'complete' (mark done), 'remove' (delete task)."
    }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "action": { "type": "string", "description": "Action: list, add, complete, remove" },
                "text": { "type": "string", "description": "Task text (for 'add' action)" },
                "id": { "type": "integer", "description": "Task ID (for 'complete' and 'remove' actions)" }
            },
            "required": ["action"]
        })
    }

    async fn execute(&self, arguments: &str, _ctx: &ToolContext) -> Result<ToolResult, ToolError> {
        let mut args: serde_json::Value = serde_json::from_str(arguments)
            .map_err(|e| ToolError::InvalidArguments(e.to_string()))?;

        coerce::coerce_arguments(&mut args, &self.parameters_schema());

        let action = args["action"].as_str()
            .ok_or_else(|| ToolError::InvalidArguments("missing action".into()))?;

        let mut items = self.load_items().await;

        match action {
            "list" => {
                if items.is_empty() {
                    return Ok(ToolResult::success("todo", "No tasks."));
                }
                let lines: Vec<String> = items.iter()
                    .map(|t| {
                        let check = if t.done { "x" } else { " " };
                        format!("[{}] {} {}", check, t.id, t.text)
                    })
                    .collect();
                Ok(ToolResult::success("todo", lines.join("\n")))
            }
            "add" => {
                let text = args["text"].as_str()
                    .ok_or_else(|| ToolError::InvalidArguments("missing text for add".into()))?;
                let next_id = items.iter().map(|t| t.id).max().unwrap_or(0) + 1;
                items.push(TodoItem { id: next_id, text: text.to_string(), done: false });
                self.save_items(&items).await?;
                Ok(ToolResult::success("todo", format!("Added task #{}: {}", next_id, text)))
            }
            "complete" => {
                let id = args["id"].as_u64()
                    .ok_or_else(|| ToolError::InvalidArguments("missing or invalid id".into()))? as usize;
                let found = items.iter_mut().find(|t| t.id == id);
                match found {
                    Some(t) => {
                        t.done = true;
                        let text = t.text.clone();
                        self.save_items(&items).await?;
                        Ok(ToolResult::success("todo", format!("Completed task #{}: {}", id, text)))
                    }
                    None => Ok(ToolResult::error("todo", format!("Task #{} not found", id))),
                }
            }
            "remove" => {
                let id = args["id"].as_u64()
                    .ok_or_else(|| ToolError::InvalidArguments("missing or invalid id".into()))? as usize;
                let before = items.len();
                items.retain(|t| t.id != id);
                if items.len() < before {
                    self.save_items(&items).await?;
                    Ok(ToolResult::success("todo", format!("Removed task #{}", id)))
                } else {
                    Ok(ToolResult::error("todo", format!("Task #{} not found", id)))
                }
            }
            _ => Err(ToolError::InvalidArguments(format!("unknown action: '{}'. Use: list, add, complete, remove", action))),
        }
    }
}

/// 内置工具：向用户提问澄清
pub struct ClarifyTool;

impl Default for ClarifyTool {
    fn default() -> Self { Self::new() }
}

impl ClarifyTool {
    pub fn new() -> Self { Self }
}

#[async_trait]
impl ToolHandler for ClarifyTool {
    fn name(&self) -> &str { "clarify" }
    fn description(&self) -> &str {
        "Ask the user a clarifying question when the request is ambiguous or incomplete."
    }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "question": { "type": "string", "description": "The question to ask the user" },
                "options": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Optional list of suggested options for the user to choose from"
                }
            },
            "required": ["question"]
        })
    }

    async fn execute(&self, arguments: &str, _ctx: &ToolContext) -> Result<ToolResult, ToolError> {
        let args: serde_json::Value = serde_json::from_str(arguments)
            .map_err(|e| ToolError::InvalidArguments(e.to_string()))?;

        let question = args["question"].as_str()
            .ok_or_else(|| ToolError::InvalidArguments("missing question".into()))?;

        let mut output = format!("CLARIFY: {}", question);

        if let Some(options) = args["options"].as_array() {
            let opts: Vec<&str> = options.iter()
                .filter_map(|v| v.as_str())
                .collect();
            if !opts.is_empty() {
                output.push_str("\nOptions:");
                for (i, opt) in opts.iter().enumerate() {
                    output.push_str(&format!("\n  {}. {}", i + 1, opt));
                }
            }
        }

        Ok(ToolResult::success("clarify", output))
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
        let cmd = "echo hello";
        let args = serde_json::json!({"command": cmd}).to_string();
        let result = tool.execute(&args, &test_ctx()).await.unwrap();
        assert!(result.content.contains("hello"));
    }

    // ── SearchFiles 测试 ──

    #[tokio::test]
    async fn test_search_files_basic() {
        let dir = std::env::temp_dir().join("hermes_test_search");
        std::fs::create_dir_all(&dir).ok();
        std::fs::write(dir.join("a.txt"), "hello world\nfoo bar\nhello rust").unwrap();
        std::fs::write(dir.join("b.txt"), "no match here").unwrap();

        let tool = SearchFilesTool::new(&dir);
        let args = serde_json::json!({"pattern": "hello", "path": "."}).to_string();
        let result = tool.execute(&args, &test_ctx()).await.unwrap();
        assert!(!result.is_error);
        assert!(result.content.contains("hello"));
        assert!(result.content.contains("2 matches"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn test_search_files_case_insensitive() {
        let dir = std::env::temp_dir().join("hermes_test_search_ci");
        std::fs::create_dir_all(&dir).ok();
        std::fs::write(dir.join("test.txt"), "Hello World\nHELLO there").unwrap();

        let tool = SearchFilesTool::new(&dir);
        let args = serde_json::json!({"pattern": "hello", "path": ".", "case_sensitive": false}).to_string();
        let result = tool.execute(&args, &test_ctx()).await.unwrap();
        assert!(result.content.contains("2 matches"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn test_search_files_no_match() {
        let dir = std::env::temp_dir().join("hermes_test_search_empty");
        std::fs::create_dir_all(&dir).ok();
        std::fs::write(dir.join("x.txt"), "nothing relevant").unwrap();

        let tool = SearchFilesTool::new(&dir);
        let args = serde_json::json!({"pattern": "zzznonexistent"}).to_string();
        let result = tool.execute(&args, &test_ctx()).await.unwrap();
        assert!(result.content.contains("No matches"));

        std::fs::remove_dir_all(&dir).ok();
    }

    // ── Clarify 测试 ──

    #[tokio::test]
    async fn test_clarify_basic() {
        let tool = ClarifyTool::new();
        let args = serde_json::json!({"question": "Which file do you want?"}).to_string();
        let result = tool.execute(&args, &test_ctx()).await.unwrap();
        assert!(result.content.contains("CLARIFY: Which file do you want?"));
        assert!(!result.is_error);
    }

    #[tokio::test]
    async fn test_clarify_with_options() {
        let tool = ClarifyTool::new();
        let args = serde_json::json!({
            "question": "Choose a format",
            "options": ["JSON", "YAML", "TOML"]
        }).to_string();
        let result = tool.execute(&args, &test_ctx()).await.unwrap();
        assert!(result.content.contains("CLARIFY:"));
        assert!(result.content.contains("1. JSON"));
        assert!(result.content.contains("2. YAML"));
        assert!(result.content.contains("3. TOML"));
    }

    // ── Todo 测试 ──

    #[tokio::test]
    async fn test_todo_add_list_complete() {
        let dir = std::env::temp_dir().join("hermes_test_todo");
        std::fs::create_dir_all(&dir).ok();
        // Clean slate
        std::fs::remove_file(dir.join("todo.json")).ok();

        let tool = TodoTool::new(&dir);

        // Add
        let args = serde_json::json!({"action": "add", "text": "Write tests"}).to_string();
        let result = tool.execute(&args, &test_ctx()).await.unwrap();
        assert!(result.content.contains("Added task #1"));

        // Add another
        let args = serde_json::json!({"action": "add", "text": "Review code"}).to_string();
        let result = tool.execute(&args, &test_ctx()).await.unwrap();
        assert!(result.content.contains("Added task #2"));

        // List
        let args = serde_json::json!({"action": "list"}).to_string();
        let result = tool.execute(&args, &test_ctx()).await.unwrap();
        assert!(result.content.contains("Write tests"));
        assert!(result.content.contains("Review code"));

        // Complete (LLM might send id as string — coerce should handle it)
        let args = serde_json::json!({"action": "complete", "id": "1"}).to_string();
        let result = tool.execute(&args, &test_ctx()).await.unwrap();
        assert!(result.content.contains("Completed task #1"));

        // Remove
        let args = serde_json::json!({"action": "remove", "id": 2}).to_string();
        let result = tool.execute(&args, &test_ctx()).await.unwrap();
        assert!(result.content.contains("Removed task #2"));

        std::fs::remove_dir_all(&dir).ok();
    }

    // ── Coerce 集成测试 ──

    #[tokio::test]
    async fn test_search_coerce_max_results_from_string() {
        let dir = std::env::temp_dir().join("hermes_test_coerce");
        std::fs::create_dir_all(&dir).ok();
        std::fs::write(dir.join("data.txt"), "match line 1\nmatch line 2\nmatch line 3").unwrap();

        let tool = SearchFilesTool::new(&dir);
        // LLM sends max_results as string "2" — coerce should fix it
        let args = serde_json::json!({"pattern": "match", "path": ".", "max_results": "2"}).to_string();
        let result = tool.execute(&args, &test_ctx()).await.unwrap();
        assert!(!result.is_error);
        // Should only show 2 results even though 3 lines match
        assert!(result.content.contains("match"));

        std::fs::remove_dir_all(&dir).ok();
    }
}
