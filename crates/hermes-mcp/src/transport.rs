use hermes_cfg::error::McpError;
use serde_json::Value;
use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;
use tracing::debug;

use crate::jsonrpc::{JsonRpcRequest, JsonRpcResponse};

/// stdio 传输层：通过子进程 stdin/stdout 通信
pub struct StdioTransport {
    child: Mutex<Child>,
    next_id: Mutex<u64>,
}

impl StdioTransport {
    pub async fn spawn(
        command: &str,
        args: &[String],
    ) -> Result<Self, McpError> {
        let child = Command::new(command)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| McpError::Transport(e.to_string()))?;

        Ok(Self {
            child: Mutex::new(child),
            next_id: Mutex::new(1),
        })
    }

    /// 发送 JSON-RPC 请求并读取响应
    pub async fn send(&self, method: &str, params: Option<Value>) -> Result<JsonRpcResponse, McpError> {
        let id = {
            let mut next = self.next_id.lock().await;
            let id = *next;
            *next += 1;
            id
        };

        let req = match params {
            Some(p) => JsonRpcRequest::new(id, method).with_params(p),
            None => JsonRpcRequest::new(id, method),
        };

        let req_str = serde_json::to_string(&req)
            .map_err(|e| McpError::Transport(e.to_string()))?;
        debug!("MCP send: {}", req_str);

        let mut child = self.child.lock().await;
        let stdin = child.stdin.as_mut()
            .ok_or_else(|| McpError::Transport("stdin not available".into()))?;
        stdin.write_all(format!("{}\n", req_str).as_bytes()).await
            .map_err(|e| McpError::Transport(e.to_string()))?;
        stdin.flush().await
            .map_err(|e| McpError::Transport(e.to_string()))?;

        let stdout = child.stdout.as_mut()
            .ok_or_else(|| McpError::Transport("stdout not available".into()))?;
        let mut reader = BufReader::new(stdout);
        let mut line = String::new();
        reader.read_line(&mut line).await
            .map_err(|e| McpError::Transport(e.to_string()))?;

        if line.is_empty() {
            return Err(McpError::ProcessExited);
        }

        debug!("MCP recv: {}", line.trim());
        let resp: JsonRpcResponse = serde_json::from_str(line.trim())
            .map_err(|e| McpError::Transport(e.to_string()))?;

        Ok(resp)
    }

    /// 发送 JSON-RPC 通知（无 id，不读取响应）
    pub async fn notify(&self, method: &str, params: Option<Value>) -> Result<(), McpError> {
        // Notifications have no "id" field per JSON-RPC spec
        let mut notification = serde_json::json!({
            "jsonrpc": "2.0",
            "method": method
        });
        if let Some(p) = params {
            notification["params"] = p;
        }

        let req_str = serde_json::to_string(&notification)
            .map_err(|e| McpError::Transport(e.to_string()))?;
        debug!("MCP notify: {}", req_str);

        let mut child = self.child.lock().await;
        let stdin = child.stdin.as_mut()
            .ok_or_else(|| McpError::Transport("stdin not available".into()))?;
        stdin.write_all(format!("{}\n", req_str).as_bytes()).await
            .map_err(|e| McpError::Transport(e.to_string()))?;
        stdin.flush().await
            .map_err(|e| McpError::Transport(e.to_string()))?;

        Ok(())
    }

    /// 关闭子进程
    pub async fn close(&self) -> Result<(), McpError> {
        let mut child = self.child.lock().await;
        child.kill().await
            .map_err(|e| McpError::Transport(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Spawn the mock MCP server via StdioTransport.
    async fn spawn_mock_transport() -> StdioTransport {
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
            .expect("CARGO_MANIFEST_DIR should be set during tests");
        let script = format!("{}/tests/fixtures/mock_mcp_server.py", manifest_dir);

        StdioTransport::spawn("python", &[script])
            .await
            .expect("Failed to spawn mock MCP server")
    }

    #[tokio::test]
    async fn test_spawn_creates_transport() {
        let transport = spawn_mock_transport().await;
        // If we got here, the process spawned successfully.
        // Clean up.
        transport.close().await.expect("close should succeed");
    }

    #[tokio::test]
    async fn test_send_initialize() {
        let transport = spawn_mock_transport().await;

        let resp = transport
            .send("initialize", Some(serde_json::json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": { "name": "test", "version": "0.1.0" }
            })))
            .await
            .expect("send initialize should succeed");

        assert!(resp.result.is_some(), "response should have a result");
        assert!(resp.error.is_none(), "response should not have an error");

        let result = resp.result.unwrap();
        assert_eq!(result["protocolVersion"], "2024-11-05");
        assert_eq!(result["serverInfo"]["name"], "mock-mcp-server");

        transport.close().await.ok();
    }

    #[tokio::test]
    async fn test_send_tools_list() {
        let transport = spawn_mock_transport().await;

        // Must initialize first (server expects it per MCP protocol)
        transport
            .send("initialize", Some(serde_json::json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": { "name": "test", "version": "0.1.0" }
            })))
            .await
            .expect("initialize should succeed");

        // Send initialized notification
        transport.notify("notifications/initialized", None).await.expect("notify should succeed");

        let resp = transport.send("tools/list", None).await.expect("tools/list should succeed");
        let result = resp.result.expect("should have result");
        let tools = result["tools"].as_array().expect("tools should be an array");

        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0]["name"], "echo");
        assert_eq!(tools[1]["name"], "add");

        transport.close().await.ok();
    }

    #[tokio::test]
    async fn test_send_unknown_method_returns_error() {
        let transport = spawn_mock_transport().await;

        let resp = transport
            .send("some_unknown_method", None)
            .await
            .expect("send should succeed even for unknown methods");

        assert!(resp.error.is_some(), "unknown method should return an error");
        let err = resp.error.unwrap();
        assert_eq!(err.code, -32601);
        assert!(err.message.contains("Unknown method"));

        transport.close().await.ok();
    }

    #[tokio::test]
    async fn test_notify_does_not_block() {
        let transport = spawn_mock_transport().await;

        // notify is fire-and-forget; it should return Ok without reading a response
        let result = transport.notify("notifications/initialized", None).await;
        assert!(result.is_ok(), "notify should succeed without waiting for response");

        transport.close().await.ok();
    }

    #[tokio::test]
    async fn test_send_increments_id() {
        let transport = spawn_mock_transport().await;

        let resp1 = transport.send("initialize", Some(serde_json::json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": { "name": "test", "version": "0.1.0" }
        }))).await.expect("first send should succeed");

        let resp2 = transport.send("tools/list", None).await.expect("second send should succeed");

        // The IDs should be sequential (1 and 2), as assigned by the transport
        assert_eq!(resp1.id, Some(1), "first request should have id 1");
        assert_eq!(resp2.id, Some(2), "second request should have id 2");

        transport.close().await.ok();
    }

    #[tokio::test]
    async fn test_spawn_nonexistent_command_fails() {
        let result = StdioTransport::spawn("this_command_does_not_exist_xyz123", &[]).await;
        assert!(result.is_err(), "spawning a nonexistent command should fail");
    }

    #[tokio::test]
    async fn test_close_idempotent() {
        let transport = spawn_mock_transport().await;
        transport.close().await.expect("first close should succeed");
        // Second close on a killed process should fail (not panic)
        let second = transport.close().await;
        // The behavior is OS-dependent, but it should not panic.
        // On some OSes, killing an already-dead child returns an error.
        let _ = second;
    }
}
