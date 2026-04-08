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

    /// 关闭子进程
    pub async fn close(&self) -> Result<(), McpError> {
        let mut child = self.child.lock().await;
        child.kill().await
            .map_err(|e| McpError::Transport(e.to_string()))
    }
}
