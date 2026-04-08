use serde::{Deserialize, Serialize};
use hermes_cfg::error::McpError;

/// JSON-RPC 2.0 请求
#[derive(Debug, Clone, Serialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: &'static str,
    pub id: u64,
    pub method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
}

impl JsonRpcRequest {
    pub fn new(id: u64, method: impl Into<String>) -> Self {
        Self {
            jsonrpc: "2.0",
            id,
            method: method.into(),
            params: None,
        }
    }

    pub fn with_params(mut self, params: serde_json::Value) -> Self {
        self.params = Some(params);
        self
    }
}

/// JSON-RPC 2.0 响应
#[derive(Debug, Clone, Deserialize)]
pub struct JsonRpcResponse {
    pub id: Option<u64>,
    pub result: Option<serde_json::Value>,
    pub error: Option<JsonRpcError>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
}

impl JsonRpcResponse {
    pub fn into_result(self) -> Result<serde_json::Value, McpError> {
        if let Some(err) = self.error {
            return Err(McpError::JsonRpc {
                code: err.code,
                message: err.message,
            });
        }
        self.result.ok_or_else(|| McpError::JsonRpc {
            code: -1,
            message: "No result in response".into(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_serialize() {
        let req = JsonRpcRequest::new(1, "tools/list");
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"jsonrpc\":\"2.0\""));
        assert!(json.contains("\"method\":\"tools/list\""));
        assert!(json.contains("\"id\":1"));
    }

    #[test]
    fn test_response_parse() {
        let json = r#"{"id":1,"result":{"tools":[]}}"#;
        let resp: JsonRpcResponse = serde_json::from_str(json).unwrap();
        assert!(resp.result.is_some());
    }

    #[test]
    fn test_response_error() {
        let json = r#"{"id":1,"error":{"code":-32600,"message":"Invalid request"}}"#;
        let resp: JsonRpcResponse = serde_json::from_str(json).unwrap();
        assert!(resp.error.is_some());
        let result = resp.into_result();
        assert!(result.is_err());
    }
}
