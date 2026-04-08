use serde::{Deserialize, Serialize};

/// SSE 流式事件统一格式
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SseEvent {
    pub event: String,
    pub data: serde_json::Value,
}

impl SseEvent {
    pub fn new(event: impl Into<String>, data: serde_json::Value) -> Self {
        Self {
            event: event.into(),
            data,
        }
    }

    /// 序列化为 SSE 协议格式 `data: {...}\n\n`
    pub fn to_sse_string(&self) -> String {
        format!(
            "event: {}\ndata: {}\n\n",
            self.event,
            serde_json::to_string(&self.data).unwrap_or_default()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sse_format() {
        let evt = SseEvent::new("token", serde_json::json!({"text": "hello"}));
        let sse = evt.to_sse_string();
        assert!(sse.starts_with("event: token\n"));
        assert!(sse.contains("data: "));
        assert!(sse.ends_with("\n\n"));
    }
}
