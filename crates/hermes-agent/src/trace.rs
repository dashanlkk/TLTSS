use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::sync::RwLock;

/// 追踪事件
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[non_exhaustive]
pub enum TraceEvent {
    UserMessage(String),
    PromptBuilt(usize),
    LlmCallStart,
    LlmCallComplete(String),
    ToolExecuted { tool: String, success: bool },
    Done,
}

/// 单条追踪记录
#[derive(Debug, Clone, Serialize)]
pub struct Trace {
    pub trace_id: String,
    pub events: Vec<TraceEvent>,
}

/// 追踪收集器
pub struct TraceCollector {
    traces: RwLock<HashMap<String, Trace>>,
}

impl Default for TraceCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl TraceCollector {
    pub fn new() -> Self {
        Self {
            traces: RwLock::new(HashMap::new()),
        }
    }

    pub async fn start(&self, trace_id: &str) {
        self.traces.write().await.insert(
            trace_id.to_string(),
            Trace {
                trace_id: trace_id.to_string(),
                events: Vec::new(),
            },
        );
    }

    pub async fn event(&self, trace_id: &str, event: TraceEvent) {
        if let Some(trace) = self.traces.write().await.get_mut(trace_id) {
            trace.events.push(event);
        }
    }

    pub async fn get(&self, trace_id: &str) -> Option<Trace> {
        self.traces.read().await.get(trace_id).cloned()
    }

    pub async fn list(&self) -> Vec<Trace> {
        self.traces.read().await.values().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_start_and_get() {
        let collector = TraceCollector::new();
        collector.start("trace-1").await;
        let trace = collector.get("trace-1").await.unwrap();
        assert_eq!(trace.trace_id, "trace-1");
        assert!(trace.events.is_empty());
    }

    #[tokio::test]
    async fn test_get_nonexistent() {
        let collector = TraceCollector::new();
        assert!(collector.get("no-such-trace").await.is_none());
    }

    #[tokio::test]
    async fn test_event_appends() {
        let collector = TraceCollector::new();
        collector.start("t1").await;
        collector.event("t1", TraceEvent::UserMessage("hello".into())).await;
        collector.event("t1", TraceEvent::PromptBuilt(3)).await;
        collector.event("t1", TraceEvent::LlmCallStart).await;
        collector.event("t1", TraceEvent::Done).await;

        let trace = collector.get("t1").await.unwrap();
        assert_eq!(trace.events.len(), 4);
        assert_eq!(trace.events[0], TraceEvent::UserMessage("hello".into()));
        assert_eq!(trace.events[1], TraceEvent::PromptBuilt(3));
        assert_eq!(trace.events[3], TraceEvent::Done);
    }

    #[tokio::test]
    async fn test_event_on_nonexistent_trace_is_noop() {
        let collector = TraceCollector::new();
        // Should not panic, just silently ignored
        collector.event("ghost", TraceEvent::Done).await;
        assert!(collector.get("ghost").await.is_none());
    }

    #[tokio::test]
    async fn test_start_overwrites_existing() {
        let collector = TraceCollector::new();
        collector.start("dup").await;
        collector.event("dup", TraceEvent::Done).await;

        // Re-start resets the trace
        collector.start("dup").await;
        let trace = collector.get("dup").await.unwrap();
        assert!(trace.events.is_empty());
    }

    #[tokio::test]
    async fn test_list_multiple_traces() {
        let collector = TraceCollector::new();
        collector.start("a").await;
        collector.start("b").await;
        collector.start("c").await;

        let list = collector.list().await;
        assert_eq!(list.len(), 3);
    }

    #[tokio::test]
    async fn test_tool_executed_event() {
        let collector = TraceCollector::new();
        collector.start("tools").await;
        collector.event("tools", TraceEvent::ToolExecuted {
            tool: "read_file".into(),
            success: true,
        }).await;
        collector.event("tools", TraceEvent::ToolExecuted {
            tool: "execute".into(),
            success: false,
        }).await;

        let trace = collector.get("tools").await.unwrap();
        assert_eq!(trace.events.len(), 2);
    }

    #[tokio::test]
    async fn test_full_lifecycle() {
        let collector = TraceCollector::new();
        collector.start("lifecycle").await;
        collector.event("lifecycle", TraceEvent::UserMessage("test".into())).await;
        collector.event("lifecycle", TraceEvent::PromptBuilt(5)).await;
        collector.event("lifecycle", TraceEvent::LlmCallStart).await;
        collector.event("lifecycle", TraceEvent::LlmCallComplete("response".into())).await;
        collector.event("lifecycle", TraceEvent::ToolExecuted { tool: "tool1".into(), success: true }).await;
        collector.event("lifecycle", TraceEvent::Done).await;

        let trace = collector.get("lifecycle").await.unwrap();
        assert_eq!(trace.events.len(), 6);
    }
}
