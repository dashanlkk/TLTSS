use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::sync::RwLock;

/// 追踪事件
#[derive(Debug, Clone, Serialize, Deserialize)]
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
