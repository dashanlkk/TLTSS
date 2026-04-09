//! Hermes Agent — 核心 Agent 状态机

pub mod session;
pub mod prompt;
pub mod agent;
pub mod memory;
pub mod trace;
pub mod retry;
pub mod compressor;
pub mod session_search;
pub mod routing;

pub use agent::Agent;
pub use session::Session;
pub use memory::MemoryStore;
pub use memory::SearchResult as MemorySearchResult;
pub use session_search::SessionSearcher;
pub use routing::{ModelRouter, RoutingTier, RoutingContext, RoutingDecision};
pub use trace::TraceCollector;
pub use retry::{RecoveryStrategy, classify_error, jittered_backoff, retry_llm_call};
pub use compressor::{CompressionConfig, compress, should_compress, estimate_tokens};
