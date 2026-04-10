//! Hermes Agent — 核心 Agent 状态机

pub mod session;
pub mod prompt;
pub mod agent;
pub mod memory;
pub mod memory_manager;
pub mod trace;
pub mod retry;
pub mod compressor;
pub mod session_search;
pub mod routing;
pub mod delegation;
pub mod delegate_tool;
pub mod context_files;

pub use agent::Agent;
pub use session::Session;
pub use memory::MemoryStore;
pub use memory::SearchResult as MemorySearchResult;
pub use memory_manager::{MemoryManager, MemoryTarget};
pub use session_search::SessionSearcher;
pub use routing::{ModelRouter, RoutingTier, RoutingContext, RoutingDecision};
pub use trace::TraceCollector;
pub use retry::{RecoveryStrategy, classify_error, jittered_backoff, retry_llm_call};
pub use compressor::{CompressionConfig, compress, should_compress, estimate_tokens};
pub use delegation::{Delegate, DelegationResult, DelegationError, ChildConfig};
pub use delegate_tool::DelegateTaskTool;
pub use context_files::{ContextFile, discover_context_files, format_context_block, MAX_CONTEXT_FILE_CHARS};
