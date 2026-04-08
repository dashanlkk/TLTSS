//! Hermes LLM — LLM 客户端

pub mod fake;
pub mod openai;
pub mod fallback;

pub use fake::FakeClient;
pub use openai::OpenAIClient;
pub use fallback::FallbackClient;
