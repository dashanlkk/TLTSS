//! Hermes Gateway — 消息平台适配

pub mod channel;
pub mod api;
pub mod adapter;

pub use channel::GatewayChannel;
pub use adapter::ApiAdapter;
