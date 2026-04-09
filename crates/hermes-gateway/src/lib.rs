//! Hermes Gateway — 消息平台适配

pub mod channel;
pub mod api;
pub mod adapter;
pub mod telegram;
pub mod discord;
pub mod slack;
pub mod runner;

pub use channel::GatewayChannel;
pub use adapter::ApiAdapter;
pub use telegram::{TelegramAdapter, GatewayManager};
pub use discord::DiscordAdapter;
pub use slack::SlackAdapter;
pub use runner::GatewayRunner;
