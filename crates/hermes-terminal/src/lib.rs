//! Hermes Terminal — 终端执行后端

pub mod local;
pub mod backend;
pub mod factory;

pub use backend::LocalBackend;
pub use factory::create_backend;
