//! Hermes Agent — 公共类型基石
//!
//! 定义整个 workspace 共享的核心数据结构、错误类型和 trait 签名。
//! 此 crate 零业务逻辑，仅包含类型契约。

pub mod error;
pub mod message;
pub mod platform;
pub mod tool;
pub mod traits;

pub mod prelude {
    pub use crate::error::*;
    pub use crate::message::*;
    pub use crate::platform::*;
    pub use crate::tool::*;
    pub use crate::traits::*;
}
