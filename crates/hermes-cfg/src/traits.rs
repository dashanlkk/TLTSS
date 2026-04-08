use async_trait::async_trait;
use futures::Stream;

use crate::error::LlmError;
use crate::message::Message;
use crate::tool::{ToolDefinition, ToolResult};
use crate::platform::SessionSource;

use std::pin::Pin;

/// LLM 客户端抽象 trait
///
/// 统一 LLM 调用接口，支持完整回复和 SSE 流式回复。
#[async_trait]
pub trait LlmClient: Send + Sync {
    /// 发送消息获取完整回复
    async fn complete(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
    ) -> Result<Message, LlmError>;

    /// 发送消息获取流式回复（逐 token）
    async fn complete_stream(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, LlmError>> + Send>>, LlmError>;

    /// 检测 provider 连通性
    async fn ping(&self) -> Result<Duration, LlmError>;
}

use std::time::Duration;

/// SSE 流式事件
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// 增量文本 token
    Delta(String),
    /// 工具调用
    ToolCall {
        id: String,
        name: String,
        arguments: String,
    },
    /// 流结束
    Done,
}

/// 工具执行抽象 trait
#[async_trait]
pub trait ToolHandler: Send + Sync {
    /// 工具名称
    fn name(&self) -> &str;

    /// 工具描述
    fn description(&self) -> &str;

    /// 参数 JSON Schema
    fn parameters_schema(&self) -> serde_json::Value;

    /// 执行工具
    async fn execute(&self, arguments: &str, context: &ToolContext) -> Result<ToolResult, crate::error::ToolError>;

    /// 导出为 OpenAI function calling 格式
    fn to_tool_definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.name().to_string(),
            description: self.description().to_string(),
            parameters: self.parameters_schema(),
        }
    }
}

/// 工具执行上下文
#[derive(Debug, Clone)]
pub struct ToolContext {
    pub session_id: String,
    pub source: SessionSource,
    pub approval_level: ApprovalLevel,
    pub is_cron_session: bool,
}

impl ToolContext {
    pub fn new(session_id: impl Into<String>, source: SessionSource) -> Self {
        Self {
            session_id: session_id.into(),
            source,
            approval_level: ApprovalLevel::AutoApprove,
            is_cron_session: false,
        }
    }

    pub fn with_approval(mut self, level: ApprovalLevel) -> Self {
        self.approval_level = level;
        self
    }

    pub fn cron_session(mut self) -> Self {
        self.is_cron_session = true;
        self
    }
}

/// 工具审批权限级别
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApprovalLevel {
    AutoApprove,
    RequireApproval,
    Blocked,
}

/// 终端执行后端抽象 trait
#[async_trait]
pub trait TerminalBackend: Send + Sync {
    /// 执行命令并返回输出
    async fn execute(
        &self,
        command: &str,
        timeout: Option<Duration>,
    ) -> Result<TerminalOutput, crate::error::TerminalError>;

    /// 关闭/清理资源
    async fn close(&self) -> Result<(), crate::error::TerminalError>;

    /// 后端名称（用于日志）
    fn backend_name(&self) -> &str;
}

/// 终端命令执行输出
#[derive(Debug, Clone)]
pub struct TerminalOutput {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
}

impl TerminalOutput {
    pub fn success(stdout: impl Into<String>) -> Self {
        Self {
            stdout: stdout.into(),
            stderr: String::new(),
            exit_code: 0,
        }
    }

    pub fn is_success(&self) -> bool {
        self.exit_code == 0
    }
}

/// 消息平台适配抽象 trait
#[async_trait]
pub trait PlatformAdapter: Send + Sync {
    /// 启动平台监听
    async fn run(&mut self) -> Result<(), crate::error::LlmError>;

    /// 向平台发送消息
    async fn send(&self, chat_id: &str, message: &str) -> Result<(), crate::error::LlmError>;

    /// 设置消息处理回调
    fn set_message_handler(&mut self, handler: Box<dyn Fn(String, SessionSource) + Send + Sync>);

    /// 平台名称
    fn platform_name(&self) -> &str;
}
