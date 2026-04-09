//! Smart Model Routing
//!
//! 基于任务复杂度的模型路由策略。
//! 简单任务用便宜模型（如 haiku），复杂任务用强模型（如 sonnet/opus）。
//! 判断依据：工具数量、消息总长度、是否包含多步工具调用。

use hermes_cfg::prelude::*;

/// 路由策略
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutingTier {
    /// 轻量模型（快速、便宜）
    Light,
    /// 标准模型（平衡）
    Standard,
    /// 强力模型（复杂推理）
    Heavy,
}

/// 路由决策上下文
#[derive(Debug, Clone)]
pub struct RoutingContext {
    /// 消息总长度（字符数）
    pub total_chars: usize,
    /// 工具定义数量
    pub tool_count: usize,
    /// 历史消息条数
    pub message_count: usize,
    /// 是否包含 tool_calls（历史中有 assistant 消息带 tool_calls）
    pub has_tool_history: bool,
}

/// 路由决策结果
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    pub tier: RoutingTier,
    pub reason: String,
}

/// 模型路由器 — 根据请求复杂度选择模型层级
pub struct ModelRouter {
    /// 轻量模型名称
    pub light_model: Option<String>,
    /// 标准模型名称
    pub standard_model: String,
    /// 强力模型名称
    pub heavy_model: Option<String>,
    /// 消息长度阈值：超过此值升级到 Standard（字符数）
    pub chars_threshold: usize,
    /// 工具数量阈值：超过此值升级到 Standard
    pub tools_threshold: usize,
}

impl ModelRouter {
    /// 创建路由器，指定标准模型名称
    pub fn new(standard_model: impl Into<String>) -> Self {
        Self {
            light_model: None,
            standard_model: standard_model.into(),
            heavy_model: None,
            chars_threshold: 2000,
            tools_threshold: 4,
        }
    }

    /// 设置轻量模型
    pub fn with_light_model(mut self, model: impl Into<String>) -> Self {
        self.light_model = Some(model.into());
        self
    }

    /// 设置强力模型
    pub fn with_heavy_model(mut self, model: impl Into<String>) -> Self {
        self.heavy_model = Some(model.into());
        self
    }

    /// 根据上下文决定使用哪个模型
    pub fn route(&self, ctx: &RoutingContext) -> RoutingDecision {
        let mut complexity_score = 0u32;

        // 长消息 → 更复杂
        if ctx.total_chars > self.chars_threshold * 2 {
            complexity_score += 2;
        } else if ctx.total_chars > self.chars_threshold {
            complexity_score += 1;
        }

        // 多工具 → 更复杂
        if ctx.tool_count > self.tools_threshold {
            complexity_score += 1;
        }

        // 有工具调用历史 → 需要推理能力
        if ctx.has_tool_history {
            complexity_score += 1;
        }

        // 长对话历史 → 上下文管理更复杂
        if ctx.message_count > 20 {
            complexity_score += 1;
        }

        let (tier, reason) = if complexity_score >= 3 {
            (
                RoutingTier::Heavy,
                format!(
                    "complexity={}: {} chars, {} tools, {} messages, tool_history={}",
                    complexity_score, ctx.total_chars, ctx.tool_count,
                    ctx.message_count, ctx.has_tool_history
                ),
            )
        } else if complexity_score >= 1 {
            (
                RoutingTier::Standard,
                format!(
                    "complexity={}: {} chars, {} tools, {} messages",
                    complexity_score, ctx.total_chars, ctx.tool_count, ctx.message_count
                ),
            )
        } else {
            (
                RoutingTier::Light,
                format!(
                    "complexity={}: simple query, {} chars, {} tools",
                    complexity_score, ctx.total_chars, ctx.tool_count
                ),
            )
        };

        RoutingDecision { tier, reason }
    }

    /// 根据路由决策返回实际模型名称
    pub fn resolve_model(&self, decision: &RoutingDecision) -> &str {
        match decision.tier {
            RoutingTier::Light => self
                .light_model
                .as_deref()
                .unwrap_or(&self.standard_model),
            RoutingTier::Standard => &self.standard_model,
            RoutingTier::Heavy => self
                .heavy_model
                .as_deref()
                .unwrap_or(&self.standard_model),
        }
    }

    /// 从消息列表构建路由上下文
    pub fn context_from_messages(messages: &[Message], tools: &[ToolDefinition]) -> RoutingContext {
        let total_chars: usize = messages.iter().map(|m| m.content.len()).sum();
        let has_tool_history = messages.iter().any(|m| {
            m.role == Role::Assistant
                && m.tool_calls.as_ref().map_or(false, |c| !c.is_empty())
        });

        RoutingContext {
            total_chars,
            tool_count: tools.len(),
            message_count: messages.len(),
            has_tool_history,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_route_simple_query() {
        let router = ModelRouter::new("claude-sonnet-4-5")
            .with_light_model("claude-haiku-4-5")
            .with_heavy_model("claude-opus-4-5");

        let ctx = RoutingContext {
            total_chars: 100,
            tool_count: 2,
            message_count: 3,
            has_tool_history: false,
        };

        let decision = router.route(&ctx);
        assert_eq!(decision.tier, RoutingTier::Light);

        let model = router.resolve_model(&decision);
        assert_eq!(model, "claude-haiku-4-5");
    }

    #[test]
    fn test_route_standard_query() {
        let router = ModelRouter::new("claude-sonnet-4-5")
            .with_light_model("claude-haiku-4-5");

        let ctx = RoutingContext {
            total_chars: 3000, // > 2000 threshold
            tool_count: 3,
            message_count: 5,
            has_tool_history: false,
        };

        let decision = router.route(&ctx);
        assert_eq!(decision.tier, RoutingTier::Standard);

        let model = router.resolve_model(&decision);
        assert_eq!(model, "claude-sonnet-4-5");
    }

    #[test]
    fn test_route_heavy_query() {
        let router = ModelRouter::new("claude-sonnet-4-5")
            .with_heavy_model("claude-opus-4-5");

        let ctx = RoutingContext {
            total_chars: 10000,
            tool_count: 8,
            message_count: 30,
            has_tool_history: true,
        };

        let decision = router.route(&ctx);
        assert_eq!(decision.tier, RoutingTier::Heavy);

        let model = router.resolve_model(&decision);
        assert_eq!(model, "claude-opus-4-5");
    }

    #[test]
    fn test_route_fallback_to_standard() {
        // 没有 light/heavy 模型配置时，都回退到 standard
        let router = ModelRouter::new("claude-sonnet-4-5");

        let ctx_light = RoutingContext {
            total_chars: 50,
            tool_count: 1,
            message_count: 2,
            has_tool_history: false,
        };
        let decision = router.route(&ctx_light);
        assert_eq!(router.resolve_model(&decision), "claude-sonnet-4-5");

        let ctx_heavy = RoutingContext {
            total_chars: 10000,
            tool_count: 10,
            message_count: 30,
            has_tool_history: true,
        };
        let decision = router.route(&ctx_heavy);
        assert_eq!(router.resolve_model(&decision), "claude-sonnet-4-5");
    }

    #[test]
    fn test_context_from_messages() {
        let messages = vec![
            Message::new_user("Hello"),
            Message::new_assistant("").with_tool_calls(vec![ToolCall {
                id: "t1".into(),
                call_type: "function".into(),
                function: FunctionCall {
                    name: "read".into(),
                    arguments: "{}".into(),
                },
            }]),
            Message::new_user("What's the result?"),
        ];
        let tools = vec![
            ToolDefinition {
                name: "read".into(),
                description: "Read a file".into(),
                parameters: serde_json::json!({}),
            },
            ToolDefinition {
                name: "write".into(),
                description: "Write a file".into(),
                parameters: serde_json::json!({}),
            },
        ];

        let ctx = ModelRouter::context_from_messages(&messages, &tools);
        assert_eq!(ctx.message_count, 3);
        assert_eq!(ctx.tool_count, 2);
        assert!(ctx.has_tool_history);
        assert!(ctx.total_chars > 0);
    }

    #[test]
    fn test_tool_history_not_flagged_for_user_only() {
        let messages = vec![
            Message::new_user("Hello"),
            Message::new_assistant("Hi there!"),
        ];
        let tools = vec![];

        let ctx = ModelRouter::context_from_messages(&messages, &tools);
        assert!(!ctx.has_tool_history);
    }

    #[test]
    fn test_complexity_score_boundaries() {
        let router = ModelRouter::new("sonnet")
            .with_light_model("haiku")
            .with_heavy_model("opus");

        // Exactly at threshold: chars=2000 → +1, tools=5 → +1 → Standard (score=2)
        let ctx = RoutingContext {
            total_chars: 2001,
            tool_count: 5,
            message_count: 3,
            has_tool_history: false,
        };
        let decision = router.route(&ctx);
        assert_eq!(decision.tier, RoutingTier::Standard);

        // Add tool_history → +1 → score=3 → Heavy
        let ctx = RoutingContext {
            total_chars: 2001,
            tool_count: 5,
            message_count: 3,
            has_tool_history: true,
        };
        let decision = router.route(&ctx);
        assert_eq!(decision.tier, RoutingTier::Heavy);
    }
}
