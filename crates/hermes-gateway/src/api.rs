use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event as SseAxumEvent, Sse},
        Json,
    },
    routing::{get, post},
    Router,
};
use futures::stream::Stream;
use hermes_cfg::platform::SessionSource;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use tokio::sync::{mpsc, broadcast};
use tower_http::cors::CorsLayer;
use tracing::{info, warn};

use crate::channel::GatewayMessage;

/// HTTP 请求体：发送消息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageRequest {
    pub chat_id: String,
    pub content: String,
}

/// HTTP 响应体
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageResponse {
    pub success: bool,
    pub reply: Option<String>,
    pub error: Option<String>,
}

/// 健康检查响应
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
}

/// SSE 流式事件（通过 broadcast channel 分发）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamMessage {
    pub event_type: String,
    pub chat_id: String,
    pub data: serde_json::Value,
}

/// HTTP Server 共享状态
#[derive(Clone)]
pub struct AppState {
    pub gateway_tx: mpsc::UnboundedSender<GatewayMessage>,
    /// SSE broadcast channel — 所有 SSE 客户端共享
    sse_tx: broadcast::Sender<StreamMessage>,
}

impl AppState {
    pub fn new(gateway_tx: mpsc::UnboundedSender<GatewayMessage>) -> Self {
        let (sse_tx, _) = broadcast::channel(256);
        Self { gateway_tx, sse_tx }
    }

    /// 广播 SSE 事件给所有订阅客户端
    pub fn broadcast_sse(&self, msg: StreamMessage) {
        // 忽略无接收者错误
        let _ = self.sse_tx.send(msg);
    }
}

/// 构建 Axum Router
pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/api/gateway/message", post(handle_message))
        .route("/api/gateway/health", get(handle_health))
        .route("/api/gateway/stream", get(handle_sse_stream))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

/// POST /api/gateway/message — 接收外部消息，转发给 Agent channel
async fn handle_message(
    State(state): State<AppState>,
    Json(req): Json<MessageRequest>,
) -> (StatusCode, Json<MessageResponse>) {
    info!("API gateway received message from chat_id={}", req.chat_id);

    let gateway_msg = GatewayMessage {
        chat_id: req.chat_id.clone(),
        content: req.content.clone(),
        source: SessionSource::api(),
    };

    match state.gateway_tx.send(gateway_msg) {
        Ok(()) => {
            // 广播给 SSE 客户端
            state.broadcast_sse(StreamMessage {
                event_type: "user_message".to_string(),
                chat_id: req.chat_id.clone(),
                data: serde_json::json!({"content": req.content}),
            });

            (
                StatusCode::OK,
                Json(MessageResponse {
                    success: true,
                    reply: None,
                    error: None,
                }),
            )
        }
        Err(e) => {
            warn!("Failed to send message to gateway: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(MessageResponse {
                    success: false,
                    reply: None,
                    error: Some(format!("Gateway send failed: {}", e)),
                }),
            )
        }
    }
}

/// GET /api/gateway/health — 健康检查
async fn handle_health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

/// GET /api/gateway/stream — SSE 流式推送
///
/// 客户端通过 EventSource 连接后，实时接收所有 gateway 事件（用户消息、AI 回复等）。
async fn handle_sse_stream(
    State(state): State<AppState>,
) -> Sse<impl Stream<Item = Result<SseAxumEvent, Infallible>>> {
    let mut rx = state.sse_tx.subscribe();

    let stream = async_stream::stream! {
        loop {
            match rx.recv().await {
                Ok(msg) => {
                    let data = serde_json::to_string(&msg).unwrap_or_default();
                    yield Ok(SseAxumEvent::default()
                        .event(&msg.event_type)
                        .data(data));
                }
                Err(broadcast::error::RecvError::Lagged(n)) => {
                    warn!("SSE client lagged by {} messages", n);
                    continue;
                }
                Err(broadcast::error::RecvError::Closed) => {
                    break;
                }
            }
        }
    };

    Sse::new(stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(std::time::Duration::from_secs(30))
            .text("ping"),
    )
}

/// 启动 HTTP 服务器
pub async fn serve(state: AppState, addr: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let app = build_router(state);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    info!("Gateway HTTP server listening on {}", addr);
    axum::serve(listener, app).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tower::ServiceExt;

    #[tokio::test]
    async fn test_health_endpoint() {
        let (tx, _rx) = mpsc::unbounded_channel();
        let state = AppState::new(tx);
        let app = build_router(state);

        let response: axum::http::Response<axum::body::Body> = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/api/gateway/health")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::OK);
    }

    #[tokio::test]
    async fn test_message_endpoint_forwards_to_channel() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let state = AppState::new(tx);
        let app = build_router(state);

        let req_body = serde_json::to_string(&MessageRequest {
            chat_id: "test-chat".into(),
            content: "hello from test".into(),
        }).unwrap();

        let response: axum::http::Response<axum::body::Body> = app
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/api/gateway/message")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(req_body))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), axum::http::StatusCode::OK);

        // 验证消息通过 channel 正确传递
        let msg = rx.recv().await.unwrap();
        assert_eq!(msg.content, "hello from test");
        assert_eq!(msg.chat_id, "test-chat");
        assert_eq!(msg.source.platform, hermes_cfg::platform::Platform::Api);
    }

    #[tokio::test]
    async fn test_message_delivery_order() {
        let (tx, mut rx) = mpsc::unbounded_channel::<GatewayMessage>();
        let state = AppState::new(tx);

        // 发送 10 条消息，验证顺序
        for i in 0..10 {
            let msg = GatewayMessage {
                chat_id: format!("chat-{}", i),
                content: format!("message {}", i),
                source: SessionSource::api(),
            };
            state.gateway_tx.send(msg).unwrap();
        }

        for i in 0..10 {
            let msg = rx.recv().await.unwrap();
            assert_eq!(msg.content, format!("message {}", i));
        }
    }

    #[test]
    fn test_sse_broadcast() {
        let (tx, _rx) = mpsc::unbounded_channel();
        let state = AppState::new(tx);
        let mut sse_rx = state.sse_tx.subscribe();

        state.broadcast_sse(StreamMessage {
            event_type: "test".to_string(),
            chat_id: "c1".to_string(),
            data: serde_json::json!({"hello": "world"}),
        });

        let msg = sse_rx.try_recv().unwrap();
        assert_eq!(msg.event_type, "test");
        assert_eq!(msg.chat_id, "c1");
    }
}
