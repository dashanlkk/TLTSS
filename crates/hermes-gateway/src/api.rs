use axum::{
    extract::State,
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use hermes_cfg::platform::SessionSource;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
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

/// HTTP Server 共享状态
#[derive(Clone)]
pub struct AppState {
    pub gateway_tx: mpsc::UnboundedSender<GatewayMessage>,
}

impl AppState {
    pub fn new(gateway_tx: mpsc::UnboundedSender<GatewayMessage>) -> Self {
        Self { gateway_tx }
    }
}

/// 构建 Axum Router
pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/api/gateway/message", post(handle_message))
        .route("/api/gateway/health", get(handle_health))
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
        Ok(()) => (
            StatusCode::OK,
            Json(MessageResponse {
                success: true,
                reply: None,
                error: None,
            }),
        ),
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
}
