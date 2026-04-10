//! Bearer token authentication middleware for Gateway HTTP endpoints.
//!
//! Reads the expected token from `HERMES_GATEWAY_TOKEN` environment variable.
//! If the variable is not set, auth is disabled (suitable for localhost dev).
//! Health endpoint (`/api/gateway/health`) is always exempt.

use axum::{
    extract::Request,
    http::{header::AUTHORIZATION, StatusCode},
    middleware::Next,
    response::Response,
};

/// Bearer token read from environment (cached on first call).
static GATEWAY_TOKEN: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();

fn get_expected_token() -> Option<&'static str> {
    GATEWAY_TOKEN
        .get_or_init(|| std::env::var("HERMES_GATEWAY_TOKEN").ok())
        .as_deref()
}

/// Axum middleware: reject requests without a valid Bearer token.
///
/// Skips auth for health endpoint and when no token is configured.
pub async fn auth_middleware(request: Request, next: Next) -> Result<Response, StatusCode> {
    let expected = match get_expected_token() {
        Some(token) => token,
        None => return Ok(next.run(request).await), // No token configured — skip auth
    };

    // Health endpoint is always exempt
    if request.uri().path() == "/api/gateway/health" {
        return Ok(next.run(request).await);
    }

    // Extract Authorization header
    let auth_header = request
        .headers()
        .get(AUTHORIZATION)
        .and_then(|v| v.to_str().ok());

    match auth_header {
        Some(header) if header.starts_with("Bearer ") => {
            let token = &header[7..];
            if token == expected {
                Ok(next.run(request).await)
            } else {
                Err(StatusCode::UNAUTHORIZED)
            }
        }
        _ => Err(StatusCode::UNAUTHORIZED),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::Body, middleware, routing::get, Router};
    use tower::ServiceExt;

    async fn ok_handler() -> &'static str {
        "ok"
    }

    #[tokio::test]
    async fn test_auth_skipped_when_no_token_configured() {
        // Ensure no env var set (test isolation)
        std::env::remove_var("HERMES_GATEWAY_TOKEN");

        let app = Router::new()
            .route("/api/gateway/message", get(ok_handler))
            .layer(middleware::from_fn(auth_middleware));

        let resp = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/api/gateway/message")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
    }
}
