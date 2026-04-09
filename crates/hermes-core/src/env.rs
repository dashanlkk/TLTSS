use tracing::info;

/// 加载 .env 文件并打印脱敏摘要
pub fn load_env() {
    if dotenvy::dotenv().is_err() {
        info!("No .env file found, using system environment");
    }
    print_env_summary();
}

fn print_env_summary() {
    let sensitive_keys = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "TELEGRAM_BOT_TOKEN",
        "DISCORD_BOT_TOKEN",
    ];

    for key in &sensitive_keys {
        match std::env::var(key) {
            Ok(val) => {
                let masked = if val.len() > 4 {
                    format!("{}***", &val[..4])
                } else {
                    "***".to_string()
                };
                info!("{} = {}", key, masked);
            }
            Err(_) => {
                info!("{} = (not set)", key);
            }
        }
    }
}

/// 获取环境变量，不存在则返回 None
pub fn get_env(key: &str) -> Option<String> {
    std::env::var(key).ok()
}

/// 获取环境变量，不存在则返回默认值
pub fn get_env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}
