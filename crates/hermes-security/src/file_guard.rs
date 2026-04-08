use glob::Pattern;
use std::sync::LazyLock;

static SENSITIVE_PATTERNS: LazyLock<Vec<Pattern>> = LazyLock::new(|| {
    let patterns = [
        ".env",
        ".env.*",
        "config.yaml",
        "config.yml",
        "config.json",
        "id_rsa",
        "id_rsa.*",
        "id_ed25519",
        "*.pem",
        "*.key",
        "*.secret",
        "credentials.json",
        "service-account*.json",
        ".gitconfig",
        ".npmrc",
        ".pypirc",
    ];
    patterns
        .iter()
        .filter_map(|p| Pattern::new(p).ok())
        .collect()
});

/// 检查文件路径是否匹配敏感文件模式
pub fn is_sensitive_file(path: &str) -> bool {
    let filename = std::path::Path::new(path)
        .file_name()
        .map(|f| f.to_string_lossy().to_string())
        .unwrap_or_default();

    SENSITIVE_PATTERNS.iter().any(|p| p.matches(&filename))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reject_sensitive_files() {
        assert!(is_sensitive_file(".env"));
        assert!(is_sensitive_file("config.yaml"));
        assert!(is_sensitive_file("id_rsa"));
        assert!(is_sensitive_file("server.pem"));
        assert!(is_sensitive_file("my_secret.key"));
        assert!(is_sensitive_file("credentials.json"));
    }

    #[test]
    fn test_allow_normal_files() {
        assert!(!is_sensitive_file("src/main.rs"));
        assert!(!is_sensitive_file("README.md"));
        assert!(!is_sensitive_file("Cargo.toml"));
        assert!(!is_sensitive_file("lib.rs"));
    }
}
