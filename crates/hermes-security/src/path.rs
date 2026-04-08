use std::path::Path;

/// 检测路径是否试图逃逸允许的工作目录
pub fn validate_path(base_dir: &Path, input_path: &str) -> Result<std::path::PathBuf, String> {
    let resolved = base_dir.join(input_path);

    // canonicalize 需要 path 存在，使用逻辑检测代替
    let base_str = base_dir.to_string_lossy().replace('\\', "/");
    let resolved_str = resolved.to_string_lossy().replace('\\', "/");

    // 检测路径遍历
    if input_path.contains("..") {
        return Err(format!("Path traversal detected: {}", input_path));
    }

    // 确保解析后路径仍在 base_dir 内
    if !resolved_str.starts_with(&base_str) {
        return Err(format!("Path escapes base directory: {}", input_path));
    }

    Ok(resolved)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_reject_traversal() {
        let base = PathBuf::from("/workspace");
        assert!(validate_path(&base, "../../../etc/passwd").is_err());
        assert!(validate_path(&base, "../../sbin/halt").is_err());
    }

    #[test]
    fn test_allow_valid_path() {
        let base = PathBuf::from("/workspace");
        assert!(validate_path(&base, "src/main.rs").is_ok());
        assert!(validate_path(&base, "./README.md").is_ok());
    }
}
