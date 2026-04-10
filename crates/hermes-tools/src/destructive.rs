//! Destructive command detection for approval gating.
//!
//! Ported from Python hermes-agent `run_agent.py`:
//! Regex-based detection of destructive terminal commands (rm, mv, sed -i,
//! truncate, dd, shred, git reset/clean/checkout, output redirects).
//! Used for automatic approval level escalation.

use regex::Regex;
use std::sync::LazyLock;

/// Patterns that indicate a destructive command.
/// When matched, the command should require explicit user approval.
static DESTRUCTIVE_PATTERNS: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    let patterns: &[&str] = &[
        // File deletion
        r"\brm\s",
        r"\brm$",
        r"\brmdir\s",
        // File move/rename (can overwrite targets)
        r"\bmv\s",
        // In-place file modification
        r"\bsed\s+.*-i",
        // Truncate files
        r"\btruncate\s",
        // Low-level disk operations
        r"\bdd\s",
        r"\bshred\s",
        // Git destructive operations
        r"\bgit\s+reset\s",
        r"\bgit\s+clean\s",
        r"\bgit\s+checkout\s+\.",
        r"\bgit\s+checkout\s+--\s",
        r"\bgit\s+push\s+.*--force",
        r"\bgit\s+push\s+.*-f\b",
        // Output redirection (can overwrite files)
        r">\s*\S",
        // Package removal
        r"\bapt(?:-get)?\s+.*(?:remove|purge)",
        r"\byum\s+.*remove",
        r"\bdnf\s+.*remove",
        r"\bpip\s+.*uninstall",
        r"\bnpm\s+.*uninstall",
        // Container/image removal
        r"\bdocker\s+.*rm\b",
        r"\bdocker\s+.*rmi\b",
        r"\bdocker\s+.*prune",
        // Kill processes
        r"\bkill\s",
        r"\bkillall\s",
        r"\bpkill\s",
        // Format/mkfs
        r"\bmkfs\b",
        // Shutdown/reboot
        r"\bshutdown\s",
        r"\breboot\b",
    ];

    patterns
        .iter()
        .filter_map(|p| Regex::new(p).ok())
        .collect()
});

/// Check if a command is potentially destructive.
///
/// Returns `true` if the command matches any known destructive pattern.
/// This is used to automatically escalate the approval level from
/// `AutoApprove` to `RequireApproval`.
pub fn is_destructive_command(command: &str) -> bool {
    let trimmed = command.trim();
    if trimmed.is_empty() {
        return false;
    }

    DESTRUCTIVE_PATTERNS.iter().any(|re| re.is_match(trimmed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rm_detected() {
        assert!(is_destructive_command("rm file.txt"));
        assert!(is_destructive_command("rm -rf /tmp/test"));
    }

    #[test]
    fn test_mv_detected() {
        assert!(is_destructive_command("mv old.txt new.txt"));
    }

    #[test]
    fn test_sed_inplace_detected() {
        assert!(is_destructive_command("sed -i 's/old/new/g' file.txt"));
    }

    #[test]
    fn test_sed_normal_not_destructive() {
        assert!(!is_destructive_command("sed 's/old/new/g' file.txt"));
    }

    #[test]
    fn test_truncate_detected() {
        assert!(is_destructive_command("truncate -s 0 file.log"));
    }

    #[test]
    fn test_dd_detected() {
        assert!(is_destructive_command("dd if=/dev/zero of=/dev/sda"));
    }

    #[test]
    fn test_git_reset_detected() {
        assert!(is_destructive_command("git reset --hard HEAD~1"));
    }

    #[test]
    fn test_git_clean_detected() {
        assert!(is_destructive_command("git clean -fd"));
    }

    #[test]
    fn test_git_checkout_dot_detected() {
        assert!(is_destructive_command("git checkout ."));
    }

    #[test]
    fn test_git_push_force_detected() {
        assert!(is_destructive_command("git push origin --force"));
        assert!(is_destructive_command("git push -f origin main"));
    }

    #[test]
    fn test_git_push_normal_not_destructive() {
        assert!(!is_destructive_command("git push origin main"));
    }

    #[test]
    fn test_redirect_detected() {
        assert!(is_destructive_command("echo hello > file.txt"));
    }

    #[test]
    fn test_kill_detected() {
        assert!(is_destructive_command("kill -9 1234"));
        assert!(is_destructive_command("killall python"));
    }

    #[test]
    fn test_safe_commands_not_destructive() {
        assert!(!is_destructive_command("ls -la"));
        assert!(!is_destructive_command("cat file.txt"));
        assert!(!is_destructive_command("echo hello world"));
        assert!(!is_destructive_command("grep pattern file.txt"));
        assert!(!is_destructive_command("cargo build"));
        assert!(!is_destructive_command("git status"));
        assert!(!is_destructive_command("git log --oneline"));
        assert!(!is_destructive_command("docker ps"));
        assert!(!is_destructive_command("docker build -t app ."));
    }

    #[test]
    fn test_empty_command_not_destructive() {
        assert!(!is_destructive_command(""));
        assert!(!is_destructive_command("   "));
    }

    #[test]
    fn test_docker_rm_detected() {
        assert!(is_destructive_command("docker rm container_id"));
    }

    #[test]
    fn test_apt_remove_detected() {
        assert!(is_destructive_command("apt-get remove package"));
        assert!(is_destructive_command("apt purge package"));
    }

    #[test]
    fn test_pip_uninstall_detected() {
        assert!(is_destructive_command("pip uninstall package"));
    }

    #[test]
    fn test_shutdown_reboot_detected() {
        assert!(is_destructive_command("shutdown -h now"));
        assert!(is_destructive_command("reboot"));
    }

    #[test]
    fn test_mkfs_detected() {
        assert!(is_destructive_command("mkfs.ext4 /dev/sda1"));
    }

    #[test]
    fn test_shred_detected() {
        assert!(is_destructive_command("shred -u secret.txt"));
    }
}
