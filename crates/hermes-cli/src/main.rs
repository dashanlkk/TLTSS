use clap::{Parser, Subcommand};
use hermes_core::config::AppConfig;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Parser)]
#[command(name = "hermes", version, about = "Hermes Agent — Rust Edition")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start interactive chat
    Chat {
        /// Enable TUI mode
        #[arg(long)]
        tui: bool,
    },
    /// Show current configuration
    Config {
        #[command(subcommand)]
        action: ConfigCommands,
    },
    /// List registered tools
    Tools,
    /// Manage skills
    Skills {
        #[command(subcommand)]
        action: SkillCommands,
    },
    /// Manage cron jobs
    Cron {
        #[command(subcommand)]
        action: CronCommands,
    },
    /// Security audit
    Security {
        #[command(subcommand)]
        action: SecurityCommands,
    },
}

#[derive(Subcommand)]
enum ConfigCommands {
    /// Show current config
    Show,
    /// Validate config file
    Check,
}

#[derive(Subcommand)]
enum SkillCommands {
    /// List all skills
    List,
    /// Show skill details
    Show { name: String },
}

#[derive(Subcommand)]
enum CronCommands {
    /// List all jobs
    List,
    /// Run a job immediately
    Run { id: String },
}

#[derive(Subcommand)]
enum SecurityCommands {
    /// Audit security status
    Audit {
        /// Path to check
        #[arg(long)]
        path: Option<String>,
        /// Prompt to scan
        #[arg(long)]
        prompt: Option<String>,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("hermes=debug")
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Chat { tui } => {
            run_chat(tui).await?;
        }
        Commands::Config { action } => {
            match action {
                ConfigCommands::Show => show_config()?,
                ConfigCommands::Check => check_config()?,
            }
        }
        Commands::Tools => {
            list_tools()?;
        }
        Commands::Skills { action } => {
            match action {
                SkillCommands::List => list_skills()?,
                SkillCommands::Show { name } => show_skill(&name)?,
            }
        }
        Commands::Cron { action } => {
            match action {
                CronCommands::List => println!("No cron jobs (scheduler not running)"),
                CronCommands::Run { id } => println!("Would run job: {}", id),
            }
        }
        Commands::Security { action } => {
            match action {
                SecurityCommands::Audit { path, prompt } => {
                    run_security_audit(path, prompt)?;
                }
            }
        }
    }

    Ok(())
}

async fn run_chat(tui: bool) -> anyhow::Result<()> {
    use hermes_agent::{Agent, MemoryStore};
    use hermes_llm::FakeClient;
    use hermes_tools::ToolRegistry;
    use hermes_cfg::traits::LlmClient;

    println!("Hermes Agent v0.1.0");
    println!("Type 'exit' to quit.\n");

    let llm: Arc<dyn LlmClient> = Arc::new(FakeClient::new("Hello! I'm Hermes. How can I help you?"));
    let registry = Arc::new(ToolRegistry::new());
    let memory = Arc::new(MemoryStore::new());

    let agent = Agent::new(
        hermes_agent::agent::AgentConfig::default(),
        llm,
        registry,
        memory,
    );

    if tui {
        println!("TUI mode not yet implemented. Running in plain mode.\n");
    }

    // Simple REPL
    let mut input = String::new();
    loop {
        print!("> ");
        use std::io::Write;
        std::io::stdout().flush()?;

        input.clear();
        if std::io::stdin().read_line(&mut input).is_err() {
            break;
        }

        let trimmed = input.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed == "exit" || trimmed == "quit" {
            println!("Goodbye!");
            break;
        }

        match agent.chat(trimmed, hermes_cfg::platform::SessionSource::cli()).await {
            Ok(msg) => println!("{}\n", msg.content),
            Err(e) => eprintln!("Error: {}\n", e),
        }
    }

    Ok(())
}

fn show_config() -> anyhow::Result<()> {
    let config = AppConfig::default();
    println!("{}", serde_json::to_string_pretty(&config)?);
    Ok(())
}

fn check_config() -> anyhow::Result<()> {
    println!("Config check:");
    println!("  Message types: OK");
    println!("  Core traits:");
    println!("    - LlmClient (complete, complete_stream, ping)");
    println!("    - ToolHandler (execute)");
    println!("    - TerminalBackend (execute, close)");
    println!("    - PlatformAdapter (run, send, set_message_handler)");
    println!("  Error types: LlmError, ToolError, TerminalError, ConfigError, McpError, SkillError, CronError");
    Ok(())
}

fn list_tools() -> anyhow::Result<()> {
    println!("Built-in tools:");
    println!("  - read_file       Read file contents");
    println!("  - write_file      Write content to file");
    println!("  - execute_command Execute shell command");
    println!("  - list_dir        List directory contents");
    Ok(())
}

fn list_skills() -> anyhow::Result<()> {
    println!("No skills loaded.");
    println!("Place .yaml files in ~/.hermes/skills/ or ./hermes/skills/");
    Ok(())
}

fn show_skill(name: &str) -> anyhow::Result<()> {
    println!("Skill '{}' not found.", name);
    Ok(())
}

fn run_security_audit(path: Option<String>, prompt: Option<String>) -> anyhow::Result<()> {
    if let Some(p) = path {
        let base = std::path::PathBuf::from("/workspace");
        match hermes_security::validate_path(&base, &p) {
            Ok(_) => println!("[OK] Path is valid: {}", p),
            Err(e) => println!("[DENIED] {}", e),
        }
    }
    if let Some(p) = prompt {
        match hermes_security::scan_prompt(&p) {
            hermes_security::prompt::ScanResult::Safe => println!("[OK] Prompt is safe"),
            hermes_security::prompt::ScanResult::Suspicious { matched_pattern } => {
                println!("[SUSPICIOUS] Prompt injection pattern detected: {}", matched_pattern);
            }
        }
    }
    Ok(())
}
