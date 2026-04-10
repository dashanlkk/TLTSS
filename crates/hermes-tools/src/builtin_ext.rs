//! Extended built-in tools — 7 tools aligned with Python hermes-agent.
//!
//! Tools: web_search, web_extract, session_search,
//!        skills_list, skill_view, vision_analyze, patch

use async_trait::async_trait;
use hermes_cfg::prelude::*;
use hermes_cfg::traits::{ToolContext, ToolHandler};
use hermes_security::validate_path;
use std::path::PathBuf;

use crate::coerce;

// ── 1. WebSearchTool ──────────────────────────────────────────────────

/// Web search tool using DuckDuckGo HTML as default backend.
///
/// Returns JSON: `{"success":true,"data":{"web":[{title,url,description,position}]}}`
pub struct WebSearchTool {
    /// Optional API key for premium search backends (Tavily/Exa).
    api_key: Option<String>,
    /// Optional search endpoint override.
    endpoint: Option<String>,
}

impl WebSearchTool {
    pub fn new() -> Self {
        Self {
            api_key: std::env::var("SEARCH_API_KEY").ok(),
            endpoint: std::env::var("SEARCH_ENDPOINT").ok(),
        }
    }

    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    pub fn with_endpoint(mut self, url: impl Into<String>) -> Self {
        self.endpoint = Some(url.into());
        self
    }
}

impl Default for WebSearchTool {
    fn default() -> Self { Self::new() }
}

#[async_trait]
impl ToolHandler for WebSearchTool {
    fn name(&self) -> &str { "web_search" }
    fn description(&self) -> &str {
        "Search the web for information. Returns a list of results with titles, URLs, and descriptions."
    }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": { "type": "string", "description": "Search query" },
                "max_results": { "type": "integer", "description": "Maximum number of results (default: 5, max: 10)" }
            },
            "required": ["query"]
        })
    }

    async fn execute(&self, arguments: &str, _ctx: &ToolContext) -> Result<ToolResult, ToolError> {
        let mut args: serde_json::Value = serde_json::from_str(arguments)
            .map_err(|e| ToolError::InvalidArguments(e.to_string()))?;
        coerce::coerce_arguments(&mut args, &self.parameters_schema());

        let query = args["query"].as_str()
            .ok_or_else(|| ToolError::InvalidArguments("missing query".into()))?;
        let max_results = args["max_results"].as_u64().unwrap_or(5).min(10) as usize;

        // Try premium backend first, fall back to DDG HTML
        if self.api_key.is_some() && self.endpoint.is_some() {
            if let Ok(result) = self.search_premium(query, max_results).await {
                return Ok(result);
            }
        }

        self.search_ddg(query, max_results).await
    }
}

impl WebSearchTool {
    /// DuckDuckGo HTML search fallback.
    async fn search_ddg(&self, query: &str, max_results: usize) -> Result<ToolResult, ToolError> {
        let url = format!(
            "https://html.duckduckgo.com/html/?q={}",
            urlencoding(query)
        );

        let response = reqwest::get(&url).await
            .map_err(|e| ToolError::ExecutionFailed(format!("Search request failed: {}", e)))?;

        let html = response.text().await
            .map_err(|e| ToolError::ExecutionFailed(format!("Failed to read response: {}", e)))?;

        let results = parse_ddg_html(&html, max_results);

        let json = serde_json::json!({
            "success": true,
            "data": {
                "web": results
            }
        });

        Ok(ToolResult::success("web_search", serde_json::to_string_pretty(&json).unwrap_or_default()))
    }

    /// Premium search backend (Tavily/Exa/etc.).
    async fn search_premium(&self, query: &str, max_results: usize) -> Result<ToolResult, ToolError> {
        let endpoint = self.endpoint.as_ref()
            .ok_or_else(|| ToolError::ExecutionFailed("Search endpoint not configured. Call initialize() first.".into()))?;
        let api_key = self.api_key.as_ref()
            .ok_or_else(|| ToolError::ExecutionFailed("Search API key not configured. Call initialize() first.".into()))?;

        let body = serde_json::json!({
            "query": query,
            "max_results": max_results,
            "api_key": api_key,
        });

        let response = reqwest::Client::new()
            .post(endpoint)
            .json(&body)
            .send()
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("Premium search failed: {}", e)))?;

        let status = response.status();
        if !status.is_success() {
            return Err(ToolError::ExecutionFailed(format!("Premium search HTTP {}", status)));
        }

        let text = response.text().await
            .map_err(|e| ToolError::ExecutionFailed(format!("Failed to read response: {}", e)))?;

        Ok(ToolResult::success("web_search", text))
    }
}

/// URL-encode a query string (basic implementation).
fn urlencoding(input: &str) -> String {
    input.chars().map(|c| {
        if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' || c == '~' {
            c.to_string()
        } else {
            format!("%{:02X}", c as u8)
        }
    }).collect()
}

/// Parse DDG HTML response to extract search results.
fn parse_ddg_html(html: &str, max_results: usize) -> Vec<serde_json::Value> {
    let mut results = Vec::new();
    let re_result = regex::Regex::new(r#"<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>"#).ok();
    let re_snippet = regex::Regex::new(r#"<a[^>]*class="result__snippet"[^>]*>(.*?)</a>"#).ok();

    if let (Some(re_result), Some(re_snippet)) = (re_result, re_snippet) {
        for (i, cap) in re_result.captures_iter(html).enumerate() {
            if i >= max_results { break; }
            let url = strip_html_tags(&cap[1]);
            let title = strip_html_tags(&cap[2]);
            let snippet = re_snippet.captures_iter(html)
                .nth(i)
                .map(|s| strip_html_tags(&s[1]))
                .unwrap_or_default();

            results.push(serde_json::json!({
                "title": title,
                "url": url,
                "description": snippet,
                "position": i + 1,
            }));
        }
    }

    results
}

/// Strip basic HTML tags from a string.
fn strip_html_tags(input: &str) -> String {
    let re = regex::Regex::new(r"<[^>]*>").unwrap_or_else(|_| regex::Regex::new("").unwrap());
    re.replace_all(input, "").trim().to_string()
}

// ── 2. WebExtractTool ─────────────────────────────────────────────────

/// Extract clean text content from URLs.
///
/// Fetches each URL, strips HTML tags, truncates to max content length.
/// Returns JSON: `{"results":[{url,content,error}]}`
pub struct WebExtractTool {
    max_content_length: usize,
}

impl WebExtractTool {
    pub fn new() -> Self {
        Self { max_content_length: 5000 }
    }

    pub fn with_max_content_length(mut self, len: usize) -> Self {
        self.max_content_length = len;
        self
    }
}

impl Default for WebExtractTool {
    fn default() -> Self { Self::new() }
}

#[async_trait]
impl ToolHandler for WebExtractTool {
    fn name(&self) -> &str { "web_extract" }
    fn description(&self) -> &str {
        "Extract clean text content from one or more URLs. Fetches each URL and strips HTML."
    }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "urls": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "List of URLs to extract content from (max 5)"
                }
            },
            "required": ["urls"]
        })
    }

    async fn execute(&self, arguments: &str, _ctx: &ToolContext) -> Result<ToolResult, ToolError> {
        let args: serde_json::Value = serde_json::from_str(arguments)
            .map_err(|e| ToolError::InvalidArguments(e.to_string()))?;

        let urls = args["urls"].as_array()
            .ok_or_else(|| ToolError::InvalidArguments("missing or invalid urls array".into()))?;

        if urls.len() > 5 {
            return Err(ToolError::InvalidArguments("maximum 5 URLs per request".into()));
        }

        let mut results = Vec::new();
        for url_val in urls {
            let url = match url_val.as_str() {
                Some(u) => u,
                None => {
                    results.push(serde_json::json!({
                        "url": format!("{:?}", url_val),
                        "content": "",
                        "error": "invalid URL format"
                    }));
                    continue;
                }
            };

            if !url.starts_with("http://") && !url.starts_with("https://") {
                results.push(serde_json::json!({
                    "url": url,
                    "content": "",
                    "error": "url must start with http:// or https://"
                }));
                continue;
            }

            match reqwest::get(url).await {
                Ok(response) => {
                    let status = response.status();
                    if !status.is_success() {
                        results.push(serde_json::json!({
                            "url": url,
                            "content": "",
                            "error": format!("HTTP {}", status)
                        }));
                        continue;
                    }
                    match response.text().await {
                        Ok(body) => {
                            let clean = strip_html_tags(&body);
                            let content = if clean.len() > self.max_content_length {
                                format!("{}...\n[truncated, {} total characters]",
                                    &clean[..self.max_content_length], clean.len())
                            } else {
                                clean
                            };
                            results.push(serde_json::json!({
                                "url": url,
                                "content": content,
                                "error": null
                            }));
                        }
                        Err(e) => {
                            results.push(serde_json::json!({
                                "url": url,
                                "content": "",
                                "error": e.to_string()
                            }));
                        }
                    }
                }
                Err(e) => {
                    results.push(serde_json::json!({
                        "url": url,
                        "content": "",
                        "error": e.to_string()
                    }));
                }
            }
        }

        let json = serde_json::json!({ "results": results });
        Ok(ToolResult::success("web_extract", serde_json::to_string_pretty(&json).unwrap_or_default()))
    }
}

// ── 3. SessionSearchTool ──────────────────────────────────────────────

/// Search through saved sessions.
///
/// Two modes:
/// - "recent": return latest N session metadata
/// - "search": keyword search through session messages
///
/// Scans `.json` files in `data_dir/sessions/` to avoid circular dependency
/// on hermes-agent's SessionSearcher.
pub struct SessionSearchTool {
    data_dir: PathBuf,
}

impl SessionSearchTool {
    pub fn new(data_dir: impl Into<PathBuf>) -> Self {
        Self { data_dir: data_dir.into() }
    }

    fn sessions_dir(&self) -> PathBuf {
        self.data_dir.join("sessions")
    }
}

#[async_trait]
impl ToolHandler for SessionSearchTool {
    fn name(&self) -> &str { "session_search" }
    fn description(&self) -> &str {
        "Search through session history. Use mode='recent' for latest sessions, mode='search' for keyword search."
    }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "mode": { "type": "string", "description": "Search mode: 'recent' or 'search' (default: recent)" },
                "query": { "type": "string", "description": "Search query (required for 'search' mode)" },
                "limit": { "type": "integer", "description": "Max results (default: 3, max: 5)" }
            },
            "required": []
        })
    }

    async fn execute(&self, arguments: &str, _ctx: &ToolContext) -> Result<ToolResult, ToolError> {
        let mut args: serde_json::Value = serde_json::from_str(arguments)
            .map_err(|e| ToolError::InvalidArguments(e.to_string()))?;
        coerce::coerce_arguments(&mut args, &self.parameters_schema());

        let mode = args["mode"].as_str().unwrap_or("recent");
        let limit = args["limit"].as_u64().unwrap_or(3).min(5) as usize;
        let query = args["query"].as_str().unwrap_or("");

        let sessions_dir = self.sessions_dir();
        if !sessions_dir.exists() {
            return Ok(ToolResult::success("session_search", r#"{"sessions":[],"count":0}"#));
        }

        match mode {
            "recent" => self.search_recent(&sessions_dir, limit).await,
            "search" => {
                if query.is_empty() {
                    return Err(ToolError::InvalidArguments("query required for search mode".into()));
                }
                self.search_keyword(&sessions_dir, query, limit).await
            }
            _ => Err(ToolError::InvalidArguments(format!("unknown mode: '{}'. Use 'recent' or 'search'", mode))),
        }
    }
}

impl SessionSearchTool {
    async fn search_recent(&self, dir: &PathBuf, limit: usize) -> Result<ToolResult, ToolError> {
        let mut entries = Vec::new();
        let mut rd = tokio::fs::read_dir(dir).await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

        while let Some(entry) = rd.next_entry().await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?
        {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("json") {
                let modified = entry.metadata().await
                    .ok()
                    .and_then(|m| m.modified().ok())
                    .map(|t| {
                        let dur = t.duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
                        dur.as_secs()
                    })
                    .unwrap_or(0);
                entries.push((path, modified));
            }
        }

        // Sort by modification time descending
        entries.sort_by(|a, b| b.1.cmp(&a.1));

        let mut sessions = Vec::new();
        for (path, _) in entries.iter().take(limit) {
            if let Ok(content) = tokio::fs::read_to_string(path).await {
                if let Ok(meta) = Self::parse_session_meta(path, &content) {
                    sessions.push(meta);
                }
            }
        }

        let json = serde_json::json!({ "sessions": sessions, "count": sessions.len() });
        Ok(ToolResult::success("session_search", serde_json::to_string_pretty(&json).unwrap_or_default()))
    }

    async fn search_keyword(&self, dir: &PathBuf, query: &str, limit: usize) -> Result<ToolResult, ToolError> {
        let query_lower = query.to_lowercase();
        let mut matches = Vec::new();
        let mut rd = tokio::fs::read_dir(dir).await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

        while let Some(entry) = rd.next_entry().await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?
        {
            if matches.len() >= limit { break; }
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") { continue; }

            if let Ok(content) = tokio::fs::read_to_string(&path).await {
                if content.to_lowercase().contains(&query_lower) {
                    if let Ok(meta) = Self::parse_session_meta(&path, &content) {
                        matches.push(meta);
                    }
                }
            }
        }

        let json = serde_json::json!({ "sessions": matches, "count": matches.len(), "query": query });
        Ok(ToolResult::success("session_search", serde_json::to_string_pretty(&json).unwrap_or_default()))
    }

    /// Extract session metadata from JSON content without depending on hermes-agent types.
    fn parse_session_meta(path: &std::path::Path, content: &str) -> Result<serde_json::Value, ()> {
        let val: serde_json::Value = serde_json::from_str(content).map_err(|_| ())?;
        let id = path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        let created = val.get("created").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let message_count = val.get("messages")
            .and_then(|v| v.as_array())
            .map(|a| a.len())
            .unwrap_or(0);

        // Preview: first user message content
        let preview = val.get("messages")
            .and_then(|v| v.as_array())
            .and_then(|msgs| msgs.iter().find_map(|m| {
                if m.get("role").and_then(|r| r.as_str()) == Some("user") {
                    m.get("content").and_then(|c| c.as_str())
                        .map(|s| if s.len() > 100 { format!("{}...", &s[..100]) } else { s.to_string() })
                } else {
                    None
                }
            }))
            .unwrap_or_default();

        Ok(serde_json::json!({
            "id": id,
            "created": created,
            "message_count": message_count,
            "preview": preview,
        }))
    }
}

// ── 4. SkillsListTool ─────────────────────────────────────────────────

/// List available skills from YAML files.
///
/// Scans `skills_dir` for `.yaml` files and parses basic metadata.
/// Returns JSON: `{"success":true,"skills":[...],"categories":[...],"count":N}`
pub struct SkillsListTool {
    skills_dir: PathBuf,
}

impl SkillsListTool {
    pub fn new(skills_dir: impl Into<PathBuf>) -> Self {
        Self { skills_dir: skills_dir.into() }
    }
}

#[async_trait]
impl ToolHandler for SkillsListTool {
    fn name(&self) -> &str { "skills_list" }
    fn description(&self) -> &str {
        "List all available skills. Optionally filter by category."
    }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "category": { "type": "string", "description": "Optional category filter" }
            },
            "required": []
        })
    }

    async fn execute(&self, arguments: &str, _ctx: &ToolContext) -> Result<ToolResult, ToolError> {
        let args: serde_json::Value = serde_json::from_str(arguments)
            .map_err(|e| ToolError::InvalidArguments(e.to_string()))?;
        let category_filter = args["category"].as_str();

        if !self.skills_dir.exists() {
            let json = serde_json::json!({"success": true, "skills": [], "categories": [], "count": 0});
            return Ok(ToolResult::success("skills_list", serde_json::to_string_pretty(&json).unwrap_or_default()));
        }

        let mut skills = Vec::new();
        let mut categories = Vec::new();
        let mut rd = tokio::fs::read_dir(&self.skills_dir).await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

        while let Some(entry) = rd.next_entry().await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?
        {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("yaml") { continue; }

            if let Ok(content) = tokio::fs::read_to_string(&path).await {
                if let Ok(meta) = Self::parse_skill_meta(&content) {
                    let skill_category = meta.get("category")
                        .and_then(|v| v.as_str())
                        .unwrap_or("general")
                        .to_string();

                    if !categories.contains(&skill_category) {
                        categories.push(skill_category.clone());
                    }

                    // Apply category filter
                    if let Some(filter) = category_filter {
                        if skill_category != filter { continue; }
                    }

                    skills.push(meta);
                }
            }
        }

        categories.sort();
        let count = skills.len();
        let json = serde_json::json!({
            "success": true,
            "skills": skills,
            "categories": categories,
            "count": count,
        });

        Ok(ToolResult::success("skills_list", serde_json::to_string_pretty(&json).unwrap_or_default()))
    }
}

impl SkillsListTool {
    /// Parse minimal skill metadata from YAML without depending on hermes-skill types.
    fn parse_skill_meta(content: &str) -> Result<serde_json::Value, ()> {
        let yaml_val: serde_yaml::Value = serde_yaml::from_str(content).map_err(|_| ())?;

        let name = yaml_val.get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("unnamed")
            .to_string();
        let description = yaml_val.get("description")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let category = yaml_val.get("category")
            .and_then(|v| v.as_str())
            .unwrap_or("general")
            .to_string();

        Ok(serde_json::json!({
            "name": name,
            "description": description,
            "category": category,
        }))
    }
}

// ── 5. SkillViewTool ──────────────────────────────────────────────────

/// View detailed content of a specific skill.
///
/// Reads the YAML file and returns full skill content including tags,
/// linked files, and environment variables.
pub struct SkillViewTool {
    skills_dir: PathBuf,
}

impl SkillViewTool {
    pub fn new(skills_dir: impl Into<PathBuf>) -> Self {
        Self { skills_dir: skills_dir.into() }
    }
}

#[async_trait]
impl ToolHandler for SkillViewTool {
    fn name(&self) -> &str { "skill_view" }
    fn description(&self) -> &str {
        "View detailed content of a specific skill by name."
    }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "name": { "type": "string", "description": "Skill name to view" },
                "file_path": { "type": "string", "description": "Optional specific file within the skill to view" }
            },
            "required": ["name"]
        })
    }

    async fn execute(&self, arguments: &str, _ctx: &ToolContext) -> Result<ToolResult, ToolError> {
        let args: serde_json::Value = serde_json::from_str(arguments)
            .map_err(|e| ToolError::InvalidArguments(e.to_string()))?;

        let name = args["name"].as_str()
            .ok_or_else(|| ToolError::InvalidArguments("missing name".into()))?;

        let skill_file = self.skills_dir.join(format!("{}.yaml", name));
        if !skill_file.exists() {
            return Ok(ToolResult::error("skill_view", format!("Skill '{}' not found", name)));
        }

        let content = tokio::fs::read_to_string(&skill_file).await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

        // Parse YAML to extract structured info
        let yaml_val: serde_yaml::Value = serde_yaml::from_str(&content).unwrap_or(serde_yaml::Value::Null);

        let tags = yaml_val.get("tags")
            .and_then(|v| v.as_sequence())
            .map(|seq| seq.iter().filter_map(|v| v.as_str().map(String::from)).collect::<Vec<_>>())
            .unwrap_or_default();

        let env_vars = yaml_val.get("env_vars")
            .and_then(|v| v.as_sequence())
            .map(|seq| seq.iter().filter_map(|v| v.as_str().map(String::from)).collect::<Vec<_>>())
            .unwrap_or_default();

        let linked_files = yaml_val.get("linked_files")
            .and_then(|v| v.as_sequence())
            .map(|seq| seq.iter().filter_map(|v| v.as_str().map(String::from)).collect::<Vec<_>>())
            .unwrap_or_default();

        // If a specific file_path is requested, read that too
        let file_content = if let Some(fp) = args["file_path"].as_str() {
            let linked = self.skills_dir.join(name).join(fp);
            if linked.exists() {
                tokio::fs::read_to_string(&linked).await.ok()
            } else {
                Some(format!("File '{}' not found in skill '{}'", fp, name))
            }
        } else {
            None
        };

        let mut result = serde_json::json!({
            "name": name,
            "content": content,
            "tags": tags,
            "env_vars": env_vars,
            "linked_files": linked_files,
        });

        if let Some(fc) = file_content {
            result["file_content"] = serde_json::json!(fc);
        }

        Ok(ToolResult::success("skill_view", serde_json::to_string_pretty(&result).unwrap_or_default()))
    }
}

// ── 6. VisionAnalyzeTool ──────────────────────────────────────────────

/// Analyze an image using a vision-capable LLM.
///
/// Downloads/reads an image, base64 encodes it, and sends to an
/// OpenAI-compatible vision API endpoint.
pub struct VisionAnalyzeTool {
    api_key: Option<String>,
    endpoint: Option<String>,
    model: String,
}

impl VisionAnalyzeTool {
    pub fn new() -> Self {
        Self {
            api_key: std::env::var("OPENAI_API_KEY")
                .or_else(|_| std::env::var("ANTHROPIC_API_KEY"))
                .ok(),
            endpoint: std::env::var("VISION_ENDPOINT").ok(),
            model: std::env::var("VISION_MODEL")
                .unwrap_or_else(|_| "gpt-4o".to_string()),
        }
    }

    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    pub fn with_endpoint(mut self, url: impl Into<String>) -> Self {
        self.endpoint = Some(url.into());
        self
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }
}

impl Default for VisionAnalyzeTool {
    fn default() -> Self { Self::new() }
}

#[async_trait]
impl ToolHandler for VisionAnalyzeTool {
    fn name(&self) -> &str { "vision_analyze" }
    fn description(&self) -> &str {
        "Analyze an image using vision AI. Provide an image URL or local path and a question about the image."
    }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "image_url": { "type": "string", "description": "URL or local path of the image to analyze" },
                "question": { "type": "string", "description": "Question or prompt about the image" }
            },
            "required": ["image_url", "question"]
        })
    }

    async fn execute(&self, arguments: &str, _ctx: &ToolContext) -> Result<ToolResult, ToolError> {
        let args: serde_json::Value = serde_json::from_str(arguments)
            .map_err(|e| ToolError::InvalidArguments(e.to_string()))?;

        let image_url = args["image_url"].as_str()
            .ok_or_else(|| ToolError::InvalidArguments("missing image_url".into()))?;
        let question = args["question"].as_str()
            .ok_or_else(|| ToolError::InvalidArguments("missing question".into()))?;

        let api_key = self.api_key.as_ref()
            .ok_or_else(|| ToolError::ExecutionFailed("No API key configured. Set OPENAI_API_KEY or ANTHROPIC_API_KEY".into()))?;

        // Obtain image bytes
        let image_bytes: Vec<u8> = if image_url.starts_with("http://") || image_url.starts_with("https://") {
            let resp = reqwest::get(image_url).await
                .map_err(|e| ToolError::ExecutionFailed(format!("Failed to download image: {}", e)))?;
            resp.bytes().await
                .map_err(|e| ToolError::ExecutionFailed(format!("Failed to read image bytes: {}", e)))?
                .to_vec()
        } else {
            // Local file path
            tokio::fs::read(image_url).await
                .map_err(|e| ToolError::ExecutionFailed(format!("Failed to read image file: {}", e)))?
        };

        let b64 = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &image_bytes);

        // Detect MIME type from extension or default to png
        let mime = if image_url.ends_with(".jpg") || image_url.ends_with(".jpeg") {
            "image/jpeg"
        } else if image_url.ends_with(".gif") {
            "image/gif"
        } else if image_url.ends_with(".webp") {
            "image/webp"
        } else {
            "image/png"
        };

        // Build OpenAI-compatible vision request
        let endpoint = self.endpoint.as_deref()
            .unwrap_or("https://api.openai.com/v1/chat/completions");

        let body = serde_json::json!({
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    { "type": "text", "text": question },
                    { "type": "image_url", "image_url": { "url": format!("data:{};base64,{}", mime, b64) } }
                ]
            }],
            "max_tokens": 1000
        });

        let response = reqwest::Client::new()
            .post(endpoint)
            .header("Authorization", format!("Bearer {}", api_key))
            .json(&body)
            .send()
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("Vision API request failed: {}", e)))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(ToolError::ExecutionFailed(format!("Vision API HTTP {}: {}", status, error_text)));
        }

        let resp_json: serde_json::Value = response.json().await
            .map_err(|e| ToolError::ExecutionFailed(format!("Failed to parse vision response: {}", e)))?;

        let analysis = resp_json["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("No analysis returned");

        let json = serde_json::json!({
            "success": true,
            "analysis": analysis,
        });

        Ok(ToolResult::success("vision_analyze", serde_json::to_string_pretty(&json).unwrap_or_default()))
    }
}

// ── 7. PatchTool ──────────────────────────────────────────────────────

/// Apply find/replace patches to files.
///
/// Mode "replace": find `old_string` and replace with `new_string`.
/// Supports `replace_all` to replace all occurrences.
/// Returns unified diff of changes.
pub struct PatchTool {
    base_dir: PathBuf,
}

impl PatchTool {
    pub fn new(base_dir: impl Into<PathBuf>) -> Self {
        Self { base_dir: base_dir.into() }
    }
}

#[async_trait]
impl ToolHandler for PatchTool {
    fn name(&self) -> &str { "patch" }
    fn description(&self) -> &str {
        "Apply a patch to a file. Use mode='replace' to find and replace text. Returns a unified diff."
    }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "mode": { "type": "string", "description": "Patch mode: 'replace' (default)" },
                "path": { "type": "string", "description": "File path to patch" },
                "old_string": { "type": "string", "description": "Text to find (for replace mode)" },
                "new_string": { "type": "string", "description": "Replacement text (for replace mode)" },
                "replace_all": { "type": "boolean", "description": "Replace all occurrences (default: false)" }
            },
            "required": ["mode", "path", "old_string", "new_string"]
        })
    }

    async fn execute(&self, arguments: &str, _ctx: &ToolContext) -> Result<ToolResult, ToolError> {
        let mut args: serde_json::Value = serde_json::from_str(arguments)
            .map_err(|e| ToolError::InvalidArguments(e.to_string()))?;
        coerce::coerce_arguments(&mut args, &self.parameters_schema());

        let mode = args["mode"].as_str().unwrap_or("replace");
        let path = args["path"].as_str()
            .ok_or_else(|| ToolError::InvalidArguments("missing path".into()))?;
        let old_string = args["old_string"].as_str()
            .ok_or_else(|| ToolError::InvalidArguments("missing old_string".into()))?;
        let new_string = args["new_string"].as_str()
            .ok_or_else(|| ToolError::InvalidArguments("missing new_string".into()))?;
        let replace_all = args["replace_all"].as_bool().unwrap_or(false);

        let validated = validate_path(&self.base_dir, path)
            .map_err(|_| ToolError::PathTraversal)?;

        match mode {
            "replace" => {
                let original = tokio::fs::read_to_string(&validated).await
                    .map_err(|e| ToolError::ExecutionFailed(format!("Failed to read file: {}", e)))?;

                if !original.contains(old_string) {
                    return Ok(ToolResult::error("patch", format!(
                        "old_string not found in {}. The exact text must match.",
                        path
                    )));
                }

                // Count occurrences
                let count = original.matches(old_string).count();
                if count > 1 && !replace_all {
                    return Err(ToolError::InvalidArguments(format!(
                        "old_string found {} times in {}. Set replace_all=true to replace all occurrences, or provide more context to uniquely identify the location.",
                        count, path
                    )));
                }

                let new_content = if replace_all {
                    original.replace(old_string, new_string)
                } else {
                    original.replacen(old_string, new_string, 1)
                };

                tokio::fs::write(&validated, &new_content).await
                    .map_err(|e| ToolError::ExecutionFailed(format!("Failed to write file: {}", e)))?;

                let diff = generate_unified_diff(path, &original, &new_content);
                let json = serde_json::json!({
                    "success": true,
                    "path": path,
                    "replacements": if replace_all { count } else { 1 },
                    "diff": diff,
                });

                Ok(ToolResult::success("patch", serde_json::to_string_pretty(&json).unwrap_or_default()))
            }
            _ => Err(ToolError::InvalidArguments(format!("unknown mode: '{}'. Use 'replace'", mode))),
        }
    }
}

/// Generate a minimal unified diff between original and new content.
fn generate_unified_diff(path: &str, original: &str, new_content: &str) -> String {
    let mut diff = format!("--- {}\n+++ {}\n", path, path);

    let old_lines: Vec<&str> = original.lines().collect();
    let new_lines: Vec<&str> = new_content.lines().collect();

    // Simple line-by-line diff: find first and last differing lines
    let first_diff = old_lines.iter().zip(new_lines.iter())
        .position(|(a, b)| a != b);

    let last_diff_old = old_lines.iter().rev().zip(new_lines.iter().rev())
        .position(|(a, b)| a != b);

    match (first_diff, last_diff_old) {
        (Some(first), Some(last_old)) => {
            let last_new = new_lines.iter().rev().zip(old_lines.iter().rev())
                .position(|(a, b)| a != b);

            let ctx = 3; // context lines
            let start = first.saturating_sub(ctx);
            let old_end = (old_lines.len() - last_old).min(old_lines.len());
            let new_end = if let Some(ln) = last_new {
                (new_lines.len() - ln).min(new_lines.len())
            } else {
                new_lines.len()
            };

            // If lengths differ, extend to end
            let old_range_end = if old_lines.len() != new_lines.len() { old_lines.len() } else { old_end.max(new_end) };
            let new_range_end = if old_lines.len() != new_lines.len() { new_lines.len() } else { old_end.max(new_end) };

            diff.push_str(&format!("@@ -{},{} +{},{} @@\n",
                start + 1, old_range_end.saturating_sub(start),
                start + 1, new_range_end.saturating_sub(start)));

            // Old lines (with context)
            for i in start..old_range_end {
                if let Some(line) = old_lines.get(i) {
                    let new_at_same = new_lines.get(i);
                    if new_at_same.is_none_or(|n| n != line) {
                        diff.push_str(&format!("-{}\n", line));
                    } else {
                        diff.push_str(&format!(" {}\n", line));
                    }
                }
            }

            // New lines that were added
            for i in start..new_range_end {
                if let Some(line) = new_lines.get(i) {
                    let old_at_same = old_lines.get(i);
                    if old_at_same.is_none_or(|o| o != line) {
                        diff.push_str(&format!("+{}\n", line));
                    }
                }
            }
        }
        _ => {
            diff.push_str("@@ no changes @@\n");
        }
    }

    diff
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use hermes_cfg::platform::SessionSource;

    fn test_ctx() -> ToolContext {
        ToolContext::new("test-session", SessionSource::cli())
    }

    // ── WebSearchTool tests ──

    #[test]
    fn test_urlencoding() {
        assert_eq!(urlencoding("hello world"), "hello%20world");
        assert_eq!(urlencoding("rust-lang"), "rust-lang");
        assert_eq!(urlencoding("a+b=c"), "a%2Bb%3Dc");
    }

    #[test]
    fn test_strip_html_tags() {
        assert_eq!(strip_html_tags("<b>hello</b>"), "hello");
        assert_eq!(strip_html_tags("no tags"), "no tags");
        assert_eq!(strip_html_tags("<a href='x'>link</a> text"), "link text");
    }

    #[test]
    fn test_parse_ddg_html_empty() {
        let results = parse_ddg_html("<html><body></body></html>", 5);
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_web_search_schema() {
        let tool = WebSearchTool::new();
        assert_eq!(tool.name(), "web_search");
        let schema = tool.parameters_schema();
        assert!(schema["properties"]["query"].is_object());
        assert!(schema["required"].as_array().unwrap().contains(&serde_json::json!("query")));
    }

    // ── WebExtractTool tests ──

    #[tokio::test]
    async fn test_web_extract_too_many_urls() {
        let tool = WebExtractTool::new();
        let urls: Vec<&str> = vec!["http://a.com", "http://b.com", "http://c.com", "http://d.com", "http://e.com", "http://f.com"];
        let args = serde_json::json!({"urls": urls}).to_string();
        let result = tool.execute(&args, &test_ctx()).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("maximum 5 URLs"));
    }

    #[tokio::test]
    async fn test_web_extract_invalid_url() {
        let tool = WebExtractTool::new();
        let args = serde_json::json!({"urls": ["ftp://bad.url"]}).to_string();
        let result = tool.execute(&args, &test_ctx()).await.unwrap();
        assert!(result.content.contains("http://"));
    }

    #[tokio::test]
    async fn test_web_extract_schema() {
        let tool = WebExtractTool::new();
        assert_eq!(tool.name(), "web_extract");
    }

    // ── SessionSearchTool tests ──

    #[tokio::test]
    async fn test_session_search_no_dir() {
        let dir = std::env::temp_dir().join("hermes_test_no_sessions");
        let _ = std::fs::remove_dir_all(&dir);

        let tool = SessionSearchTool::new(&dir);
        let args = serde_json::json!({"mode": "recent"}).to_string();
        let result = tool.execute(&args, &test_ctx()).await.unwrap();
        assert!(result.content.contains("count"));
    }

    #[tokio::test]
    async fn test_session_search_recent() {
        let dir = std::env::temp_dir().join("hermes_test_sessions");
        let sessions_dir = dir.join("sessions");
        std::fs::create_dir_all(&sessions_dir).ok();

        let session = serde_json::json!({
            "created": "2025-01-01T00:00:00Z",
            "messages": [
                {"role": "user", "content": "Hello world test message"},
                {"role": "assistant", "content": "Hi there"}
            ]
        });
        std::fs::write(sessions_dir.join("test-session.json"), serde_json::to_string(&session).unwrap()).unwrap();

        let tool = SessionSearchTool::new(&dir);
        let args = serde_json::json!({"mode": "recent", "limit": 5}).to_string();
        let result = tool.execute(&args, &test_ctx()).await.unwrap();
        assert!(result.content.contains("test-session"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn test_session_search_keyword() {
        let dir = std::env::temp_dir().join("hermes_test_sessions_kw");
        let sessions_dir = dir.join("sessions");
        std::fs::create_dir_all(&sessions_dir).ok();

        let session = serde_json::json!({
            "created": "2025-01-01T00:00:00Z",
            "messages": [
                {"role": "user", "content": "Tell me about Rust programming"},
            ]
        });
        std::fs::write(sessions_dir.join("rust-session.json"), serde_json::to_string(&session).unwrap()).unwrap();

        let tool = SessionSearchTool::new(&dir);
        let args = serde_json::json!({"mode": "search", "query": "Rust"}).to_string();
        let result = tool.execute(&args, &test_ctx()).await.unwrap();
        assert!(result.content.contains("rust-session"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn test_session_search_missing_query() {
        let dir = std::env::temp_dir().join("hermes_test_sessions_err");
        std::fs::create_dir_all(dir.join("sessions")).ok();

        let tool = SessionSearchTool::new(&dir);
        let args = serde_json::json!({"mode": "search"}).to_string();
        let result = tool.execute(&args, &test_ctx()).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("query required"));

        std::fs::remove_dir_all(&dir).ok();
    }

    // ── SkillsListTool tests ──

    #[tokio::test]
    async fn test_skills_list_empty() {
        let dir = std::env::temp_dir().join("hermes_test_skills_empty");
        let _ = std::fs::remove_dir_all(&dir);
        // dir doesn't exist

        let tool = SkillsListTool::new(&dir);
        let args = serde_json::json!({}).to_string();
        let result = tool.execute(&args, &test_ctx()).await.unwrap();
        assert!(result.content.contains("\"count\": 0"));
    }

    #[tokio::test]
    async fn test_skills_list_with_skills() {
        let dir = std::env::temp_dir().join("hermes_test_skills");
        std::fs::create_dir_all(&dir).ok();

        let yaml = "name: test-skill\ndescription: A test skill\ncategory: testing\n";
        std::fs::write(dir.join("test-skill.yaml"), yaml).unwrap();

        let tool = SkillsListTool::new(&dir);
        let args = serde_json::json!({}).to_string();
        let result = tool.execute(&args, &test_ctx()).await.unwrap();
        assert!(result.content.contains("test-skill"));
        assert!(result.content.contains("testing"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn test_skills_list_category_filter() {
        let dir = std::env::temp_dir().join("hermes_test_skills_filter");
        std::fs::create_dir_all(&dir).ok();

        std::fs::write(dir.join("a.yaml"), "name: skill-a\ncategory: coding\n").unwrap();
        std::fs::write(dir.join("b.yaml"), "name: skill-b\ncategory: writing\n").unwrap();

        let tool = SkillsListTool::new(&dir);
        let args = serde_json::json!({"category": "coding"}).to_string();
        let result = tool.execute(&args, &test_ctx()).await.unwrap();
        assert!(result.content.contains("skill-a"));
        assert!(!result.content.contains("skill-b"));

        std::fs::remove_dir_all(&dir).ok();
    }

    // ── SkillViewTool tests ──

    #[tokio::test]
    async fn test_skill_view_found() {
        let dir = std::env::temp_dir().join("hermes_test_skill_view");
        std::fs::create_dir_all(&dir).ok();

        let yaml = "name: my-skill\ndescription: A skill\ntags:\n  - test\nenv_vars:\n  - API_KEY\n";
        std::fs::write(dir.join("my-skill.yaml"), yaml).unwrap();

        let tool = SkillViewTool::new(&dir);
        let args = serde_json::json!({"name": "my-skill"}).to_string();
        let result = tool.execute(&args, &test_ctx()).await.unwrap();
        assert!(result.content.contains("my-skill"));
        assert!(result.content.contains("API_KEY"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn test_skill_view_not_found() {
        let dir = std::env::temp_dir().join("hermes_test_skill_view_nf");
        std::fs::create_dir_all(&dir).ok();

        let tool = SkillViewTool::new(&dir);
        let args = serde_json::json!({"name": "nonexistent"}).to_string();
        let result = tool.execute(&args, &test_ctx()).await.unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("not found"));

        std::fs::remove_dir_all(&dir).ok();
    }

    // ── VisionAnalyzeTool tests ──

    #[test]
    fn test_vision_schema() {
        let tool = VisionAnalyzeTool::new();
        assert_eq!(tool.name(), "vision_analyze");
        let schema = tool.parameters_schema();
        assert!(schema["required"].as_array().unwrap().contains(&serde_json::json!("image_url")));
        assert!(schema["required"].as_array().unwrap().contains(&serde_json::json!("question")));
    }

    #[tokio::test]
    async fn test_vision_no_api_key() {
        let tool = VisionAnalyzeTool {
            api_key: None,
            endpoint: None,
            model: "test".to_string(),
        };
        let args = serde_json::json!({"image_url": "test.png", "question": "what?"}).to_string();
        let result = tool.execute(&args, &test_ctx()).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("No API key"));
    }

    // ── PatchTool tests ──

    #[tokio::test]
    async fn test_patch_replace_single() {
        let dir = std::env::temp_dir().join("hermes_test_patch");
        std::fs::create_dir_all(&dir).ok();
        std::fs::write(dir.join("file.txt"), "hello world\nfoo bar\nhello rust").unwrap();

        let tool = PatchTool::new(&dir);
        let args = serde_json::json!({
            "mode": "replace",
            "path": "file.txt",
            "old_string": "foo bar",
            "new_string": "baz qux"
        }).to_string();
        let result = tool.execute(&args, &test_ctx()).await.unwrap();
        assert!(!result.is_error);

        let patched = std::fs::read_to_string(dir.join("file.txt")).unwrap();
        assert!(patched.contains("baz qux"));
        assert!(!patched.contains("foo bar"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn test_patch_replace_all() {
        let dir = std::env::temp_dir().join("hermes_test_patch_all");
        std::fs::create_dir_all(&dir).ok();
        std::fs::write(dir.join("file.txt"), "aaa\nbbb\naaa\nccc\naaa").unwrap();

        let tool = PatchTool::new(&dir);
        let args = serde_json::json!({
            "mode": "replace",
            "path": "file.txt",
            "old_string": "aaa",
            "new_string": "XXX",
            "replace_all": true
        }).to_string();
        let result = tool.execute(&args, &test_ctx()).await.unwrap();
        assert!(!result.is_error);

        let patched = std::fs::read_to_string(dir.join("file.txt")).unwrap();
        assert_eq!(patched, "XXX\nbbb\nXXX\nccc\nXXX");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn test_patch_old_not_found() {
        let dir = std::env::temp_dir().join("hermes_test_patch_nf");
        std::fs::create_dir_all(&dir).ok();
        std::fs::write(dir.join("file.txt"), "unchanged content").unwrap();

        let tool = PatchTool::new(&dir);
        let args = serde_json::json!({
            "mode": "replace",
            "path": "file.txt",
            "old_string": "does not exist",
            "new_string": "something"
        }).to_string();
        let result = tool.execute(&args, &test_ctx()).await.unwrap();
        assert!(result.is_error);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn test_patch_multiple_without_replace_all() {
        let dir = std::env::temp_dir().join("hermes_test_patch_multi");
        std::fs::create_dir_all(&dir).ok();
        std::fs::write(dir.join("file.txt"), "aaa bbb aaa").unwrap();

        let tool = PatchTool::new(&dir);
        let args = serde_json::json!({
            "mode": "replace",
            "path": "file.txt",
            "old_string": "aaa",
            "new_string": "zzz"
        }).to_string();
        let result = tool.execute(&args, &test_ctx()).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("found 2 times"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn test_patch_path_traversal() {
        let dir = std::env::temp_dir().join("hermes_test_patch_safe");
        std::fs::create_dir_all(&dir).ok();

        let tool = PatchTool::new(&dir);
        let args = serde_json::json!({
            "mode": "replace",
            "path": "../../../etc/passwd",
            "old_string": "x",
            "new_string": "y"
        }).to_string();
        let result = tool.execute(&args, &test_ctx()).await;
        assert!(result.is_err());
    }

    // ── Unified diff tests ──

    #[test]
    fn test_generate_unified_diff_no_change() {
        let diff = generate_unified_diff("test.txt", "hello\nworld", "hello\nworld");
        assert!(diff.contains("no changes"));
    }

    #[test]
    fn test_generate_unified_diff_with_change() {
        let diff = generate_unified_diff("test.txt", "line1\nline2\nline3", "line1\nmodified\nline3");
        assert!(diff.contains("-line2") || diff.contains("+modified"));
    }
}
