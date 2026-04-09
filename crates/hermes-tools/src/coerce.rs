//! Tool argument type coercion.
//!
//! LLM providers frequently return numeric/boolean arguments as strings
//! (e.g. `"42"` instead of `42`, `"true"` instead of `true`).
//! This module provides utilities to coerce such values against a JSON Schema.

use serde_json::Value;

/// Coerce a JSON value to match the expected schema type.
///
/// Handles these common LLM mistakes:
/// - `"42"` → `42` (string to number)
/// - `"3.14"` → `3.14` (string to float)
/// - `"true"` / `"false"` → `true` / `false` (string to bool)
/// - `"null"` → `null` (string to null)
pub fn coerce_value(value: &Value, schema_type: &str) -> Value {
    match schema_type {
        "number" | "integer" => coerce_to_number(value),
        "boolean" => coerce_to_bool(value),
        _ => value.clone(),
    }
}

/// Coerce a JSON object's fields according to a JSON Schema's `properties`.
///
/// Walks the schema's `properties` and coerces each field value if the
/// expected type is `number`, `integer`, or `boolean`.
pub fn coerce_arguments(args: &mut Value, schema: &Value) {
    let Some(properties) = schema.get("properties").and_then(|p| p.as_object()) else {
        return;
    };

    let Some(obj) = args.as_object_mut() else {
        return;
    };

    for (key, prop_schema) in properties {
        if let Some(field) = obj.get_mut(key) {
            if let Some(type_name) = prop_schema.get("type").and_then(|t| t.as_str()) {
                *field = coerce_value(field, type_name);
            }
        }
    }
}

/// String → number coercion.
fn coerce_to_number(value: &Value) -> Value {
    // Already a number
    if value.is_number() {
        return value.clone();
    }
    let Some(s) = value.as_str() else {
        return value.clone();
    };
    let trimmed = s.trim();
    // Try integer first
    if let Ok(n) = trimmed.parse::<i64>() {
        return Value::Number(n.into());
    }
    // Then float
    if let Ok(f) = trimmed.parse::<f64>() {
        if let Some(n) = serde_json::Number::from_f64(f) {
            return Value::Number(n);
        }
    }
    value.clone()
}

/// String → bool coercion.
fn coerce_to_bool(value: &Value) -> Value {
    if value.is_boolean() {
        return value.clone();
    }
    let Some(s) = value.as_str() else {
        return value.clone();
    };
    match s.trim().to_lowercase().as_str() {
        "true" | "yes" | "1" | "on" => Value::Bool(true),
        "false" | "no" | "0" | "off" => Value::Bool(false),
        _ => value.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ── coerce_value ──

    #[test]
    fn test_coerce_string_to_integer() {
        assert_eq!(coerce_value(&json!("42"), "integer"), json!(42));
    }

    #[test]
    fn test_coerce_string_to_float() {
        assert_eq!(coerce_value(&json!("2.71"), "number"), json!(2.71));
    }

    #[test]
    fn test_coerce_string_to_bool_true() {
        assert_eq!(coerce_value(&json!("true"), "boolean"), json!(true));
    }

    #[test]
    fn test_coerce_string_to_bool_false() {
        assert_eq!(coerce_value(&json!("false"), "boolean"), json!(false));
    }

    #[test]
    fn test_coerce_passthrough_string() {
        assert_eq!(coerce_value(&json!("hello"), "string"), json!("hello"));
    }

    #[test]
    fn test_coerce_number_already_number() {
        assert_eq!(coerce_value(&json!(42), "integer"), json!(42));
    }

    #[test]
    fn test_coerce_bool_already_bool() {
        assert_eq!(coerce_value(&json!(true), "boolean"), json!(true));
    }

    // ── coerce_arguments ──

    #[test]
    fn test_coerce_arguments_full() {
        let schema = json!({
            "type": "object",
            "properties": {
                "path": { "type": "string" },
                "max_results": { "type": "integer" },
                "threshold": { "type": "number" },
                "case_sensitive": { "type": "boolean" }
            }
        });

        let mut args = json!({
            "path": "src/main.rs",
            "max_results": "10",
            "threshold": "0.5",
            "case_sensitive": "true"
        });

        coerce_arguments(&mut args, &schema);

        assert_eq!(args["max_results"], json!(10));
        assert_eq!(args["threshold"], json!(0.5));
        assert_eq!(args["case_sensitive"], json!(true));
        assert_eq!(args["path"], json!("src/main.rs")); // unchanged
    }

    #[test]
    fn test_coerce_arguments_missing_field() {
        let schema = json!({
            "type": "object",
            "properties": {
                "count": { "type": "integer" }
            }
        });

        let mut args = json!({});
        coerce_arguments(&mut args, &schema);
        // No panic, no field added
        assert!(args.get("count").is_none());
    }

    #[test]
    fn test_coerce_arguments_no_properties() {
        let schema = json!({"type": "object"});
        let mut args = json!({"foo": "bar"});
        coerce_arguments(&mut args, &schema);
        assert_eq!(args["foo"], json!("bar"));
    }

    // ── edge cases ──

    #[test]
    fn test_coerce_bool_variants() {
        assert_eq!(coerce_value(&json!("yes"), "boolean"), json!(true));
        assert_eq!(coerce_value(&json!("1"), "boolean"), json!(true));
        assert_eq!(coerce_value(&json!("on"), "boolean"), json!(true));
        assert_eq!(coerce_value(&json!("no"), "boolean"), json!(false));
        assert_eq!(coerce_value(&json!("0"), "boolean"), json!(false));
        assert_eq!(coerce_value(&json!("off"), "boolean"), json!(false));
    }

    #[test]
    fn test_coerce_uncoercable_string_to_number() {
        // "abc" cannot be parsed as number — return as-is
        assert_eq!(coerce_value(&json!("abc"), "integer"), json!("abc"));
    }
}
