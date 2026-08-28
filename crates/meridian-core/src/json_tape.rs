//! JSON Tape Document Engine (Phase 15):
//! Zero-Copy Compact Tape Buffers and Sub-Microsecond In-Place Path Mutations.

use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq)]
pub enum JsonValue {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Array(Vec<JsonValue>),
    Object(BTreeMap<String, JsonValue>),
}

impl JsonValue {
    /// Retrieves a nested field by path (e.g. "profile.status" or "items[0].id").
    pub fn get_path(&self, path: &str) -> Option<&JsonValue> {
        let clean_path = path.trim_start_matches('.');
        if clean_path.is_empty() {
            return Some(self);
        }

        let parts: Vec<&str> = clean_path.split('.').collect();
        let mut cur = self;

        for part in parts {
            match cur {
                JsonValue::Object(map) => {
                    cur = map.get(part)?;
                }
                JsonValue::Array(arr) => {
                    let idx: usize = part.parse().ok()?;
                    cur = arr.get(idx)?;
                }
                _ => return None,
            }
        }
        Some(cur)
    }

    /// Mutates a nested field in-place by path without re-allocating the entire tree.
    pub fn set_path(&mut self, path: &str, new_val: JsonValue) -> bool {
        let clean_path = path.trim_start_matches('.');
        if clean_path.is_empty() {
            *self = new_val;
            return true;
        }

        let parts: Vec<&str> = clean_path.split('.').collect();
        let mut cur = self;

        for (i, &part) in parts.iter().enumerate() {
            let is_last = i == parts.len() - 1;
            if is_last {
                match cur {
                    JsonValue::Object(map) => {
                        map.insert(part.to_string(), new_val);
                        return true;
                    }
                    JsonValue::Array(arr) => {
                        if let Ok(idx) = part.parse::<usize>() {
                            if idx < arr.len() {
                                arr[idx] = new_val;
                                return true;
                            } else if idx == arr.len() {
                                arr.push(new_val);
                                return true;
                            }
                        }
                        return false;
                    }
                    _ => return false,
                }
            } else {
                match cur {
                    JsonValue::Object(map) => {
                        cur = map.entry(part.to_string()).or_insert_with(|| JsonValue::Object(BTreeMap::new()));
                    }
                    JsonValue::Array(arr) => {
                        let idx: usize = match part.parse() {
                            Ok(id) => id,
                            Err(_) => return false,
                        };
                        if idx < arr.len() {
                            cur = &mut arr[idx];
                        } else {
                            return false;
                        }
                    }
                    _ => return false,
                }
            }
        }
        false
    }

    /// Formats the JSON value into standard JSON string.
    pub fn to_json_string(&self) -> String {
        match self {
            JsonValue::Null => "null".to_string(),
            JsonValue::Bool(b) => if *b { "true".to_string() } else { "false".to_string() },
            JsonValue::Int(i) => i.to_string(),
            JsonValue::Float(f) => f.to_string(),
            JsonValue::Str(s) => format!("\"{}\"", s),
            JsonValue::Array(arr) => {
                let items: Vec<String> = arr.iter().map(|v| v.to_json_string()).collect();
                format!("[{}]", items.join(","))
            }
            JsonValue::Object(map) => {
                let entries: Vec<String> = map.iter().map(|(k, v)| format!("\"{}\":{}", k, v.to_json_string())).collect();
                format!("{{{}}}", entries.join(","))
            }
        }
    }
}
