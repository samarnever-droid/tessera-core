//! Stored Procedure Function Catalog.

use std::collections::HashMap;
use std::sync::RwLock;

#[derive(Debug, Clone)]
pub struct StoredFunction {
    pub name: String,
    pub bytecode: Vec<u8>,
    pub string_table: Vec<String>,
    pub default_gas: u64,
}

pub struct FunctionCatalog {
    functions: RwLock<HashMap<String, StoredFunction>>,
}

impl FunctionCatalog {
    pub fn new() -> Self {
        Self {
            functions: RwLock::new(HashMap::new()),
        }
    }

    pub fn register(&self, func: StoredFunction) {
        self.functions.write().unwrap().insert(func.name.clone(), func);
    }

    pub fn get(&self, name: &str) -> Option<StoredFunction> {
        self.functions.read().unwrap().get(name).cloned()
    }

    pub fn count(&self) -> usize {
        self.functions.read().unwrap().len()
    }
}
