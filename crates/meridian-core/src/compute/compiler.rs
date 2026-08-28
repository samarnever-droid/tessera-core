//! Small L++ (MCR Mini-Script) Single-Pass Bytecode Compiler.

use crate::compute::opcodes::*;

pub struct Compiler {
    bytecode: Vec<u8>,
    strings: Vec<String>,
}

impl Compiler {
    pub fn new() -> Self {
        Self {
            bytecode: Vec::new(),
            strings: Vec::new(),
        }
    }

    pub fn push_int(&mut self, val: i64) {
        self.bytecode.push(OP_PUSH_INT);
        self.bytecode.extend_from_slice(&val.to_le_bytes());
    }

    pub fn push_str(&mut self, s: &str) {
        let idx = self.strings.len() as u32;
        self.strings.push(s.to_string());
        self.bytecode.push(OP_PUSH_STR);
        self.bytecode.extend_from_slice(&idx.to_le_bytes());
    }

    pub fn load_key(&mut self, key: &str) {
        self.push_str(key);
        self.bytecode.push(OP_LOAD_KEY);
    }

    pub fn delta_sum(&mut self, key: &str, delta: i64) {
        self.push_str(key);
        self.push_int(delta);
        self.bytecode.push(OP_DELTA_SUM);
    }

    pub fn op(&mut self, opcode: u8) {
        self.bytecode.push(opcode);
    }

    pub fn finish(mut self) -> (Vec<u8>, Vec<String>) {
        if self.bytecode.is_empty() || *self.bytecode.last().unwrap() != OP_RETURN {
            self.bytecode.push(OP_RETURN);
        }
        (self.bytecode, self.strings)
    }
}
