//! MCR-VM Execution Engine: Stack-Based, Gas-Metered Deterministic VM.

use crate::delta::DeltaOp;
use crate::compute::opcodes::*;

pub const DEFAULT_GAS_BUDGET: u64 = 10_000;
pub const MAX_STACK_DEPTH: usize = 256;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VmError {
    GasExhausted,
    StackOverflow,
    StackUnderflow,
    DivisionByZero,
    InvalidOpcode(u8),
    UnexpectedEof,
}

#[derive(Debug, Clone)]
pub struct VmResult {
    pub return_value: i64,
    pub gas_consumed: u64,
    pub staged_deltas: Vec<DeltaOp>,
}

pub struct MeridianVM {
    stack: [i64; MAX_STACK_DEPTH],
    sp: usize,
    pc: usize,
    gas_remaining: u64,
    gas_initial: u64,
    staged_deltas: Vec<DeltaOp>,
    string_table: Vec<String>,
}

impl MeridianVM {
    pub fn new(gas_limit: u64) -> Self {
        Self {
            stack: [0; MAX_STACK_DEPTH],
            sp: 0,
            pc: 0,
            gas_remaining: if gas_limit == 0 { DEFAULT_GAS_BUDGET } else { gas_limit },
            gas_initial: if gas_limit == 0 { DEFAULT_GAS_BUDGET } else { gas_limit },
            staged_deltas: Vec::new(),
            string_table: Vec::new(),
        }
    }

    #[inline]
    fn push(&mut self, val: i64) -> Result<(), VmError> {
        if self.sp >= MAX_STACK_DEPTH {
            return Err(VmError::StackOverflow);
        }
        self.stack[self.sp] = val;
        self.sp += 1;
        Ok(())
    }

    #[inline]
    fn pop(&mut self) -> Result<i64, VmError> {
        if self.sp == 0 {
            return Err(VmError::StackUnderflow);
        }
        self.sp -= 1;
        Ok(self.stack[self.sp])
    }

    #[inline]
    fn consume_gas(&mut self, cost: u64) -> Result<(), VmError> {
        if self.gas_remaining < cost {
            self.gas_remaining = 0;
            return Err(VmError::GasExhausted);
        }
        self.gas_remaining -= cost;
        Ok(())
    }

    pub fn add_string(&mut self, s: impl Into<String>) -> usize {
        let idx = self.string_table.len();
        self.string_table.push(s.into());
        idx
    }

    /// Executes bytecode against simulated or provided state store.
    pub fn execute(
        &mut self,
        bytecode: &[u8],
        state_reader: impl Fn(&str) -> Option<i64>,
    ) -> Result<VmResult, VmError> {
        self.pc = 0;
        self.sp = 0;
        self.staged_deltas.clear();

        while self.pc < bytecode.len() {
            let opcode = bytecode[self.pc];
            self.pc += 1;

            self.consume_gas(opcode_gas_cost(opcode))?;

            match opcode {
                OP_PUSH_INT => {
                    if self.pc + 8 > bytecode.len() {
                        return Err(VmError::UnexpectedEof);
                    }
                    let val = i64::from_le_bytes(bytecode[self.pc..self.pc + 8].try_into().unwrap());
                    self.pc += 8;
                    self.push(val)?;
                }
                OP_PUSH_STR => {
                    if self.pc + 4 > bytecode.len() {
                        return Err(VmError::UnexpectedEof);
                    }
                    let str_idx = u32::from_le_bytes(bytecode[self.pc..self.pc + 4].try_into().unwrap()) as i64;
                    self.pc += 4;
                    self.push(str_idx)?;
                }
                OP_POP => {
                    let _ = self.pop()?;
                }
                OP_LOAD_KEY => {
                    let str_idx = self.pop()? as usize;
                    let key = self.string_table.get(str_idx).map(|s| s.as_str()).unwrap_or("");
                    let val = state_reader(key).unwrap_or(0);
                    self.push(val)?;
                }
                OP_STORE_KEY => {
                    let val = self.pop()?;
                    let str_idx = self.pop()? as usize;
                    let _key = self.string_table.get(str_idx).cloned().unwrap_or_default();
                    self.staged_deltas.push(DeltaOp::Sum { delta: val });
                }
                OP_DELTA_SUM => {
                    let delta_val = self.pop()?;
                    let str_idx = self.pop()? as usize;
                    let _key = self.string_table.get(str_idx).cloned().unwrap_or_default();
                    self.staged_deltas.push(DeltaOp::Sum { delta: delta_val });
                }
                OP_HLL_ADD => {
                    let elem_hash = self.pop()? as u64;
                    let _str_idx = self.pop()? as usize;
                    self.staged_deltas.push(DeltaOp::HllAdd { element_hash: elem_hash });
                }
                OP_ADD => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.push(a.wrapping_add(b))?;
                }
                OP_SUB => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.push(a.wrapping_sub(b))?;
                }
                OP_MUL => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.push(a.wrapping_mul(b))?;
                }
                OP_DIV => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    if b == 0 {
                        return Err(VmError::DivisionByZero);
                    }
                    self.push(a / b)?;
                }
                OP_CMP_EQ => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.push(if a == b { 1 } else { 0 })?;
                }
                OP_CMP_LT => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.push(if a < b { 1 } else { 0 })?;
                }
                OP_CMP_GT => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.push(if a > b { 1 } else { 0 })?;
                }
                OP_JUMP => {
                    if self.pc + 4 > bytecode.len() {
                        return Err(VmError::UnexpectedEof);
                    }
                    let target = u32::from_le_bytes(bytecode[self.pc..self.pc + 4].try_into().unwrap()) as usize;
                    self.pc = target;
                }
                OP_JUMP_IF => {
                    if self.pc + 4 > bytecode.len() {
                        return Err(VmError::UnexpectedEof);
                    }
                    let target = u32::from_le_bytes(bytecode[self.pc..self.pc + 4].try_into().unwrap()) as usize;
                    self.pc += 4;
                    let cond = self.pop()?;
                    if cond != 0 {
                        self.pc = target;
                    }
                }
                OP_RETURN => {
                    let ret_val = if self.sp > 0 { self.pop()? } else { 0 };
                    return Ok(VmResult {
                        return_value: ret_val,
                        gas_consumed: self.gas_initial - self.gas_remaining,
                        staged_deltas: self.staged_deltas.clone(),
                    });
                }
                _ => return Err(VmError::InvalidOpcode(opcode)),
            }
        }

        Ok(VmResult {
            return_value: if self.sp > 0 { self.pop()? } else { 0 },
            gas_consumed: self.gas_initial - self.gas_remaining,
            staged_deltas: self.staged_deltas.clone(),
        })
    }
}
