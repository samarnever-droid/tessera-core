//! MCR-VM (Phase 18): Opcode Definitions, Encodings, and Gas Tables.

pub const OP_PUSH_INT: u8 = 0x01;
pub const OP_PUSH_STR: u8 = 0x02;
pub const OP_POP: u8      = 0x03;

pub const OP_LOAD_KEY: u8 = 0x10;
pub const OP_STORE_KEY: u8= 0x11;
pub const OP_DELTA_SUM: u8= 0x12;
pub const OP_HLL_ADD: u8  = 0x13;
pub const OP_JSON_GET: u8 = 0x14;
pub const OP_JSON_SET: u8 = 0x15;
pub const OP_ZSET_ADD: u8 = 0x16;

pub const OP_ADD: u8      = 0x20;
pub const OP_SUB: u8      = 0x21;
pub const OP_MUL: u8      = 0x22;
pub const OP_DIV: u8      = 0x23;

pub const OP_CMP_EQ: u8   = 0x30;
pub const OP_CMP_LT: u8   = 0x31;
pub const OP_CMP_GT: u8   = 0x32;

pub const OP_JUMP: u8     = 0x40;
pub const OP_JUMP_IF: u8  = 0x41;

pub const OP_RETURN: u8   = 0xFF;

#[inline]
pub fn opcode_gas_cost(opcode: u8) -> u64 {
    match opcode {
        OP_PUSH_INT | OP_PUSH_STR | OP_POP => 1,
        OP_ADD | OP_SUB | OP_CMP_EQ | OP_CMP_LT | OP_CMP_GT => 1,
        OP_MUL => 2,
        OP_DIV => 4,
        OP_JUMP => 2,
        OP_JUMP_IF => 3,
        OP_DELTA_SUM | OP_HLL_ADD => 8,
        OP_LOAD_KEY => 10,
        OP_JSON_GET | OP_ZSET_ADD => 12,
        OP_STORE_KEY | OP_JSON_SET => 15,
        OP_RETURN => 1,
        _ => 1,
    }
}
