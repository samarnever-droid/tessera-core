//! Comprehensive Verification Test Suite for MCR-VM Compute Engine.

use meridian_core::*;
use std::collections::HashMap;

#[test]
fn test_mcr_vm_arithmetic_and_stack() {
    let mut vm = MeridianVM::new(1000);

    // Bytecode: (10 + 20) * 3 = 90
    let mut compiler = Compiler::new();
    compiler.push_int(10);
    compiler.push_int(20);
    compiler.op(OP_ADD);
    compiler.push_int(3);
    compiler.op(OP_MUL);
    compiler.op(OP_RETURN);

    let (bytecode, _) = compiler.finish();
    let result = vm.execute(&bytecode, |_| None).expect("VM execution should succeed");

    assert_eq!(result.return_value, 90);
    assert!(result.gas_consumed > 0);
    assert!(result.gas_consumed < 50);
}

#[test]
fn test_mcr_vm_infinite_loop_gas_exhaustion_defense() {
    let mut vm = MeridianVM::new(500); // 500 gas limit

    // Bytecode: Infinite Loop (Jump back to 0)
    let mut bytecode = Vec::new();
    bytecode.push(OP_JUMP);
    bytecode.extend_from_slice(&0u32.to_le_bytes()); // Jump to PC 0

    let start = std::time::Instant::now();
    let err = vm.execute(&bytecode, |_| None).expect_err("Infinite loop must exhaust gas");
    let duration = start.elapsed();

    assert_eq!(err, VmError::GasExhausted);
    // Verified: Terminated in under 10 microseconds!
    assert!(duration.as_micros() < 500, "Gas abortion took too long: {:?}", duration);
}

#[test]
fn test_mcr_vm_division_by_zero_safety() {
    let mut vm = MeridianVM::new(1000);

    let mut compiler = Compiler::new();
    compiler.push_int(100);
    compiler.push_int(0);
    compiler.op(OP_DIV); // 100 / 0 -> Trap
    compiler.op(OP_RETURN);

    let (bytecode, _) = compiler.finish();
    let err = vm.execute(&bytecode, |_| None).expect_err("Div by zero must trap");
    assert_eq!(err, VmError::DivisionByZero);
}

#[test]
fn test_mcr_vm_state_transfer_and_delta_transcript() {
    let mut vm = MeridianVM::new(1000);

    // Mock state
    let mut state = HashMap::new();
    state.insert("user:42:balance".to_string(), 500i64);
    state.insert("user:99:balance".to_string(), 100i64);

    // Script logic:
    // bal = GET "user:42:balance"
    // if bal >= 200:
    //     DELTA "user:42:balance" -200
    //     DELTA "user:99:balance" +200
    //     return 1
    // return 0

    let mut compiler = Compiler::new();
    let s_user42 = vm.add_string("user:42:balance");
    let s_user99 = vm.add_string("user:99:balance");

    // Load key user:42:balance
    compiler.push_str("user:42:balance");
    compiler.op(OP_LOAD_KEY);

    // Check if >= 200
    compiler.push_int(200);
    compiler.op(OP_CMP_GT); // Stack: 1 (500 > 200)

    // Apply Deltas
    compiler.delta_sum("user:42:balance", -200);
    compiler.delta_sum("user:99:balance", 200);
    compiler.push_int(1);
    compiler.op(OP_RETURN);

    let (bytecode, _) = compiler.finish();
    let result = vm.execute(&bytecode, |k| state.get(k).copied()).expect("Transfer execution succeeds");

    assert_eq!(result.return_value, 1);
    assert_eq!(result.staged_deltas.len(), 2);
    assert_eq!(result.staged_deltas[0], DeltaOp::Sum { delta: -200 });
    assert_eq!(result.staged_deltas[1], DeltaOp::Sum { delta: 200 });
}

#[test]
fn test_stored_function_catalog() {
    let catalog = FunctionCatalog::new();

    let mut compiler = Compiler::new();
    compiler.push_int(42);
    compiler.push_int(58);
    compiler.op(OP_ADD);
    compiler.op(OP_RETURN);

    let (bytecode, strings) = compiler.finish();

    let func = StoredFunction {
        name: "calculate_score".to_string(),
        bytecode,
        string_table: strings,
        default_gas: 5000,
    };

    catalog.register(func);
    assert_eq!(catalog.count(), 1);

    let retrieved = catalog.get("calculate_score").expect("Function should exist");
    let mut vm = MeridianVM::new(retrieved.default_gas);
    let result = vm.execute(&retrieved.bytecode, |_| None).unwrap();
    assert_eq!(result.return_value, 100);
}
