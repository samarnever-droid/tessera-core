use meridian_core::*;

#[test]
fn test_phase5_oracle_dependency_invalidation() {
    let oracle = OracleIndex::new();

    // Register user profile cache
    oracle.register_deps(b"user:42:profile".to_vec(), vec![
        Dep::row("users", 42),
        Dep::table("users"),
    ]);

    // Register order list cache
    oracle.register_deps(b"user:42:orders".to_vec(), vec![
        Dep::row("users", 42),
        Dep::table("orders"),
    ]);

    // Register global settings
    oracle.register_deps(b"config:global".to_vec(), vec![
        Dep::table("config"),
    ]);

    assert_eq!(oracle.total_tracked_keys(), 3);

    // Invalidate user 42 row -> should invalidate both user:42 keys, but NOT config:global
    let invalidated = oracle.invalidate_by_dep(&Dep::row("users", 42));
    assert_eq!(invalidated.len(), 2);
    assert!(invalidated.contains(&b"user:42:profile".to_vec()));
    assert!(invalidated.contains(&b"user:42:orders".to_vec()));
    assert!(!invalidated.contains(&b"config:global".to_vec()));

    assert_eq!(oracle.total_tracked_keys(), 1);
}

#[test]
fn test_phase6_cdc_watermark_and_token_bucket() {
    let wm = WatermarkTracker::new();

    assert!(wm.advance_lsn("postgres_main", 100, 1000));
    assert!(wm.advance_lsn("postgres_main", 200, 2000));
    // Replay must be rejected
    assert!(!wm.advance_lsn("postgres_main", 150, 2500));
    assert_eq!(wm.get_watermark("postgres_main"), 200);

    let bucket = OriginTokenBucket::new(10, 5.0);
    assert!(bucket.try_acquire(5, 1000));
    assert!(bucket.try_acquire(5, 1000));
    assert!(!bucket.try_acquire(1, 1000)); // Exhausted
}

#[test]
fn test_phase7_delta_in_place_maintenance() {
    // 1. In-place SUM
    let init_sum = 1000i64.to_le_bytes().to_vec();
    let updated_sum = apply_delta(&init_sum, &DeltaOp::Sum { delta: 250 });
    let val = i64::from_le_bytes(updated_sum[0..8].try_into().unwrap());
    assert_eq!(val, 1250);

    // 2. In-place COUNT
    let init_cnt = 5i64.to_le_bytes().to_vec();
    let updated_cnt = apply_delta(&init_cnt, &DeltaOp::Count { delta: 3 });
    let cnt = i64::from_le_bytes(updated_cnt[0..8].try_into().unwrap());
    assert_eq!(cnt, 8);

    // 3. In-place GROUP BY
    let init_gb = b"dept_1=100;dept_2=50";
    let updated_gb = apply_delta(init_gb, &DeltaOp::GroupBy { group: "dept_1".to_string(), delta: 25 });
    let str_gb = String::from_utf8(updated_gb).unwrap();
    assert!(str_gb.contains("dept_1=125"));
    assert!(str_gb.contains("dept_2=50"));

    // 4. Auditor
    let auditor = DifferentialAuditor::new(100); // 10%
    auditor.record_audit_result(true);
    assert_eq!(auditor.total_mismatches(), 0);
}

#[test]
fn test_phase8_chronos_snapshot_isolation() {
    let chronos = ChronosStore::new();

    chronos.append_version(b"account:101", 100, b"balance:1000".to_vec());
    chronos.append_version(b"account:101", 200, b"balance:1500".to_vec());
    chronos.append_version(b"account:101", 300, b"balance:2000".to_vec());

    // Snapshot pinned at LSN 150 -> reads version 100
    let snap150 = chronos.open_snapshot(150);
    let val150 = chronos.read_snapshot(snap150, b"account:101").unwrap();
    assert_eq!(val150, b"balance:1000");

    // Snapshot pinned at LSN 250 -> reads version 200
    let snap250 = chronos.open_snapshot(250);
    let val250 = chronos.read_snapshot(snap250, b"account:101").unwrap();
    assert_eq!(val250, b"balance:1500");

    // Snapshot pinned at LSN 350 -> reads version 300
    let snap350 = chronos.open_snapshot(350);
    let val350 = chronos.read_snapshot(snap350, b"account:101").unwrap();
    assert_eq!(val350, b"balance:2000");

    chronos.close_snapshot(snap150);
    chronos.close_snapshot(snap250);
    chronos.close_snapshot(snap350);
}

#[test]
fn test_phase9_price_system_dual_ascent() {
    let engine = DualAscentEngine::new(0.5);
    // Under heavy DRAM usage (ratio = 1.5), price increases
    engine.step_update(1.5, 0.8, 0.5, 0.2);
    let prices = engine.prices.read().unwrap();
    assert!(prices.lambda_dram > 1.0);
}

#[test]
fn test_phase10_spectrum_fidelity_types() {
    let exact_approx = Approx::new("user_full_data".to_string(), FidelityLevel::Exact, 0.0);
    assert_eq!(exact_approx.unwrap_exact(), Some("user_full_data".to_string()));

    let proj_approx = Approx::new("user_summary".to_string(), FidelityLevel::Projected, 0.05);
    assert_eq!(proj_approx.unwrap_exact(), None);
}

#[test]
fn test_phase11_deadline_scheduler() {
    let sched = DeadlineScheduler::new(500);
    // 200us elapsed out of 500us budget, tier needs 100us -> ServeExact
    let action1 = sched.evaluate_tier(200, 500, 100);
    assert_eq!(action1, DegradeAction::ServeExact);

    // 480us elapsed out of 500us budget, tier needs 100us -> CancelWithDeadlineExceeded
    let action2 = sched.evaluate_tier(480, 500, 100);
    assert_eq!(action2, DegradeAction::CancelWithDeadlineExceeded);
}

#[test]
fn test_phase12_flash_and_mesh() {
    let flash = FlashTier::new(1.2);
    flash.put(b"item:99".to_vec(), b"ssd_payload".to_vec());
    assert_eq!(flash.get(b"item:99"), Some(b"ssd_payload".to_vec()));

    let mesh = MeshCluster::new("127.0.0.1:7717");
    mesh.add_node("127.0.0.1:7718");
    let target_node = mesh.locate_node(b"key:abc");
    assert!(target_node.starts_with("127.0.0.1:771"));
}

#[test]
fn test_in_engine_cdc_to_delta_pipeline() {
    let engine = Engine::new(EngineOptions {
        total_entries: 1024,
        ..Default::default()
    });

    // 1. Store initial balance in engine
    engine.set(b"user:42:balance", &1000i64.to_le_bytes());

    // 2. Register maintenance plan: when user 42 updates, apply SUM
    engine.register_maintenance_plan(
        b"user:42:balance",
        DeltaOp::Sum { delta: 250 },
        vec![Dep::row("users", 42)],
    );

    // 3. Ingest CDC Update Record from DB WAL
    let record = CdcRecord {
        lsn: 1001,
        source: "samardb_main".to_string(),
        table: "users".to_string(),
        key_id: 42,
        op: CdcOp::Update,
        before_bytes: None,
        after_bytes: None,
        timestamp_ms: 1000,
    };

    let affected = engine.ingest_cdc(&record);
    assert_eq!(affected, 1);

    // 4. Verify in-place updated balance directly in Engine
    let cur_bytes = engine.get(b"user:42:balance").unwrap();
    let balance = i64::from_le_bytes(cur_bytes[0..8].try_into().unwrap());
    assert_eq!(balance, 1250); // 1000 + 250 = 1250! Zero origin queries!
}
