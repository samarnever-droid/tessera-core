//! Comprehensive Verification Test Suite for Dynamic Resharding and Multi-Region WAN Mesh.

use meridian_core::*;

#[test]
fn test_crc16_hash_slot_mapping_and_routing() {
    let table = SlotTable::new(1); // Default owner = Node 1

    // Assign Slots 0..5000 to Node 1, 5001..10000 to Node 2, 10001..16383 to Node 3
    table.assign_range(0, 5000, 1);
    table.assign_range(5001, 10000, 2);
    table.assign_range(10001, 16383, 3);

    let key_alpha = b"user:100:profile";
    let slot_alpha = get_slot(key_alpha);
    assert!(slot_alpha < 16384);

    let (node, is_migrating) = table.route_key(key_alpha);
    assert!(node >= 1 && node <= 3);
    assert!(!is_migrating);
}

#[test]
fn test_zero_downtime_3_phase_slot_migration() {
    let table = SlotTable::new(100); // Node 100
    let controller = MigrationController::new();
    let slot = 42;

    // 1. Initiate Migration of Slot 42 from Node 100 -> Node 200
    controller.start_migration(slot, 100, 200, &table);

    let state1 = table.get_slot_state(slot);
    assert_eq!(state1, SlotState::Migrating { source: 100, target: 200 });

    // 2. Advance through Bulk Snapshot
    controller.advance_to_delta_catchup(slot, 500); // 500 bulk keys transferred

    // 3. Live mutations occurring during migration -> Forward deltas
    controller.forward_delta(slot);
    controller.forward_delta(slot);

    let task = controller.get_task(slot).unwrap();
    assert_eq!(task.phase, MigrationPhase::DeltaCatchup);
    assert_eq!(task.keys_transferred, 500);
    assert_eq!(task.deltas_forwarded, 2);

    // 4. Atomic Cutover
    assert!(controller.commit_cutover(slot, &table));
    let state_final = table.get_slot_state(slot);
    assert_eq!(state_final, SlotState::Stable { owner: 200 }); // Now stably owned by Node 200!
}

#[test]
fn test_multi_region_active_active_wan_conflict_resolution() {
    let us_mesh = WanMesh::new(1); // Region 1 (US)
    let eu_mesh = WanMesh::new(2); // Region 2 (EU)

    // US writes "title" = "Architect"
    let c1 = us_mesh.apply_local_write(b"job:title", b"Architect");
    assert_eq!(c1.region_id, 1);

    // Drain US outbound queue and replicate to EU
    let deltas = us_mesh.drain_outbound();
    assert_eq!(deltas.len(), 1);

    assert!(eu_mesh.receive_wan_delta(deltas[0].clone()));
    assert_eq!(eu_mesh.get_value(b"job:title"), Some(b"Architect".to_vec()));

    // EU performs concurrent causal update with higher timestamp
    std::thread::sleep(std::time::Duration::from_millis(2));
    let _c2 = eu_mesh.apply_local_write(b"job:title", b"Distinguished Engineer");

    let eu_deltas = eu_mesh.drain_outbound();
    assert_eq!(eu_deltas.len(), 1);

    // Replicate EU update back to US -> LWW-CRDT adopts the higher causal update!
    assert!(us_mesh.receive_wan_delta(eu_deltas[0].clone()));
    assert_eq!(us_mesh.get_value(b"job:title"), Some(b"Distinguished Engineer".to_vec()));
}
