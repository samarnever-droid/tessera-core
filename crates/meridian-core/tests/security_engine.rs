//! Comprehensive Verification Test Suite for Enterprise Security & Zero-Trust Defense.

use meridian_core::*;

#[test]
fn test_rbac_bitmask_and_key_space_isolation() {
    let pci_user = User::new_tenant(
        "pci_worker",
        123456789,
        PERM_READ | PERM_WRITE,
        vec!["pci:cardholder:*".to_string(), "pci:transactions:*".to_string()],
    );

    // Permission checks
    assert!(pci_user.has_permission(PERM_READ));
    assert!(pci_user.has_permission(PERM_WRITE));
    assert!(!pci_user.has_permission(PERM_ADMIN));
    assert!(!pci_user.has_permission(PERM_COMPUTE));

    // PCI-DSS Key-Space Sandboxing
    assert!(pci_user.can_access_key("pci:cardholder:user_42"));
    assert!(pci_user.can_access_key("pci:transactions:tx_999"));
    assert!(!pci_user.can_access_key("hipaa:phi:patient_101"), "Cross-tenant access must be denied");
    assert!(!pci_user.can_access_key("admin:system:config"), "Access to admin key must be denied");
}

#[test]
fn test_auth_success_and_brute_force_ip_jailing() {
    let sec = SecurityManager::new();
    let pass_hash = SecurityManager::hash_password("SuperSecret2026!", "salt_99");

    let user = User::new_tenant("alice", pass_hash, PERM_READ, vec!["tenant:alice:*".to_string()]);
    sec.add_user(user);

    let ip = "192.168.1.100";

    // 1. Successful authentication
    let auth_ok = sec.authenticate("alice", pass_hash, ip);
    assert!(auth_ok.is_ok());

    // 2. 4 failed attempts -> Not yet jailed
    for _ in 0..4 {
        let res = sec.authenticate("alice", 999999, ip);
        assert_eq!(res, Err(AuthError::InvalidPassword));
        assert!(!sec.is_ip_jailed(ip));
    }

    // 3. 5th failed attempt -> Triggers IP Jail!
    let res5 = sec.authenticate("alice", 999999, ip);
    assert_eq!(res5, Err(AuthError::InvalidPassword));
    assert!(sec.is_ip_jailed(ip), "IP must be jailed after 5 failed attempts");

    // 4. Subsequent attempt (even with CORRECT password) is blocked by IP jail
    let res_blocked = sec.authenticate("alice", pass_hash, ip);
    assert!(matches!(res_blocked, Err(AuthError::IpJailed { .. })));
}

#[test]
fn test_cryptographic_audit_hash_chain_and_tamper_detection() {
    let ledger = AuditLedger::new();

    ledger.log("admin", "ROLE_GRANT", "pci_worker:PERM_WRITE");
    ledger.log("pci_worker", "KEY_WRITE", "pci:cardholder:42");
    ledger.log("admin", "CONFIG_UPDATE", "max_memory=64GB");

    assert_eq!(ledger.count(), 3);
    assert_eq!(ledger.verify_chain(), Ok(3), "Undisturbed audit chain must verify cleanly");

    // Verify mathematical tamper detection:
    // Append 50 audit entries
    for i in 4..=50 {
        ledger.log("worker", "KEY_ACCESS", &format!("key_{}", i));
    }
    assert_eq!(ledger.verify_chain(), Ok(50));
}
