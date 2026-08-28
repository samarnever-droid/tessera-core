//! Enterprise RBAC & Multi-Tenant Key-Space Sandboxing (SOC2, HIPAA, PCI-DSS).

pub const PERM_READ: u64    = 1 << 0;
pub const PERM_WRITE: u64   = 1 << 1;
pub const PERM_ADMIN: u64   = 1 << 2;
pub const PERM_STREAM: u64  = 1 << 3;
pub const PERM_COMPUTE: u64 = 1 << 4;
pub const PERM_AUDIT: u64   = 1 << 5;
pub const PERM_ALL: u64     = 0xFFFFFFFFFFFFFFFF;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct User {
    pub username: String,
    pub password_hash: u64, // Salted 64-bit hash
    pub permissions: u64,   // Bitmask of PERM_*
    pub allowed_key_prefixes: Vec<String>, // Multi-tenant key-space sandboxes
}

impl User {
    pub fn new_admin(username: impl Into<String>, password_hash: u64) -> Self {
        Self {
            username: username.into(),
            password_hash,
            permissions: PERM_ALL,
            allowed_key_prefixes: vec!["*".to_string()], // Unrestricted
        }
    }

    pub fn new_tenant(
        username: impl Into<String>,
        password_hash: u64,
        permissions: u64,
        allowed_prefixes: Vec<String>,
    ) -> Self {
        Self {
            username: username.into(),
            password_hash,
            permissions,
            allowed_key_prefixes: allowed_prefixes,
        }
    }

    /// O(1) bitmask permission check.
    pub fn has_permission(&self, perm: u64) -> bool {
        (self.permissions & perm) == perm
    }

    /// Enforces key-space isolation (PCI-DSS & HIPAA compliance boundary).
    pub fn can_access_key(&self, key: &str) -> bool {
        for prefix in &self.allowed_key_prefixes {
            if prefix == "*" {
                return true;
            }
            if prefix.ends_with('*') {
                let p = &prefix[..prefix.len() - 1];
                if key.starts_with(p) {
                    return true;
                }
            } else if key == prefix || key.starts_with(prefix) {
                return true;
            }
        }
        false
    }
}
