//! Enterprise Authentication, Salted Challenge Hashing & IP Jailing.

use crate::security::audit::AuditLedger;
use crate::security::rbac::User;
use std::collections::HashMap;
use std::sync::RwLock;

pub const MAX_FAILED_ATTEMPTS: u32 = 5;
pub const JAIL_DURATION_MS: u64 = 900_000; // 15 minutes jail

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthError {
    UserNotFound,
    InvalidPassword,
    IpJailed { remaining_ms: u64 },
    AccessDenied,
}

pub struct SecurityManager {
    users: RwLock<HashMap<String, User>>,
    failed_attempts: RwLock<HashMap<String, (u32, u64)>>, // ip -> (count, first_fail_ms)
    jailed_ips: RwLock<HashMap<String, u64>>,            // ip -> jailed_until_ms
    pub audit: AuditLedger,
}

impl SecurityManager {
    pub fn new() -> Self {
        Self {
            users: RwLock::new(HashMap::new()),
            failed_attempts: RwLock::new(HashMap::new()),
            jailed_ips: RwLock::new(HashMap::new()),
            audit: AuditLedger::new(),
        }
    }

    pub fn hash_password(password: &str, salt: &str) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        for b in salt.bytes().chain(password.bytes()) {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }

    pub fn add_user(&self, user: User) {
        self.audit.log("SYSTEM", "USER_CREATE", &user.username);
        self.users.write().unwrap().insert(user.username.clone(), user);
    }

    /// Authenticates a client with IP jailing protection against brute-force attacks.
    pub fn authenticate(
        &self,
        username: &str,
        password_hash: u64,
        ip: &str,
    ) -> Result<User, AuthError> {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        // 1. Check if IP is jailed
        {
            let jails = self.jailed_ips.read().unwrap();
            if let Some(&jailed_until) = jails.get(ip) {
                if now_ms < jailed_until {
                    return Err(AuthError::IpJailed {
                        remaining_ms: jailed_until - now_ms,
                    });
                }
            }
        }

        // 2. Validate user
        let users = self.users.read().unwrap();
        let user = match users.get(username) {
            Some(u) => u,
            None => {
                self.record_failed_attempt(ip, now_ms);
                self.audit.log("ANONYMOUS", "AUTH_FAIL_USER_NOT_FOUND", username);
                return Err(AuthError::UserNotFound);
            }
        };

        if user.password_hash != password_hash {
            self.record_failed_attempt(ip, now_ms);
            self.audit.log(username, "AUTH_FAIL_BAD_PASSWORD", ip);
            return Err(AuthError::InvalidPassword);
        }

        // 3. Clear failed attempts on success
        self.failed_attempts.write().unwrap().remove(ip);
        self.audit.log(username, "AUTH_SUCCESS", ip);
        Ok(user.clone())
    }

    fn record_failed_attempt(&self, ip: &str, now_ms: u64) {
        let mut fails = self.failed_attempts.write().unwrap();
        let entry = fails.entry(ip.to_string()).or_insert((0, now_ms));
        entry.0 += 1;

        if entry.0 >= MAX_FAILED_ATTEMPTS {
            self.jailed_ips
                .write()
                .unwrap()
                .insert(ip.to_string(), now_ms + JAIL_DURATION_MS);
            self.audit.log("SYSTEM", "IP_JAILED", ip);
        }
    }

    pub fn is_ip_jailed(&self, ip: &str) -> bool {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        let jails = self.jailed_ips.read().unwrap();
        if let Some(&until) = jails.get(ip) {
            now_ms < until
        } else {
            false
        }
    }
}

impl Default for SecurityManager {
    fn default() -> Self {
        Self::new()
    }
}
