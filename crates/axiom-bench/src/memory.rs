//! Peak RSS memory tracking for benchmark reporting.

use sysinfo::{Pid, ProcessRefreshKind, RefreshKind, System};

/// Reads current process Resident Set Size (RSS) memory in megabytes (MB).
pub fn current_rss_mb() -> f64 {
    let pid = Pid::from_u32(std::process::id());
    let mut sys = System::new_with_specifics(
        RefreshKind::new().with_processes(ProcessRefreshKind::new().with_memory()),
    );
    sys.refresh_processes_specifics(ProcessRefreshKind::new().with_memory());

    if let Some(process) = sys.process(pid) {
        process.memory() as f64 / (1024.0 * 1024.0)
    } else {
        0.0
    }
}
