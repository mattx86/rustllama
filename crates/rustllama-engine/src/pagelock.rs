//! Windows memory-residency: VirtualLock the model's hot weight tiers
//! into the process working set so the OS can't page them to the
//! pagefile, then PrefetchVirtualMemory them in as one coalesced read.
//!
//! Gated by `RUSTLLAMA_LOCK_RAM_MB` (default OFF). `auto` locks the
//! always-hot Tier-0 set within available RAM; an explicit MB value
//! caps the budget. Cold MoE routed-expert blobs (Tier 2) are normally
//! left pageable so the OS can still relieve pressure.
//!
//! On non-Windows targets the whole surface is a no-op.

use std::sync::Arc;
use std::sync::OnceLock;

use rustllama_models::llama_arch::LlamaModel;

/// Parsed `RUSTLLAMA_LOCK_RAM_MB` budget directive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockBudget {
    /// Disabled — no working-set changes, no locking.
    Off,
    /// Lock the Tier-0 always-hot set, clamped to available RAM.
    Auto,
    /// Lock up to this many bytes (tier-ordered), clamped to available RAM.
    Bytes(u64),
}

/// Read `RUSTLLAMA_LOCK_RAM_MB` once. `unset`/`0`/empty → Off;
/// `auto` (case-insensitive) → Auto; a positive integer → Bytes(MB).
pub fn lock_ram_budget() -> LockBudget {
    static CELL: OnceLock<LockBudget> = OnceLock::new();
    *CELL.get_or_init(|| match std::env::var("RUSTLLAMA_LOCK_RAM_MB") {
        Ok(v) => {
            let v = v.trim();
            if v.eq_ignore_ascii_case("auto") {
                LockBudget::Auto
            } else if v.is_empty() {
                LockBudget::Off
            } else {
                match v.parse::<u64>() {
                    Ok(0) => LockBudget::Off,
                    Ok(mb) => LockBudget::Bytes(mb.saturating_mul(1024 * 1024)),
                    Err(_) => LockBudget::Off,
                }
            }
        }
        Err(_) => LockBudget::Off,
    })
}

// ---------------------------------------------------------------------------
// Windows implementation
// ---------------------------------------------------------------------------
#[cfg(windows)]
mod sys {
    use windows_sys::Win32::Foundation::BOOL;
    use windows_sys::Win32::System::Memory::{
        PrefetchVirtualMemory, SetProcessWorkingSetSizeEx, VirtualLock, VirtualUnlock,
        WIN32_MEMORY_RANGE_ENTRY,
    };
    use windows_sys::Win32::System::ProcessStatus::{
        QueryWorkingSetEx, PSAPI_WORKING_SET_EX_INFORMATION,
    };
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    // QUOTA_LIMITS_HARDWS_MIN_ENABLE (= 1) — the SETPROCESSWORKINGSETSIZEEX_FLAGS
    // alias is a plain u32, so the literal matches the param type.
    const QUOTA_LIMITS_HARDWS_MIN_ENABLE: u32 = 0x0000_0001;
    /// `PSAPI_WORKING_SET_EX_BLOCK.Flags`: Valid is bit 0, Locked is
    /// bit 22 (in the "valid" layout). We bit-test these directly since
    /// windows-sys exposes the union as a single `usize`.
    const WS_FLAG_VALID: usize = 1 << 0;
    const WS_FLAG_LOCKED: usize = 1 << 22;

    pub fn mem_status() -> Option<(u64 /*total*/, u64 /*avail*/)> {
        let mut s: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
        s.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
        // SAFETY: `s` is a correctly-sized, zero-initialized buffer with
        // dwLength set as the API requires.
        let ok: BOOL = unsafe { GlobalMemoryStatusEx(&mut s) };
        if ok == 0 {
            None
        } else {
            Some((s.ullTotalPhys, s.ullAvailPhys))
        }
    }

    /// Raise the process working-set minimum so subsequent VirtualLock
    /// calls (which require locked pages to fit within the min WS) can
    /// pin `intended_bytes`. Returns false on failure (caller falls back
    /// to fail-soft per-range locking, which will then mostly no-op).
    pub fn prepare_working_set(intended_bytes: u64, total_phys: u64) -> bool {
        // Headroom above the locked set for code + scratch + transient
        // allocations; clamp the floor to 90% of physical RAM.
        let headroom: u64 = 256 * 1024 * 1024;
        let cap = total_phys / 100 * 90;
        let min_ws = (intended_bytes.saturating_add(headroom)).min(cap);
        let max_ws = total_phys; // OS caps in practice; generous max avoids trimming.
        // SAFETY: pseudo-handle from GetCurrentProcess is always valid.
        let ok: BOOL = unsafe {
            SetProcessWorkingSetSizeEx(
                GetCurrentProcess(),
                min_ws as usize,
                max_ws as usize,
                QUOTA_LIMITS_HARDWS_MIN_ENABLE,
            )
        };
        ok != 0
    }

    /// VirtualLock a single range. No-op (returns false) on null/zero.
    pub fn lock_range(addr: usize, len: usize) -> bool {
        if addr == 0 || len == 0 {
            return false;
        }
        // SAFETY: addr/len describe a live, CPU-resident weight buffer
        // owned by the model for the registry's lifetime.
        let ok: BOOL = unsafe { VirtualLock(addr as *const core::ffi::c_void, len) };
        ok != 0
    }

    /// VirtualUnlock a single range (best-effort, errors ignored).
    pub fn unlock_range(addr: usize, len: usize) {
        if addr == 0 || len == 0 {
            return;
        }
        // SAFETY: same range previously locked; unlocking is always safe.
        unsafe {
            let _ = VirtualUnlock(addr as *const core::ffi::c_void, len);
        }
    }

    /// PrefetchVirtualMemory over all locked ranges in one call.
    pub fn prefetch(regions: &[(usize, usize)]) {
        if regions.is_empty() {
            return;
        }
        let entries: Vec<WIN32_MEMORY_RANGE_ENTRY> = regions
            .iter()
            .map(|&(addr, len)| WIN32_MEMORY_RANGE_ENTRY {
                VirtualAddress: addr as *mut core::ffi::c_void,
                NumberOfBytes: len,
            })
            .collect();
        // SAFETY: entries point at live mapped ranges; count matches.
        unsafe {
            let _ = PrefetchVirtualMemory(
                GetCurrentProcess(),
                entries.len(),
                entries.as_ptr(),
                0,
            );
        }
    }

    /// Diagnostics: count how many of `regions`' first pages report the
    /// Locked working-set bit. Returns (locked_pages_seen, probed).
    pub fn count_locked(regions: &[(usize, usize)]) -> (u64, u64) {
        let mut locked = 0u64;
        let mut probed = 0u64;
        for &(addr, _len) in regions {
            if addr == 0 {
                continue;
            }
            let mut info = PSAPI_WORKING_SET_EX_INFORMATION {
                VirtualAddress: addr as *mut core::ffi::c_void,
                VirtualAttributes: unsafe { std::mem::zeroed() },
            };
            // SAFETY: single-element query over a valid address.
            let ok: BOOL = unsafe {
                QueryWorkingSetEx(
                    GetCurrentProcess(),
                    (&mut info as *mut PSAPI_WORKING_SET_EX_INFORMATION).cast(),
                    std::mem::size_of::<PSAPI_WORKING_SET_EX_INFORMATION>() as u32,
                )
            };
            if ok != 0 {
                probed += 1;
                // SAFETY: union read — `Flags` aliases the bitfield; we
                // only bit-test it, never interpret the typed view.
                let flags = unsafe { info.VirtualAttributes.Flags };
                if flags & WS_FLAG_VALID != 0 && flags & WS_FLAG_LOCKED != 0 {
                    locked += 1;
                }
            }
        }
        (locked, probed)
    }
}

/// Highest working-set floor (bytes intended for VirtualLock) granted
/// so far. Raise-only: shrinking the min while pages are locked would
/// make later `VirtualLock` calls fail, so refreshes only ever grow it.
static WS_FLOOR_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Bytes the pagelock tier intends to lock (stamped by
/// [`lock_model_into_ram`]); the expert-cache budget is added on top
/// at refresh time, so the two consumers share one floor.
static PAGELOCK_INTENDED_BYTES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Physical memory status: `(total, available)` bytes. `None` on
/// non-Windows targets or API failure. Public so the memory-budget
/// planner shares the same source of truth as the lock logic.
pub fn memory_status() -> Option<(u64, u64)> {
    #[cfg(windows)]
    {
        sys::mem_status()
    }
    #[cfg(not(windows))]
    {
        None
    }
}

/// (Re-)raise the process working-set floor to cover the pagelock
/// tier's intended bytes plus the current MoE expert-cache budget.
/// Called at model load and again on elastic budget changes; no-op
/// when the current floor already covers the need. Returns `false`
/// when the OS refused the raise (subsequent `VirtualLock`s will
/// likely no-op and the affected weights simply stream).
pub fn refresh_working_set_floor() -> bool {
    use std::sync::atomic::Ordering;
    let needed = PAGELOCK_INTENDED_BYTES
        .load(Ordering::Relaxed)
        .saturating_add(rustllama_models::accel::moe_expert_cache_max_bytes());
    if needed == 0 || needed <= WS_FLOOR_BYTES.load(Ordering::Relaxed) {
        return true;
    }
    #[cfg(windows)]
    {
        let Some((total_phys, _)) = sys::mem_status() else {
            return false;
        };
        if sys::prepare_working_set(needed, total_phys) {
            WS_FLOOR_BYTES.store(needed, Ordering::Relaxed);
            true
        } else {
            false
        }
    }
    #[cfg(not(windows))]
    {
        true
    }
}

/// Holds the set of VirtualLock'd ranges + an `Arc<LlamaModel>` clone so
/// the backing buffers outlive the unlock on drop. Dropping the registry
/// VirtualUnlocks every range.
pub struct LockRegistry {
    regions: Vec<(usize, usize)>,
    locked_bytes: u64,
    // Keep the model alive at least as long as the locked ranges.
    _model: Arc<LlamaModel>,
}

impl LockRegistry {
    pub fn locked_bytes(&self) -> u64 {
        self.locked_bytes
    }

    pub fn region_count(&self) -> usize {
        self.regions.len()
    }

    /// Diagnostics: (#regions reporting Locked bit, #probed).
    #[cfg(windows)]
    pub fn verify_locked(&self) -> (u64, u64) {
        sys::count_locked(&self.regions)
    }
    #[cfg(not(windows))]
    pub fn verify_locked(&self) -> (u64, u64) {
        (0, 0)
    }
}

#[cfg(windows)]
impl Drop for LockRegistry {
    fn drop(&mut self) {
        for &(addr, len) in &self.regions {
            sys::unlock_range(addr, len);
        }
        tracing::info!(
            regions = self.regions.len(),
            locked_mb = self.locked_bytes / (1024 * 1024),
            "pagelock: released VirtualLock ranges on engine drop"
        );
    }
}

/// Lock the model's hot weight tiers into RAM per `RUSTLLAMA_LOCK_RAM_MB`.
/// Returns `Some(registry)` when any range was locked (caller stores it
/// on the engine so it unlocks on drop), `None` when disabled, on a
/// non-Windows target, or when nothing could be locked.
#[cfg(windows)]
pub fn lock_model_into_ram(model: Arc<LlamaModel>) -> Option<LockRegistry> {
    let budget_kind = lock_ram_budget();
    // The MoE expert-pin cache VirtualLocks through the same process
    // working-set quota, so its budget must be part of the floor we
    // request here — otherwise an expert-cache-only config silently
    // fails to pin past the default working set.
    let expert_budget = rustllama_models::accel::moe_expert_cache_max_bytes();
    if budget_kind == LockBudget::Off && expert_budget == 0 {
        return None;
    }

    let (total_phys, avail_phys) = match sys::mem_status() {
        Some(v) => v,
        None => {
            tracing::warn!("pagelock: GlobalMemoryStatusEx failed; skipping lock");
            return None;
        }
    };

    if budget_kind == LockBudget::Off {
        // Pagelock itself is off — just raise the working-set floor
        // for the expert-pin cache and return.
        if refresh_working_set_floor() {
            tracing::info!(
                expert_cache_mb = expert_budget / (1024 * 1024),
                "pagelock: working-set floor raised for the MoE expert-pin cache"
            );
        } else {
            tracing::warn!(
                "pagelock: SetProcessWorkingSetSizeEx failed; expert-pin cache \
                 VirtualLocks will likely no-op (experts stream instead)"
            );
        }
        return None;
    }

    // Enumerate targets, dedup by backing address, and sort by tier so the
    // greedy budget fills always-hot ranges before cold expert blobs.
    let mut targets = model.collect_lock_targets();
    targets.sort_by_key(|t| t.tier);
    let mut seen = std::collections::HashSet::new();
    targets.retain(|t| t.addr != 0 && t.len != 0 && seen.insert(t.addr));

    let tier0_total: u64 = targets
        .iter()
        .filter(|t| t.tier == 0)
        .map(|t| t.len as u64)
        .sum();

    // Compute the byte budget, clamped against available RAM. The
    // expert-pin cache will VirtualLock up to its own budget later in
    // the load sequence, so treat those bytes as already spoken for —
    // otherwise the two budgets independently "fit" and jointly
    // overcommit.
    let two_gb: u64 = 2 * 1024 * 1024 * 1024;
    let reserve = two_gb.max(total_phys / 4);
    let max_lockable = avail_phys.saturating_sub(reserve.saturating_add(expert_budget));
    let budget: u64 = match budget_kind {
        LockBudget::Off => return None,
        LockBudget::Auto => tier0_total.min(max_lockable),
        LockBudget::Bytes(n) => {
            let clamped = n.min(avail_phys.saturating_sub(two_gb.saturating_add(expert_budget)));
            if clamped < n {
                tracing::warn!(
                    requested_mb = n / (1024 * 1024),
                    clamped_mb = clamped / (1024 * 1024),
                    avail_mb = avail_phys / (1024 * 1024),
                    "pagelock: RUSTLLAMA_LOCK_RAM_MB clamped to fit available RAM"
                );
            }
            clamped
        }
    };

    if budget == 0 {
        tracing::warn!(
            avail_mb = avail_phys / (1024 * 1024),
            "pagelock: no headroom to lock weights (budget resolved to 0); skipping"
        );
        return None;
    }

    // Greedily lock tier-ordered targets up to the (capped) budget. Size
    // the working-set floor to what we actually intend to lock.
    let mut intended: u64 = 0;
    let mut to_lock: Vec<(usize, usize)> = Vec::new();
    for t in &targets {
        let len = t.len as u64;
        if intended + len > budget {
            continue; // skip oversized; keep filling smaller later targets in this/next tier
        }
        intended += len;
        to_lock.push((t.addr, t.len));
    }

    PAGELOCK_INTENDED_BYTES.store(intended, std::sync::atomic::Ordering::Relaxed);
    if !refresh_working_set_floor() {
        tracing::warn!(
            "pagelock: SetProcessWorkingSetSizeEx failed (SE_INC_WORKING_SET_NAME absent?); \
             large VirtualLock will likely fail — proceeding fail-soft"
        );
    }

    let mut regions: Vec<(usize, usize)> = Vec::with_capacity(to_lock.len());
    let mut locked_bytes: u64 = 0;
    let mut failed = 0usize;
    for (addr, len) in to_lock {
        if sys::lock_range(addr, len) {
            regions.push((addr, len));
            locked_bytes += len as u64;
        } else {
            failed += 1;
        }
    }

    if regions.is_empty() {
        tracing::warn!(
            failed,
            "pagelock: locked 0 ranges (all VirtualLock calls failed); not retaining registry"
        );
        return None;
    }

    sys::prefetch(&regions);

    tracing::info!(
        budget_mb = budget / (1024 * 1024),
        tier0_mb = tier0_total / (1024 * 1024),
        locked_mb = locked_bytes / (1024 * 1024),
        regions = regions.len(),
        failed,
        "pagelock: VirtualLock'd hot weight tiers into RAM + prefetched"
    );

    Some(LockRegistry {
        regions,
        locked_bytes,
        _model: model,
    })
}

#[cfg(not(windows))]
pub fn lock_model_into_ram(_model: Arc<LlamaModel>) -> Option<LockRegistry> {
    None
}
