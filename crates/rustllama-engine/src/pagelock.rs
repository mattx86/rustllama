//! Memory-residency: pin the model's hot weight tiers into RAM so the OS
//! can't page them to the pagefile/swap.
//!
//! - **Windows:** `VirtualLock` into the process working set (after raising
//!   the working-set minimum with `SetProcessWorkingSetSizeEx`), then
//!   `PrefetchVirtualMemory` the ranges in as one coalesced read.
//! - **POSIX (M7):** `mlock`/`munlock` the same tier-ordered ranges, capped
//!   by the `RLIMIT_MEMLOCK` soft limit — when the limit is below the
//!   request we lock what the limit allows and `warn!` the shortfall rather
//!   than failing the load (fail-soft, matching the Windows path).
//!
//! Gated by `RUSTLLAMA_LOCK_RAM_MB` (default OFF). `auto` locks the
//! always-hot Tier-0 set within available RAM / the memlock limit; an
//! explicit MB value caps the budget. Cold MoE routed-expert blobs (Tier 2)
//! are normally left pageable so the OS can still relieve pressure.
//!
//! On a target that is neither Windows nor Unix the whole surface is a no-op.

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
    // T3: the directive comes from the typed `LoadOptions` (via the accel
    // override cell) when the engine set it, else the legacy
    // `RUSTLLAMA_LOCK_RAM_MB` env var (back-compat). Still parsed once.
    *CELL.get_or_init(|| match rustllama_models::accel::lock_ram_mb_directive() {
        Some(v) => {
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
        None => LockBudget::Off,
    })
}

/// Clamp a requested lock size (bytes) to what the POSIX `RLIMIT_MEMLOCK`
/// soft limit permits, after reserving bytes already spoken for (the MoE
/// expert-pin cache locks through the SAME per-process limit, so the two
/// budgets must not independently "fit"). Returns `(granted, shortfall)`:
/// `granted` is how many bytes may be locked and `shortfall = requested -
/// granted` (0 when the full request fits under the limit).
/// `rlimit_cur == u64::MAX` models `RLIM_INFINITY` (unlimited — typical for
/// root or a container with no memlock cap) and never clamps.
///
/// Pure + host-agnostic so the clamp math is unit-tested on any platform
/// (the `mlock` behaviour itself is validated on the Linux pod). Only the
/// `#[cfg(unix)]` lock path consumes it in a non-test build.
#[cfg_attr(not(unix), allow(dead_code))]
fn clamp_to_memlock_limit(requested: u64, rlimit_cur: u64, reserved: u64) -> (u64, u64) {
    if rlimit_cur == u64::MAX {
        return (requested, 0);
    }
    let headroom = rlimit_cur.saturating_sub(reserved);
    let granted = requested.min(headroom);
    (granted, requested - granted)
}

/// Project how many bytes the pagelock tier will pin for this model under the
/// current `RUSTLLAMA_LOCK_RAM_MB` directive, WITHOUT locking anything.
///
/// The cross-platform memory-budget planner ([`crate::memory_budget`])
/// reserves this so the (also-pinned) MoE expert cache is never sized onto
/// RAM the loader is about to lock — the same accounting the Windows path got
/// only from the pagelock side before M7. `0` when locking is off
/// (`LockBudget::Off`) → the planner is byte-identical to before this hook.
///
/// Platform-agnostic (no OS calls — it walks the model's lock targets, which
/// `lock_model_into_ram` does too): `Auto` projects the Tier-0 always-hot
/// set; an explicit `Bytes(n)` projects `min(n, total lockable)`. The actual
/// lock later clamps this against available RAM / the `RLIMIT_MEMLOCK` limit,
/// so this is an UPPER bound on what gets pinned — the safe side to reserve.
pub fn projected_lock_bytes(model: &LlamaModel) -> u64 {
    let budget_kind = lock_ram_budget();
    if budget_kind == LockBudget::Off {
        return 0;
    }
    let mut targets = model.collect_lock_targets();
    let mut seen = std::collections::HashSet::new();
    targets.retain(|t| t.addr != 0 && t.len != 0 && seen.insert(t.addr));
    let tier0_total: u64 = targets
        .iter()
        .filter(|t| t.tier == 0)
        .map(|t| t.len as u64)
        .sum();
    let all_total: u64 = targets.iter().map(|t| t.len as u64).sum();
    match budget_kind {
        LockBudget::Off => 0,
        LockBudget::Auto => tier0_total,
        LockBudget::Bytes(n) => n.min(all_total),
    }
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
            let _ = PrefetchVirtualMemory(GetCurrentProcess(), entries.len(), entries.as_ptr(), 0);
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

// ---------------------------------------------------------------------------
// POSIX implementation (M7) — Linux / macOS
// ---------------------------------------------------------------------------
#[cfg(unix)]
mod sys_unix {
    /// `RLIMIT_MEMLOCK` soft limit in bytes — the per-process ceiling on how
    /// much locked (unswappable) memory `mlock` will grant. `None` on
    /// `getrlimit` failure; `Some(u64::MAX)` for `RLIM_INFINITY` (unlimited).
    pub fn memlock_limit() -> Option<u64> {
        // SAFETY: zero-initialized POD out-param; getrlimit fills it from a
        // constant resource id. rc==0 means the fields are valid.
        let mut rl: libc::rlimit = unsafe { std::mem::zeroed() };
        let rc = unsafe { libc::getrlimit(libc::RLIMIT_MEMLOCK, &mut rl) };
        if rc != 0 {
            return None;
        }
        if rl.rlim_cur == libc::RLIM_INFINITY {
            Some(u64::MAX)
        } else {
            Some(rl.rlim_cur as u64)
        }
    }

    /// `mlock` a single range. No-op (returns false) on null/zero; returns
    /// false (fail-soft) when the kernel refuses (e.g. `ENOMEM`/`EPERM`).
    pub fn lock_range(addr: usize, len: usize) -> bool {
        if addr == 0 || len == 0 {
            return false;
        }
        // SAFETY: addr/len describe a live, CPU-resident weight buffer owned
        // by the model for the registry's lifetime.
        let rc = unsafe { libc::mlock(addr as *const libc::c_void, len) };
        rc == 0
    }

    /// `munlock` a single range (best-effort, errors ignored).
    pub fn unlock_range(addr: usize, len: usize) {
        if addr == 0 || len == 0 {
            return;
        }
        // SAFETY: same range previously mlock'd; munlock is always safe.
        unsafe {
            let _ = libc::munlock(addr as *const libc::c_void, len);
        }
    }

    /// Physical memory `(total, available)` bytes via `sysconf`, so the
    /// cross-platform memory-budget planner runs on POSIX too (previously it
    /// only ran on Windows, where `GlobalMemoryStatusEx` is available).
    /// Linux (and most SVR4) expose `_SC_PHYS_PAGES` / `_SC_AVPHYS_PAGES`;
    /// macOS does NOT define them in `libc` (its queries are `sysctl`
    /// `hw.memsize` + the `vm_statistics64` Mach call), so it has its OWN arm
    /// below (M7-mac-mem) rather than this `sysconf` path. `None` on any
    /// `sysconf` failure.
    #[cfg(target_os = "linux")]
    pub fn mem_status() -> Option<(u64, u64)> {
        // SAFETY: sysconf takes a constant resource id, no pointers.
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        let phys = unsafe { libc::sysconf(libc::_SC_PHYS_PAGES) };
        let avail = unsafe { libc::sysconf(libc::_SC_AVPHYS_PAGES) };
        if page <= 0 || phys <= 0 {
            return None;
        }
        let total = (phys as u64).saturating_mul(page as u64);
        let free = if avail > 0 {
            (avail as u64).saturating_mul(page as u64)
        } else {
            0
        };
        Some((total, free))
    }

    /// Physical memory `(total, available)` bytes on macOS (M7-mac-mem).
    /// macOS does not implement `_SC_PHYS_PAGES` / `_SC_AVPHYS_PAGES`, so the
    /// Linux `sysconf` arm above cannot serve it; this uses the native
    /// BSD/Mach queries instead, so the cross-platform memory-budget +
    /// pagelock planner runs on Apple hosts too (both now `mlock` RAM there):
    ///   * TOTAL  — `sysctl(CTL_HW, HW_MEMSIZE)`, the documented 64-bit count
    ///     of installed physical RAM (the legacy `HW_PHYSMEM` is a 32-bit int
    ///     that wraps above 2 GiB, so it is deliberately NOT used).
    ///   * AVAIL  — `host_statistics64(HOST_VM_INFO64)` → `vm_statistics64`,
    ///     with `(free + inactive)` pages × page size the readily-reclaimable
    ///     RAM, the macOS analogue of Linux's `_SC_AVPHYS_PAGES`. Inactive
    ///     pages are clean / file-backed and reclaimed on demand; speculative
    ///     and purgeable pages are deliberately NOT counted, keeping the
    ///     figure conservative (the planner must never OVER-estimate free RAM).
    /// Fail-soft, like the Linux arm: a failing `sysctl` → `None`; a failing
    /// Mach call → `Some((total, 0))` (total known, free unknown → the planner
    /// treats the box as tight rather than guessing, matching how the Linux
    /// arm reports `0` when `_SC_AVPHYS_PAGES` is unavailable).
    ///
    /// WRITE-BLIND: this arm is `cfg`'d out on Windows / Linux, so it is NOT
    /// exercised in CI here — correctness is by review (libc symbols + Mach
    /// ABI) and on-device confirmation on a Mac. The Windows / Linux arms are
    /// untouched and stay byte-identical.
    #[cfg(target_os = "macos")]
    pub fn mem_status() -> Option<(u64, u64)> {
        // --- total physical RAM: sysctl hw.memsize (64-bit) ---
        let mut mib: [libc::c_int; 2] = [libc::CTL_HW, libc::HW_MEMSIZE];
        let mut total: u64 = 0;
        let mut total_len = std::mem::size_of::<u64>();
        // SAFETY: `mib` is a valid 2-element MIB; `total` / `total_len` are a
        // correctly-sized out-buffer and its length. `newp` is null (read
        // only), so sysctl only writes `total`.
        let rc = unsafe {
            libc::sysctl(
                mib.as_mut_ptr(),
                mib.len() as libc::c_uint,
                &mut total as *mut u64 as *mut libc::c_void,
                &mut total_len,
                std::ptr::null_mut(),
                0,
            )
        };
        if rc != 0 || total == 0 {
            return None;
        }

        // --- available RAM: host_statistics64(HOST_VM_INFO64) page counts ---
        // SAFETY: sysconf takes a constant resource id, no pointers.
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if page <= 0 {
            // Total is still useful; report free as unknown (0) — see above.
            return Some((total, 0));
        }
        let mut vm: libc::vm_statistics64 = unsafe { std::mem::zeroed() };
        // Element count of the out-struct in `integer_t` units — the canonical
        // value of `HOST_VM_INFO64_COUNT`, computed here so we do not depend on
        // libc exposing that particular constant.
        let mut count = (std::mem::size_of::<libc::vm_statistics64>()
            / std::mem::size_of::<libc::integer_t>())
            as libc::mach_msg_type_number_t;
        // SAFETY: `mach_host_self()` yields a valid host port; `vm` is a zeroed,
        // correctly-sized out-struct and `count` its element count (matching
        // the HOST_VM_INFO64 flavor above).
        let kr = unsafe {
            libc::host_statistics64(
                libc::mach_host_self(),
                libc::HOST_VM_INFO64,
                &mut vm as *mut libc::vm_statistics64 as libc::host_info64_t,
                &mut count,
            )
        };
        if kr != libc::KERN_SUCCESS {
            return Some((total, 0));
        }
        let page = page as u64;
        let avail = (vm.free_count as u64)
            .saturating_add(vm.inactive_count as u64)
            .saturating_mul(page);
        Some((total, avail))
    }

    /// Other POSIX (e.g. the BSDs): no portable page-count query wired yet →
    /// `None`, so the planner stays off there (today's behavior), exactly as
    /// before this macOS arm was added.
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    pub fn mem_status() -> Option<(u64, u64)> {
        None
    }
}

/// Highest working-set floor (bytes intended for VirtualLock) granted
/// so far. Raise-only: shrinking the min while pages are locked would
/// make later `VirtualLock` calls fail, so refreshes only ever grow it.
static WS_FLOOR_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Bytes the pagelock tier intends to lock (stamped by
/// [`lock_model_into_ram`]); the expert-cache budget is added on top
/// at refresh time, so the two consumers share one floor.
static PAGELOCK_INTENDED_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Physical memory status: `(total, available)` bytes. `None` on API
/// failure or an unsupported target. Public so the memory-budget planner
/// shares the same source of truth as the lock logic.
pub fn memory_status() -> Option<(u64, u64)> {
    #[cfg(windows)]
    {
        sys::mem_status()
    }
    // M7 (cross-platform planner): now that `lock_ram_mb` pins RAM on POSIX
    // too, the memory-budget planner must run there to reserve it. Linux
    // reports via `sysconf`; macOS via `sysctl` + `host_statistics64`
    // (M7-mac-mem, below); any other POSIX returns `None` (planner stays off
    // there).
    #[cfg(unix)]
    {
        sys_unix::mem_status()
    }
    #[cfg(not(any(windows, unix)))]
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
    /// POSIX has no per-page "is this locked?" query (`mincore` reports
    /// residency, not the locked state), so report the retained — i.e.
    /// successfully `mlock`'d — region count as both locked and probed.
    #[cfg(all(unix, not(windows)))]
    pub fn verify_locked(&self) -> (u64, u64) {
        let n = self.regions.len() as u64;
        (n, n)
    }
    #[cfg(not(any(windows, unix)))]
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

#[cfg(all(unix, not(windows)))]
impl Drop for LockRegistry {
    fn drop(&mut self) {
        for &(addr, len) in &self.regions {
            sys_unix::unlock_range(addr, len);
        }
        tracing::info!(
            regions = self.regions.len(),
            locked_mb = self.locked_bytes / (1024 * 1024),
            "pagelock: released mlock ranges on engine drop"
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

/// POSIX page-lock (M7): `mlock` the model's hot weight tiers into RAM per
/// `RUSTLLAMA_LOCK_RAM_MB`, capped by the `RLIMIT_MEMLOCK` soft limit. On a
/// limit below the request we lock what the limit allows and `warn!` the
/// shortfall instead of failing the load (fail-soft). `None` when disabled
/// or when nothing could be locked.
#[cfg(all(unix, not(windows)))]
pub fn lock_model_into_ram(model: Arc<LlamaModel>) -> Option<LockRegistry> {
    let budget_kind = lock_ram_budget();
    if budget_kind == LockBudget::Off {
        // POSIX has no working-set-floor analogue to raise for the MoE
        // expert-pin cache; `mlock` draws straight from the memlock limit,
        // so with pagelock itself off there is nothing to do here.
        return None;
    }

    // Enumerate targets, dedup by backing address, sort by tier so the
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
    let all_total: u64 = targets.iter().map(|t| t.len as u64).sum();

    // Bytes the pagelock tier would like to lock before the RLIMIT clamp.
    let requested: u64 = match budget_kind {
        LockBudget::Off => return None,
        LockBudget::Auto => tier0_total,
        LockBudget::Bytes(n) => n.min(all_total),
    };

    // The MoE expert-pin cache `mlock`s through the SAME per-process limit
    // later in the load, so treat its budget as already spoken for.
    let expert_reserve = rustllama_models::accel::moe_expert_cache_max_bytes();
    let rlimit_cur = sys_unix::memlock_limit().unwrap_or(u64::MAX);
    let (granted, shortfall) = clamp_to_memlock_limit(requested, rlimit_cur, expert_reserve);
    let mb = |b: u64| b / (1024 * 1024);
    if shortfall > 0 {
        tracing::warn!(
            requested_mb = mb(requested),
            granted_mb = mb(granted),
            shortfall_mb = mb(shortfall),
            rlimit_memlock_mb = if rlimit_cur == u64::MAX {
                u64::MAX
            } else {
                mb(rlimit_cur)
            },
            expert_reserve_mb = mb(expert_reserve),
            "pagelock: RLIMIT_MEMLOCK below request — locking what the limit \
             allows (raise with `ulimit -l` / systemd `LimitMEMLOCK=`)"
        );
    }
    if granted == 0 {
        tracing::warn!(
            requested_mb = mb(requested),
            "pagelock: no memlock headroom (RLIMIT_MEMLOCK exhausted); skipping"
        );
        return None;
    }

    // Greedily lock tier-ordered targets up to the (capped) budget.
    let mut intended: u64 = 0;
    let mut to_lock: Vec<(usize, usize)> = Vec::new();
    for t in &targets {
        let len = t.len as u64;
        if intended + len > granted {
            continue; // skip oversized; keep filling smaller later targets
        }
        intended += len;
        to_lock.push((t.addr, t.len));
    }

    let mut regions: Vec<(usize, usize)> = Vec::with_capacity(to_lock.len());
    let mut locked_bytes: u64 = 0;
    let mut failed = 0usize;
    for (addr, len) in to_lock {
        if sys_unix::lock_range(addr, len) {
            regions.push((addr, len));
            locked_bytes += len as u64;
        } else {
            failed += 1;
        }
    }

    if regions.is_empty() {
        tracing::warn!(
            failed,
            "pagelock: mlock locked 0 ranges (all mlock calls failed); not retaining registry"
        );
        return None;
    }

    tracing::info!(
        requested_mb = mb(requested),
        tier0_mb = mb(tier0_total),
        granted_mb = mb(granted),
        locked_mb = mb(locked_bytes),
        regions = regions.len(),
        failed,
        "pagelock: mlock'd hot weight tiers into RAM"
    );

    Some(LockRegistry {
        regions,
        locked_bytes,
        _model: model,
    })
}

#[cfg(not(any(windows, unix)))]
pub fn lock_model_into_ram(_model: Arc<LlamaModel>) -> Option<LockRegistry> {
    None
}

#[cfg(test)]
mod tests {
    use super::clamp_to_memlock_limit;

    // M7: the RLIMIT_MEMLOCK clamp math is host-agnostic, so it runs on any
    // platform (including the Windows dev box) even though `mlock` itself is
    // only exercised on the Linux pod.
    #[test]
    fn memlock_clamp_math() {
        // Full request fits comfortably under the limit → no shortfall.
        assert_eq!(clamp_to_memlock_limit(1000, 4000, 0), (1000, 0));
        // Request exceeds the limit → lock what's allowed, report the rest.
        assert_eq!(clamp_to_memlock_limit(5000, 4000, 0), (4000, 1000));
        // Reserved bytes (the expert-pin cache) eat into the headroom first.
        assert_eq!(clamp_to_memlock_limit(3000, 4000, 1500), (2500, 500));
        // Reserved alone exceeds the limit → nothing granted, full shortfall.
        assert_eq!(clamp_to_memlock_limit(3000, 1000, 2000), (0, 3000));
        // Exactly at the limit is not a shortfall.
        assert_eq!(clamp_to_memlock_limit(4000, 4000, 0), (4000, 0));
        // RLIM_INFINITY (modeled as u64::MAX) never clamps, reserve ignored.
        assert_eq!(
            clamp_to_memlock_limit(u64::MAX, u64::MAX, 123),
            (u64::MAX, 0)
        );
        // Saturating headroom: reserved > limit clamps to 0 (no underflow).
        assert_eq!(clamp_to_memlock_limit(10, 5, 9), (0, 10));
        // Reserved leaves a sliver of headroom below the request.
        assert_eq!(clamp_to_memlock_limit(10, 5, 4), (1, 9));
    }
}
