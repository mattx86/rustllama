//! Intel AMX (Advanced Matrix Extensions) bf16 tile GEMM — an optional fast
//! path for the F16·F32 prefill GEMM (`gemm_f16_w_f32_a`) on Sapphire Rapids+
//! server CPUs.
//!
//! WRITE-BLIND + inline-asm. Rust's AMX tile intrinsics (`_tile_dpbf16ps`, …)
//! are nightly-only (`x86_amx_intrinsics`, rust-lang/rust#126622), but this
//! crate must build on the repo's stable 1.83 toolchain, so the TMUL
//! instructions are emitted through `core::arch::asm!` with the standard x86
//! mnemonics (which LLVM's integrated assembler accepts regardless of target
//! features — inline asm is opaque to the feature system). There is no AMX
//! hardware on the author's box, so this is authored blind and SELF-VALIDATES
//! at runtime: [`amx_gemm_usable`] executes the kernel against a scalar bf16
//! reference ONCE and enables the fast path only where the output matches —
//! fail-closed, so a wrong (unvalidated) encoding degrades to the existing
//! scalar/SIMD GEMM instead of corrupting a result.
//!
//! Safety / fault model: a tile instruction `#GP`s if the OS has not granted
//! `XTILEDATA` permission to the thread, so on Linux we gate on an explicit
//! `arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)` (Windows enables AMX
//! through the normal XSAVE path; macOS/non-x86 have no AMX). On a CPU without
//! AMX the feature is simply never detected and NO tile instruction is ever
//! reached, so the dev box and CI compile the asm but never run it.
//!
//! Numerics: AMX bf16 truncates both operands to bf16 (8-bit mantissa), so the
//! result is LOSSY vs the F16·F32 reference — the self-check grades against a
//! *bf16* reference (same truncation) so it measures kernel correctness, not
//! bf16 rounding; the fast path is still only taken for matmuls large enough
//! that the prefill GEMM dominates.
//!
//! Tile layout for `tdpbf16ps C, A, B` (C[M,N] f32 += A[M,K] bf16 · B[K,N]
//! bf16), all three tiles a uniform 16 rows × 64 bytes:
//!   - C: 16 rows × 16 f32  (colsb = 64)
//!   - A: 16 rows × 32 bf16 (colsb = 64), plain row-major M×K
//!   - B: 16 rows × 32 bf16 (colsb = 64), VNNI-packed: row `kk` holds the
//!        interleaved pair (X[2*kk][n], X[2*kk+1][n]) for n in 0..16, i.e.
//!        32 bf16 = two K-values per output column.
//! M/N remainders are handled by zero-padding the packed A/B buffers to the
//! full tile and copying back only the valid M×N sub-block of C; K remainders
//! by zero-padding the last K-tile (an odd K pads the trailing VNNI pair).

#![cfg(target_arch = "x86_64")]

use half::{bf16, f16};
use std::sync::OnceLock;

/// AMX bf16 tile dims: C is TM×TN f32, A is TM×TK bf16, B is (TK/2)×(2·TN) bf16.
const TM: usize = 16;
const TN: usize = 16;
const TK: usize = 32;
/// Every tile is 16 rows × 64 bytes with this layout.
const TILE_ROWS: u8 = 16;
const TILE_COLSB: u16 = 64;

/// 64-byte `ldtilecfg` payload. Palette 1; tiles 0/1/2 configured to 16×64,
/// the rest left zero (unused).
#[repr(C, align(64))]
struct TileConfig {
    palette_id: u8,
    start_row: u8,
    _reserved0: [u8; 14],
    colsb: [u16; 8],
    rows: [u8; 8],
}

impl TileConfig {
    fn new() -> Self {
        let mut colsb = [0u16; 8];
        let mut rows = [0u8; 8];
        // tmm0 = C, tmm1 = A, tmm2 = B.
        for t in 0..3 {
            colsb[t] = TILE_COLSB;
            rows[t] = TILE_ROWS;
        }
        TileConfig {
            palette_id: 1,
            start_row: 0,
            _reserved0: [0u8; 14],
            colsb,
            rows,
        }
    }
}

/// Request `XTILEDATA` permission for this thread (Linux ≥ 5.16). Without it the
/// first tile instruction `#GP`s. Returns `true` on success; non-Linux targets
/// enable AMX through the normal XSAVE path and return `true`.
#[cfg(target_os = "linux")]
fn request_amx_tiledata() -> bool {
    use std::arch::asm;
    // arch_prctl(ARCH_REQ_XCOMP_PERM = 0x1023, XFEATURE_XTILEDATA = 18).
    const ARCH_REQ_XCOMP_PERM: u64 = 0x1023;
    const XFEATURE_XTILEDATA: u64 = 18;
    const SYS_ARCH_PRCTL: u64 = 158;
    let ret: i64;
    // SAFETY: a plain syscall with register inputs; clobbers rcx/r11 per the
    // Linux syscall ABI. No memory is touched.
    unsafe {
        asm!(
            "syscall",
            inlateout("rax") SYS_ARCH_PRCTL => ret,
            in("rdi") ARCH_REQ_XCOMP_PERM,
            in("rsi") XFEATURE_XTILEDATA,
            lateout("rcx") _,
            lateout("r11") _,
            options(nostack)
        );
    }
    ret == 0
}

#[cfg(not(target_os = "linux"))]
fn request_amx_tiledata() -> bool {
    true
}

/// Detect AMX (tile + bf16) via raw CPUID + XCR0. The `is_x86_feature_detected!`
/// macro can't be used here: its `"amx-tile"` / `"amx-bf16"` arms are themselves
/// gated behind the nightly `x86_amx_intrinsics` feature, so they don't compile
/// on stable. CPUID leaf 7 (EDX bit 24 = AMX-TILE, bit 22 = AMX-BF16) plus XCR0
/// bits 17/18 (the OS has enabled XTILECFG + XTILEDATA state) are the same bits
/// that macro would read, and `__cpuid_count` / `xgetbv` are stable.
fn amx_cpu_supported() -> bool {
    use std::arch::x86_64::__cpuid_count;
    // SAFETY: CPUID is always available on x86-64.
    let leaf7 = unsafe { __cpuid_count(7, 0) };
    let amx_tile = (leaf7.edx >> 24) & 1 != 0;
    let amx_bf16 = (leaf7.edx >> 22) & 1 != 0;
    if !(amx_tile && amx_bf16) {
        return false;
    }
    // OSXSAVE (CPUID.1:ECX bit 27) must be set before XGETBV is valid.
    let leaf1 = unsafe { __cpuid_count(1, 0) };
    if (leaf1.ecx >> 27) & 1 == 0 {
        return false;
    }
    // XCR0 bits 17 (XTILECFG) + 18 (XTILEDATA): the OS saves/restores AMX tile
    // state → tile instructions are usable. Read via the `xgetbv` instruction
    // directly (avoids any intrinsic-stability question).
    let xcr0 = unsafe { read_xcr0() };
    (xcr0 >> 17) & 0b11 == 0b11
}

/// Read XCR0 (`xgetbv` with ECX=0). SAFETY: valid only when OSXSAVE is set
/// (checked by the sole caller).
unsafe fn read_xcr0() -> u64 {
    use std::arch::asm;
    let eax: u32;
    let edx: u32;
    asm!(
        "xgetbv",
        in("ecx") 0u32,
        out("eax") eax,
        out("edx") edx,
        options(nostack, nomem, preserves_flags)
    );
    ((edx as u64) << 32) | (eax as u64)
}

#[inline]
fn f16_to_bf16_bits(v: f16) -> u16 {
    bf16::from_f32(v.to_f32()).to_bits()
}

#[inline]
fn f32_to_bf16_bits(v: f32) -> u16 {
    bf16::from_f32(v).to_bits()
}

/// Pack a TM×TK bf16 A-tile (row-major) from `w[M,K]` at block (mb, kb),
/// zero-padding the M and K remainders.
fn pack_a(w: &[f16], a_buf: &mut [u16; TM * TK], m: usize, k: usize, mb: usize, kb: usize) {
    a_buf.fill(0);
    let mrows = (m - mb).min(TM);
    let kcols = (k - kb).min(TK);
    for i in 0..mrows {
        let w_row = &w[(mb + i) * k..];
        let dst = &mut a_buf[i * TK..];
        for p in 0..kcols {
            dst[p] = f16_to_bf16_bits(w_row[kb + p]);
        }
    }
}

/// Pack a VNNI B-tile from `x[K,N]` (row-major) at block (kb, nb): row `kk`
/// holds the interleaved pair (x[kb+2kk][n], x[kb+2kk+1][n]) for n in the
/// N-block. Zero-pads the N and K remainders (an odd K zeros the trailing
/// element of the last pair).
fn pack_b_vnni(x: &[f32], b_buf: &mut [u16; (TK / 2) * (2 * TN)], n: usize, k: usize, kb: usize, nb: usize) {
    b_buf.fill(0);
    let ncols = (n - nb).min(TN);
    let kcols = (k - kb).min(TK);
    for kk in 0..(TK / 2) {
        let k0 = kb + 2 * kk;
        let k1 = k0 + 1;
        let row = &mut b_buf[kk * (2 * TN)..];
        for j in 0..ncols {
            if 2 * kk < kcols {
                row[2 * j] = f32_to_bf16_bits(x[k0 * n + (nb + j)]);
            }
            if 2 * kk + 1 < kcols {
                row[2 * j + 1] = f32_to_bf16_bits(x[k1 * n + (nb + j)]);
            }
        }
    }
}

/// One AMX bf16 GEMM block: `C_tile[16,16] += A_tile · B_tile` using tmm0 (C),
/// tmm1 (A), tmm2 (B). The caller has already run `ldtilecfg` and `tilezero
/// tmm0`. `stride` is 64 (bytes/row) for all three tiles.
///
/// SAFETY: requires AMX to be present and `XTILEDATA` permitted (see
/// [`amx_gemm_usable`]); `a_ptr`/`b_ptr` point at 16×64-byte packed tiles.
#[inline]
unsafe fn amx_block_mul(a_ptr: *const u16, b_ptr: *const u16) {
    use std::arch::asm;
    let stride: usize = 64;
    asm!(
        "tileloadd tmm1, [{a} + {s}]",
        "tileloadd tmm2, [{b} + {s}]",
        "tdpbf16ps tmm0, tmm1, tmm2",
        a = in(reg) a_ptr,
        b = in(reg) b_ptr,
        s = in(reg) stride,
        options(nostack, preserves_flags)
    );
}

/// AMX bf16 GEMM: `c[M,N] = w[M,K] · x[K,N]` (same shape contract as
/// `gemm_f16_w_f32_a`), computed in bf16 tiles with f32 accumulation.
///
/// SAFETY: caller must ensure AMX is usable ([`amx_gemm_usable`] returned
/// true) on THIS thread; shapes must satisfy `w.len()==m*k`, `x.len()==k*n`,
/// `c.len()==m*n`.
pub unsafe fn gemm_bf16_amx(w: &[f16], x: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    use std::arch::asm;
    let cfg = TileConfig::new();
    // SAFETY: cfg is a 64-byte aligned payload; ldtilecfg reads 64 bytes.
    asm!("ldtilecfg [{c}]", c = in(reg) &cfg as *const TileConfig, options(nostack, readonly, preserves_flags));

    let mut a_buf = [0u16; TM * TK];
    let mut b_buf = [0u16; (TK / 2) * (2 * TN)];
    let mut c_buf = [0f32; TM * TN];
    let stride: usize = 64;

    let mut mb = 0;
    while mb < m {
        let mrows = (m - mb).min(TM);
        let mut nb = 0;
        while nb < n {
            let ncols = (n - nb).min(TN);
            asm!("tilezero tmm0", options(nostack, preserves_flags));
            let mut kb = 0;
            while kb < k {
                pack_a(w, &mut a_buf, m, k, mb, kb);
                pack_b_vnni(x, &mut b_buf, n, k, kb, nb);
                amx_block_mul(a_buf.as_ptr(), b_buf.as_ptr());
                kb += TK;
            }
            // Store the accumulator tile, then scatter the valid sub-block.
            asm!(
                "tilestored [{c} + {s}], tmm0",
                c = in(reg) c_buf.as_mut_ptr(),
                s = in(reg) stride,
                options(nostack)
            );
            for i in 0..mrows {
                for j in 0..ncols {
                    c[(mb + i) * n + (nb + j)] = c_buf[i * TN + j];
                }
            }
            nb += TN;
        }
        mb += TM;
    }
    asm!("tilerelease", options(nostack, preserves_flags));
}

/// Scalar bf16 reference for the self-check: truncates both operands to bf16
/// exactly as the tile path does, so a match isolates kernel/encoding
/// correctness from bf16 rounding.
fn gemm_bf16_ref(w: &[f16], x: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0f32;
            for p in 0..k {
                let a = bf16::from_f32(w[i * k + p].to_f32()).to_f32();
                let b = bf16::from_f32(x[p * n + j]).to_f32();
                acc += a * b;
            }
            c[i * n + j] = acc;
        }
    }
}

/// Whether the AMX bf16 GEMM fast path is usable on this host — cached. True
/// only when: the CPU reports `amx-tile` + `amx-bf16`, the OS granted tile
/// permission, AND the kernel reproduces the scalar bf16 reference on a probe
/// matrix (so a wrong write-blind encoding fails closed). Any failure (or a
/// non-AMX CPU) returns false and NO tile instruction is executed afterward.
pub fn amx_gemm_usable() -> bool {
    static USABLE: OnceLock<bool> = OnceLock::new();
    *USABLE.get_or_init(|| {
        if !amx_cpu_supported() {
            return false;
        }
        if !request_amx_tiledata() {
            return false;
        }
        // Self-check on a probe that crosses several tiles in every dim
        // (M=20, N=20, K=40 → 2×2 M/N tiles + 2 K-tiles with remainders).
        let (m, n, k) = (20usize, 20usize, 40usize);
        let w: Vec<f16> = (0..m * k)
            .map(|i| f16::from_f32((((i as i64 * 2_654_435_761) % 97) as f32 - 48.0) * 0.03))
            .collect();
        let x: Vec<f32> = (0..k * n)
            .map(|i| (((i as i64 * 40_503) % 89) as f32 - 44.0) * 0.02)
            .collect();
        let mut got = vec![0f32; m * n];
        let mut want = vec![0f32; m * n];
        gemm_bf16_ref(&w, &x, &mut want, m, n, k);
        // SAFETY: feature-detected + permission granted above; shapes match.
        unsafe { gemm_bf16_amx(&w, &x, &mut got, m, n, k) };
        got.iter().zip(&want).all(|(a, b)| {
            let denom = b.abs().max(1e-4);
            (a - b).abs() / denom <= 0.02
        })
    })
}

// ============================================================
// AMX-INT8 tile GEMM (tdpbssd) — the int8 sibling of the bf16 path above, for
// W8A8 quant (e.g. Q8_0) prefill on Xeon. Reuses the same 16×64 TileConfig: an
// int8 A tile is 16 rows × 64 i8 (K=64), a VNNI B tile is (K/4)=16 rows × 4·N
// i8, and the C tile is 16×16 i32 — all 16 rows × 64 bytes. The int8 dot folds
// 4 K-values per i32 lane, so K per tile is 64 (vs 32 for bf16) and B is
// 4-packed (vs 2-packed). WRITE-BLIND (no AMX hardware here); self-check-gated.
// ============================================================

/// int8 A/B/C tile dims (A 16×64 i8, B VNNI (K/4)×(4·N) i8, C 16×16 i32).
const TK_I8: usize = 64;

/// AMX-INT8 presence: AMX-TILE (CPUID.7.0:EDX[24]) + AMX-INT8 (EDX[25]) + OS
/// XTILEDATA (XCR0 17/18), same gating as [`amx_cpu_supported`].
fn amx_int8_supported() -> bool {
    use std::arch::x86_64::__cpuid_count;
    let leaf7 = unsafe { __cpuid_count(7, 0) };
    let amx_tile = (leaf7.edx >> 24) & 1 != 0;
    let amx_int8 = (leaf7.edx >> 25) & 1 != 0;
    if !(amx_tile && amx_int8) {
        return false;
    }
    let leaf1 = unsafe { __cpuid_count(1, 0) };
    if (leaf1.ecx >> 27) & 1 == 0 {
        return false;
    }
    let xcr0 = unsafe { read_xcr0() };
    (xcr0 >> 17) & 0b11 == 0b11
}

/// Pack a TM×TK_I8 i8 A-tile (row-major) from `a[M,K]` at block (mb, kb),
/// zero-padding the M/K remainders.
fn pack_a_i8(a: &[i8], a_buf: &mut [i8; TM * TK_I8], m: usize, k: usize, mb: usize, kb: usize) {
    a_buf.fill(0);
    let mrows = (m - mb).min(TM);
    let kcols = (k - kb).min(TK_I8);
    for i in 0..mrows {
        let src = &a[(mb + i) * k..];
        let dst = &mut a_buf[i * TK_I8..];
        for p in 0..kcols {
            dst[p] = src[kb + p];
        }
    }
}

/// Pack a VNNI int8 B-tile from `b[K,N]` (row-major) at block (kb, nb): row `kk`
/// holds, per output column n, the 4 consecutive K-values b[kb+4kk+{0..4}][n].
fn pack_b_i8_vnni(b: &[i8], b_buf: &mut [i8; (TK_I8 / 4) * (4 * TN)], n: usize, k: usize, kb: usize, nb: usize) {
    b_buf.fill(0);
    let ncols = (n - nb).min(TN);
    let kcols = (k - kb).min(TK_I8);
    for kk in 0..(TK_I8 / 4) {
        let row = &mut b_buf[kk * (4 * TN)..];
        for j in 0..ncols {
            for r in 0..4 {
                let ki = 4 * kk + r;
                if ki < kcols {
                    row[j * 4 + r] = b[(kb + ki) * n + (nb + j)];
                }
            }
        }
    }
}

/// One AMX-INT8 block: `C_tile[16,16] i32 += A_tile · B_tile` (tmm0/1/2). The
/// caller has run `ldtilecfg` + `tilezero tmm0`.
///
/// SAFETY: AMX present + XTILEDATA permitted; ptrs point at 16×64-byte tiles.
#[inline]
unsafe fn amx_block_mul_i8(a_ptr: *const i8, b_ptr: *const i8) {
    use std::arch::asm;
    let stride: usize = 64;
    asm!(
        "tileloadd tmm1, [{a} + {s}]",
        "tileloadd tmm2, [{b} + {s}]",
        "tdpbssd tmm0, tmm1, tmm2",
        a = in(reg) a_ptr,
        b = in(reg) b_ptr,
        s = in(reg) stride,
        options(nostack, preserves_flags)
    );
}

/// AMX-INT8 GEMM: `c[M,N] i32 = a[M,K] i8 · b[K,N] i8`. The reusable int8 tile
/// primitive; a Q8_0 W8A8 prefill wrapper (dequant block scales + quantize the
/// activation) layers on top — a follow-up, since the quant prefill path is
/// matvec-shaped today.
///
/// SAFETY: caller ensures AMX-INT8 is usable ([`amx_i8_gemm_usable`]); shapes
/// satisfy `a.len()==m*k`, `b.len()==k*n`, `c.len()==m*n`.
// Not yet wired into a quant prefill path (follow-up); kept available.
#[allow(dead_code)]
pub unsafe fn gemm_i8_amx(a: &[i8], b: &[i8], c: &mut [i32], m: usize, n: usize, k: usize) {
    use std::arch::asm;
    let cfg = TileConfig::new();
    asm!("ldtilecfg [{c}]", c = in(reg) &cfg as *const TileConfig, options(nostack, readonly, preserves_flags));
    let mut a_buf = [0i8; TM * TK_I8];
    let mut b_buf = [0i8; (TK_I8 / 4) * (4 * TN)];
    let mut c_buf = [0i32; TM * TN];
    let stride: usize = 64;
    let mut mb = 0;
    while mb < m {
        let mrows = (m - mb).min(TM);
        let mut nb = 0;
        while nb < n {
            let ncols = (n - nb).min(TN);
            asm!("tilezero tmm0", options(nostack, preserves_flags));
            let mut kb = 0;
            while kb < k {
                pack_a_i8(a, &mut a_buf, m, k, mb, kb);
                pack_b_i8_vnni(b, &mut b_buf, n, k, kb, nb);
                amx_block_mul_i8(a_buf.as_ptr(), b_buf.as_ptr());
                kb += TK_I8;
            }
            asm!(
                "tilestored [{c} + {s}], tmm0",
                c = in(reg) c_buf.as_mut_ptr(),
                s = in(reg) stride,
                options(nostack)
            );
            for i in 0..mrows {
                for j in 0..ncols {
                    c[(mb + i) * n + (nb + j)] = c_buf[i * TN + j];
                }
            }
            nb += TN;
        }
        mb += TM;
    }
    asm!("tilerelease", options(nostack, preserves_flags));
}

/// Scalar int8 GEMM reference for the self-check (exact integer math).
fn gemm_i8_ref(a: &[i8], b: &[i8], c: &mut [i32], m: usize, n: usize, k: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0i32;
            for p in 0..k {
                acc += a[i * k + p] as i32 * b[p * n + j] as i32;
            }
            c[i * n + j] = acc;
        }
    }
}

/// Whether the AMX-INT8 GEMM is usable — cached. True only when the CPU has
/// AMX-INT8, the OS granted tile permission, AND the kernel reproduces the
/// scalar int8 reference EXACTLY on a probe (integer math is exact → a mismatch
/// is a wrong encoding, fail closed).
#[allow(dead_code)]
pub fn amx_i8_gemm_usable() -> bool {
    static USABLE: OnceLock<bool> = OnceLock::new();
    *USABLE.get_or_init(|| {
        if !amx_int8_supported() || !request_amx_tiledata() {
            return false;
        }
        // Probe crossing M/N/K tiles (M=20, N=20, K=130 → tiles + remainders).
        let (m, n, k) = (20usize, 20usize, 130usize);
        let a: Vec<i8> = (0..m * k).map(|i| (((i as i64 * 2_654_435_761) % 255) - 127) as i8).collect();
        let b: Vec<i8> = (0..k * n).map(|i| (((i as i64 * 40_503) % 255) - 127) as i8).collect();
        let mut got = vec![0i32; m * n];
        let mut want = vec![0i32; m * n];
        gemm_i8_ref(&a, &b, &mut want, m, n, k);
        // SAFETY: AMX-INT8 present + permitted above; shapes match.
        unsafe { gemm_i8_amx(&a, &b, &mut got, m, n, k) };
        got == want
    })
}

// ============================================================
// AMX-INT8 W8A8 Q8_0 PREFILL GEMM — wires the int8 tile primitive above into the
// batched (prefill) matvec→GEMM path (`matvec_tensor_batched`). Q8_0's weight IS
// int8 (losslessly: 32 i8 quants + one f16 block scale per 32-block), so "W8" is
// EXACT; the activation is quantized to int8 per 32-block (the SAME scheme as the
// VNNI decode path, `vnni::quantize_activation_i8`), so the only loss is the
// activation quant — hence LOSSY, OPT-IN (reuses `vnni::set_vnni_w8a8`, the
// existing W8A8 switch) and additionally self-check-gated. WRITE-BLIND (no
// AMX-INT8 on the author's box): off AMX the self-check returns false and the
// caller runs the AVX/scalar per-row matvec — BYTE-IDENTICAL. Validated on an AMX
// host by the parity test below + the opt-in A/B.
//
// A Q8_0 block's int8 dot is 32-wide; AMX-INT8's K-tile is 64, so each block is
// zero-padded into the low 32 K-lanes. Per-BLOCK (not per-64-K-tile) accumulation
// is required because the scale `d[row]·xscale[token]` varies per k-block — the
// int32 tile dot is scaled block-by-block into an f32 accumulator, exactly the
// order `vnni::matvec_q8_0_w8a8_scalar` sums, so the two agree to int rounding
// (the self-check grades against it).
// ============================================================

/// AMX-INT8 W8A8 Q8_0 prefill GEMM: `out[t*m + i] = Σ_b d[i,b]·xscale[t,b]·(Σ_j
/// wq[i,b,j]·xq[t,b,j])` over 32-blocks `b`, output rows `i in 0..m`, token rows
/// `t in 0..n_rows`. `w_bytes` is Q8_0-packed `[m × (k/32) × 34]`; `xq`/`xscales`
/// are the pre-quantized activation (`[n_rows × k]` i8 + `[n_rows × k/32]` f32
/// per-block scale) from [`crate::vnni::quantize_activation_i8`]. `out` is
/// `[n_rows × m]` f32 (row-major per token — the layout the per-row matvec writes).
///
/// SAFETY: caller confirmed [`amx_q8_0_prefill_usable`] on THIS thread (AMX-INT8
/// present + OS-permitted + self-checked); shapes satisfy `k % 32 == 0`,
/// `w_bytes.len() == m*(k/32)*34`, `xq.len() == n_rows*k`,
/// `xscales.len() == n_rows*(k/32)`, `out.len() == n_rows*m`.
pub unsafe fn gemm_q8_0_w8a8_amx(
    w_bytes: &[u8],
    xq: &[i8],
    xscales: &[f32],
    out: &mut [f32],
    m: usize,
    n_rows: usize,
    k: usize,
) {
    use std::arch::asm;
    const QK: usize = 32;
    const BLOCK_BYTES: usize = 34;
    let blocks = k / QK;

    let cfg = TileConfig::new();
    // SAFETY: 64-byte aligned payload; ldtilecfg reads 64 bytes.
    asm!("ldtilecfg [{c}]", c = in(reg) &cfg as *const TileConfig, options(nostack, readonly, preserves_flags));

    // tmm1 = weight A tile (16 rows × 64 i8, block padded 32→64), tmm2 = VNNI
    // activation B tile, tmm0 = the int32 C accumulator tile.
    let mut a_buf = [0i8; TM * TK_I8];
    let mut b_buf = [0i8; (TK_I8 / 4) * (4 * TN)];
    let mut c_buf = [0i32; TM * TN];
    let stride: usize = 64;

    let mut mb = 0;
    while mb < m {
        let mrows = (m - mb).min(TM);
        let mut nb = 0;
        while nb < n_rows {
            let ncols = (n_rows - nb).min(TN);
            // f32 accumulators for this (≤16 weight rows × ≤16 tokens) output block.
            let mut acc = [0f32; TM * TN];
            for b in 0..blocks {
                // Weight A tile: rows mb..mb+mrows, this block's 32 i8 quants into
                // the low 32 K-lanes (plain row-major), 32..64 left zero.
                a_buf.fill(0);
                for i in 0..mrows {
                    let off = (mb + i) * blocks * BLOCK_BYTES + b * BLOCK_BYTES + 2;
                    let dst = &mut a_buf[i * TK_I8..];
                    for j in 0..QK {
                        dst[j] = w_bytes[off + j] as i8;
                    }
                }
                // Activation B tile (VNNI): token cols nb..nb+ncols, this block's
                // 32 i8 (4 consecutive K per VNNI row; kk in 0..8 valid, rest zero).
                b_buf.fill(0);
                for kk in 0..(TK_I8 / 4) {
                    let row = &mut b_buf[kk * (4 * TN)..];
                    for jcol in 0..ncols {
                        for r in 0..4 {
                            let ki = 4 * kk + r;
                            if ki < QK {
                                row[jcol * 4 + r] = xq[(nb + jcol) * k + b * QK + ki];
                            }
                        }
                    }
                }
                asm!("tilezero tmm0", options(nostack, preserves_flags));
                amx_block_mul_i8(a_buf.as_ptr(), b_buf.as_ptr());
                asm!(
                    "tilestored [{c} + {s}], tmm0",
                    c = in(reg) c_buf.as_mut_ptr(),
                    s = in(reg) stride,
                    options(nostack)
                );
                // Scale this block's int32 dots by d[row]·xscale[token] into the
                // f32 accumulator (per-block, matching the scalar reference order).
                for i in 0..mrows {
                    let off = (mb + i) * blocks * BLOCK_BYTES + b * BLOCK_BYTES;
                    let d = f16::from_le_bytes([w_bytes[off], w_bytes[off + 1]]).to_f32();
                    for t in 0..ncols {
                        let p = c_buf[i * TN + t];
                        acc[i * TN + t] += d * xscales[(nb + t) * blocks + b] * p as f32;
                    }
                }
            }
            // Scatter the valid sub-block to the per-token output rows.
            for i in 0..mrows {
                for t in 0..ncols {
                    out[(nb + t) * m + (mb + i)] = acc[i * TN + t];
                }
            }
            nb += TN;
        }
        mb += TM;
    }
    asm!("tilerelease", options(nostack, preserves_flags));
}

/// Whether the AMX-INT8 W8A8 Q8_0 prefill GEMM is usable — cached. True only when
/// the CPU has AMX-INT8, the OS granted tile permission, AND the kernel reproduces
/// the scalar W8A8 reference ([`crate::vnni::matvec_q8_0_w8a8_scalar`]) on a probe
/// (same activation quant → a mismatch is a wrong tile encoding, fail closed). Any
/// failure / non-AMX CPU returns false and NO tile instruction runs afterward.
pub fn amx_q8_0_prefill_usable() -> bool {
    static USABLE: OnceLock<bool> = OnceLock::new();
    *USABLE.get_or_init(|| {
        if !amx_int8_supported() || !request_amx_tiledata() {
            return false;
        }
        // Probe crossing M/N/K tiles + remainders: m=20, n_rows=20, k=128 (4
        // blocks → 2×2 row/token tiles, 4 per-block tile ops with M/N remainders).
        let (m, n_rows, k) = (20usize, 20usize, 128usize);
        let blocks = k / 32;
        let w = crate::vnni::synth_q8_0(m, k, 2_654_435_761);
        let mut xq = vec![0i8; n_rows * k];
        let mut xscales = vec![0f32; n_rows * blocks];
        for t in 0..n_rows {
            let x: Vec<f32> = (0..k)
                .map(|j| ((((t * k + j) as i64 * 40_503) % 211) as f32 - 105.0) * 0.01)
                .collect();
            crate::vnni::quantize_activation_i8(
                &x,
                k,
                &mut xq[t * k..(t + 1) * k],
                &mut xscales[t * blocks..(t + 1) * blocks],
            );
        }
        let mut got = vec![0f32; n_rows * m];
        // SAFETY: AMX-INT8 present + permitted above; shapes match.
        unsafe { gemm_q8_0_w8a8_amx(&w, &xq, &xscales, &mut got, m, n_rows, k) };
        for t in 0..n_rows {
            let mut want = vec![0f32; m];
            crate::vnni::matvec_q8_0_w8a8_scalar(
                &w,
                &xq[t * k..(t + 1) * k],
                &xscales[t * blocks..(t + 1) * blocks],
                &mut want,
                m,
                k,
            );
            for i in 0..m {
                let g = got[t * m + i];
                let denom = want[i].abs().max(1e-4);
                if (g - want[i]).abs() / denom > 0.02 {
                    return false;
                }
            }
        }
        true
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn amx_probe_matches_reference_or_skips() {
        // On a non-AMX host this just returns false (no tile instruction runs).
        // On an AMX host the self-check must pass (the kernel == bf16 ref).
        let usable = amx_gemm_usable();
        if !usable {
            eprintln!("amx: not usable on this host (expected off AMX hardware) — SKIP");
            return;
        }
        // Re-run a fresh parity check at a different shape to confirm.
        let (m, n, k) = (33usize, 17usize, 64usize);
        let w: Vec<f16> = (0..m * k).map(|i| f16::from_f32(((i % 7) as f32 - 3.0) * 0.1)).collect();
        let x: Vec<f32> = (0..k * n).map(|i| ((i % 5) as f32 - 2.0) * 0.05).collect();
        let mut got = vec![0f32; m * n];
        let mut want = vec![0f32; m * n];
        gemm_bf16_ref(&w, &x, &mut want, m, n, k);
        unsafe { gemm_bf16_amx(&w, &x, &mut got, m, n, k) };
        for (a, b) in got.iter().zip(&want) {
            let denom = b.abs().max(1e-4);
            assert!((a - b).abs() / denom <= 0.02, "amx {a} vs ref {b}");
        }
    }

    #[test]
    fn amx_int8_probe_matches_reference_or_skips() {
        if !amx_i8_gemm_usable() {
            eprintln!("amx-int8: not usable on this host (expected off AMX-INT8 hw) — SKIP");
            return;
        }
        // Fresh exact-int parity at a different shape.
        let (m, n, k) = (33usize, 17usize, 128usize);
        let a: Vec<i8> = (0..m * k).map(|i| ((i % 251) as i32 - 125) as i8).collect();
        let b: Vec<i8> = (0..k * n).map(|i| ((i % 247) as i32 - 123) as i8).collect();
        let mut got = vec![0i32; m * n];
        let mut want = vec![0i32; m * n];
        gemm_i8_ref(&a, &b, &mut want, m, n, k);
        unsafe { gemm_i8_amx(&a, &b, &mut got, m, n, k) };
        assert_eq!(got, want, "AMX-INT8 GEMM != scalar int8 reference");
    }

    /// The AMX W8A8 Q8_0 prefill GEMM must reproduce the scalar W8A8 reference
    /// (one row per token) within int rounding. On a non-AMX-INT8 host the
    /// self-check returns false (no tile runs) and this SKIPs — matching the
    /// write-blind, fail-closed-to-AVX contract. On an AMX host it must pass.
    #[test]
    fn amx_q8_0_w8a8_prefill_matches_scalar_or_skips() {
        if !amx_q8_0_prefill_usable() {
            eprintln!("amx-int8 Q8_0 prefill: not usable on this host (expected off AMX-INT8 hw) — SKIP");
            return;
        }
        // Fresh shapes crossing tiles + M/N/K remainders (k multiple of 32).
        let (m, n_rows, k) = (37usize, 23usize, 160usize);
        let blocks = k / 32;
        let w = crate::vnni::synth_q8_0(m, k, 99_173);
        let mut xq = vec![0i8; n_rows * k];
        let mut xscales = vec![0f32; n_rows * blocks];
        for t in 0..n_rows {
            let x: Vec<f32> = (0..k)
                .map(|j| ((((t * 7 + j * 3 + 1) % 97) as f32) - 48.0) * 0.02)
                .collect();
            crate::vnni::quantize_activation_i8(
                &x,
                k,
                &mut xq[t * k..(t + 1) * k],
                &mut xscales[t * blocks..(t + 1) * blocks],
            );
        }
        let mut got = vec![0f32; n_rows * m];
        unsafe { gemm_q8_0_w8a8_amx(&w, &xq, &xscales, &mut got, m, n_rows, k) };
        for t in 0..n_rows {
            let mut want = vec![0f32; m];
            crate::vnni::matvec_q8_0_w8a8_scalar(
                &w,
                &xq[t * k..(t + 1) * k],
                &xscales[t * blocks..(t + 1) * blocks],
                &mut want,
                m,
                k,
            );
            for i in 0..m {
                let g = got[t * m + i];
                let denom = want[i].abs().max(1e-4);
                assert!(
                    (g - want[i]).abs() / denom <= 0.02,
                    "amx q8_0 prefill t={t} row={i}: {g} vs ref {}",
                    want[i]
                );
            }
        }
    }
}
