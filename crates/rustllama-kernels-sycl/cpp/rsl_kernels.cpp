// rustllama SYCL kernel translation unit.
//
// Phase 3 ships the first real GPU kernel: `rsl_gemm_f16` — a naive
// tiled F16 GEMM. The other entry points still stub in this phase;
// they fill in incrementally as the model forward pass moves piece-by-
// piece onto the GPU. Compiles to a no-op when SYCL isn't available
// (so non-Intel build hosts can still link without SYCL headers).
//
// Build:
//   `crates/rustllama-kernels-sycl/build.rs` invokes Intel `icx`:
//      icx -fsycl -fsycl-targets=spir64_gen -O3 -DRUSTLLAMA_HAS_SYCL cpp/rsl_kernels.cpp
//   When `RUSTLLAMA_HAS_SYCL` is undefined (no Intel oneAPI toolchain)
//   the stubs below compile under a stock C++ toolchain, which keeps
//   the build working without SYCL headers.

#include "../include/rsl_kernels.h"

#include <algorithm>
#include <cmath> // INFINITY — MSVC leaks it via <windows.h>; clang/icpx needs this
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <sstream>
#include <string>

// Stash the most-recent USM allocation diagnostic. Set on every
// failed `sycl::malloc_shared`; consumed (and cleared) by the Rust
// side via `rsl_consume_last_usm_alloc_diag` so a diagnostic event
// surfaces in the engine's tracing log. Defined here so both the
// allocator and the consume-fn link to the same instance.
std::string g_last_usm_alloc_diag;

#if !defined(RUSTLLAMA_HAS_SYCL)
#error "rustllama-kernels-sycl is real-only; build with icx/icpx -fsycl -DRUSTLLAMA_HAS_SYCL (see build.rs)."
#endif
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>

// IQ-family codebook tables, generated from rustllama-gguf at
// build time (see kernels-sycl/build.rs `generate_iq_grids_inl`).
// Pulled in via the build script's `-I$OUT_DIR` include path.
#include "iq_grids.inl"

#ifdef _WIN32
#include <windows.h>
#endif

struct rsl_stream {
    sycl::queue q;
    // Copies of the queue's device + context, kept by value so the
    // interop accessors below have stable addresses to hand out. The
    // SYCL runtime copies devices/contexts cheaply (they're
    // reference-counted handles), so the extra fields cost ~16 bytes.
    // Required because `sycl::queue::get_device()` / `get_context()`
    // return BY VALUE, which means no stable address to take a
    // pointer of — and oneDNN's SYCL interop functions need raw
    // `sycl::device*` / `sycl::context*` pointers.
    sycl::device dev;
    sycl::context ctx;
    // Reusable device-USM scratch for embedding-lookup token IDs.
    // Sized lazily on first use; grows monotonically to the largest
    // batch seen this run. Avoids the per-call `malloc_device` +
    // `wait()` round-trip that dominated decode (n_ids=1 per token).
    // Freed in `rsl_stream_destroy` below.
    int32_t* embed_ids_scratch = nullptr;
    std::size_t embed_ids_capacity = 0;
};

namespace {

// Convert a u16 IEEE-754 binary16 bit pattern to f32. SYCL's
// `sycl::half` lacks a direct from_bits, so we round-trip via
// memcpy onto a sycl::half-shaped storage. Used only on the device
// side to load A/B values.
inline float bits_to_f32(uint16_t bits) {
    sycl::half h;
    std::memcpy(&h, &bits, sizeof(h));
    return static_cast<float>(h);
}

inline uint16_t f32_to_bits(float v) {
    sycl::half h = static_cast<sycl::half>(v);
    uint16_t bits;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}

// Local work-group size for our 1D kernels. Hand-picked instead of
// letting SYCL auto-derive from `MAX_WORK_GROUPS_3D`, because the
// Iris Xe / Tiger Lake compute-runtime (driver `32.0.101.7076` and
// also `32.0.101.7xxx` from March 2026) reports the limit as
// `{u32::MAX, u32::MAX, u32::MAX}`, which makes the SYCL runtime
// over-allocate / crash when sizing the work-group internally.
//
// 64 is the Intel iGPU sweet spot: matches the typical sub-group
// size on Gen12LP EUs, fits multiple work-groups per slice, and
// works on every Intel GPU we care about (Iris Xe / Arc / Lunar
// Lake). The autotuner (phase 5) will sweep this per `(device,
// kernel, problem-shape)` once we have correctness baselines.
constexpr std::size_t RSL_LWS = 64;

// Pad a logical work-item count up to a multiple of `RSL_LWS` so
// it's a valid nd_range global size. The kernel body then bounds-
// checks `get_global_id(0) >= N` and bails on the tail items.
constexpr std::size_t round_up_to_lws(std::size_t n) {
    return ((n + RSL_LWS - 1) / RSL_LWS) * RSL_LWS;
}

// Per-thread error state. The catch handlers in the
// `RSL_FFI_BODY_*` macros increment this counter and copy the
// exception message into `g_last_error`. Rust consumes both via
// `rsl_consume_error_count` + `rsl_get_last_error_message`
// after every FFI call so a swallowed exception turns into a
// SyclError on the Rust side, which the engine treats as
// "kernel failed — fall back to CPU" rather than silently
// producing garbage output.
thread_local int g_rsl_error_count = 0;
thread_local std::string g_rsl_last_error;

// Side channel for surfacing the *exact* `ze_result_t` from a
// failed `zeMemAllocHost` call inside the L0 import path. The
// public return value of `rsl_try_import_win32_handle_as_usm`
// stays at our 0-4 category code so the Rust enum stays stable;
// the L0 code goes here for diagnostics. Rust drains it via
// `rsl_consume_last_l0_import_code` after each call.
thread_local uint32_t g_rsl_last_l0_import_code = 0;

// Log a SYCL/C++ exception to stderr, record it for Rust to
// consume, and swallow it. Used by the `RSL_FFI_*` macros below
// so a thrown `sycl::exception` (OOM, kernel-launch failure,
// JIT error, etc.) never escapes an `extern "C"` boundary into
// Rust — which would otherwise abort the process with
// `fatal runtime error: Rust cannot catch foreign exceptions`.
inline void log_ffi_exception(const char* fn, const char* what) {
    std::fprintf(stderr, "[rsl-sycl] %s threw: %s\n", fn, what);
    std::fflush(stderr);
    ++g_rsl_error_count;
    // Best-effort string copy; if `string::operator=` itself
    // throws (allocator OOM in an OOM scenario) we just leave
    // the previous message in place. The counter is what Rust
    // gates on; the message is only diagnostic.
    try {
        g_rsl_last_error = std::string(fn) + ": " + what;
    } catch (...) {
        // ignore — counter is enough to signal failure
    }
}

// ------------------------------------------------------------
// Quantized-KV dequant helpers (device-side). Byte-exact ports of
// the CPU reference dequant in rustllama-kernels-cpu
// (q4_0_kv.rs / nvfp4.rs / turboquant.rs). Each writes one
// dequantized KV row of `head_dim` f32 into `row`. Used by the
// quantized-KV FlashAttention kernels below, which dequantize a
// whole K (then V) row per kv position — same element order as the
// CPU kernels' dequant-into-scratch + online_softmax_attn_f32_scratch
// so `doctor --sycl-parity` matches within tolerance.
// ------------------------------------------------------------

// Cap on head_dim so the per-work-item dequant buffer is fixed-size
// (private memory). The kernels reject larger head_dim so the caller
// falls back to the CPU kernel.
constexpr int RSL_FLASH_MAX_HEAD_DIM = 256;

// Q4_0 KV row: 18 B / 32 elems, embedded f16 scale;
// weight = (nibble - 8) * d.
inline void deq_q4_0_row(const uint8_t* p, float* row, int head_dim) {
    int nb = head_dim / 32;
    for (int b = 0; b < nb; ++b) {
        const uint8_t* blk = p + b * 18;
        uint16_t d_bits = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
        float d = bits_to_f32(d_bits);
        const uint8_t* qs = blk + 2;
        int o = b * 32;
        for (int j = 0; j < 16; ++j) {
            row[o + j]      = (float)((int)(qs[j] & 0x0F) - 8) * d;
            row[o + j + 16] = (float)((int)(qs[j] >> 4) - 8) * d;
        }
    }
}

// FP8 E4M3 -> f32 (byte-exact port of nvfp4::e4m3_to_f32). Named
// distinctly from the NVFP4 matvec TU's `e4m3_to_f32_dev` (same
// anonymous namespace) to avoid a redefinition; uses `ldexp` (exact
// 2^e scaling) to match the CPU reference's `powi(2, e)` bit-for-bit.
// The 0x7F/0xFF NaN slot never occurs for real KV scales.
inline float rsl_flash_e4m3_to_f32(uint8_t b) {
    bool sign = (b & 0x80) != 0;
    int exp = (b >> 3) & 0x0F;
    int mant = b & 0x07;
    float val;
    if (exp == 0x0F && mant == 0x07) {
        val = sycl::nan(0u);
    } else if (exp == 0) {
        val = (float)mant * (1.0f / 512.0f);  // subnormal: mant * 2^-9
    } else {
        float m = 1.0f + (float)mant / 8.0f;  // normal: (1 + mant/8) * 2^(exp-7)
        val = sycl::ldexp(m, exp - 7);
    }
    return sign ? -val : val;
}

// NVFP4 KV row: 9 B / 16 elems, interleaved nibbles (lo->2j, hi->2j+1)
// + FP8 E4M3 scale byte. Codebook = E2M1 signed table.
inline void deq_nvfp4_row(const uint8_t* p, float* row, int head_dim) {
    const float codebook[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    int nb = head_dim / 16;
    for (int b = 0; b < nb; ++b) {
        const uint8_t* blk = p + b * 9;
        float scale = rsl_flash_e4m3_to_f32(blk[8]);
        int o = b * 16;
        for (int j = 0; j < 8; ++j) {
            uint8_t byte = blk[j];
            int lo = byte & 0x0F;
            int hi = (byte >> 4) & 0x0F;
            row[o + j * 2]     = codebook[lo] * scale;
            row[o + j * 2 + 1] = codebook[hi] * scale;
        }
    }
}

// TurboQuant KV row: unpack signed codes (bits in {1,2,4,8}), multiply
// by the row's separate scale, then inverse WHT (forward butterfly +
// 1/N). Matches turboquant::dequantize_row's op order (scale BEFORE the
// transform). head_dim must be a power of two.
inline void deq_tq_row(const uint8_t* p, float scale, int bits,
                       float* row, int head_dim) {
    int max_level = (bits == 1) ? 1 : ((bits == 2) ? 1 : ((bits == 4) ? 7 : 127));
    unsigned mask = (bits == 1) ? 0x1u : ((bits == 2) ? 0x3u : ((bits == 4) ? 0xFu : 0xFFu));
    for (int i = 0; i < head_dim; ++i) {
        int bit_off = i * bits;
        int byte_idx = bit_off >> 3;
        int bit_in = bit_off & 7;
        unsigned cell = (unsigned)p[byte_idx] >> bit_in;
        if (bit_in + bits > 8) {
            int spill = bit_in + bits - 8;
            int shift = bits - spill;
            cell |= (unsigned)p[byte_idx + 1] << shift;
        }
        unsigned u = cell & mask;
        int code = (bits == 1) ? ((u == 0) ? -1 : 1) : ((int)u - max_level);
        row[i] = (float)code * scale;
    }
    // Inverse WHT = forward WHT (butterfly) then * (1/N).
    for (int h = 1; h < head_dim; h <<= 1) {
        for (int i = 0; i < head_dim; i += (h << 1)) {
            for (int j = i; j < i + h; ++j) {
                float a = row[j];
                float bqv = row[j + h];
                row[j]     = a + bqv;
                row[j + h] = a - bqv;
            }
        }
    }
    float inv_n = 1.0f / (float)head_dim;
    for (int i = 0; i < head_dim; ++i) row[i] *= inv_n;
}

}  // namespace

// Wrap a function body in `try { ... } catch` so any SYCL or std
// exception is logged + swallowed instead of unwinding past the
// C ABI. `RSL_FFI_BODY_VOID(name, ...)` is for `void`-returning
// entry points; `RSL_FFI_BODY_RET(name, default, ...)` is for the
// few that return a value (returns `default` on exception).
//
// Usage:
//   void rsl_foo(...) RSL_FFI_BODY_VOID("rsl_foo", { ...body... })
//   int  rsl_bar(...) RSL_FFI_BODY_RET("rsl_bar", -1, { ...body... })
#define RSL_FFI_BODY_VOID(fn_name, ...)                                 \
    {                                                                    \
        /* Clear any stale error from a prior consume-less caller so   */ \
        /* the post-call `rsl_consume_error_count` reflects only THIS   */ \
        /* call's outcome. */                                            \
        g_rsl_error_count = 0;                                           \
        try __VA_ARGS__                                                  \
        catch (const std::exception& e) { log_ffi_exception(fn_name, e.what()); } \
        catch (...) { log_ffi_exception(fn_name, "unknown C++ exception"); } \
    }

#define RSL_FFI_BODY_RET(fn_name, default_val, ...)                     \
    {                                                                    \
        g_rsl_error_count = 0;                                           \
        try __VA_ARGS__                                                  \
        catch (const std::exception& e) { log_ffi_exception(fn_name, e.what()); return default_val; } \
        catch (...) { log_ffi_exception(fn_name, "unknown C++ exception"); return default_val; } \
    }


extern "C" {


rsl_stream* rsl_stream_create(int device_index) RSL_FFI_BODY_RET("rsl_stream_create", nullptr, {
    // Backend selection.
    //
    // Default policy: prefer the **Level Zero** backend over OpenCL.
    // L0 enables the USM-shared-allocation import fast-path and is
    // typically 20-50% faster on Intel iGPU/Arc for our kernel mix.
    // OpenCL is the universal fallback — kept available for drivers
    // where L0 is missing or buggy. Override via the env var
    // `RUSTLLAMA_SYCL_BACKEND`:
    //
    //   - unset / "auto" / "level_zero" → prefer L0, fall back to
    //     OpenCL with a runtime warning if no L0 GPU is visible.
    //   - "opencl"                      → force OpenCL (skip L0 even
    //     when available; useful when an L0 driver bug surfaces).
    //   - "any"                         → take the SYCL runtime's
    //     default selection (legacy behavior).
    //
    // ONEAPI_DEVICE_SELECTOR env var (Intel's official knob) is
    // honored too — if the SYCL runtime has already filtered the
    // device list, we operate on that filtered set. Our preference
    // is additive: we further-filter within whatever the runtime
    // gives us.
    const char* prefer_env = std::getenv("RUSTLLAMA_SYCL_BACKEND");
    std::string prefer = prefer_env ? std::string(prefer_env) : std::string("level_zero");
    // Normalize to lowercase + trim hyphens for matching.
    for (auto& c : prefer) { c = static_cast<char>(std::tolower(static_cast<unsigned char>(c))); }

    auto all_gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (all_gpus.empty()) {
        return nullptr;
    }

    // Partition by backend so we can prefer one without losing the
    // others as a fallback.
    std::vector<sycl::device> l0_gpus;
    std::vector<sycl::device> ocl_gpus;
    std::vector<sycl::device> other_gpus;
    for (auto& d : all_gpus) {
        auto b = d.get_backend();
        if (b == sycl::backend::ext_oneapi_level_zero) {
            l0_gpus.push_back(d);
        } else if (b == sycl::backend::opencl) {
            ocl_gpus.push_back(d);
        } else {
            other_gpus.push_back(d);
        }
    }

    // Pick the working set based on the preference + availability.
    std::vector<sycl::device>* chosen = nullptr;
    const char* chosen_label = "";
    if (prefer == "opencl") {
        if (!ocl_gpus.empty()) {
            chosen = &ocl_gpus;
            chosen_label = "opencl (forced)";
        } else if (!l0_gpus.empty()) {
            chosen = &l0_gpus;
            chosen_label = "level_zero (opencl requested but unavailable)";
        } else {
            chosen = &other_gpus;
            chosen_label = "other (opencl + l0 both unavailable)";
        }
    } else if (prefer == "any" || prefer == "default") {
        chosen = &all_gpus;
        chosen_label = "any (runtime default)";
    } else {
        // "level_zero", "l0", "auto", anything-else → prefer L0.
        if (!l0_gpus.empty()) {
            chosen = &l0_gpus;
            chosen_label = "level_zero (preferred)";
        } else if (!ocl_gpus.empty()) {
            chosen = &ocl_gpus;
            chosen_label = "opencl (level_zero unavailable, fell back)";
        } else {
            chosen = &other_gpus;
            chosen_label = "other (level_zero + opencl unavailable)";
        }
    }
    if (chosen == nullptr || chosen->empty()) {
        return nullptr;
    }
    if (device_index < 0 ||
        static_cast<std::size_t>(device_index) >= chosen->size()) {
        return nullptr;
    }

    // In-order queue: enforces FIFO ordering on submitted command
    // groups and lets the SYCL runtime use a leaner internal
    // command-pool. On Intel iGPU (Iris Xe / Arc) with oneAPI
    // 2026, out-of-order queues were observed to throw
    // `std::bad_alloc` on every `q.submit` after a few hundred
    // calls — the in-order path is the supported fast path on
    // those drivers. Our forward pass is inherently sequential
    // (every kernel `.wait()`s its predecessor anyway), so we
    // lose nothing.
    sycl::property_list props{ sycl::property::queue::in_order() };
    sycl::queue q((*chosen)[device_index], props);
    // Copy the device + context out of the queue so the interop
    // accessors below have stable addresses. SYCL device/context
    // handles are reference-counted, so the copies don't duplicate
    // underlying runtime resources — just bump refcounts.
    auto dev = q.get_device();
    auto ctx = q.get_context();
    auto* s = new (std::nothrow) rsl_stream{
        std::move(q), std::move(dev), std::move(ctx)
    };
    // Diagnostic: leave the chosen-backend label in a per-process
    // static so the Rust side's `rsl_stream_backend` log can pair it
    // with the numeric backend id. The label is set on every stream
    // creation but the log only fires once per process.
    static const char* g_backend_label = "";
    g_backend_label = chosen_label;
    (void)g_backend_label; // currently used only for future telemetry
    return s;
})

// SYCL interop accessors — return raw pointers to the queue/device/
// context stored on the `rsl_stream`. Used by `rustllama-onednn-sys`
// to construct `dnnl_engine_t` + `dnnl_stream_t` via
// `dnnl_sycl_interop_engine_create` / `dnnl_sycl_interop_stream_create`
// without exposing SYCL types across the FFI boundary.
//
// SAFETY contract:
//   - The returned pointers alias the corresponding field on
//     `*stream` and are valid for as long as `*stream` outlives them.
//   - Callers MUST NOT call `delete` on these pointers; the
//     `rsl_stream` owner does that on `rsl_stream_destroy`.
//   - Returns nullptr when `stream == nullptr` so the safe wrappers
//     can handle a None-input gracefully.
void* rsl_stream_sycl_queue(rsl_stream* stream) {
    if (!stream) return nullptr;
    return static_cast<void*>(&stream->q);
}
void* rsl_stream_sycl_device(rsl_stream* stream) {
    if (!stream) return nullptr;
    return static_cast<void*>(&stream->dev);
}
void* rsl_stream_sycl_context(rsl_stream* stream) {
    if (!stream) return nullptr;
    return static_cast<void*>(&stream->ctx);
}

// Returns the numeric `sycl::backend` enum value of the queue's
// runtime backend. Used by Rust to log via `tracing::info!` (which
// reaches `gui.log`) rather than C++ `fprintf(stderr)` (which the
// Tauri GUI process discards). Returns -1 on null input.
//
// Decoding the value Rust-side:
//   0 = host (cpu)
//   1 = opencl
//   2 = ext_oneapi_level_zero        ← L0 import path eligible
//   3 = ext_oneapi_cuda
//   4 = ext_oneapi_hip
//   5 = ext_oneapi_native_cpu
// (Underlying enum is implementation-defined; mapping comes from
// the SYCL spec + libsycl9's `backend_types.hpp`.)
int rsl_stream_backend(rsl_stream* s) RSL_FFI_BODY_RET("rsl_stream_backend", -1, {
    if (s == nullptr) return -1;
    return static_cast<int>(s->q.get_backend());
})

void rsl_stream_destroy(rsl_stream* s) RSL_FFI_BODY_VOID("rsl_stream_destroy", {
    if (s != nullptr) {
        if (s->embed_ids_scratch != nullptr) {
            sycl::free(s->embed_ids_scratch, s->q);
            s->embed_ids_scratch = nullptr;
            s->embed_ids_capacity = 0;
        }
    }
    delete s;
})

int rsl_sycl_device_count(void) RSL_FFI_BODY_RET("rsl_sycl_device_count", 0, {
    auto gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
    return static_cast<int>(gpus.size());
})

// Per-thread error state accessors used by the Rust safe wrappers
// to detect a swallowed exception after every FFI call. These are
// intentionally NOT wrapped in `RSL_FFI_BODY_*` so a failure to
// read the counter doesn't itself bump the counter and produce a
// feedback loop.
int rsl_consume_error_count(void) {
    int n = g_rsl_error_count;
    g_rsl_error_count = 0;
    return n;
}

void rsl_get_last_error_message(char* buf, int capacity) {
    if (buf == nullptr || capacity <= 0) {
        return;
    }
    const std::size_t to_copy =
        std::min<std::size_t>(g_rsl_last_error.size(),
                              static_cast<std::size_t>(capacity - 1));
    std::memcpy(buf, g_rsl_last_error.data(), to_copy);
    buf[to_copy] = '\0';
}

// Returns the most recent `zeMemAllocHost` `ze_result_t` from
// the L0 import path, then resets it to 0. Diagnostic-only; the
// public import call's return value is the stable surface.
uint32_t rsl_consume_last_l0_import_code(void) {
    uint32_t c = g_rsl_last_l0_import_code;
    g_rsl_last_l0_import_code = 0;
    return c;
}

namespace {
// Write a UTF-8 string into `buf` with capacity `cap` (includes the
// trailing NUL slot). Truncates on overflow and always NUL-terminates.
// No-op when `buf == nullptr` or `cap <= 0`.
inline void write_truncated(const std::string& src, char* buf, int cap) {
    if (buf == nullptr || cap <= 0) {
        return;
    }
    const std::size_t to_copy =
        std::min<std::size_t>(src.size(), static_cast<std::size_t>(cap - 1));
    std::memcpy(buf, src.data(), to_copy);
    buf[to_copy] = '\0';
}
}  // namespace

int rsl_sycl_device_info(int device_index,
                         char* name_out, int name_capacity,
                         char* driver_out, int driver_capacity,
                         uint32_t* vendor_id_out,
                         uint64_t* vram_bytes_out,
                         uint8_t* uuid_out,
                         uint8_t* is_integrated_out) {
    if (device_index < 0) {
        return -1;
    }
    // A zeroed UUID is the "unsupported / unknown" sentinel the Rust side
    // falls back from; fill it whenever the device exposes one.
    if (uuid_out != nullptr) {
        std::memset(uuid_out, 0, 16);
    }
    // Default 0 = discrete (dedicated VRAM); overwritten below from the
    // device's host_unified_memory flag inside the try.
    if (is_integrated_out != nullptr) {
        *is_integrated_out = 0;
    }
    auto gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (static_cast<std::size_t>(device_index) >= gpus.size()) {
        return -1;
    }
    try {
        const sycl::device& dev = gpus[device_index];
        write_truncated(dev.get_info<sycl::info::device::name>(),
                        name_out, name_capacity);
        write_truncated(dev.get_info<sycl::info::device::driver_version>(),
                        driver_out, driver_capacity);
        if (vendor_id_out != nullptr) {
            *vendor_id_out =
                static_cast<uint32_t>(dev.get_info<sycl::info::device::vendor_id>());
        }
        if (vram_bytes_out != nullptr) {
            *vram_bytes_out =
                static_cast<uint64_t>(dev.get_info<sycl::info::device::global_mem_size>());
        }
        // Integrated vs discrete. `host_unified_memory` is true for a GPU
        // whose memory is physically shared with the host (integrated iGPU
        // like Iris Xe — no separate dedicated VRAM), false for a discrete
        // GPU with its own VRAM (Intel Arc). The placement planner uses
        // this to decide whether `vram_only` can be enforced.
        // NOTE: the authoritative alternative, if host_unified_memory ever
        // proves unreliable across backends, is the Level-Zero property
        // `ZE_DEVICE_PROPERTY_FLAG_INTEGRATED` read via L0 interop
        // (`sycl::get_native<sycl::backend::ext_oneapi_level_zero>(dev)` →
        // `zeDeviceGetProperties`).
        if (is_integrated_out != nullptr) {
            *is_integrated_out =
                dev.get_info<sycl::info::device::host_unified_memory>() ? 1 : 0;
        }
        // Stable, driver-invariant device UUID (Level-Zero exposes it via the
        // Intel SYCL extension). Guarded by the aspect so it never throws on a
        // backend (e.g. OpenCL) that doesn't support it — leaving the zeroed
        // sentinel so the fingerprint falls back to vendor/name/vram.
#ifdef SYCL_EXT_INTEL_DEVICE_INFO
        if (uuid_out != nullptr &&
            dev.has(sycl::aspect::ext_intel_device_info_uuid)) {
            auto id = dev.get_info<sycl::ext::intel::info::device::uuid>();
            std::memcpy(uuid_out, id.data(),
                        std::min<std::size_t>(id.size(), 16));
        }
#endif
        return 0;
    } catch (...) {
        // Any SYCL exception during get_info — leave outputs at
        // their post-write_truncated state (could be partially
        // written if name succeeded but driver_version threw) and
        // signal failure.
        return -1;
    }
}

// USM-shared allocator. `sycl::malloc_shared` returns memory page-
// mapped to both host and device; on integrated GPUs (Iris Xe with
// shared LPDDR) this is the same physical pages, on discrete GPUs
// the runtime migrates pages on first device access.
void* rsl_usm_alloc_shared(rsl_stream* s, std::size_t n_bytes) RSL_FFI_BODY_RET("rsl_usm_alloc_shared", nullptr, {
    if (s == nullptr || n_bytes == 0) {
        return nullptr;
    }
    // First try: USM shared. On integrated GPUs (Iris Xe) shared
    // is the same physical pages as host; on discrete Arc the
    // runtime migrates on first device access.
    void* p = sycl::malloc_shared(n_bytes, s->q);
    if (p != nullptr) {
        return p;
    }
    // Diagnostic + fallback. The Intel L0 driver on Iris Xe was
    // observed to fail `malloc_shared` while OpenCL succeeded for
    // the same byte count on the same device. USM aspects vary
    // per backend: L0 may not advertise `usm_shared_allocations`
    // even when the OpenCL path does. Probe the aspect support
    // and fall back to USM host (which on integrated graphics is
    // the same physical memory — GPU + CPU both read it through
    // the same LPDDR controller).
    bool has_shared = s->dev.has(sycl::aspect::usm_shared_allocations);
    bool has_host   = s->dev.has(sycl::aspect::usm_host_allocations);
    bool has_device = s->dev.has(sycl::aspect::usm_device_allocations);
    // Stash the diagnostic in a per-process slot so the Rust side
    // can log it via tracing (C++ fprintf is invisible in the Tauri
    // GUI). Updated on each failure; the Rust side picks it up via
    // `rsl_consume_last_usm_alloc_diag`.
    {
        static std::mutex diag_mu;
        std::lock_guard<std::mutex> g(diag_mu);
        std::ostringstream os;
        os << "malloc_shared failed: bytes=" << n_bytes
           << " has_shared=" << has_shared
           << " has_host=" << has_host
           << " has_device=" << has_device;
        // The diagnostic ring is read by Rust the next time it polls
        // — keep just the most recent message to avoid unbounded
        // growth on repeated failures. `g_last_usm_alloc_diag` is the
        // file-scope std::string defined above.
        ::g_last_usm_alloc_diag = os.str();
    }
    // Host USM fallback. Returns nullptr if even this isn't
    // supported; the caller treats nullptr as "USM unavailable"
    // and falls back to the CPU dispatch path. The
    // `sycl::malloc_host` path is universally supported on every
    // SYCL implementation we ship against, so this should land.
    if (has_host) {
        return sycl::malloc_host(n_bytes, s->q);
    }
    return nullptr;
})

void rsl_usm_free(rsl_stream* s, void* ptr) RSL_FFI_BODY_VOID("rsl_usm_free", {
    if (s == nullptr || ptr == nullptr) {
        return;
    }
    sycl::free(ptr, s->q);
})

// Dedicated-VRAM weight tier: allocate device-local USM and populate
// it from a host source in one shot. `sycl::malloc_device` returns
// memory that is NOT host-accessible — on the integrated Iris Xe it
// is carved from the BIOS-reserved "dedicated" VRAM aperture (shown
// as Dedicated GPU memory), which is separate from the shared-LPDDR /
// system-RAM pool. So a weight placed here fills dedicated VRAM AND
// relieves host-RAM pressure (unlike malloc_shared, which on an iGPU
// is double-resident host RAM). Because device memory can't be
// written through a host pointer, the copy goes through an explicit
// blocking `q.memcpy` H2D (the same idiom the transient host-pointer
// kernels use). Returns the populated device pointer, or nullptr when
// device USM is unavailable / the alloc fails — the caller then falls
// back to the shared tier. Free with rsl_usm_free (sycl::free is
// tier-agnostic).
void* rsl_usm_alloc_device_from_host(rsl_stream* s, const void* src_host, std::size_t n_bytes)
        RSL_FFI_BODY_RET("rsl_usm_alloc_device_from_host", nullptr, {
    if (s == nullptr || src_host == nullptr || n_bytes == 0) {
        return nullptr;
    }
    if (!s->dev.has(sycl::aspect::usm_device_allocations)) {
        return nullptr;
    }
    void* p = sycl::malloc_device(n_bytes, s->q);
    if (p == nullptr) {
        return nullptr;
    }
    try {
        s->q.memcpy(p, src_host, n_bytes).wait();
    } catch (...) {
        // Free the device allocation before surfacing the failure so a
        // transient copy error can't leak the VRAM aperture.
        sycl::free(p, s->q);
        return nullptr;
    }
    return p;
})

// Pull the most-recent USM allocation diagnostic into `dst_buf` and
// clear it. Returns the byte length copied (excluding NUL); 0 when
// no diagnostic has been stored since the last consume call. Buffer
// is NUL-terminated when capacity allows. Single-shot so the Rust
// side can log each failure exactly once per occurrence.
int rsl_consume_last_usm_alloc_diag(char* dst_buf, int capacity) RSL_FFI_BODY_RET("rsl_consume_last_usm_alloc_diag", 0, {
    if (dst_buf == nullptr || capacity <= 0) {
        return 0;
    }
    static std::mutex diag_mu;
    std::string take;
    {
        std::lock_guard<std::mutex> g(diag_mu);
        take = std::move(::g_last_usm_alloc_diag);
        ::g_last_usm_alloc_diag.clear();
    }
    if (take.empty()) {
        dst_buf[0] = '\0';
        return 0;
    }
    int n = static_cast<int>(std::min<std::size_t>(take.size(), static_cast<std::size_t>(capacity - 1)));
    std::memcpy(dst_buf, take.data(), static_cast<std::size_t>(n));
    dst_buf[n] = '\0';
    return n;
})

// Naive tiled F16 GEMM: C[M,N] = A[M,K] @ B[K,N], row-major.
//
// Tile dimensions are conservative (16x16) so the same kernel runs on
// integrated GPUs (Iris Xe, 96 EUs, shared LPDDR) and discrete Arc
// devices (A380/A770) without retuning. The autotuner (`rustllama-tuner`,
// phase 5) will sweep tile_m/tile_n/tile_k once we have correctness here.
//
// Accumulation is f32 on-chip to avoid the precision loss that f16
// accumulation incurs over long K; the final write rounds back to f16.
void rsl_gemm_f16(rsl_stream* s,
                  const uint16_t* A, const uint16_t* B, uint16_t* C,
                  int M, int N, int K,
                  int lda, int ldb, int ldc) RSL_FFI_BODY_VOID("rsl_gemm_f16", {
    if (s == nullptr || A == nullptr || B == nullptr || C == nullptr) {
        return;
    }
    constexpr int TILE = 16;
    auto& q = s->q;

    // Allocate USM device buffers, copy in A/B, run kernel, copy out C.
    // Device-side USM keeps the kernel zero-copy on systems where the
    // GPU shares LPDDR with the CPU; on discrete Arc it falls back to
    // an explicit DMA, both handled by the SYCL runtime.
    const std::size_t a_sz = static_cast<std::size_t>(M) * K * sizeof(uint16_t);
    const std::size_t b_sz = static_cast<std::size_t>(K) * N * sizeof(uint16_t);
    const std::size_t c_sz = static_cast<std::size_t>(M) * N * sizeof(uint16_t);
    auto* a_d = sycl::malloc_device<uint16_t>(M * K, q);
    auto* b_d = sycl::malloc_device<uint16_t>(K * N, q);
    auto* c_d = sycl::malloc_device<uint16_t>(M * N, q);
    if (!a_d || !b_d || !c_d) {
        if (a_d) sycl::free(a_d, q);
        if (b_d) sycl::free(b_d, q);
        if (c_d) sycl::free(c_d, q);
        return;
    }
    q.memcpy(a_d, A, a_sz);
    q.memcpy(b_d, B, b_sz);
    q.wait();

    const int M_tiles = (M + TILE - 1) / TILE;
    const int N_tiles = (N + TILE - 1) / TILE;

    sycl::range<2> global(M_tiles * TILE, N_tiles * TILE);
    sycl::range<2> local(TILE, TILE);

    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<2>(global, local),
            [=](sycl::nd_item<2> it) {
                const int row = static_cast<int>(it.get_global_id(0));
                const int col = static_cast<int>(it.get_global_id(1));
                if (row >= M || col >= N) {
                    return;
                }
                float acc = 0.0f;
                for (int p = 0; p < K; ++p) {
                    const uint16_t a_bits = a_d[row * lda + p];
                    const uint16_t b_bits = b_d[p * ldb + col];
                    acc += bits_to_f32(a_bits) * bits_to_f32(b_bits);
                }
                c_d[row * ldc + col] = f32_to_bits(acc);
            });
    }).wait();

    q.memcpy(C, c_d, c_sz).wait();
    sycl::free(a_d, q);
    sycl::free(b_d, q);
    sycl::free(c_d, q);
})

// ----- Phase 3.1 kernels: RMSNorm, RoPE, softmax-attn, SwiGLU, embedding -----

// Per-row RMS normalization: y[r,i] = w[i] * x[r,i] / sqrt(mean(x[r,:]^2) + eps).
// One work-item per row — d-dimensional reduction inside the kernel. Suitable
// when n_rows is small (single decode position) up through a few hundred (a
// prefill batch). The reduction is sequential within a work-item, but rows are
// independent so the GPU's many EUs stay busy across them.
void rsl_rmsnorm(rsl_stream* s,
                 const uint16_t* x, const uint16_t* w, uint16_t* y,
                 int n_rows, int d, float eps) RSL_FFI_BODY_VOID("rsl_rmsnorm", {
    if (s == nullptr || x == nullptr || w == nullptr || y == nullptr) return;
    if (n_rows <= 0 || d <= 0) return;
    auto& q = s->q;
    const std::size_t total = static_cast<std::size_t>(n_rows) * d;
    auto* xd = sycl::malloc_device<uint16_t>(total, q);
    auto* wd = sycl::malloc_device<uint16_t>(d, q);
    auto* yd = sycl::malloc_device<uint16_t>(total, q);
    if (!xd || !wd || !yd) {
        if (xd) sycl::free(xd, q);
        if (wd) sycl::free(wd, q);
        if (yd) sycl::free(yd, q);
        return;
    }
    q.memcpy(xd, x, total * sizeof(uint16_t));
    q.memcpy(wd, w, d * sizeof(uint16_t));
    q.wait();
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(round_up_to_lws(n_rows)),
                              sycl::range<1>(RSL_LWS)),
            [=](sycl::nd_item<1> it) {
                const int r = static_cast<int>(it.get_global_id(0));
                if (r >= n_rows) return;
                const int base = r * d;
                float sum_sq = 0.0f;
                for (int i = 0; i < d; ++i) {
                    float v = bits_to_f32(xd[base + i]);
                    sum_sq += v * v;
                }
                float scale = 1.0f / sycl::sqrt(sum_sq / static_cast<float>(d) + eps);
                for (int i = 0; i < d; ++i) {
                    float v = bits_to_f32(xd[base + i]);
                    float wv = bits_to_f32(wd[i]);
                    yd[base + i] = f32_to_bits(v * scale * wv);
                }
            });
    }).wait();
    q.memcpy(y, yd, total * sizeof(uint16_t)).wait();
    sycl::free(xd, q);
    sycl::free(wd, q);
    sycl::free(yd, q);
})

// USM-resident FlashAttention decode. Fuses Q·Kᵀ, softmax, and ·V
// via the online-softmax recurrence. One work-item per head:
// walks kv_len positions sequentially, maintains running max/sum/
// output. The cross-`t` dependency rules out parallelism inside a
// single head, but the n_heads-wide outer loop saturates the
// device on typical configs (n_heads = 32+ for 7B-class models).
//
// Mirrors the CPU `gqa_attention_flash_decode` algorithm exactly
// so bit-for-bit parity holds (up to float reduction order).
void rsl_flash_attn_decode_usm(rsl_stream* s,
                               const uint16_t* q_usm,
                               const uint16_t* k_usm,
                               const uint16_t* v_usm,
                               uint16_t* out_usm,
                               int n_heads, int n_kv_heads,
                               int head_dim, int max_ctx, int kv_len) RSL_FFI_BODY_VOID("rsl_flash_attn_decode_usm", {
    if (s == nullptr || q_usm == nullptr || k_usm == nullptr || v_usm == nullptr || out_usm == nullptr) return;
    if (n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0 || max_ctx <= 0 || kv_len <= 0) {
        // kv_len == 0 case: zero the output and return. We use the
        // submit pattern below to keep the device active.
        if (out_usm != nullptr && n_heads > 0 && head_dim > 0 && s != nullptr) {
            auto& q = s->q;
            std::size_t total = static_cast<std::size_t>(n_heads) * head_dim;
            q.memset(out_usm, 0, total * sizeof(uint16_t)).wait();
        }
        return;
    }
    if ((n_heads % n_kv_heads) != 0) return;
    auto& q = s->q;
    const int n_gqa = n_heads / n_kv_heads;
    const float scale = 1.0f / sycl::sqrt(static_cast<float>(head_dim));
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(round_up_to_lws(n_heads)),
                              sycl::range<1>(RSL_LWS)),
            [=](sycl::nd_item<1> it) {
                const int hh = static_cast<int>(it.get_global_id(0));
                if (hh >= n_heads) return;
                const int kv_h = hh / n_gqa;
                const int q_base = hh * head_dim;
                const int out_base = hh * head_dim;
            // Zero accumulator.
            for (int i = 0; i < head_dim; ++i) {
                out_usm[out_base + i] = 0;
            }
            float m = -INFINITY;
            float l = 0.0f;
            for (int t = 0; t < kv_len; ++t) {
                const int k_off = (kv_h * max_ctx + t) * head_dim;
                // Q · K row.
                float s_dot = 0.0f;
                for (int i = 0; i < head_dim; ++i) {
                    float qv = bits_to_f32(q_usm[q_base + i]);
                    float kv = bits_to_f32(k_usm[k_off + i]);
                    s_dot += qv * kv;
                }
                s_dot *= scale;
                // Online softmax update.
                const float m_new = sycl::fmax(m, s_dot);
                const float rescale = sycl::isfinite(m) ? sycl::exp(m - m_new) : 0.0f;
                const float p = sycl::exp(s_dot - m_new);
                l = l * rescale + p;
                // V accumulation with rescale.
                const int v_off = (kv_h * max_ctx + t) * head_dim;
                for (int i = 0; i < head_dim; ++i) {
                    const float cur = bits_to_f32(out_usm[out_base + i]);
                    const float vv = bits_to_f32(v_usm[v_off + i]);
                    out_usm[out_base + i] = f32_to_bits(cur * rescale + p * vv);
                }
                m = m_new;
            }
            const float inv_l = (l > 0.0f) ? 1.0f / l : 0.0f;
            for (int i = 0; i < head_dim; ++i) {
                const float cur = bits_to_f32(out_usm[out_base + i]);
                out_usm[out_base + i] = f32_to_bits(cur * inv_l);
            }
        });
    }).wait();
})

// ============================================================
// FlashAttention v2 (sub-group cooperation)
// ============================================================
//
// Same algorithmic shape as `rsl_flash_attn_decode_usm` (online
// softmax over a single query attending to `kv_len` cached K/V
// positions), but each attention head is processed by a *sub-group*
// of SG_SIZE work-items instead of a single WI. The head_dim is
// striped across SG lanes: lane `i` owns dimensions
// `[i, i+SG_SIZE, i+2*SG_SIZE, ...]`. The per-position Q·K dot
// product is computed by every WI contributing its partial sum,
// then `sycl::reduce_over_group` collapses to the full dot in one
// SG-wide reduction tree (`O(log SG_SIZE)` rather than the v1
// `O(head_dim)` per-WI loop).
//
// Wins over v1:
//   - Dot product is parallel across SG lanes; the linear scan
//     becomes a per-lane partial + one reduce.
//   - V accumulation is parallel across lanes (each lane updates
//     its own slice of the accumulator).
//   - Register pressure per WI drops by ~16x for the Q/output
//     storage — fewer spills on heads with `head_dim ≥ 96`.
//   - Sub-group reductions on Intel Xe map directly onto the
//     `sub_group_shuffle_*` primitives the IR exposes; the
//     compiler doesn't have to invent the reduction tree.
//
// Constraints (caller-enforced; the Rust dispatch hook falls back
// to v1 when these aren't met):
//   - `head_dim % SG_SIZE == 0` — striping requires exact
//     divisibility. SG_SIZE = 16 covers head_dim ∈ {64, 80, 96,
//     112, 128, 160, 192, 256}. head_dim ∈ {48, 88} (rare) fall
//     back.
//   - `head_dim / SG_SIZE <= MAX_DIM_PER_LANE` — caps private-
//     reg array size at compile time so the compiler can unroll
//     the per-lane inner loops. MAX_DIM_PER_LANE = 16 → head_dim
//     ≤ 256.
//   - `n_heads % n_kv_heads == 0` — same GQA invariant as v1.
//
// Layout (matches v1 exactly):
//   q:   [n_heads, head_dim]                  fp16 (USM)
//   k:   [n_kv_heads, max_ctx, head_dim]      fp16 (USM)
//   v:   [n_kv_heads, max_ctx, head_dim]      fp16 (USM)
//   out: [n_heads, head_dim]                  fp16 (USM, written)
//
// Per-WI floating-point op order differs from v1 (parallel partial
// sums with SG reduction vs. serial accumulation), so output is
// *numerically equivalent* not bit-identical. Parity tests gate
// element-wise difference at <= 1e-2 of the v1 / CPU reference.
void rsl_flash_attn_decode_v2_usm(rsl_stream* s,
                                  const uint16_t* q_usm,
                                  const uint16_t* k_usm,
                                  const uint16_t* v_usm,
                                  uint16_t* out_usm,
                                  int n_heads, int n_kv_heads,
                                  int head_dim, int max_ctx, int kv_len) RSL_FFI_BODY_VOID("rsl_flash_attn_decode_v2_usm", {
    if (s == nullptr || q_usm == nullptr || k_usm == nullptr || v_usm == nullptr || out_usm == nullptr) return;
    if (n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0 || max_ctx <= 0) return;
    if (kv_len <= 0) {
        auto& q = s->q;
        std::size_t total = static_cast<std::size_t>(n_heads) * head_dim;
        q.memset(out_usm, 0, total * sizeof(uint16_t)).wait();
        return;
    }
    if ((n_heads % n_kv_heads) != 0) return;

    constexpr int SG_SIZE = 16;
    constexpr int MAX_DIM_PER_LANE = 16;  // head_dim <= 256
    if ((head_dim % SG_SIZE) != 0) return;
    const int dim_per_lane = head_dim / SG_SIZE;
    if (dim_per_lane > MAX_DIM_PER_LANE) return;

    const int n_gqa = n_heads / n_kv_heads;
    const float scale = 1.0f / sycl::sqrt(static_cast<float>(head_dim));

    auto& q_queue = s->q;
    // Global = n_heads sub-groups × SG_SIZE WIs each. Local = SG_SIZE
    // (one sub-group per WG). Keeping HEADS_PER_WG = 1 simplifies
    // dispatch; v2.1 can pack multiple heads per WG once profiling
    // confirms that's a win on Iris Xe.
    const std::size_t global_size =
        static_cast<std::size_t>(n_heads) * static_cast<std::size_t>(SG_SIZE);
    q_queue.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(global_size),
                              sycl::range<1>(SG_SIZE)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                auto sg = it.get_sub_group();
                const int hh = static_cast<int>(it.get_group(0));
                const int lane = static_cast<int>(it.get_local_id(0));
                if (hh >= n_heads) return;
                const int kv_h = hh / n_gqa;
                const int q_base = hh * head_dim;
                const int out_base = hh * head_dim;
                const int lane_off = lane * dim_per_lane;

                // Load this lane's slice of Q into private regs.
                float q_slice[MAX_DIM_PER_LANE];
                for (int d = 0; d < dim_per_lane; ++d) {
                    q_slice[d] = bits_to_f32(q_usm[q_base + lane_off + d]);
                }
                // Output accumulator slice (private).
                float out_slice[MAX_DIM_PER_LANE];
                for (int d = 0; d < dim_per_lane; ++d) {
                    out_slice[d] = 0.0f;
                }

                float m_state = -INFINITY;
                float l_state = 0.0f;

                for (int t = 0; t < kv_len; ++t) {
                    const int kv_off = (kv_h * max_ctx + t) * head_dim;
                    // Partial dot — this lane's contribution.
                    float partial = 0.0f;
                    for (int d = 0; d < dim_per_lane; ++d) {
                        const float kv = bits_to_f32(k_usm[kv_off + lane_off + d]);
                        partial += q_slice[d] * kv;
                    }
                    // SG-wide sum → every lane sees the full dot.
                    float full_dot = sycl::reduce_over_group(
                        sg, partial, sycl::plus<float>());
                    full_dot *= scale;
                    // Online softmax update. Every lane computes the
                    // same values (deterministic; m_state/l_state
                    // start identical across lanes and update from
                    // the broadcast full_dot).
                    const float m_new = sycl::fmax(m_state, full_dot);
                    const float rescale =
                        sycl::isfinite(m_state) ? sycl::exp(m_state - m_new) : 0.0f;
                    const float p = sycl::exp(full_dot - m_new);
                    l_state = l_state * rescale + p;
                    m_state = m_new;
                    // V accumulation — each lane updates its slice.
                    for (int d = 0; d < dim_per_lane; ++d) {
                        const float vv = bits_to_f32(v_usm[kv_off + lane_off + d]);
                        out_slice[d] = out_slice[d] * rescale + p * vv;
                    }
                }

                const float inv_l = (l_state > 0.0f) ? 1.0f / l_state : 0.0f;
                for (int d = 0; d < dim_per_lane; ++d) {
                    out_usm[out_base + lane_off + d] = f32_to_bits(out_slice[d] * inv_l);
                }
            });
    }).wait();
})

// ============================================================
// FlashAttention v3 (SLM K/V tiling + sub-group cooperation)
// ============================================================
//
// Builds on v2's sub-group striping by adding *shared local memory*
// (SLM) tiles of K and V. The outer kv-position loop is chunked into
// blocks of `KV_TILE` positions; on each outer iteration the whole
// sub-group cooperatively loads the K_tile + V_tile slabs from USM
// into SLM, then the inner per-position loop reads from SLM only.
//
// Why this is a win on Intel GPUs (Iris Xe / Arc) even at
// HEADS_PER_WG=1:
//   - SLM has deterministic per-EU latency (~5 cycles) and bandwidth
//     that doesn't depend on cache state. USM reads can hit any
//     level of the cache hierarchy and stall the EU unpredictably.
//   - Loads are coalesced cooperatively: all SG_SIZE lanes load
//     adjacent fp16 elements in one transaction, which the GPU
//     vectorizes into a single 32-byte block load. The v2 kernel
//     does per-lane 8-element strided reads, which the L1 may
//     re-coalesce but not as reliably.
//   - The K and V tiles in SLM are reused across every inner-loop
//     iteration of the block, so the per-position dot product reads
//     are pure SLM (one local-memory op per element).
//
// SLM sizing: KV_TILE * head_dim * 2 bytes (K) + same for V.
// With KV_TILE=32 and head_dim<=256 → max 32KB per WG. Iris Xe has
// 64KB SLM per WG, so we have comfortable headroom. For head_dim=64
// (Llama-3-8B): 32*64*4 = 8KB. For head_dim=128 (Qwen2.5-Coder-7B):
// 16KB. For head_dim=256 (some Mistral variants): 32KB.
//
// HEADS_PER_WG=1 in this first cut: one sub-group per WG, so each
// WG processes exactly one attention head. A later v3.1 can pack
// HEADS_PER_WG = n_gqa heads into one WG (when GQA grouping allows
// it) so the K/V tile is loaded once and reused across all heads
// in the group — that's the bigger win.
//
// Constraints (same as v2, plus the SLM check at dispatch time):
//   - head_dim % SG_SIZE == 0  (SG_SIZE = 16; head_dim ∈ {16, 32,
//     48 … 256} covered)
//   - head_dim / SG_SIZE <= MAX_DIM_PER_LANE  (head_dim <= 256)
//   - n_heads % n_kv_heads == 0
//
// Numerical equivalence (not bit-identical) to v2 / v1 / CPU
// reference — same online-softmax recurrence; only the memory path
// changes.
// Templated kernel-body helper for `rsl_flash_attn_decode_v3_usm`.
// Templating on `DIM_PER_LANE` lets the compiler fully unroll the
// inner Q-dot / V-rescale loops at compile time (the most common
// values — 4 for head_dim=64, 8 for head_dim=128 — are the hot
// paths in practice). Falls back to the runtime-sized loop when the
// caller's head_dim doesn't match a specialized DIM_PER_LANE.
// KV_TILE is a compile-time constant. Templating it (E2 attempted
// a 3 × 3 sweep over DIM_PER_LANE × KV_TILE) regressed the icx
// build with "no matching function for call" for the 32/64 variants
// and was never validated on real hardware. Keeping the documented
// default = 32 here unblocks the SYCL build; reintroducing the
// autotuned KV_TILE sweep requires a GPU-validated kernel
// re-implementation as a follow-up.
//
// The surrounding `extern "C" {` (opened at line 180 for the
// FFI exports) gives every declaration C linkage by default, but
// C++ templates are not allowed under C linkage. Wrap the
// template + the generic kernel below in `extern "C++"` so they
// keep C++ linkage while the FFI dispatchers immediately after
// stay under the outer extern "C". Matches the pattern used by
// the other templated kernels in this file (Q4_K_M, Q5_K_M, etc.,
// which close + reopen extern "C" around their templates).
extern "C++" {
template <int DIM_PER_LANE, int SG_SIZE, int KV_TILE_T>
inline void flash_attn_decode_v3_tpl(sycl::queue& q_queue,
                                     const uint16_t* q_usm,
                                     const uint16_t* k_usm,
                                     const uint16_t* v_usm,
                                     uint16_t* out_usm,
                                     int n_heads, int n_kv_heads,
                                     int head_dim, int max_ctx, int kv_len) {
    // H11: KV_TILE is now templated so the dispatcher can pick
    // 16/32/64 based on kv_len. 32 was the historic default.
    constexpr int KV_TILE = KV_TILE_T;
    const int n_gqa = n_heads / n_kv_heads;
    const float scale = 1.0f / sycl::sqrt(static_cast<float>(head_dim));
    const std::size_t slm_elems = static_cast<std::size_t>(KV_TILE) * head_dim;
    const std::size_t global_size =
        static_cast<std::size_t>(n_heads) * static_cast<std::size_t>(SG_SIZE);
    q_queue.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint16_t, 1> k_tile(sycl::range<1>(slm_elems), h);
        sycl::local_accessor<uint16_t, 1> v_tile(sycl::range<1>(slm_elems), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(global_size),
                              sycl::range<1>(SG_SIZE)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                auto sg = it.get_sub_group();
                const int hh = static_cast<int>(it.get_group(0));
                const int lane = static_cast<int>(it.get_local_id(0));
                if (hh >= n_heads) return;
                const int kv_h = hh / n_gqa;
                const int q_base = hh * head_dim;
                const int out_base = hh * head_dim;
                const int lane_off = lane * DIM_PER_LANE;

                // Q slice in private regs (compile-time unrolled).
                float q_slice[DIM_PER_LANE];
                #pragma unroll
                for (int d = 0; d < DIM_PER_LANE; ++d) {
                    q_slice[d] = bits_to_f32(q_usm[q_base + lane_off + d]);
                }
                float out_slice[DIM_PER_LANE];
                #pragma unroll
                for (int d = 0; d < DIM_PER_LANE; ++d) {
                    out_slice[d] = 0.0f;
                }
                float m_state = -INFINITY;
                float l_state = 0.0f;

                for (int kv_base = 0; kv_base < kv_len; kv_base += KV_TILE) {
                    const int kv_block_end = (kv_base + KV_TILE < kv_len)
                        ? (kv_base + KV_TILE) : kv_len;
                    const int kv_block_size = kv_block_end - kv_base;
                    const int tile_elems = kv_block_size * head_dim;

                    for (int i = lane; i < tile_elems; i += SG_SIZE) {
                        const int t = i / head_dim;
                        const int d = i % head_dim;
                        const int src_off = (kv_h * max_ctx + kv_base + t) * head_dim + d;
                        k_tile[i] = k_usm[src_off];
                        v_tile[i] = v_usm[src_off];
                    }
                    sycl::group_barrier(it.get_group());

                    for (int t_off = 0; t_off < kv_block_size; ++t_off) {
                        const int tile_off = t_off * head_dim;
                        float partial = 0.0f;
                        #pragma unroll
                        for (int d = 0; d < DIM_PER_LANE; ++d) {
                            const float kv = bits_to_f32(k_tile[tile_off + lane_off + d]);
                            partial += q_slice[d] * kv;
                        }
                        float full_dot = sycl::reduce_over_group(
                            sg, partial, sycl::plus<float>());
                        full_dot *= scale;
                        const float m_new = sycl::fmax(m_state, full_dot);
                        const float rescale =
                            sycl::isfinite(m_state) ? sycl::exp(m_state - m_new) : 0.0f;
                        const float p = sycl::exp(full_dot - m_new);
                        l_state = l_state * rescale + p;
                        m_state = m_new;
                        #pragma unroll
                        for (int d = 0; d < DIM_PER_LANE; ++d) {
                            const float vv = bits_to_f32(v_tile[tile_off + lane_off + d]);
                            out_slice[d] = out_slice[d] * rescale + p * vv;
                        }
                    }

                    sycl::group_barrier(it.get_group());
                }

                const float inv_l = (l_state > 0.0f) ? 1.0f / l_state : 0.0f;
                #pragma unroll
                for (int d = 0; d < DIM_PER_LANE; ++d) {
                    out_usm[out_base + lane_off + d] = f32_to_bits(out_slice[d] * inv_l);
                }
            });
    }).wait();
}

// Generic-`dim_per_lane` fallback kernel — used when head_dim doesn't
// match one of the compile-time-specialized variants above. Same
// algorithm as the templated form; just `dim_per_lane` is runtime.
inline void flash_attn_decode_v3_generic(sycl::queue& q_queue,
                                         const uint16_t* q_usm,
                                         const uint16_t* k_usm,
                                         const uint16_t* v_usm,
                                         uint16_t* out_usm,
                                         int n_heads, int n_kv_heads,
                                         int head_dim, int max_ctx, int kv_len,
                                         int dim_per_lane) {
    constexpr int SG_SIZE = 16;
    constexpr int MAX_DIM_PER_LANE = 16;
    constexpr int KV_TILE = 32;
    const int n_gqa = n_heads / n_kv_heads;
    const float scale = 1.0f / sycl::sqrt(static_cast<float>(head_dim));
    const std::size_t slm_elems = static_cast<std::size_t>(KV_TILE) * head_dim;
    const std::size_t global_size =
        static_cast<std::size_t>(n_heads) * static_cast<std::size_t>(SG_SIZE);
    q_queue.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint16_t, 1> k_tile(sycl::range<1>(slm_elems), h);
        sycl::local_accessor<uint16_t, 1> v_tile(sycl::range<1>(slm_elems), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(global_size),
                              sycl::range<1>(SG_SIZE)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                auto sg = it.get_sub_group();
                const int hh = static_cast<int>(it.get_group(0));
                const int lane = static_cast<int>(it.get_local_id(0));
                if (hh >= n_heads) return;
                const int kv_h = hh / n_gqa;
                const int q_base = hh * head_dim;
                const int out_base = hh * head_dim;
                const int lane_off = lane * dim_per_lane;

                float q_slice[MAX_DIM_PER_LANE];
                for (int d = 0; d < dim_per_lane; ++d) {
                    q_slice[d] = bits_to_f32(q_usm[q_base + lane_off + d]);
                }
                float out_slice[MAX_DIM_PER_LANE];
                for (int d = 0; d < dim_per_lane; ++d) {
                    out_slice[d] = 0.0f;
                }
                float m_state = -INFINITY;
                float l_state = 0.0f;

                for (int kv_base = 0; kv_base < kv_len; kv_base += KV_TILE) {
                    const int kv_block_end = (kv_base + KV_TILE < kv_len)
                        ? (kv_base + KV_TILE) : kv_len;
                    const int kv_block_size = kv_block_end - kv_base;
                    const int tile_elems = kv_block_size * head_dim;

                    for (int i = lane; i < tile_elems; i += SG_SIZE) {
                        const int t = i / head_dim;
                        const int d = i % head_dim;
                        const int src_off = (kv_h * max_ctx + kv_base + t) * head_dim + d;
                        k_tile[i] = k_usm[src_off];
                        v_tile[i] = v_usm[src_off];
                    }
                    sycl::group_barrier(it.get_group());

                    for (int t_off = 0; t_off < kv_block_size; ++t_off) {
                        const int tile_off = t_off * head_dim;
                        float partial = 0.0f;
                        for (int d = 0; d < dim_per_lane; ++d) {
                            const float kv = bits_to_f32(k_tile[tile_off + lane_off + d]);
                            partial += q_slice[d] * kv;
                        }
                        float full_dot = sycl::reduce_over_group(
                            sg, partial, sycl::plus<float>());
                        full_dot *= scale;
                        const float m_new = sycl::fmax(m_state, full_dot);
                        const float rescale =
                            sycl::isfinite(m_state) ? sycl::exp(m_state - m_new) : 0.0f;
                        const float p = sycl::exp(full_dot - m_new);
                        l_state = l_state * rescale + p;
                        m_state = m_new;
                        for (int d = 0; d < dim_per_lane; ++d) {
                            const float vv = bits_to_f32(v_tile[tile_off + lane_off + d]);
                            out_slice[d] = out_slice[d] * rescale + p * vv;
                        }
                    }

                    sycl::group_barrier(it.get_group());
                }

                const float inv_l = (l_state > 0.0f) ? 1.0f / l_state : 0.0f;
                for (int d = 0; d < dim_per_lane; ++d) {
                    out_usm[out_base + lane_off + d] = f32_to_bits(out_slice[d] * inv_l);
                }
            });
    }).wait();
}
}  // extern "C++" — restore C linkage for the FFI dispatcher below

void rsl_flash_attn_decode_v3_usm(rsl_stream* s,
                                  const uint16_t* q_usm,
                                  const uint16_t* k_usm,
                                  const uint16_t* v_usm,
                                  uint16_t* out_usm,
                                  int n_heads, int n_kv_heads,
                                  int head_dim, int max_ctx, int kv_len) RSL_FFI_BODY_VOID("rsl_flash_attn_decode_v3_usm", {
    if (s == nullptr || q_usm == nullptr || k_usm == nullptr || v_usm == nullptr || out_usm == nullptr) return;
    if (n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0 || max_ctx <= 0) return;
    if (kv_len <= 0) {
        auto& q = s->q;
        std::size_t total = static_cast<std::size_t>(n_heads) * head_dim;
        q.memset(out_usm, 0, total * sizeof(uint16_t)).wait();
        return;
    }
    if ((n_heads % n_kv_heads) != 0) return;

    constexpr int SG_SIZE = 16;
    constexpr int MAX_DIM_PER_LANE = 16;
    if ((head_dim % SG_SIZE) != 0) return;
    const int dim_per_lane = head_dim / SG_SIZE;
    if (dim_per_lane > MAX_DIM_PER_LANE) return;

    auto& q_queue = s->q;
    // H11: adaptive KV_TILE selection based on kv_len. Smaller
    // tiles (16) reduce SLM pressure for short context; larger
    // tiles (64) amortize the per-tile load/barrier overhead for
    // long context. Crossover points empirical on Iris Xe — tune
    // via the `RUSTLLAMA_FLASH_V3_KV_TILE` env var if needed (set
    // explicit 16/32/64 to override; absent or 0 uses the heuristic).
    const char* env_tile = std::getenv("RUSTLLAMA_FLASH_V3_KV_TILE");
    int effective_tile = 32;
    if (env_tile) {
        int v = std::atoi(env_tile);
        if (v == 16 || v == 32 || v == 64) {
            effective_tile = v;
        } else if (v == 0 || v == -1) {
            // Auto mode — fall through to kv_len heuristic.
            effective_tile = (kv_len < 512) ? 16 : ((kv_len >= 4096) ? 64 : 32);
        }
    } else {
        effective_tile = (kv_len < 512) ? 16 : ((kv_len >= 4096) ? 64 : 32);
    }
    // Dispatch on (dim_per_lane, KV_TILE) — 3 × 3 = 9 explicit arms.
    switch (dim_per_lane) {
        case 4:
            switch (effective_tile) {
                case 16: flash_attn_decode_v3_tpl<4, SG_SIZE, 16>(q_queue, q_usm, k_usm, v_usm, out_usm, n_heads, n_kv_heads, head_dim, max_ctx, kv_len); break;
                case 64: flash_attn_decode_v3_tpl<4, SG_SIZE, 64>(q_queue, q_usm, k_usm, v_usm, out_usm, n_heads, n_kv_heads, head_dim, max_ctx, kv_len); break;
                default: flash_attn_decode_v3_tpl<4, SG_SIZE, 32>(q_queue, q_usm, k_usm, v_usm, out_usm, n_heads, n_kv_heads, head_dim, max_ctx, kv_len); break;
            }
            break;
        case 8:
            switch (effective_tile) {
                case 16: flash_attn_decode_v3_tpl<8, SG_SIZE, 16>(q_queue, q_usm, k_usm, v_usm, out_usm, n_heads, n_kv_heads, head_dim, max_ctx, kv_len); break;
                case 64: flash_attn_decode_v3_tpl<8, SG_SIZE, 64>(q_queue, q_usm, k_usm, v_usm, out_usm, n_heads, n_kv_heads, head_dim, max_ctx, kv_len); break;
                default: flash_attn_decode_v3_tpl<8, SG_SIZE, 32>(q_queue, q_usm, k_usm, v_usm, out_usm, n_heads, n_kv_heads, head_dim, max_ctx, kv_len); break;
            }
            break;
        case 16:
            switch (effective_tile) {
                case 16: flash_attn_decode_v3_tpl<16, SG_SIZE, 16>(q_queue, q_usm, k_usm, v_usm, out_usm, n_heads, n_kv_heads, head_dim, max_ctx, kv_len); break;
                case 64: flash_attn_decode_v3_tpl<16, SG_SIZE, 64>(q_queue, q_usm, k_usm, v_usm, out_usm, n_heads, n_kv_heads, head_dim, max_ctx, kv_len); break;
                default: flash_attn_decode_v3_tpl<16, SG_SIZE, 32>(q_queue, q_usm, k_usm, v_usm, out_usm, n_heads, n_kv_heads, head_dim, max_ctx, kv_len); break;
            }
            break;
        default:
            flash_attn_decode_v3_generic(
                q_queue, q_usm, k_usm, v_usm, out_usm,
                n_heads, n_kv_heads, head_dim, max_ctx, kv_len,
                dim_per_lane);
            break;
    }
})

// USM-resident FlashAttention prefill. Computes attention for `n_new`
// new query positions in one launch. One work-item per `(head, q_pos)`
// pair → `n_heads * n_new` parallelism. Each work-item walks `t` from
// 0 to `kv_len_base + q_pos` (causal mask), maintaining its own
// online-softmax state — no cross-WI dependencies. Per-WI floating-
// point op order matches the CPU scalar reference's inner loop, so
// the two should produce bit-identical output for matching inputs
// (modulo small `sycl::exp` / `sycl::sqrt` precision differences vs
// libm, which haven't shown up in the decode parity tests).
//
// Inputs (all USM-resident, f32):
//   q_usm:       `[N, n_heads, head_dim]`        row-major
//   k_cache_usm: `[n_kv_heads, max_ctx, head_dim]`
//   v_cache_usm: `[n_kv_heads, max_ctx, head_dim]`
//   out_usm:     `[N, n_heads, head_dim]`        row-major (zeroed by kernel)
//
// Caller guarantees `kv_len_base + n_new <= max_ctx` and
// `n_heads % n_kv_heads == 0`. The K/V cache must already contain
// the n_new new rows appended at positions `[kv_len_base,
// kv_len_base + n_new)` — the caller's append-then-attend pattern.
void rsl_flash_attn_prefill_usm(rsl_stream* s,
                                const float* q_usm,
                                const float* k_cache_usm,
                                const float* v_cache_usm,
                                float* out_usm,
                                int n_heads, int n_kv_heads,
                                int head_dim, int max_ctx,
                                int kv_len_base, int n_new) RSL_FFI_BODY_VOID("rsl_flash_attn_prefill_usm", {
    if (s == nullptr || q_usm == nullptr || k_cache_usm == nullptr
        || v_cache_usm == nullptr || out_usm == nullptr) return;
    if (n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0 || max_ctx <= 0) return;
    if (n_new <= 0) return;  // matches CPU reference's early-return
    if ((n_heads % n_kv_heads) != 0) return;
    if (kv_len_base < 0 || kv_len_base + n_new > max_ctx) return;
    auto& q = s->q;
    const int n_gqa = n_heads / n_kv_heads;
    const float scale = 1.0f / sycl::sqrt(static_cast<float>(head_dim));
    // 2D dispatch: x = head index (n_heads), y = query position
    // (n_new). LWS in x mirrors the decode kernel's RSL_LWS (so
    // sub-group occupancy on the same device class is preserved);
    // LWS in y = 1 keeps each row of work-items independent so no
    // cross-q_pos sync is needed.
    const std::size_t global_h = round_up_to_lws(n_heads);
    sycl::range<2> global(global_h, static_cast<std::size_t>(n_new));
    sycl::range<2> local(RSL_LWS, 1);
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<2>(global, local),
            [=](sycl::nd_item<2> it) {
                const int hh = static_cast<int>(it.get_global_id(0));
                const int q_pos = static_cast<int>(it.get_global_id(1));
                if (hh >= n_heads || q_pos >= n_new) return;
                const int kv_h = hh / n_gqa;
                const int q_off = (q_pos * n_heads + hh) * head_dim;
                const int out_off = q_off;  // out shape matches Q
                // Causal mask: query at new-batch position `q_pos`
                // attends absolute positions `[0, kv_len_base + q_pos]`
                // inclusive (its own newly-appended K/V row included).
                const int kv_len_for_q = kv_len_base + q_pos + 1;
                for (int i = 0; i < head_dim; ++i) {
                    out_usm[out_off + i] = 0.0f;
                }
                float m_state = -INFINITY;
                float l_state = 0.0f;
                for (int t = 0; t < kv_len_for_q; ++t) {
                    const int kv_off = (kv_h * max_ctx + t) * head_dim;
                    float s_dot = 0.0f;
                    for (int i = 0; i < head_dim; ++i) {
                        s_dot += q_usm[q_off + i] * k_cache_usm[kv_off + i];
                    }
                    s_dot *= scale;
                    const float m_new = sycl::fmax(m_state, s_dot);
                    const float rescale =
                        sycl::isfinite(m_state) ? sycl::exp(m_state - m_new) : 0.0f;
                    const float p = sycl::exp(s_dot - m_new);
                    l_state = l_state * rescale + p;
                    for (int i = 0; i < head_dim; ++i) {
                        out_usm[out_off + i] =
                            out_usm[out_off + i] * rescale + p * v_cache_usm[kv_off + i];
                    }
                    m_state = m_new;
                }
                const float inv_l = (l_state > 0.0f) ? 1.0f / l_state : 0.0f;
                for (int i = 0; i < head_dim; ++i) {
                    out_usm[out_off + i] *= inv_l;
                }
            });
    }).wait();
})

// ============================================================
// Quantized-KV FlashAttention (F32 Q / F32 out, packed K/V
// dequantized on the fly). Same online-softmax recurrence as the
// v1 F32 decode/prefill above; only the K/V source differs. Each
// work-item dequantizes one K (then V) row into a private buffer per
// kv position via the deq_* helpers, reproducing the CPU reference's
// dequant-then-attend float op order (q4_0_kv.rs / nvfp4.rs /
// turboquant.rs). One WI per head (decode) / per (head,q_pos)
// (prefill). head_dim capped at RSL_FLASH_MAX_HEAD_DIM; kernels
// early-return on larger / mis-shaped inputs (caller falls back to
// CPU). Q is F32, K/V are packed bytes; TQ also takes per-row f32
// scales (indexed kv_h*max_ctx + t) + bit width.
// ============================================================

void rsl_flash_attn_decode_q4_0_usm(rsl_stream* s,
                                    const float* q_usm,
                                    const void* k_packed_usm,
                                    const void* v_packed_usm,
                                    float* out_usm,
                                    int n_heads, int n_kv_heads,
                                    int head_dim, int max_ctx,
                                    int kv_len) RSL_FFI_BODY_VOID("rsl_flash_attn_decode_q4_0_usm", {
    if (s == nullptr || q_usm == nullptr || k_packed_usm == nullptr
        || v_packed_usm == nullptr || out_usm == nullptr) return;
    if (n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0 || max_ctx <= 0) return;
    if ((n_heads % n_kv_heads) != 0) return;
    if (head_dim > RSL_FLASH_MAX_HEAD_DIM || (head_dim % 32) != 0) return;
    auto& q = s->q;
    if (kv_len <= 0) {
        std::size_t total = static_cast<std::size_t>(n_heads) * head_dim;
        q.memset(out_usm, 0, total * sizeof(float)).wait();
        return;
    }
    const int n_gqa = n_heads / n_kv_heads;
    const float scale = 1.0f / sycl::sqrt(static_cast<float>(head_dim));
    const int bytes_per_row = (head_dim / 32) * 18;
    const uint8_t* k_bytes = static_cast<const uint8_t*>(k_packed_usm);
    const uint8_t* v_bytes = static_cast<const uint8_t*>(v_packed_usm);
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(round_up_to_lws(n_heads)),
                              sycl::range<1>(RSL_LWS)),
            [=](sycl::nd_item<1> it) {
                const int hh = static_cast<int>(it.get_global_id(0));
                if (hh >= n_heads) return;
                const int kv_h = hh / n_gqa;
                const int q_base = hh * head_dim;
                const int out_base = hh * head_dim;
                float row[RSL_FLASH_MAX_HEAD_DIM];
                for (int i = 0; i < head_dim; ++i) out_usm[out_base + i] = 0.0f;
                float m = -INFINITY, l = 0.0f;
                for (int t = 0; t < kv_len; ++t) {
                    const uint8_t* kp = k_bytes + static_cast<std::size_t>(kv_h * max_ctx + t) * bytes_per_row;
                    deq_q4_0_row(kp, row, head_dim);
                    float s_dot = 0.0f;
                    for (int i = 0; i < head_dim; ++i) s_dot += q_usm[q_base + i] * row[i];
                    s_dot *= scale;
                    const float m_new = sycl::fmax(m, s_dot);
                    const float rescale = sycl::isfinite(m) ? sycl::exp(m - m_new) : 0.0f;
                    const float p = sycl::exp(s_dot - m_new);
                    l = l * rescale + p;
                    const uint8_t* vp = v_bytes + static_cast<std::size_t>(kv_h * max_ctx + t) * bytes_per_row;
                    deq_q4_0_row(vp, row, head_dim);
                    for (int i = 0; i < head_dim; ++i)
                        out_usm[out_base + i] = out_usm[out_base + i] * rescale + p * row[i];
                    m = m_new;
                }
                const float inv_l = (l > 0.0f) ? 1.0f / l : 0.0f;
                for (int i = 0; i < head_dim; ++i) out_usm[out_base + i] *= inv_l;
            });
    }).wait();
})

void rsl_flash_attn_prefill_q4_0_usm(rsl_stream* s,
                                     const float* q_usm,
                                     const void* k_packed_usm,
                                     const void* v_packed_usm,
                                     float* out_usm,
                                     int n_heads, int n_kv_heads,
                                     int head_dim, int max_ctx,
                                     int kv_len_base, int n_new) RSL_FFI_BODY_VOID("rsl_flash_attn_prefill_q4_0_usm", {
    if (s == nullptr || q_usm == nullptr || k_packed_usm == nullptr
        || v_packed_usm == nullptr || out_usm == nullptr) return;
    if (n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0 || max_ctx <= 0 || n_new <= 0) return;
    if ((n_heads % n_kv_heads) != 0) return;
    if (head_dim > RSL_FLASH_MAX_HEAD_DIM || (head_dim % 32) != 0) return;
    if (kv_len_base < 0 || kv_len_base + n_new > max_ctx) return;
    auto& q = s->q;
    const int n_gqa = n_heads / n_kv_heads;
    const float scale = 1.0f / sycl::sqrt(static_cast<float>(head_dim));
    const int bytes_per_row = (head_dim / 32) * 18;
    const uint8_t* k_bytes = static_cast<const uint8_t*>(k_packed_usm);
    const uint8_t* v_bytes = static_cast<const uint8_t*>(v_packed_usm);
    const std::size_t global_h = round_up_to_lws(n_heads);
    sycl::range<2> global(global_h, static_cast<std::size_t>(n_new));
    sycl::range<2> local(RSL_LWS, 1);
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<2>(global, local),
            [=](sycl::nd_item<2> it) {
                const int hh = static_cast<int>(it.get_global_id(0));
                const int q_pos = static_cast<int>(it.get_global_id(1));
                if (hh >= n_heads || q_pos >= n_new) return;
                const int kv_h = hh / n_gqa;
                const int q_off = (q_pos * n_heads + hh) * head_dim;
                const int out_off = q_off;
                const int kv_len_for_q = kv_len_base + q_pos + 1;
                float row[RSL_FLASH_MAX_HEAD_DIM];
                for (int i = 0; i < head_dim; ++i) out_usm[out_off + i] = 0.0f;
                float m = -INFINITY, l = 0.0f;
                for (int t = 0; t < kv_len_for_q; ++t) {
                    const uint8_t* kp = k_bytes + static_cast<std::size_t>(kv_h * max_ctx + t) * bytes_per_row;
                    deq_q4_0_row(kp, row, head_dim);
                    float s_dot = 0.0f;
                    for (int i = 0; i < head_dim; ++i) s_dot += q_usm[q_off + i] * row[i];
                    s_dot *= scale;
                    const float m_new = sycl::fmax(m, s_dot);
                    const float rescale = sycl::isfinite(m) ? sycl::exp(m - m_new) : 0.0f;
                    const float p = sycl::exp(s_dot - m_new);
                    l = l * rescale + p;
                    const uint8_t* vp = v_bytes + static_cast<std::size_t>(kv_h * max_ctx + t) * bytes_per_row;
                    deq_q4_0_row(vp, row, head_dim);
                    for (int i = 0; i < head_dim; ++i)
                        out_usm[out_off + i] = out_usm[out_off + i] * rescale + p * row[i];
                    m = m_new;
                }
                const float inv_l = (l > 0.0f) ? 1.0f / l : 0.0f;
                for (int i = 0; i < head_dim; ++i) out_usm[out_off + i] *= inv_l;
            });
    }).wait();
})

// ---- Q8_0-KV decode / prefill (i8 slab + per-row f32 scale) ----
// Byte-exact port of the CPU reference gqa_attention_flash_decode_q8_0 /
// _prefill_q8_0 (rustllama-kernels-cpu/src/lib.rs). Unlike the Q4_0/NVFP4/TQ
// KV kernels above, the engine's KvLayer::Q8_0 stores K/V as a PLAIN i8 slab
// [n_kv_heads, max_ctx, head_dim] (one byte per element, NO 34B/32 GGUF
// blocks) + a separate per-row absmax f32 scale in k_scales/v_scales =
// [n_kv_heads*max_ctx] (indexed kv_h*max_ctx + t, same shape as the TQ
// scales). The CPU Q8_0 reference does NOT materialize a dequantized row — it
// dots the raw i8 codes then factors the row scale out of the inner loop
// (s *= k_scale*attn_scale; out += (p*v_scale)*i8). We reproduce that
// factoring exactly for float parity, so there is NO private row buffer here
// and head_dim is unconstrained (no block-alignment, no power-of-two).
void rsl_flash_attn_decode_q8_0_usm(rsl_stream* s,
                                    const float* q_usm,
                                    const void* k_packed_usm,
                                    const void* v_packed_usm,
                                    const float* k_scales_usm,
                                    const float* v_scales_usm,
                                    float* out_usm,
                                    int n_heads, int n_kv_heads,
                                    int head_dim, int max_ctx,
                                    int kv_len) RSL_FFI_BODY_VOID("rsl_flash_attn_decode_q8_0_usm", {
    if (s == nullptr || q_usm == nullptr || k_packed_usm == nullptr
        || v_packed_usm == nullptr || k_scales_usm == nullptr
        || v_scales_usm == nullptr || out_usm == nullptr) return;
    if (n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0 || max_ctx <= 0) return;
    if ((n_heads % n_kv_heads) != 0) return;
    auto& q = s->q;
    if (kv_len <= 0) {
        std::size_t total = static_cast<std::size_t>(n_heads) * head_dim;
        q.memset(out_usm, 0, total * sizeof(float)).wait();
        return;
    }
    const int n_gqa = n_heads / n_kv_heads;
    const float scale = 1.0f / sycl::sqrt(static_cast<float>(head_dim));
    const int8_t* k_bytes = static_cast<const int8_t*>(k_packed_usm);
    const int8_t* v_bytes = static_cast<const int8_t*>(v_packed_usm);
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(round_up_to_lws(n_heads)),
                              sycl::range<1>(RSL_LWS)),
            [=](sycl::nd_item<1> it) {
                const int hh = static_cast<int>(it.get_global_id(0));
                if (hh >= n_heads) return;
                const int kv_h = hh / n_gqa;
                const int q_base = hh * head_dim;
                const int out_base = hh * head_dim;
                for (int i = 0; i < head_dim; ++i) out_usm[out_base + i] = 0.0f;
                float m = -INFINITY, l = 0.0f;
                for (int t = 0; t < kv_len; ++t) {
                    const std::size_t ridx = static_cast<std::size_t>(kv_h * max_ctx + t);
                    const int8_t* kp = k_bytes + ridx * head_dim;
                    float s_dot = 0.0f;
                    for (int i = 0; i < head_dim; ++i)
                        s_dot += q_usm[q_base + i] * static_cast<float>(kp[i]);
                    s_dot *= k_scales_usm[ridx] * scale;
                    const float m_new = sycl::fmax(m, s_dot);
                    const float rescale = sycl::isfinite(m) ? sycl::exp(m - m_new) : 0.0f;
                    const float p = sycl::exp(s_dot - m_new);
                    l = l * rescale + p;
                    const int8_t* vp = v_bytes + ridx * head_dim;
                    const float p_eff = p * v_scales_usm[ridx];
                    for (int i = 0; i < head_dim; ++i)
                        out_usm[out_base + i] = out_usm[out_base + i] * rescale
                                              + p_eff * static_cast<float>(vp[i]);
                    m = m_new;
                }
                const float inv_l = (l > 0.0f) ? 1.0f / l : 0.0f;
                for (int i = 0; i < head_dim; ++i) out_usm[out_base + i] *= inv_l;
            });
    }).wait();
})

void rsl_flash_attn_prefill_q8_0_usm(rsl_stream* s,
                                     const float* q_usm,
                                     const void* k_packed_usm,
                                     const void* v_packed_usm,
                                     const float* k_scales_usm,
                                     const float* v_scales_usm,
                                     float* out_usm,
                                     int n_heads, int n_kv_heads,
                                     int head_dim, int max_ctx,
                                     int kv_len_base, int n_new) RSL_FFI_BODY_VOID("rsl_flash_attn_prefill_q8_0_usm", {
    if (s == nullptr || q_usm == nullptr || k_packed_usm == nullptr
        || v_packed_usm == nullptr || k_scales_usm == nullptr
        || v_scales_usm == nullptr || out_usm == nullptr) return;
    if (n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0 || max_ctx <= 0 || n_new <= 0) return;
    if ((n_heads % n_kv_heads) != 0) return;
    if (kv_len_base < 0 || kv_len_base + n_new > max_ctx) return;
    auto& q = s->q;
    const int n_gqa = n_heads / n_kv_heads;
    const float scale = 1.0f / sycl::sqrt(static_cast<float>(head_dim));
    const int8_t* k_bytes = static_cast<const int8_t*>(k_packed_usm);
    const int8_t* v_bytes = static_cast<const int8_t*>(v_packed_usm);
    const std::size_t global_h = round_up_to_lws(n_heads);
    sycl::range<2> global(global_h, static_cast<std::size_t>(n_new));
    sycl::range<2> local(RSL_LWS, 1);
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<2>(global, local),
            [=](sycl::nd_item<2> it) {
                const int hh = static_cast<int>(it.get_global_id(0));
                const int q_pos = static_cast<int>(it.get_global_id(1));
                if (hh >= n_heads || q_pos >= n_new) return;
                const int kv_h = hh / n_gqa;
                const int q_off = (q_pos * n_heads + hh) * head_dim;
                const int out_off = q_off;
                const int kv_len_for_q = kv_len_base + q_pos + 1;
                for (int i = 0; i < head_dim; ++i) out_usm[out_off + i] = 0.0f;
                float m = -INFINITY, l = 0.0f;
                for (int t = 0; t < kv_len_for_q; ++t) {
                    const std::size_t ridx = static_cast<std::size_t>(kv_h * max_ctx + t);
                    const int8_t* kp = k_bytes + ridx * head_dim;
                    float s_dot = 0.0f;
                    for (int i = 0; i < head_dim; ++i)
                        s_dot += q_usm[q_off + i] * static_cast<float>(kp[i]);
                    s_dot *= k_scales_usm[ridx] * scale;
                    const float m_new = sycl::fmax(m, s_dot);
                    const float rescale = sycl::isfinite(m) ? sycl::exp(m - m_new) : 0.0f;
                    const float p = sycl::exp(s_dot - m_new);
                    l = l * rescale + p;
                    const int8_t* vp = v_bytes + ridx * head_dim;
                    const float p_eff = p * v_scales_usm[ridx];
                    for (int i = 0; i < head_dim; ++i)
                        out_usm[out_off + i] = out_usm[out_off + i] * rescale
                                             + p_eff * static_cast<float>(vp[i]);
                    m = m_new;
                }
                const float inv_l = (l > 0.0f) ? 1.0f / l : 0.0f;
                for (int i = 0; i < head_dim; ++i) out_usm[out_off + i] *= inv_l;
            });
    }).wait();
})

void rsl_flash_attn_decode_nvfp4_usm(rsl_stream* s,
                                     const float* q_usm,
                                     const void* k_packed_usm,
                                     const void* v_packed_usm,
                                     float* out_usm,
                                     int n_heads, int n_kv_heads,
                                     int head_dim, int max_ctx,
                                     int kv_len) RSL_FFI_BODY_VOID("rsl_flash_attn_decode_nvfp4_usm", {
    if (s == nullptr || q_usm == nullptr || k_packed_usm == nullptr
        || v_packed_usm == nullptr || out_usm == nullptr) return;
    if (n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0 || max_ctx <= 0) return;
    if ((n_heads % n_kv_heads) != 0) return;
    if (head_dim > RSL_FLASH_MAX_HEAD_DIM || (head_dim % 16) != 0) return;
    auto& q = s->q;
    if (kv_len <= 0) {
        std::size_t total = static_cast<std::size_t>(n_heads) * head_dim;
        q.memset(out_usm, 0, total * sizeof(float)).wait();
        return;
    }
    const int n_gqa = n_heads / n_kv_heads;
    const float scale = 1.0f / sycl::sqrt(static_cast<float>(head_dim));
    const int bytes_per_row = (head_dim / 16) * 9;
    const uint8_t* k_bytes = static_cast<const uint8_t*>(k_packed_usm);
    const uint8_t* v_bytes = static_cast<const uint8_t*>(v_packed_usm);
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(round_up_to_lws(n_heads)),
                              sycl::range<1>(RSL_LWS)),
            [=](sycl::nd_item<1> it) {
                const int hh = static_cast<int>(it.get_global_id(0));
                if (hh >= n_heads) return;
                const int kv_h = hh / n_gqa;
                const int q_base = hh * head_dim;
                const int out_base = hh * head_dim;
                float row[RSL_FLASH_MAX_HEAD_DIM];
                for (int i = 0; i < head_dim; ++i) out_usm[out_base + i] = 0.0f;
                float m = -INFINITY, l = 0.0f;
                for (int t = 0; t < kv_len; ++t) {
                    const uint8_t* kp = k_bytes + static_cast<std::size_t>(kv_h * max_ctx + t) * bytes_per_row;
                    deq_nvfp4_row(kp, row, head_dim);
                    float s_dot = 0.0f;
                    for (int i = 0; i < head_dim; ++i) s_dot += q_usm[q_base + i] * row[i];
                    s_dot *= scale;
                    const float m_new = sycl::fmax(m, s_dot);
                    const float rescale = sycl::isfinite(m) ? sycl::exp(m - m_new) : 0.0f;
                    const float p = sycl::exp(s_dot - m_new);
                    l = l * rescale + p;
                    const uint8_t* vp = v_bytes + static_cast<std::size_t>(kv_h * max_ctx + t) * bytes_per_row;
                    deq_nvfp4_row(vp, row, head_dim);
                    for (int i = 0; i < head_dim; ++i)
                        out_usm[out_base + i] = out_usm[out_base + i] * rescale + p * row[i];
                    m = m_new;
                }
                const float inv_l = (l > 0.0f) ? 1.0f / l : 0.0f;
                for (int i = 0; i < head_dim; ++i) out_usm[out_base + i] *= inv_l;
            });
    }).wait();
})

void rsl_flash_attn_prefill_nvfp4_usm(rsl_stream* s,
                                      const float* q_usm,
                                      const void* k_packed_usm,
                                      const void* v_packed_usm,
                                      float* out_usm,
                                      int n_heads, int n_kv_heads,
                                      int head_dim, int max_ctx,
                                      int kv_len_base, int n_new) RSL_FFI_BODY_VOID("rsl_flash_attn_prefill_nvfp4_usm", {
    if (s == nullptr || q_usm == nullptr || k_packed_usm == nullptr
        || v_packed_usm == nullptr || out_usm == nullptr) return;
    if (n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0 || max_ctx <= 0 || n_new <= 0) return;
    if ((n_heads % n_kv_heads) != 0) return;
    if (head_dim > RSL_FLASH_MAX_HEAD_DIM || (head_dim % 16) != 0) return;
    if (kv_len_base < 0 || kv_len_base + n_new > max_ctx) return;
    auto& q = s->q;
    const int n_gqa = n_heads / n_kv_heads;
    const float scale = 1.0f / sycl::sqrt(static_cast<float>(head_dim));
    const int bytes_per_row = (head_dim / 16) * 9;
    const uint8_t* k_bytes = static_cast<const uint8_t*>(k_packed_usm);
    const uint8_t* v_bytes = static_cast<const uint8_t*>(v_packed_usm);
    const std::size_t global_h = round_up_to_lws(n_heads);
    sycl::range<2> global(global_h, static_cast<std::size_t>(n_new));
    sycl::range<2> local(RSL_LWS, 1);
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<2>(global, local),
            [=](sycl::nd_item<2> it) {
                const int hh = static_cast<int>(it.get_global_id(0));
                const int q_pos = static_cast<int>(it.get_global_id(1));
                if (hh >= n_heads || q_pos >= n_new) return;
                const int kv_h = hh / n_gqa;
                const int q_off = (q_pos * n_heads + hh) * head_dim;
                const int out_off = q_off;
                const int kv_len_for_q = kv_len_base + q_pos + 1;
                float row[RSL_FLASH_MAX_HEAD_DIM];
                for (int i = 0; i < head_dim; ++i) out_usm[out_off + i] = 0.0f;
                float m = -INFINITY, l = 0.0f;
                for (int t = 0; t < kv_len_for_q; ++t) {
                    const uint8_t* kp = k_bytes + static_cast<std::size_t>(kv_h * max_ctx + t) * bytes_per_row;
                    deq_nvfp4_row(kp, row, head_dim);
                    float s_dot = 0.0f;
                    for (int i = 0; i < head_dim; ++i) s_dot += q_usm[q_off + i] * row[i];
                    s_dot *= scale;
                    const float m_new = sycl::fmax(m, s_dot);
                    const float rescale = sycl::isfinite(m) ? sycl::exp(m - m_new) : 0.0f;
                    const float p = sycl::exp(s_dot - m_new);
                    l = l * rescale + p;
                    const uint8_t* vp = v_bytes + static_cast<std::size_t>(kv_h * max_ctx + t) * bytes_per_row;
                    deq_nvfp4_row(vp, row, head_dim);
                    for (int i = 0; i < head_dim; ++i)
                        out_usm[out_off + i] = out_usm[out_off + i] * rescale + p * row[i];
                    m = m_new;
                }
                const float inv_l = (l > 0.0f) ? 1.0f / l : 0.0f;
                for (int i = 0; i < head_dim; ++i) out_usm[out_off + i] *= inv_l;
            });
    }).wait();
})

void rsl_flash_attn_decode_tq_usm(rsl_stream* s,
                                  const float* q_usm,
                                  const void* k_packed_usm,
                                  const void* v_packed_usm,
                                  const float* k_scales_usm,
                                  const float* v_scales_usm,
                                  int bits,
                                  float* out_usm,
                                  int n_heads, int n_kv_heads,
                                  int head_dim, int max_ctx,
                                  int kv_len) RSL_FFI_BODY_VOID("rsl_flash_attn_decode_tq_usm", {
    if (s == nullptr || q_usm == nullptr || k_packed_usm == nullptr
        || v_packed_usm == nullptr || k_scales_usm == nullptr
        || v_scales_usm == nullptr || out_usm == nullptr) return;
    if (bits != 1 && bits != 2 && bits != 4 && bits != 8) return;
    if (n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0 || max_ctx <= 0) return;
    if ((n_heads % n_kv_heads) != 0) return;
    if (head_dim > RSL_FLASH_MAX_HEAD_DIM || (head_dim & (head_dim - 1)) != 0) return;
    auto& q = s->q;
    if (kv_len <= 0) {
        std::size_t total = static_cast<std::size_t>(n_heads) * head_dim;
        q.memset(out_usm, 0, total * sizeof(float)).wait();
        return;
    }
    const int n_gqa = n_heads / n_kv_heads;
    const float scale = 1.0f / sycl::sqrt(static_cast<float>(head_dim));
    const int bytes_per_row = (head_dim * bits + 7) / 8;
    const uint8_t* k_bytes = static_cast<const uint8_t*>(k_packed_usm);
    const uint8_t* v_bytes = static_cast<const uint8_t*>(v_packed_usm);
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(round_up_to_lws(n_heads)),
                              sycl::range<1>(RSL_LWS)),
            [=](sycl::nd_item<1> it) {
                const int hh = static_cast<int>(it.get_global_id(0));
                if (hh >= n_heads) return;
                const int kv_h = hh / n_gqa;
                const int q_base = hh * head_dim;
                const int out_base = hh * head_dim;
                float row[RSL_FLASH_MAX_HEAD_DIM];
                for (int i = 0; i < head_dim; ++i) out_usm[out_base + i] = 0.0f;
                float m = -INFINITY, l = 0.0f;
                for (int t = 0; t < kv_len; ++t) {
                    const std::size_t ridx = static_cast<std::size_t>(kv_h * max_ctx + t);
                    deq_tq_row(k_bytes + ridx * bytes_per_row, k_scales_usm[ridx], bits, row, head_dim);
                    float s_dot = 0.0f;
                    for (int i = 0; i < head_dim; ++i) s_dot += q_usm[q_base + i] * row[i];
                    s_dot *= scale;
                    const float m_new = sycl::fmax(m, s_dot);
                    const float rescale = sycl::isfinite(m) ? sycl::exp(m - m_new) : 0.0f;
                    const float p = sycl::exp(s_dot - m_new);
                    l = l * rescale + p;
                    deq_tq_row(v_bytes + ridx * bytes_per_row, v_scales_usm[ridx], bits, row, head_dim);
                    for (int i = 0; i < head_dim; ++i)
                        out_usm[out_base + i] = out_usm[out_base + i] * rescale + p * row[i];
                    m = m_new;
                }
                const float inv_l = (l > 0.0f) ? 1.0f / l : 0.0f;
                for (int i = 0; i < head_dim; ++i) out_usm[out_base + i] *= inv_l;
            });
    }).wait();
})

void rsl_flash_attn_prefill_tq_usm(rsl_stream* s,
                                   const float* q_usm,
                                   const void* k_packed_usm,
                                   const void* v_packed_usm,
                                   const float* k_scales_usm,
                                   const float* v_scales_usm,
                                   int bits,
                                   float* out_usm,
                                   int n_heads, int n_kv_heads,
                                   int head_dim, int max_ctx,
                                   int kv_len_base, int n_new) RSL_FFI_BODY_VOID("rsl_flash_attn_prefill_tq_usm", {
    if (s == nullptr || q_usm == nullptr || k_packed_usm == nullptr
        || v_packed_usm == nullptr || k_scales_usm == nullptr
        || v_scales_usm == nullptr || out_usm == nullptr) return;
    if (bits != 1 && bits != 2 && bits != 4 && bits != 8) return;
    if (n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0 || max_ctx <= 0 || n_new <= 0) return;
    if ((n_heads % n_kv_heads) != 0) return;
    if (head_dim > RSL_FLASH_MAX_HEAD_DIM || (head_dim & (head_dim - 1)) != 0) return;
    if (kv_len_base < 0 || kv_len_base + n_new > max_ctx) return;
    auto& q = s->q;
    const int n_gqa = n_heads / n_kv_heads;
    const float scale = 1.0f / sycl::sqrt(static_cast<float>(head_dim));
    const int bytes_per_row = (head_dim * bits + 7) / 8;
    const uint8_t* k_bytes = static_cast<const uint8_t*>(k_packed_usm);
    const uint8_t* v_bytes = static_cast<const uint8_t*>(v_packed_usm);
    const std::size_t global_h = round_up_to_lws(n_heads);
    sycl::range<2> global(global_h, static_cast<std::size_t>(n_new));
    sycl::range<2> local(RSL_LWS, 1);
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<2>(global, local),
            [=](sycl::nd_item<2> it) {
                const int hh = static_cast<int>(it.get_global_id(0));
                const int q_pos = static_cast<int>(it.get_global_id(1));
                if (hh >= n_heads || q_pos >= n_new) return;
                const int kv_h = hh / n_gqa;
                const int q_off = (q_pos * n_heads + hh) * head_dim;
                const int out_off = q_off;
                const int kv_len_for_q = kv_len_base + q_pos + 1;
                float row[RSL_FLASH_MAX_HEAD_DIM];
                for (int i = 0; i < head_dim; ++i) out_usm[out_off + i] = 0.0f;
                float m = -INFINITY, l = 0.0f;
                for (int t = 0; t < kv_len_for_q; ++t) {
                    const std::size_t ridx = static_cast<std::size_t>(kv_h * max_ctx + t);
                    deq_tq_row(k_bytes + ridx * bytes_per_row, k_scales_usm[ridx], bits, row, head_dim);
                    float s_dot = 0.0f;
                    for (int i = 0; i < head_dim; ++i) s_dot += q_usm[q_off + i] * row[i];
                    s_dot *= scale;
                    const float m_new = sycl::fmax(m, s_dot);
                    const float rescale = sycl::isfinite(m) ? sycl::exp(m - m_new) : 0.0f;
                    const float p = sycl::exp(s_dot - m_new);
                    l = l * rescale + p;
                    deq_tq_row(v_bytes + ridx * bytes_per_row, v_scales_usm[ridx], bits, row, head_dim);
                    for (int i = 0; i < head_dim; ++i)
                        out_usm[out_off + i] = out_usm[out_off + i] * rescale + p * row[i];
                    m = m_new;
                }
                const float inv_l = (l > 0.0f) ? 1.0f / l : 0.0f;
                for (int i = 0; i < head_dim; ++i) out_usm[out_off + i] *= inv_l;
            });
    }).wait();
})

// FlashAttention v2 prefill (sub-group cooperation, F32 K/V cache).
//
// Same sub-group striping pattern as `rsl_flash_attn_decode_v2_usm`:
// SG_SIZE = 16 work-items cooperate on one (head, q_pos) pair, each
// holding head_dim/SG_SIZE elements of Q + accumulator in private
// registers. Per-kv-position Q·K dot is computed via partial sums +
// `sycl::reduce_over_group`. Causal mask still drives the per-q
// loop bound: query at q_pos attends `[0, kv_len_base + q_pos]`
// inclusive.
//
// Dispatch shape: nd_range<2> = (n_heads*SG_SIZE, n_new) with
// local = (SG_SIZE, 1). Each (head, q_pos) gets its own sub-group;
// no cross-WI dependencies between rows.
//
// Constraints (caller-enforced; dispatch falls back to v1 on
// mismatch):
//   - head_dim % SG_SIZE == 0
//   - head_dim <= 256 (MAX_DIM_PER_LANE)
//   - n_heads % n_kv_heads == 0 (GQA)
//
// Same numerical-equivalence claim as the v2 decode: outputs match
// the v1 / CPU reference to within fp16 reduction tolerance, but
// not bit-identically (parallel partial sums vs serial accumulate).
void rsl_flash_attn_prefill_v2_usm(rsl_stream* s,
                                   const float* q_usm,
                                   const float* k_cache_usm,
                                   const float* v_cache_usm,
                                   float* out_usm,
                                   int n_heads, int n_kv_heads,
                                   int head_dim, int max_ctx,
                                   int kv_len_base, int n_new) RSL_FFI_BODY_VOID("rsl_flash_attn_prefill_v2_usm", {
    if (s == nullptr || q_usm == nullptr || k_cache_usm == nullptr
        || v_cache_usm == nullptr || out_usm == nullptr) return;
    if (n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0 || max_ctx <= 0) return;
    if (n_new <= 0) return;
    if ((n_heads % n_kv_heads) != 0) return;
    if (kv_len_base < 0 || kv_len_base + n_new > max_ctx) return;

    constexpr int SG_SIZE = 16;
    constexpr int MAX_DIM_PER_LANE = 16;  // head_dim <= 256
    if ((head_dim % SG_SIZE) != 0) return;
    const int dim_per_lane = head_dim / SG_SIZE;
    if (dim_per_lane > MAX_DIM_PER_LANE) return;

    const int n_gqa = n_heads / n_kv_heads;
    const float scale = 1.0f / sycl::sqrt(static_cast<float>(head_dim));
    auto& queue = s->q;

    // 2D dispatch: x = n_heads * SG_SIZE (one SG per head), y = n_new
    // (one row of SGs per new query position). Local = (SG_SIZE, 1).
    sycl::range<2> global(static_cast<std::size_t>(n_heads) * SG_SIZE,
                          static_cast<std::size_t>(n_new));
    sycl::range<2> local(SG_SIZE, 1);
    queue.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<2>(global, local),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                auto sg = it.get_sub_group();
                const int hh = static_cast<int>(it.get_group(0));
                const int q_pos = static_cast<int>(it.get_global_id(1));
                const int lane = static_cast<int>(it.get_local_id(0));
                if (hh >= n_heads || q_pos >= n_new) return;
                const int kv_h = hh / n_gqa;
                const int q_base = (q_pos * n_heads + hh) * head_dim;
                const int out_base = q_base;
                const int lane_off = lane * dim_per_lane;
                // Causal: query at q_pos attends positions
                // `[0, kv_len_base + q_pos]` inclusive (own row
                // included since the caller appended the new K/V
                // rows before invocation).
                const int kv_len_for_q = kv_len_base + q_pos + 1;

                float q_slice[MAX_DIM_PER_LANE];
                for (int d = 0; d < dim_per_lane; ++d) {
                    q_slice[d] = q_usm[q_base + lane_off + d];
                }
                float out_slice[MAX_DIM_PER_LANE];
                for (int d = 0; d < dim_per_lane; ++d) {
                    out_slice[d] = 0.0f;
                }
                float m_state = -INFINITY;
                float l_state = 0.0f;

                for (int t = 0; t < kv_len_for_q; ++t) {
                    const int kv_off = (kv_h * max_ctx + t) * head_dim;
                    float partial = 0.0f;
                    for (int d = 0; d < dim_per_lane; ++d) {
                        partial += q_slice[d] * k_cache_usm[kv_off + lane_off + d];
                    }
                    float full_dot = sycl::reduce_over_group(
                        sg, partial, sycl::plus<float>());
                    full_dot *= scale;
                    const float m_new = sycl::fmax(m_state, full_dot);
                    const float rescale =
                        sycl::isfinite(m_state) ? sycl::exp(m_state - m_new) : 0.0f;
                    const float p = sycl::exp(full_dot - m_new);
                    l_state = l_state * rescale + p;
                    m_state = m_new;
                    for (int d = 0; d < dim_per_lane; ++d) {
                        out_slice[d] = out_slice[d] * rescale
                            + p * v_cache_usm[kv_off + lane_off + d];
                    }
                }

                const float inv_l = (l_state > 0.0f) ? 1.0f / l_state : 0.0f;
                for (int d = 0; d < dim_per_lane; ++d) {
                    out_usm[out_base + lane_off + d] = out_slice[d] * inv_l;
                }
            });
    }).wait();
})

// FlashAttention v3 prefill (SLM K/V tiling, F32 K/V cache).
//
// Combines v2's sub-group cooperation with SLM-resident K/V tiles
// loaded cooperatively per outer block of kv positions. Same
// constraints as v3 decode (head_dim % 16 == 0, ≤ 256). 2D dispatch
// `(n_heads*SG_SIZE, n_new)` with one SG per (head, q_pos); each SG
// owns one work-group so SLM tiles aren't shared across rows of
// queries.
//
// Per-q causal mask drives a tighter inner-loop bound:
// `kv_len_for_q = kv_len_base + q_pos + 1`. Tiles past that point
// are skipped entirely; partial tiles at the boundary still load
// the whole tile (cooperative load) but the inner loop processes
// only the in-mask positions.
void rsl_flash_attn_prefill_v3_usm(rsl_stream* s,
                                   const float* q_usm,
                                   const float* k_cache_usm,
                                   const float* v_cache_usm,
                                   float* out_usm,
                                   int n_heads, int n_kv_heads,
                                   int head_dim, int max_ctx,
                                   int kv_len_base, int n_new) RSL_FFI_BODY_VOID("rsl_flash_attn_prefill_v3_usm", {
    if (s == nullptr || q_usm == nullptr || k_cache_usm == nullptr
        || v_cache_usm == nullptr || out_usm == nullptr) return;
    if (n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0 || max_ctx <= 0) return;
    if (n_new <= 0) return;
    if ((n_heads % n_kv_heads) != 0) return;
    if (kv_len_base < 0 || kv_len_base + n_new > max_ctx) return;

    constexpr int SG_SIZE = 16;
    constexpr int MAX_DIM_PER_LANE = 16;
    constexpr int KV_TILE = 32;
    if ((head_dim % SG_SIZE) != 0) return;
    const int dim_per_lane = head_dim / SG_SIZE;
    if (dim_per_lane > MAX_DIM_PER_LANE) return;

    const int n_gqa = n_heads / n_kv_heads;
    const float scale = 1.0f / sycl::sqrt(static_cast<float>(head_dim));
    const std::size_t slm_elems = static_cast<std::size_t>(KV_TILE) * head_dim;

    auto& queue = s->q;
    sycl::range<2> global(static_cast<std::size_t>(n_heads) * SG_SIZE,
                          static_cast<std::size_t>(n_new));
    sycl::range<2> local(SG_SIZE, 1);
    queue.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> k_tile(sycl::range<1>(slm_elems), h);
        sycl::local_accessor<float, 1> v_tile(sycl::range<1>(slm_elems), h);
        h.parallel_for(
            sycl::nd_range<2>(global, local),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                auto sg = it.get_sub_group();
                const int hh = static_cast<int>(it.get_group(0));
                const int q_pos = static_cast<int>(it.get_global_id(1));
                const int lane = static_cast<int>(it.get_local_id(0));
                if (hh >= n_heads || q_pos >= n_new) return;
                const int kv_h = hh / n_gqa;
                const int q_base = (q_pos * n_heads + hh) * head_dim;
                const int out_base = q_base;
                const int lane_off = lane * dim_per_lane;
                const int kv_len_for_q = kv_len_base + q_pos + 1;

                float q_slice[MAX_DIM_PER_LANE];
                for (int d = 0; d < dim_per_lane; ++d) {
                    q_slice[d] = q_usm[q_base + lane_off + d];
                }
                float out_slice[MAX_DIM_PER_LANE];
                for (int d = 0; d < dim_per_lane; ++d) {
                    out_slice[d] = 0.0f;
                }
                float m_state = -INFINITY;
                float l_state = 0.0f;

                for (int kv_base = 0; kv_base < kv_len_for_q; kv_base += KV_TILE) {
                    const int kv_block_end = (kv_base + KV_TILE < kv_len_for_q)
                        ? (kv_base + KV_TILE) : kv_len_for_q;
                    const int kv_block_size = kv_block_end - kv_base;
                    const int tile_elems = kv_block_size * head_dim;

                    for (int i = lane; i < tile_elems; i += SG_SIZE) {
                        const int t = i / head_dim;
                        const int d = i % head_dim;
                        const int src_off = (kv_h * max_ctx + kv_base + t) * head_dim + d;
                        k_tile[i] = k_cache_usm[src_off];
                        v_tile[i] = v_cache_usm[src_off];
                    }
                    sycl::group_barrier(it.get_group());

                    for (int t_off = 0; t_off < kv_block_size; ++t_off) {
                        const int tile_off = t_off * head_dim;
                        float partial = 0.0f;
                        for (int d = 0; d < dim_per_lane; ++d) {
                            partial += q_slice[d] * k_tile[tile_off + lane_off + d];
                        }
                        float full_dot = sycl::reduce_over_group(
                            sg, partial, sycl::plus<float>());
                        full_dot *= scale;
                        const float m_new = sycl::fmax(m_state, full_dot);
                        const float rescale =
                            sycl::isfinite(m_state) ? sycl::exp(m_state - m_new) : 0.0f;
                        const float p = sycl::exp(full_dot - m_new);
                        l_state = l_state * rescale + p;
                        m_state = m_new;
                        for (int d = 0; d < dim_per_lane; ++d) {
                            out_slice[d] = out_slice[d] * rescale
                                + p * v_tile[tile_off + lane_off + d];
                        }
                    }
                    sycl::group_barrier(it.get_group());
                }

                const float inv_l = (l_state > 0.0f) ? 1.0f / l_state : 0.0f;
                for (int d = 0; d < dim_per_lane; ++d) {
                    out_usm[out_base + lane_off + d] = out_slice[d] * inv_l;
                }
            });
    }).wait();
})

// USM-resident Q8_0 weight × F32 activation matvec.
// One work-item per output row. Each walks K = (K/32) blocks of
// 32 i8 weights, dequants inline (× per-block f32 scale), and
// accumulates the dot product with the f32 activation.
void rsl_matvec_q8_0_f32_usm(rsl_stream* s,
                             const int8_t* w_q_usm,
                             const float* w_scales_usm,
                             const float* x_usm,
                             float* out_usm,
                             int M, int K) RSL_FFI_BODY_VOID("rsl_matvec_q8_0_f32_usm", {
    if (s == nullptr || w_q_usm == nullptr || w_scales_usm == nullptr
        || x_usm == nullptr || out_usm == nullptr) {
        return;
    }
    if (M <= 0 || K <= 0 || (K % 32) != 0) {
        return;
    }
    auto& q = s->q;
    const int blocks_per_row = K / 32;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(round_up_to_lws(M)),
                              sycl::range<1>(RSL_LWS)),
            [=](sycl::nd_item<1> it) {
                const int m = static_cast<int>(it.get_global_id(0));
                if (m >= M) return;
                const int row_off = m * K;
                const int scales_off = m * blocks_per_row;
                float acc = 0.0f;
                for (int b = 0; b < blocks_per_row; ++b) {
                    const float scale = w_scales_usm[scales_off + b];
                    float block_dot = 0.0f;
                    const int b_off = row_off + b * 32;
                    const int x_off = b * 32;
                    for (int d = 0; d < 32; ++d) {
                        const int8_t w_i8 = w_q_usm[b_off + d];
                        block_dot += static_cast<float>(w_i8) * x_usm[x_off + d];
                    }
                    acc += scale * block_dot;
                }
                out_usm[m] = acc;
            });
    }).wait();
})

// ============================================================
// Level Zero host-memory import
// ============================================================
//
// Path that lets the engine reuse GGUF mmap'd pages directly as a
// device-accessible USM allocation instead of memcpy'ing every weight
// tensor into a fresh `sycl::malloc_shared` region. On Iris Xe shared
// memory this saves ~4 GB of RAM duplication for a 7B Q4_K_M model
// + ~150 ms of memcpy at load time. On dGPUs with separate VRAM it
// would be a smaller win (the memcpy is over PCIe vs in-DRAM), but
// the same code path applies.
//
// The mechanism: `zeMemAllocHost` accepts a chained
// `ze_external_memory_import_win32_handle_t` descriptor that asks
// the L0 driver to import an existing Win32 NT handle (typically a
// file-mapping SECTION handle from `CreateFileMappingW`) as device-
// accessible host memory. The resulting pointer is owned by L0 (must
// be freed with `zeMemFree`) but the underlying pages are still the
// same physical RAM as the host mmap.
//
// We bypass any L0 SDK / header dependency: the oneAPI compiler
// install doesn't bundle `level_zero/ze_api.h`, and the driver-
// supplied `ze_loader.dll` doesn't ship a matching `.lib`. Instead
// we hand-roll the few types we need (mirroring the upstream
// `ze_api.h` v1.13) and resolve the function pointers at runtime via
// `GetProcAddress`. Failure modes degrade cleanly: the engine's
// caller checks the return code and falls back to its existing
// copy-to-USM path.
//
// Lifetime: imported pointers must be freed with
// `rsl_release_imported_usm` before the SYCL queue's context is
// destroyed. The engine pairs each `rsl_try_import_*` with a release
// on tensor eviction / model unload.

namespace {

// Type aliases for the L0 ABI subset we use. Hand-rolled to match
// the upstream `level_zero/ze_api.h` v1.13 — see comments below for
// each struct's spec-defined field order.
using rsl_ze_result_t = uint32_t;
using rsl_ze_structure_type_t = uint32_t;
using rsl_ze_host_mem_alloc_flags_t = uint32_t;
using rsl_ze_external_memory_type_flags_t = uint32_t;

// Spec constants pinned to their numeric values so this TU has no
// header dependency. `ZE_BIT(n)` in the upstream header expands to
// `(1 << n)`.
//   ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC                = 0x16 (ze_api.h line 290)
//   ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_WIN32       = 0x22 (ze_api.h line 302)
//   ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32            = ZE_BIT(2) = 4 ("an NT handle")
constexpr rsl_ze_result_t RSL_ZE_RESULT_SUCCESS = 0;
constexpr rsl_ze_structure_type_t RSL_ZE_STYPE_HOST_MEM_ALLOC_DESC = 0x16;
constexpr rsl_ze_structure_type_t RSL_ZE_STYPE_EXTERNAL_MEMORY_IMPORT_WIN32 = 0x22;
constexpr rsl_ze_external_memory_type_flags_t RSL_ZE_EXT_MEM_TYPE_OPAQUE_WIN32 = (1u << 2);

// Layout from `typedef struct _ze_host_mem_alloc_desc_t` at
// `ze_api.h` line 6409. Three fields: stype, pNext, flags.
struct rsl_ze_host_mem_alloc_desc {
    rsl_ze_structure_type_t stype;
    const void* pNext;
    rsl_ze_host_mem_alloc_flags_t flags;
};

// Layout from `typedef struct _ze_external_memory_import_win32_handle_t`
// at `ze_api.h` line 7051. Five fields: stype, pNext, flags, handle, name.
// `name` is `const void*` (NOT `const wchar_t*`) per the spec — used
// only for named imports; we always pass `nullptr` since we have the
// HANDLE directly.
struct rsl_ze_external_memory_import_win32_handle {
    rsl_ze_structure_type_t stype;
    const void* pNext;
    rsl_ze_external_memory_type_flags_t flags;
    void* handle;
    const void* name;
};

// L0 function pointer types. Calling convention: L0 v1.13 defines
// `ZE_APICALL` as the default platform calling convention. On x64
// Windows this is the standard ABI, which Rust/C++ `extern "C"`
// already matches. (32-bit Windows would need `__stdcall`; we don't
// support 32-bit.)
using rsl_zeMemAllocHost_fn = rsl_ze_result_t (*)(
    ze_context_handle_t hContext,
    const rsl_ze_host_mem_alloc_desc* host_desc,
    size_t size,
    size_t alignment,
    void** pptr);

using rsl_zeMemFree_fn = rsl_ze_result_t (*)(
    ze_context_handle_t hContext,
    void* ptr);

// Process-singleton loader for the two L0 entry points we use. The
// constructor runs once (via the meyers-singleton pattern in a
// `static` local) and resolves `zeMemAllocHost` + `zeMemFree` from
// `ze_loader.dll`. On non-Windows or when the DLL / symbols are
// missing, `available` stays false and every subsequent
// `rsl_try_import_*` call returns the "L0 unavailable" code.
struct L0ImportLoader {
    rsl_zeMemAllocHost_fn mem_alloc_host = nullptr;
    rsl_zeMemFree_fn mem_free = nullptr;
    bool available = false;
};

const L0ImportLoader& get_l0_import_loader() {
    static const L0ImportLoader loader = []() -> L0ImportLoader {
#ifdef _WIN32
        HMODULE h = LoadLibraryW(L"ze_loader.dll");
        if (h == nullptr) {
            return {};
        }
        auto alloc = reinterpret_cast<rsl_zeMemAllocHost_fn>(
            reinterpret_cast<void*>(GetProcAddress(h, "zeMemAllocHost")));
        auto free_fn = reinterpret_cast<rsl_zeMemFree_fn>(
            reinterpret_cast<void*>(GetProcAddress(h, "zeMemFree")));
        if (alloc == nullptr || free_fn == nullptr) {
            return {};
        }
        return L0ImportLoader{alloc, free_fn, true};
#else
        // Linux path TBD — `libze_loader.so.1` + `dlopen`/`dlsym`.
        // Not implemented in Step 1b since the user's target is
        // Windows-first; can be added without changing the calling
        // contract.
        return {};
#endif
    }();
    return loader;
}

}  // namespace

// Try to import an existing Win32 file-mapping HANDLE (typically from
// `CreateFileMappingW` over a GGUF file) as a device-accessible host
// memory allocation. On success, `*out_dev_ptr` is a pointer that
// can be passed to USM kernels exactly as if it were the result of
// `sycl::malloc_shared(size, queue)`. The backing pages are the same
// physical RAM as the host mmap — no memcpy involved.
//
// Return codes:
//   0  success; `*out_dev_ptr` is set
//   1  invalid arguments (null stream, handle, or output)
//   2  the SYCL queue isn't backed by Level Zero (e.g. OpenCL fallback)
//   3  the L0 loader DLL or its `zeMemAllocHost` symbol isn't available
//   4  `zeMemAllocHost` returned a non-success code; driver doesn't
//      support importing this handle type, or some other runtime error
int rsl_try_import_win32_handle_as_usm(
    rsl_stream* stream,
    void* mapping_handle,
    size_t size,
    void** out_dev_ptr) RSL_FFI_BODY_RET(
    "rsl_try_import_win32_handle_as_usm", 1, {
    if (stream == nullptr || mapping_handle == nullptr
        || out_dev_ptr == nullptr || size == 0) {
        return 1;
    }
    auto backend = stream->q.get_backend();
    if (backend != sycl::backend::ext_oneapi_level_zero) {
        // Diagnostic: log the numeric backend value so the caller
        // can tell OpenCL (commonly 2) from CUDA / HIP / etc. The
        // sycl::backend enum's underlying type is implementation-
        // defined; cast through int for the printout.
        std::fprintf(stderr,
            "[rsl-sycl] rsl_try_import_win32_handle_as_usm: queue "
            "backend is %d (need ext_oneapi_level_zero = %d). Set "
            "ONEAPI_DEVICE_SELECTOR=level_zero:gpu before stream "
            "creation to force L0.\n",
            static_cast<int>(backend),
            static_cast<int>(sycl::backend::ext_oneapi_level_zero));
        return 2;
    }
    const L0ImportLoader& loader = get_l0_import_loader();
    if (!loader.available) {
        return 3;
    }
    // SYCL → L0 interop: extract the underlying ze_context_handle_t.
    // The header forward-declares this type and the backend trait
    // specialization in `backend_traits_level_zero.hpp` maps
    // `backend::ext_oneapi_level_zero + context` to it.
    auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
        stream->q.get_context());
    rsl_ze_external_memory_import_win32_handle import_desc = {
        RSL_ZE_STYPE_EXTERNAL_MEMORY_IMPORT_WIN32,
        nullptr,
        RSL_ZE_EXT_MEM_TYPE_OPAQUE_WIN32,
        mapping_handle,
        nullptr,
    };
    rsl_ze_host_mem_alloc_desc host_desc = {
        RSL_ZE_STYPE_HOST_MEM_ALLOC_DESC,
        &import_desc,
        0u,
    };
    void* ptr = nullptr;
    rsl_ze_result_t r = loader.mem_alloc_host(ze_ctx, &host_desc, size, 0, &ptr);
    if (r != RSL_ZE_RESULT_SUCCESS || ptr == nullptr) {
        // Stash the actual L0 result code so Rust can log it via
        // tracing (and so we can distinguish e.g. UNSUPPORTED_FEATURE
        // = 0x78000003 from INVALID_ARGUMENT = 0x78000004 from
        // OUT_OF_HOST_MEMORY = 0x70000002). Without this side
        // channel the engine just sees "code 4" and can't tell what
        // the driver actually objected to.
        g_rsl_last_l0_import_code = r;
        std::fprintf(stderr,
            "[rsl-sycl] rsl_try_import_win32_handle_as_usm: "
            "zeMemAllocHost returned 0x%x (size=%zu)\n",
            static_cast<unsigned>(r), size);
        return 4;
    }
    g_rsl_last_l0_import_code = 0;
    *out_dev_ptr = ptr;
    return 0;
})

// Diagnostic-only: try a bare `zeMemAllocHost` (no import
// descriptor chained on pNext) to verify our basic struct layout +
// L0 call sequence are correct. If THIS fails, the problem is our
// own code; if it succeeds but `rsl_try_import_win32_handle_as_usm`
// fails, the problem is specifically the import descriptor or the
// handle origin. Returns the same category codes as the import call
// (0 success / 2 not L0 / 3 loader missing / 4 driver rejected),
// and stashes the L0 result code in the same side channel.
int rsl_try_alloc_host_baseline(rsl_stream* stream, size_t size, void** out_dev_ptr)
    RSL_FFI_BODY_RET("rsl_try_alloc_host_baseline", 1, {
    if (stream == nullptr || out_dev_ptr == nullptr || size == 0) {
        return 1;
    }
    if (stream->q.get_backend() != sycl::backend::ext_oneapi_level_zero) {
        return 2;
    }
    const L0ImportLoader& loader = get_l0_import_loader();
    if (!loader.available) {
        return 3;
    }
    auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
        stream->q.get_context());
    // host_desc with no pNext — pure base allocation.
    rsl_ze_host_mem_alloc_desc host_desc = {
        RSL_ZE_STYPE_HOST_MEM_ALLOC_DESC,
        nullptr,
        0u,
    };
    void* ptr = nullptr;
    rsl_ze_result_t r = loader.mem_alloc_host(ze_ctx, &host_desc, size, 0, &ptr);
    if (r != RSL_ZE_RESULT_SUCCESS || ptr == nullptr) {
        g_rsl_last_l0_import_code = r;
        return 4;
    }
    g_rsl_last_l0_import_code = 0;
    // Free immediately — this is just a sanity probe.
    loader.mem_free(ze_ctx, ptr);
    *out_dev_ptr = ptr;  // returned for the caller's info but the
                         // allocation is already freed; don't deref.
    return 0;
})

// Free a pointer obtained from `rsl_try_import_win32_handle_as_usm`.
// Safe to call with `dev_ptr == nullptr` (no-op). Must run on the
// same SYCL queue's context that produced the import — the L0 spec
// requires the context match.
void rsl_release_imported_usm(rsl_stream* stream, void* dev_ptr)
    RSL_FFI_BODY_VOID("rsl_release_imported_usm", {
    if (stream == nullptr || dev_ptr == nullptr) {
        return;
    }
    const L0ImportLoader& loader = get_l0_import_loader();
    if (!loader.available) {
        return;
    }
    if (stream->q.get_backend() != sycl::backend::ext_oneapi_level_zero) {
        return;
    }
    auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
        stream->q.get_context());
    rsl_ze_result_t r = loader.mem_free(ze_ctx, dev_ptr);
    if (r != RSL_ZE_RESULT_SUCCESS) {
        std::fprintf(stderr,
            "[rsl-sycl] rsl_release_imported_usm: zeMemFree returned 0x%x\n",
            static_cast<unsigned>(r));
    }
})

// USM-resident Q8_0 packed matvec. Same algorithm as the separated
// variant, but reads the raw GGUF on-disk block layout (34 bytes per
// block: 2-byte f16 scale + 32 i8 weights). No per-load repack
// needed — the engine just memcpy's the GGUF tensor bytes into USM
// once and dispatches against the same pointer for every forward.
// Close extern "C" so the template below has C++ linkage. Re-opened
// just before the dispatcher (same pattern as Q4_K above).
}  // extern "C"

template <std::size_t LWS_T>
inline void matvec_q8_0_packed_f32_usm_impl(
    sycl::queue& q,
    const void* w_bytes_usm,
    const float* x_usm,
    float* out_usm,
    int M, int K) {
    constexpr std::size_t LWS = LWS_T;
    const int blocks_per_row = K / 32;
    const int bytes_per_row = blocks_per_row * 34;
    const uint8_t* w_bytes = static_cast<const uint8_t*>(w_bytes_usm);
    const std::size_t global =
        ((static_cast<std::size_t>(M) + LWS - 1) / LWS) * LWS;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(LWS)),
            [=](sycl::nd_item<1> it) {
                const int m = static_cast<int>(it.get_global_id(0));
                if (m >= M) return;
                const uint8_t* row = w_bytes + m * bytes_per_row;
                float acc = 0.0f;
                for (int b = 0; b < blocks_per_row; ++b) {
                    const uint8_t* blk = row + b * 34;
                    uint16_t scale_bits =
                        static_cast<uint16_t>(blk[0]) |
                        (static_cast<uint16_t>(blk[1]) << 8);
                    const float scale = bits_to_f32(scale_bits);
                    const int x_off = b * 32;
                    float block_dot = 0.0f;
                    for (int d = 0; d < 32; ++d) {
                        const int8_t w_i8 = static_cast<int8_t>(blk[2 + d]);
                        block_dot += static_cast<float>(w_i8) * x_usm[x_off + d];
                    }
                    acc += scale * block_dot;
                }
                out_usm[m] = acc;
            });
    }).wait();
}

extern "C" {

void rsl_matvec_q8_0_packed_f32_usm(rsl_stream* s,
                                    const void* w_bytes_usm,
                                    const float* x_usm,
                                    float* out_usm,
                                    int M, int K,
                                    int lws) RSL_FFI_BODY_VOID(
    "rsl_matvec_q8_0_packed_f32_usm", {
    if (s == nullptr || w_bytes_usm == nullptr
        || x_usm == nullptr || out_usm == nullptr) {
        return;
    }
    if (M <= 0 || K <= 0 || (K % 32) != 0) {
        return;
    }
    auto& q = s->q;
    const int eff_lws = (lws <= 0) ? static_cast<int>(RSL_LWS) : lws;
    switch (eff_lws) {
        case 16:  matvec_q8_0_packed_f32_usm_impl<16>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 32:  matvec_q8_0_packed_f32_usm_impl<32>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 64:  matvec_q8_0_packed_f32_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 128: matvec_q8_0_packed_f32_usm_impl<128>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 256: matvec_q8_0_packed_f32_usm_impl<256>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        default:  matvec_q8_0_packed_f32_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
    }
})

// H4: Q8_0 gate+up FUSED matvec. Shares x_usm loads across both
// matvecs; one kernel launch instead of two.
}  // extern "C" — close so we can declare a template
template <std::size_t LWS_T>
inline void matvec_q8_0_gate_up_fused_usm_impl(
    sycl::queue& q,
    const void* gate_w_bytes_usm,
    const void* up_w_bytes_usm,
    const float* x_usm,
    float* gate_out_usm,
    float* up_out_usm,
    int M, int K) {
    constexpr std::size_t LWS = LWS_T;
    const int blocks_per_row = K / 32;
    const int bytes_per_row = blocks_per_row * 34;
    const uint8_t* g_bytes = static_cast<const uint8_t*>(gate_w_bytes_usm);
    const uint8_t* u_bytes = static_cast<const uint8_t*>(up_w_bytes_usm);
    const std::size_t global =
        ((static_cast<std::size_t>(M) + LWS - 1) / LWS) * LWS;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(LWS)),
            [=](sycl::nd_item<1> it) {
                const int m = static_cast<int>(it.get_global_id(0));
                if (m >= M) return;
                const uint8_t* g_row = g_bytes + m * bytes_per_row;
                const uint8_t* u_row = u_bytes + m * bytes_per_row;
                float gate_acc = 0.0f;
                float up_acc = 0.0f;
                for (int b = 0; b < blocks_per_row; ++b) {
                    const uint8_t* g_blk = g_row + b * 34;
                    const uint8_t* u_blk = u_row + b * 34;
                    const float g_scale = bits_to_f32(
                        static_cast<uint16_t>(g_blk[0])
                        | (static_cast<uint16_t>(g_blk[1]) << 8));
                    const float u_scale = bits_to_f32(
                        static_cast<uint16_t>(u_blk[0])
                        | (static_cast<uint16_t>(u_blk[1]) << 8));
                    const int x_off = b * 32;
                    float g_block_dot = 0.0f;
                    float u_block_dot = 0.0f;
                    for (int d = 0; d < 32; ++d) {
                        const float xv = x_usm[x_off + d];
                        g_block_dot += static_cast<float>(static_cast<int8_t>(g_blk[2 + d])) * xv;
                        u_block_dot += static_cast<float>(static_cast<int8_t>(u_blk[2 + d])) * xv;
                    }
                    gate_acc += g_scale * g_block_dot;
                    up_acc += u_scale * u_block_dot;
                }
                gate_out_usm[m] = gate_acc;
                up_out_usm[m] = up_acc;
            });
    }).wait();
}
extern "C" {

void rsl_matvec_q8_0_gate_up_fused_usm(rsl_stream* s,
                                       const void* gate_w_bytes_usm,
                                       const void* up_w_bytes_usm,
                                       const float* x_usm,
                                       float* gate_out_usm,
                                       float* up_out_usm,
                                       int M, int K,
                                       int lws) RSL_FFI_BODY_VOID(
    "rsl_matvec_q8_0_gate_up_fused_usm", {
    if (s == nullptr || gate_w_bytes_usm == nullptr || up_w_bytes_usm == nullptr
        || x_usm == nullptr || gate_out_usm == nullptr || up_out_usm == nullptr) {
        return;
    }
    if (M <= 0 || K <= 0 || (K % 32) != 0) {
        return;
    }
    auto& q = s->q;
    const int eff_lws = (lws <= 0) ? static_cast<int>(RSL_LWS) : lws;
    switch (eff_lws) {
        case 16:  matvec_q8_0_gate_up_fused_usm_impl<16>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        case 32:  matvec_q8_0_gate_up_fused_usm_impl<32>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        case 64:  matvec_q8_0_gate_up_fused_usm_impl<64>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        case 128: matvec_q8_0_gate_up_fused_usm_impl<128>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        case 256: matvec_q8_0_gate_up_fused_usm_impl<256>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        default:  matvec_q8_0_gate_up_fused_usm_impl<64>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
    }
})

// USM-resident Q4_K_M packed matvec — templated on local
// work-group size (LWS). The autotuner (phase 5) sweeps the
// candidate set {16, 32, 64, 128, 256} per (device, problem
// shape) and caches the winner; engine call sites pass the cached
// LWS via the `lws` parameter on the FFI entry point. Passing
// `lws == 0` selects the hand-picked default of 64 (Intel iGPU
// Gen12LP sweet spot, matches the typical sub-group size × 8).
//
// One template instantiation per candidate LWS lives in the
// compiled SPIR-V binary. The dispatcher below switches on the
// runtime value. Adding a candidate = add a `case` here.
//
// All instantiations are byte-identical except for the `LWS`
// constexpr baked into the `nd_range`'s local size; the kernel
// body is unchanged, so they produce bit-identical outputs (gated
// by `matvec_q4_k_packed_f32_usm_lws_parity` test).
//
// Mirrors the CPU scalar reference exactly: each super-block of
// 256 weights carries a (d, dmin) f16 pair, 12 bytes of packed
// 6-bit (sc, mn) pairs for 8 sub-blocks, and 128 bytes of 4-bit
// packed weights. Pairs of sub-blocks share a 32-byte qs chunk
// (low/high nibble). One work-item per output row.
// Close the outer extern "C" block (opened at the top of the SYCL
// implementation section) so we can define a C++ template here.
// Templates can't live inside extern "C" — the language forbids
// template declarations with C linkage. We reopen extern "C" right
// before the dispatcher's `extern "C"` definition so its linkage
// continues to match the header's prior declaration.
}  // extern "C" — temporary close for the template

// File-scope (not in anonymous namespace) so the extern "C"
// dispatcher below can see this template after it's been declared.
// `static inline` would give internal linkage but template
// instantiations need to share linkage across the dispatcher's
// switch arms — leaving them as a non-static `inline` template is
// the conventional choice. The template's body still expands once
// per LWS variant via the explicit instantiations in the dispatcher.
template <std::size_t LWS_T>
inline void matvec_q4_k_packed_f32_usm_impl(
    sycl::queue& q,
    const void* w_bytes_usm,
    const float* x_usm,
    float* out_usm,
    int M, int K) {
    constexpr std::size_t LWS = LWS_T;
    const int blocks_per_row = K / 256;
    const int bytes_per_row = blocks_per_row * 144;
    const uint8_t* w_bytes = static_cast<const uint8_t*>(w_bytes_usm);
    const std::size_t global =
        ((static_cast<std::size_t>(M) + LWS - 1) / LWS) * LWS;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(LWS)),
            [=](sycl::nd_item<1> it) {
                const int m = static_cast<int>(it.get_global_id(0));
                if (m >= M) return;
                const uint8_t* row = w_bytes + m * bytes_per_row;
                float acc = 0.0f;
                for (int b = 0; b < blocks_per_row; ++b) {
                    const uint8_t* blk = row + b * 144;
                    uint16_t d_bits = static_cast<uint16_t>(blk[0])
                                      | (static_cast<uint16_t>(blk[1]) << 8);
                    uint16_t dmin_bits = static_cast<uint16_t>(blk[2])
                                         | (static_cast<uint16_t>(blk[3]) << 8);
                    const float d = bits_to_f32(d_bits);
                    const float dmin = bits_to_f32(dmin_bits);
                    const uint8_t* sb = blk + 4;
                    uint8_t sc[8];
                    uint8_t mn[8];
                    for (int j = 0; j < 8; ++j) {
                        if (j < 4) {
                            sc[j] = sb[j] & 0x3F;
                            mn[j] = sb[j + 4] & 0x3F;
                        } else {
                            sc[j] = (sb[j + 4] & 0x0F) | ((sb[j - 4] >> 6) << 4);
                            mn[j] = (sb[j + 4] >> 4) | ((sb[j] >> 6) << 4);
                        }
                    }
                    const uint8_t* qs = blk + 16;
                    const int x_base = b * 256;
                    for (int group = 0; group < 4; ++group) {
                        const uint8_t* qc = qs + group * 32;
                        const float d_lo = d * static_cast<float>(sc[group * 2]);
                        const float m_lo = dmin * static_cast<float>(mn[group * 2]);
                        const float d_hi = d * static_cast<float>(sc[group * 2 + 1]);
                        const float m_hi = dmin * static_cast<float>(mn[group * 2 + 1]);
                        const int x_lo_off = x_base + group * 64;
                        const int x_hi_off = x_lo_off + 32;
                        for (int l = 0; l < 32; ++l) {
                            const uint8_t qb = qc[l];
                            const float q_lo = static_cast<float>(qb & 0x0F);
                            const float q_hi = static_cast<float>(qb >> 4);
                            acc += (d_lo * q_lo - m_lo) * x_usm[x_lo_off + l];
                            acc += (d_hi * q_hi - m_hi) * x_usm[x_hi_off + l];
                        }
                    }
                }
                out_usm[m] = acc;
            });
    }).wait();
}

// Reopen extern "C" so the dispatcher's linkage matches the
// `extern "C"` declaration in the header. The closing brace of
// this re-opened block matches the original outer-block close at
// the end of the SYCL implementation section.
extern "C" {

void rsl_matvec_q4_k_packed_f32_usm(rsl_stream* s,
                                    const void* w_bytes_usm,
                                    const float* x_usm,
                                    float* out_usm,
                                    int M, int K,
                                    int lws) RSL_FFI_BODY_VOID(
    "rsl_matvec_q4_k_packed_f32_usm", {
    if (s == nullptr || w_bytes_usm == nullptr
        || x_usm == nullptr || out_usm == nullptr) {
        return;
    }
    if (M <= 0 || K <= 0 || (K % 256) != 0) {
        return;
    }
    auto& q = s->q;
    // lws == 0 → use the hand-picked default (RSL_LWS). The autotuner
    // overrides this on each call with the per-(device, problem-shape)
    // winner from the cache. Unknown values fall back to the default
    // rather than triggering a kernel launch failure.
    const int eff_lws = (lws <= 0) ? static_cast<int>(RSL_LWS) : lws;
    switch (eff_lws) {
        case 16:
            matvec_q4_k_packed_f32_usm_impl<16>(q, w_bytes_usm, x_usm, out_usm, M, K);
            break;
        case 32:
            matvec_q4_k_packed_f32_usm_impl<32>(q, w_bytes_usm, x_usm, out_usm, M, K);
            break;
        case 64:
            matvec_q4_k_packed_f32_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K);
            break;
        case 128:
            matvec_q4_k_packed_f32_usm_impl<128>(q, w_bytes_usm, x_usm, out_usm, M, K);
            break;
        case 256:
            matvec_q4_k_packed_f32_usm_impl<256>(q, w_bytes_usm, x_usm, out_usm, M, K);
            break;
        default:
            matvec_q4_k_packed_f32_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K);
            break;
    }
})

// H4: Q4_K gate+up FUSED matvec. Same per-block math as the
// single-row Q4_K matvec, but each work-item dispatches TWO dot
// products (against `w_gate` and `w_up`) for the same input row
// `x_usm`. Wins over two separate dispatches on three axes:
//   1. Activation cache: x is loaded once, used for both matvecs.
//   2. Dispatch: one kernel launch instead of two.
//   3. Q-dequant reuse: gate and up share decode-loop overhead.
//
// Both `gate_w_bytes_usm` and `up_w_bytes_usm` must point to Q4_K-
// formatted USM buffers (M × blocks_per_row × 144 bytes each); gate
// and up MUST share the same M and K. Outputs are `gate_out_usm`
// and `up_out_usm`, both `M` f32 elements in USM.
}  // extern "C" — close so we can declare a template
template <std::size_t LWS_T>
inline void matvec_q4_k_gate_up_fused_usm_impl(
    sycl::queue& q,
    const void* gate_w_bytes_usm,
    const void* up_w_bytes_usm,
    const float* x_usm,
    float* gate_out_usm,
    float* up_out_usm,
    int M, int K) {
    constexpr std::size_t LWS = LWS_T;
    const int blocks_per_row = K / 256;
    const int bytes_per_row = blocks_per_row * 144;
    const uint8_t* gate_w_bytes = static_cast<const uint8_t*>(gate_w_bytes_usm);
    const uint8_t* up_w_bytes = static_cast<const uint8_t*>(up_w_bytes_usm);
    const std::size_t global =
        ((static_cast<std::size_t>(M) + LWS - 1) / LWS) * LWS;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(LWS)),
            [=](sycl::nd_item<1> it) {
                const int m = static_cast<int>(it.get_global_id(0));
                if (m >= M) return;
                const uint8_t* gate_row = gate_w_bytes + m * bytes_per_row;
                const uint8_t* up_row   = up_w_bytes   + m * bytes_per_row;
                float gate_acc = 0.0f;
                float up_acc = 0.0f;
                for (int b = 0; b < blocks_per_row; ++b) {
                    // Unpack gate block.
                    const uint8_t* g_blk = gate_row + b * 144;
                    const float g_d = bits_to_f32(static_cast<uint16_t>(g_blk[0])
                                                  | (static_cast<uint16_t>(g_blk[1]) << 8));
                    const float g_dmin = bits_to_f32(static_cast<uint16_t>(g_blk[2])
                                                     | (static_cast<uint16_t>(g_blk[3]) << 8));
                    uint8_t g_sc[8], g_mn[8];
                    {
                        const uint8_t* sb = g_blk + 4;
                        for (int j = 0; j < 8; ++j) {
                            if (j < 4) {
                                g_sc[j] = sb[j] & 0x3F;
                                g_mn[j] = sb[j + 4] & 0x3F;
                            } else {
                                g_sc[j] = (sb[j + 4] & 0x0F) | ((sb[j - 4] >> 6) << 4);
                                g_mn[j] = (sb[j + 4] >> 4) | ((sb[j] >> 6) << 4);
                            }
                        }
                    }
                    const uint8_t* g_qs = g_blk + 16;
                    // Unpack up block (same layout).
                    const uint8_t* u_blk = up_row + b * 144;
                    const float u_d = bits_to_f32(static_cast<uint16_t>(u_blk[0])
                                                  | (static_cast<uint16_t>(u_blk[1]) << 8));
                    const float u_dmin = bits_to_f32(static_cast<uint16_t>(u_blk[2])
                                                     | (static_cast<uint16_t>(u_blk[3]) << 8));
                    uint8_t u_sc[8], u_mn[8];
                    {
                        const uint8_t* sb = u_blk + 4;
                        for (int j = 0; j < 8; ++j) {
                            if (j < 4) {
                                u_sc[j] = sb[j] & 0x3F;
                                u_mn[j] = sb[j + 4] & 0x3F;
                            } else {
                                u_sc[j] = (sb[j + 4] & 0x0F) | ((sb[j - 4] >> 6) << 4);
                                u_mn[j] = (sb[j + 4] >> 4) | ((sb[j] >> 6) << 4);
                            }
                        }
                    }
                    const uint8_t* u_qs = u_blk + 16;
                    const int x_base = b * 256;
                    // Per-group loop. Loads x_usm once per element +
                    // accumulates into both gate and up.
                    for (int group = 0; group < 4; ++group) {
                        const float g_d_lo = g_d * static_cast<float>(g_sc[group * 2]);
                        const float g_m_lo = g_dmin * static_cast<float>(g_mn[group * 2]);
                        const float g_d_hi = g_d * static_cast<float>(g_sc[group * 2 + 1]);
                        const float g_m_hi = g_dmin * static_cast<float>(g_mn[group * 2 + 1]);
                        const float u_d_lo = u_d * static_cast<float>(u_sc[group * 2]);
                        const float u_m_lo = u_dmin * static_cast<float>(u_mn[group * 2]);
                        const float u_d_hi = u_d * static_cast<float>(u_sc[group * 2 + 1]);
                        const float u_m_hi = u_dmin * static_cast<float>(u_mn[group * 2 + 1]);
                        const uint8_t* g_qc = g_qs + group * 32;
                        const uint8_t* u_qc = u_qs + group * 32;
                        const int x_lo_off = x_base + group * 64;
                        const int x_hi_off = x_lo_off + 32;
                        for (int l = 0; l < 32; ++l) {
                            const float x_lo = x_usm[x_lo_off + l];
                            const float x_hi = x_usm[x_hi_off + l];
                            const uint8_t gqb = g_qc[l];
                            gate_acc += (g_d_lo * static_cast<float>(gqb & 0x0F) - g_m_lo) * x_lo;
                            gate_acc += (g_d_hi * static_cast<float>(gqb >> 4)   - g_m_hi) * x_hi;
                            const uint8_t uqb = u_qc[l];
                            up_acc += (u_d_lo * static_cast<float>(uqb & 0x0F) - u_m_lo) * x_lo;
                            up_acc += (u_d_hi * static_cast<float>(uqb >> 4)   - u_m_hi) * x_hi;
                        }
                    }
                }
                gate_out_usm[m] = gate_acc;
                up_out_usm[m] = up_acc;
            });
    }).wait();
}
extern "C" {

void rsl_matvec_q4_k_gate_up_fused_usm(rsl_stream* s,
                                       const void* gate_w_bytes_usm,
                                       const void* up_w_bytes_usm,
                                       const float* x_usm,
                                       float* gate_out_usm,
                                       float* up_out_usm,
                                       int M, int K,
                                       int lws) RSL_FFI_BODY_VOID(
    "rsl_matvec_q4_k_gate_up_fused_usm", {
    if (s == nullptr || gate_w_bytes_usm == nullptr || up_w_bytes_usm == nullptr
        || x_usm == nullptr || gate_out_usm == nullptr || up_out_usm == nullptr) {
        return;
    }
    if (M <= 0 || K <= 0 || (K % 256) != 0) {
        return;
    }
    auto& q = s->q;
    const int eff_lws = (lws <= 0) ? static_cast<int>(RSL_LWS) : lws;
    switch (eff_lws) {
        case 16:  matvec_q4_k_gate_up_fused_usm_impl<16>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        case 32:  matvec_q4_k_gate_up_fused_usm_impl<32>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        case 64:  matvec_q4_k_gate_up_fused_usm_impl<64>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        case 128: matvec_q4_k_gate_up_fused_usm_impl<128>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        case 256: matvec_q4_k_gate_up_fused_usm_impl<256>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        default:  matvec_q4_k_gate_up_fused_usm_impl<64>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
    }
})

// USM-resident Q5_K_M packed matvec. Same super-block-of-256
// layout as Q4_K_M plus a 32-byte `qh` section holding the 5th bit
// per weight. Mirrors the CPU scalar reference exactly.
}  // extern "C" — temporary close for the Q5_K template

template <std::size_t LWS_T>
inline void matvec_q5_k_packed_f32_usm_impl(
    sycl::queue& q,
    const void* w_bytes_usm,
    const float* x_usm,
    float* out_usm,
    int M, int K) {
    constexpr std::size_t LWS = LWS_T;
    const int blocks_per_row = K / 256;
    const int bytes_per_row = blocks_per_row * 176;
    const uint8_t* w_bytes = static_cast<const uint8_t*>(w_bytes_usm);
    const std::size_t global =
        ((static_cast<std::size_t>(M) + LWS - 1) / LWS) * LWS;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(LWS)),
            [=](sycl::nd_item<1> it) {
                const int m = static_cast<int>(it.get_global_id(0));
                if (m >= M) return;
                const uint8_t* row = w_bytes + m * bytes_per_row;
                float acc = 0.0f;
                for (int b = 0; b < blocks_per_row; ++b) {
                    const uint8_t* blk = row + b * 176;
                    uint16_t d_bits = static_cast<uint16_t>(blk[0])
                                      | (static_cast<uint16_t>(blk[1]) << 8);
                    uint16_t dmin_bits = static_cast<uint16_t>(blk[2])
                                         | (static_cast<uint16_t>(blk[3]) << 8);
                    const float d = bits_to_f32(d_bits);
                    const float dmin = bits_to_f32(dmin_bits);
                    const uint8_t* sb = blk + 4;
                    uint8_t sc[8];
                    uint8_t mn[8];
                    for (int j = 0; j < 8; ++j) {
                        if (j < 4) {
                            sc[j] = sb[j] & 0x3F;
                            mn[j] = sb[j + 4] & 0x3F;
                        } else {
                            sc[j] = (sb[j + 4] & 0x0F) | ((sb[j - 4] >> 6) << 4);
                            mn[j] = (sb[j + 4] >> 4) | ((sb[j] >> 6) << 4);
                        }
                    }
                    const uint8_t* qh = blk + 16;
                    const uint8_t* qs = blk + 48;
                    const int x_base = b * 256;
                    for (int group = 0; group < 4; ++group) {
                        const uint8_t* qc = qs + group * 32;
                        const float d_lo = d * static_cast<float>(sc[group * 2]);
                        const float m_lo = dmin * static_cast<float>(mn[group * 2]);
                        const float d_hi = d * static_cast<float>(sc[group * 2 + 1]);
                        const float m_hi = dmin * static_cast<float>(mn[group * 2 + 1]);
                        const int bit_lo = group * 2;
                        const int bit_hi = group * 2 + 1;
                        const int x_lo_off = x_base + group * 64;
                        const int x_hi_off = x_lo_off + 32;
                        for (int l = 0; l < 32; ++l) {
                            const uint8_t qb = qc[l];
                            const uint8_t qhb = qh[l];
                            const uint32_t lo = static_cast<uint32_t>(qb & 0x0F)
                                | (static_cast<uint32_t>((qhb >> bit_lo) & 1) << 4);
                            const uint32_t hi = static_cast<uint32_t>(qb >> 4)
                                | (static_cast<uint32_t>((qhb >> bit_hi) & 1) << 4);
                            acc += (d_lo * static_cast<float>(lo) - m_lo) * x_usm[x_lo_off + l];
                            acc += (d_hi * static_cast<float>(hi) - m_hi) * x_usm[x_hi_off + l];
                        }
                    }
                }
                out_usm[m] = acc;
            });
    }).wait();
}

extern "C" {

void rsl_matvec_q5_k_packed_f32_usm(rsl_stream* s,
                                    const void* w_bytes_usm,
                                    const float* x_usm,
                                    float* out_usm,
                                    int M, int K,
                                    int lws) RSL_FFI_BODY_VOID(
    "rsl_matvec_q5_k_packed_f32_usm", {
    if (s == nullptr || w_bytes_usm == nullptr
        || x_usm == nullptr || out_usm == nullptr) {
        return;
    }
    if (M <= 0 || K <= 0 || (K % 256) != 0) {
        return;
    }
    auto& q = s->q;
    const int eff_lws = (lws <= 0) ? static_cast<int>(RSL_LWS) : lws;
    switch (eff_lws) {
        case 16:  matvec_q5_k_packed_f32_usm_impl<16>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 32:  matvec_q5_k_packed_f32_usm_impl<32>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 64:  matvec_q5_k_packed_f32_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 128: matvec_q5_k_packed_f32_usm_impl<128>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 256: matvec_q5_k_packed_f32_usm_impl<256>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        default:  matvec_q5_k_packed_f32_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
    }
})

}  // extern "C" — close for the Q5_K gate+up fused template

// H4: Q5_K gate + up fused matvec. Same per-block dequant as
// matvec_q5_k_packed_f32_usm_impl, but one work-item computes BOTH
// `gate_out[m]` and `up_out[m]`. Each `x_usm[…]` is loaded once per
// element and feeds both accumulators — halves activation-cache
// pressure on top of saving one Level-Zero launch per FFN expert.
template <std::size_t LWS_T>
inline void matvec_q5_k_gate_up_fused_usm_impl(
    sycl::queue& q,
    const void* gate_w_bytes_usm,
    const void* up_w_bytes_usm,
    const float* x_usm,
    float* gate_out_usm,
    float* up_out_usm,
    int M, int K) {
    constexpr std::size_t LWS = LWS_T;
    const int blocks_per_row = K / 256;
    const int bytes_per_row = blocks_per_row * 176;
    const uint8_t* gate_w_bytes = static_cast<const uint8_t*>(gate_w_bytes_usm);
    const uint8_t* up_w_bytes   = static_cast<const uint8_t*>(up_w_bytes_usm);
    const std::size_t global =
        ((static_cast<std::size_t>(M) + LWS - 1) / LWS) * LWS;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(LWS)),
            [=](sycl::nd_item<1> it) {
                const int m = static_cast<int>(it.get_global_id(0));
                if (m >= M) return;
                const uint8_t* g_row = gate_w_bytes + m * bytes_per_row;
                const uint8_t* u_row = up_w_bytes   + m * bytes_per_row;
                float gate_acc = 0.0f;
                float up_acc = 0.0f;
                for (int b = 0; b < blocks_per_row; ++b) {
                    const uint8_t* g_blk = g_row + b * 176;
                    const uint8_t* u_blk = u_row + b * 176;
                    const float g_d = bits_to_f32(static_cast<uint16_t>(g_blk[0])
                                                   | (static_cast<uint16_t>(g_blk[1]) << 8));
                    const float g_dmin = bits_to_f32(static_cast<uint16_t>(g_blk[2])
                                                      | (static_cast<uint16_t>(g_blk[3]) << 8));
                    const float u_d = bits_to_f32(static_cast<uint16_t>(u_blk[0])
                                                   | (static_cast<uint16_t>(u_blk[1]) << 8));
                    const float u_dmin = bits_to_f32(static_cast<uint16_t>(u_blk[2])
                                                      | (static_cast<uint16_t>(u_blk[3]) << 8));
                    uint8_t g_sc[8], g_mn[8], u_sc[8], u_mn[8];
                    {
                        const uint8_t* gs = g_blk + 4;
                        const uint8_t* us = u_blk + 4;
                        for (int j = 0; j < 8; ++j) {
                            if (j < 4) {
                                g_sc[j] = gs[j] & 0x3F;
                                g_mn[j] = gs[j + 4] & 0x3F;
                                u_sc[j] = us[j] & 0x3F;
                                u_mn[j] = us[j + 4] & 0x3F;
                            } else {
                                g_sc[j] = (gs[j + 4] & 0x0F) | ((gs[j - 4] >> 6) << 4);
                                g_mn[j] = (gs[j + 4] >> 4) | ((gs[j] >> 6) << 4);
                                u_sc[j] = (us[j + 4] & 0x0F) | ((us[j - 4] >> 6) << 4);
                                u_mn[j] = (us[j + 4] >> 4) | ((us[j] >> 6) << 4);
                            }
                        }
                    }
                    const uint8_t* g_qh = g_blk + 16;
                    const uint8_t* g_qs = g_blk + 48;
                    const uint8_t* u_qh = u_blk + 16;
                    const uint8_t* u_qs = u_blk + 48;
                    const int x_base = b * 256;
                    for (int group = 0; group < 4; ++group) {
                        const uint8_t* g_qc = g_qs + group * 32;
                        const uint8_t* u_qc = u_qs + group * 32;
                        const float g_d_lo = g_d * static_cast<float>(g_sc[group * 2]);
                        const float g_m_lo = g_dmin * static_cast<float>(g_mn[group * 2]);
                        const float g_d_hi = g_d * static_cast<float>(g_sc[group * 2 + 1]);
                        const float g_m_hi = g_dmin * static_cast<float>(g_mn[group * 2 + 1]);
                        const float u_d_lo = u_d * static_cast<float>(u_sc[group * 2]);
                        const float u_m_lo = u_dmin * static_cast<float>(u_mn[group * 2]);
                        const float u_d_hi = u_d * static_cast<float>(u_sc[group * 2 + 1]);
                        const float u_m_hi = u_dmin * static_cast<float>(u_mn[group * 2 + 1]);
                        const int bit_lo = group * 2;
                        const int bit_hi = group * 2 + 1;
                        const int x_lo_off = x_base + group * 64;
                        const int x_hi_off = x_lo_off + 32;
                        for (int l = 0; l < 32; ++l) {
                            const float x_lo = x_usm[x_lo_off + l];
                            const float x_hi = x_usm[x_hi_off + l];
                            const uint8_t g_qb = g_qc[l];
                            const uint8_t g_qhb = g_qh[l];
                            const uint32_t g_lo = static_cast<uint32_t>(g_qb & 0x0F)
                                | (static_cast<uint32_t>((g_qhb >> bit_lo) & 1) << 4);
                            const uint32_t g_hi = static_cast<uint32_t>(g_qb >> 4)
                                | (static_cast<uint32_t>((g_qhb >> bit_hi) & 1) << 4);
                            gate_acc += (g_d_lo * static_cast<float>(g_lo) - g_m_lo) * x_lo;
                            gate_acc += (g_d_hi * static_cast<float>(g_hi) - g_m_hi) * x_hi;
                            const uint8_t u_qb = u_qc[l];
                            const uint8_t u_qhb = u_qh[l];
                            const uint32_t u_lo = static_cast<uint32_t>(u_qb & 0x0F)
                                | (static_cast<uint32_t>((u_qhb >> bit_lo) & 1) << 4);
                            const uint32_t u_hi = static_cast<uint32_t>(u_qb >> 4)
                                | (static_cast<uint32_t>((u_qhb >> bit_hi) & 1) << 4);
                            up_acc += (u_d_lo * static_cast<float>(u_lo) - u_m_lo) * x_lo;
                            up_acc += (u_d_hi * static_cast<float>(u_hi) - u_m_hi) * x_hi;
                        }
                    }
                }
                gate_out_usm[m] = gate_acc;
                up_out_usm[m] = up_acc;
            });
    }).wait();
}
extern "C" {

void rsl_matvec_q5_k_gate_up_fused_usm(rsl_stream* s,
                                       const void* gate_w_bytes_usm,
                                       const void* up_w_bytes_usm,
                                       const float* x_usm,
                                       float* gate_out_usm,
                                       float* up_out_usm,
                                       int M, int K,
                                       int lws) RSL_FFI_BODY_VOID(
    "rsl_matvec_q5_k_gate_up_fused_usm", {
    if (s == nullptr || gate_w_bytes_usm == nullptr || up_w_bytes_usm == nullptr
        || x_usm == nullptr || gate_out_usm == nullptr || up_out_usm == nullptr) {
        return;
    }
    if (M <= 0 || K <= 0 || (K % 256) != 0) {
        return;
    }
    auto& q = s->q;
    const int eff_lws = (lws <= 0) ? static_cast<int>(RSL_LWS) : lws;
    switch (eff_lws) {
        case 16:  matvec_q5_k_gate_up_fused_usm_impl<16>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        case 32:  matvec_q5_k_gate_up_fused_usm_impl<32>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        case 64:  matvec_q5_k_gate_up_fused_usm_impl<64>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        case 128: matvec_q5_k_gate_up_fused_usm_impl<128>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        case 256: matvec_q5_k_gate_up_fused_usm_impl<256>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        default:  matvec_q5_k_gate_up_fused_usm_impl<64>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
    }
})

// USM-resident Q6_K packed matvec. Same super-block-of-256 layout
// shape as Q4_K_M and Q5_K_M but with a different scales+bits
// encoding: 128 bytes of low-4-bit nibbles + 64 bytes of high-2-bit
// pairs + 16 bytes of i8 per-sub-block scales + 2 bytes of f16
// super-block scale (210 bytes total).
//
// Each 6-bit weight is reconstructed as
//   `((ql_nibble | (qh_bits << 4)) - 32)`  (signed [-32, 31]).
// The 8 sub-block scales sit in `scales[]` indexed by `is = l/16 +
// n*8`. The inner loop pattern mirrors the CPU scalar reference
// exactly so parity holds.
}  // extern "C" — temporary close for the Q6_K template

template <std::size_t LWS_T>
inline void matvec_q6_k_packed_f32_usm_impl(
    sycl::queue& q,
    const void* w_bytes_usm,
    const float* x_usm,
    float* out_usm,
    int M, int K) {
    constexpr std::size_t LWS = LWS_T;
    const int blocks_per_row = K / 256;
    const int bytes_per_row = blocks_per_row * 210;
    const uint8_t* w_bytes = static_cast<const uint8_t*>(w_bytes_usm);
    const std::size_t global =
        ((static_cast<std::size_t>(M) + LWS - 1) / LWS) * LWS;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(LWS)),
            [=](sycl::nd_item<1> it) {
                const int m = static_cast<int>(it.get_global_id(0));
                if (m >= M) return;
                const uint8_t* row = w_bytes + m * bytes_per_row;
                float acc = 0.0f;
                for (int b = 0; b < blocks_per_row; ++b) {
                    const uint8_t* blk = row + b * 210;
                    const uint8_t* ql = blk;
                    const uint8_t* qh = blk + 128;
                    const int8_t* scales = reinterpret_cast<const int8_t*>(blk + 192);
                    uint16_t d_bits = static_cast<uint16_t>(blk[208])
                                      | (static_cast<uint16_t>(blk[209]) << 8);
                    const float d = bits_to_f32(d_bits);
                    const int x_base = b * 256;
                    for (int n = 0; n < 2; ++n) {
                        for (int l = 0; l < 32; ++l) {
                            const int is = l / 16 + n * 8;
                            const int qh_byte = qh[32 * n + l];
                            const int q1 = (static_cast<int>(ql[64 * n + l] & 0x0F)
                                | ((qh_byte >> 0) & 0x03) << 4) - 32;
                            const int q2 = (static_cast<int>(ql[64 * n + l + 32] & 0x0F)
                                | ((qh_byte >> 2) & 0x03) << 4) - 32;
                            const int q3 = (static_cast<int>(ql[64 * n + l] >> 4)
                                | ((qh_byte >> 4) & 0x03) << 4) - 32;
                            const int q4 = (static_cast<int>(ql[64 * n + l + 32] >> 4)
                                | ((qh_byte >> 6) & 0x03) << 4) - 32;
                            const float s0 = static_cast<float>(scales[is]);
                            const float s1 = static_cast<float>(scales[is + 2]);
                            const float s2 = static_cast<float>(scales[is + 4]);
                            const float s3 = static_cast<float>(scales[is + 6]);
                            const int base = n * 128 + l;
                            acc += d * s0 * static_cast<float>(q1) * x_usm[x_base + base];
                            acc += d * s1 * static_cast<float>(q2) * x_usm[x_base + base + 32];
                            acc += d * s2 * static_cast<float>(q3) * x_usm[x_base + base + 64];
                            acc += d * s3 * static_cast<float>(q4) * x_usm[x_base + base + 96];
                        }
                    }
                }
                out_usm[m] = acc;
            });
    }).wait();
}

extern "C" {

void rsl_matvec_q6_k_packed_f32_usm(rsl_stream* s,
                                    const void* w_bytes_usm,
                                    const float* x_usm,
                                    float* out_usm,
                                    int M, int K,
                                    int lws) RSL_FFI_BODY_VOID(
    "rsl_matvec_q6_k_packed_f32_usm", {
    if (s == nullptr || w_bytes_usm == nullptr
        || x_usm == nullptr || out_usm == nullptr) {
        return;
    }
    if (M <= 0 || K <= 0 || (K % 256) != 0) {
        return;
    }
    auto& q = s->q;
    const int eff_lws = (lws <= 0) ? static_cast<int>(RSL_LWS) : lws;
    switch (eff_lws) {
        case 16:  matvec_q6_k_packed_f32_usm_impl<16>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 32:  matvec_q6_k_packed_f32_usm_impl<32>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 64:  matvec_q6_k_packed_f32_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 128: matvec_q6_k_packed_f32_usm_impl<128>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 256: matvec_q6_k_packed_f32_usm_impl<256>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        default:  matvec_q6_k_packed_f32_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
    }
})

}  // extern "C" — close for the Q6_K gate+up fused template

// H4: Q6_K gate + up fused matvec — see q4_k_gate_up_fused for the
// fusion pattern. Per-block dequant arithmetic mirrors
// matvec_q6_k_packed_f32_usm_impl exactly; x loads are shared
// between gate and up accumulators per element.
template <std::size_t LWS_T>
inline void matvec_q6_k_gate_up_fused_usm_impl(
    sycl::queue& q,
    const void* gate_w_bytes_usm,
    const void* up_w_bytes_usm,
    const float* x_usm,
    float* gate_out_usm,
    float* up_out_usm,
    int M, int K) {
    constexpr std::size_t LWS = LWS_T;
    const int blocks_per_row = K / 256;
    const int bytes_per_row = blocks_per_row * 210;
    const uint8_t* gate_w_bytes = static_cast<const uint8_t*>(gate_w_bytes_usm);
    const uint8_t* up_w_bytes   = static_cast<const uint8_t*>(up_w_bytes_usm);
    const std::size_t global =
        ((static_cast<std::size_t>(M) + LWS - 1) / LWS) * LWS;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(LWS)),
            [=](sycl::nd_item<1> it) {
                const int m = static_cast<int>(it.get_global_id(0));
                if (m >= M) return;
                const uint8_t* g_row = gate_w_bytes + m * bytes_per_row;
                const uint8_t* u_row = up_w_bytes   + m * bytes_per_row;
                float gate_acc = 0.0f;
                float up_acc = 0.0f;
                for (int b = 0; b < blocks_per_row; ++b) {
                    const uint8_t* g_blk = g_row + b * 210;
                    const uint8_t* u_blk = u_row + b * 210;
                    const uint8_t* g_ql = g_blk;
                    const uint8_t* g_qh = g_blk + 128;
                    const int8_t* g_scales = reinterpret_cast<const int8_t*>(g_blk + 192);
                    const float g_d = bits_to_f32(static_cast<uint16_t>(g_blk[208])
                                                  | (static_cast<uint16_t>(g_blk[209]) << 8));
                    const uint8_t* u_ql = u_blk;
                    const uint8_t* u_qh = u_blk + 128;
                    const int8_t* u_scales = reinterpret_cast<const int8_t*>(u_blk + 192);
                    const float u_d = bits_to_f32(static_cast<uint16_t>(u_blk[208])
                                                  | (static_cast<uint16_t>(u_blk[209]) << 8));
                    const int x_base = b * 256;
                    for (int n = 0; n < 2; ++n) {
                        for (int l = 0; l < 32; ++l) {
                            const int is = l / 16 + n * 8;
                            const int g_qh_byte = g_qh[32 * n + l];
                            const int u_qh_byte = u_qh[32 * n + l];
                            const int g_q1 = (static_cast<int>(g_ql[64 * n + l] & 0x0F)
                                | ((g_qh_byte >> 0) & 0x03) << 4) - 32;
                            const int g_q2 = (static_cast<int>(g_ql[64 * n + l + 32] & 0x0F)
                                | ((g_qh_byte >> 2) & 0x03) << 4) - 32;
                            const int g_q3 = (static_cast<int>(g_ql[64 * n + l] >> 4)
                                | ((g_qh_byte >> 4) & 0x03) << 4) - 32;
                            const int g_q4 = (static_cast<int>(g_ql[64 * n + l + 32] >> 4)
                                | ((g_qh_byte >> 6) & 0x03) << 4) - 32;
                            const int u_q1 = (static_cast<int>(u_ql[64 * n + l] & 0x0F)
                                | ((u_qh_byte >> 0) & 0x03) << 4) - 32;
                            const int u_q2 = (static_cast<int>(u_ql[64 * n + l + 32] & 0x0F)
                                | ((u_qh_byte >> 2) & 0x03) << 4) - 32;
                            const int u_q3 = (static_cast<int>(u_ql[64 * n + l] >> 4)
                                | ((u_qh_byte >> 4) & 0x03) << 4) - 32;
                            const int u_q4 = (static_cast<int>(u_ql[64 * n + l + 32] >> 4)
                                | ((u_qh_byte >> 6) & 0x03) << 4) - 32;
                            const float g_s0 = static_cast<float>(g_scales[is]);
                            const float g_s1 = static_cast<float>(g_scales[is + 2]);
                            const float g_s2 = static_cast<float>(g_scales[is + 4]);
                            const float g_s3 = static_cast<float>(g_scales[is + 6]);
                            const float u_s0 = static_cast<float>(u_scales[is]);
                            const float u_s1 = static_cast<float>(u_scales[is + 2]);
                            const float u_s2 = static_cast<float>(u_scales[is + 4]);
                            const float u_s3 = static_cast<float>(u_scales[is + 6]);
                            const int base = n * 128 + l;
                            const float x0 = x_usm[x_base + base];
                            const float x1 = x_usm[x_base + base + 32];
                            const float x2 = x_usm[x_base + base + 64];
                            const float x3 = x_usm[x_base + base + 96];
                            gate_acc += g_d * g_s0 * static_cast<float>(g_q1) * x0;
                            gate_acc += g_d * g_s1 * static_cast<float>(g_q2) * x1;
                            gate_acc += g_d * g_s2 * static_cast<float>(g_q3) * x2;
                            gate_acc += g_d * g_s3 * static_cast<float>(g_q4) * x3;
                            up_acc   += u_d * u_s0 * static_cast<float>(u_q1) * x0;
                            up_acc   += u_d * u_s1 * static_cast<float>(u_q2) * x1;
                            up_acc   += u_d * u_s2 * static_cast<float>(u_q3) * x2;
                            up_acc   += u_d * u_s3 * static_cast<float>(u_q4) * x3;
                        }
                    }
                }
                gate_out_usm[m] = gate_acc;
                up_out_usm[m] = up_acc;
            });
    }).wait();
}
extern "C" {

void rsl_matvec_q6_k_gate_up_fused_usm(rsl_stream* s,
                                       const void* gate_w_bytes_usm,
                                       const void* up_w_bytes_usm,
                                       const float* x_usm,
                                       float* gate_out_usm,
                                       float* up_out_usm,
                                       int M, int K,
                                       int lws) RSL_FFI_BODY_VOID(
    "rsl_matvec_q6_k_gate_up_fused_usm", {
    if (s == nullptr || gate_w_bytes_usm == nullptr || up_w_bytes_usm == nullptr
        || x_usm == nullptr || gate_out_usm == nullptr || up_out_usm == nullptr) {
        return;
    }
    if (M <= 0 || K <= 0 || (K % 256) != 0) {
        return;
    }
    auto& q = s->q;
    const int eff_lws = (lws <= 0) ? static_cast<int>(RSL_LWS) : lws;
    switch (eff_lws) {
        case 16:  matvec_q6_k_gate_up_fused_usm_impl<16>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        case 32:  matvec_q6_k_gate_up_fused_usm_impl<32>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        case 64:  matvec_q6_k_gate_up_fused_usm_impl<64>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        case 128: matvec_q6_k_gate_up_fused_usm_impl<128>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        case 256: matvec_q6_k_gate_up_fused_usm_impl<256>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        default:  matvec_q6_k_gate_up_fused_usm_impl<64>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
    }
})

// F4: USM-resident IQ4_NL packed matvec. 18 bytes per 32-weight
// block:
//   { d: f16, qs: [u8; 16] }
// Each `qs` byte's low/high nibbles index `KVALUES_IQ4XS` (16-entry
// signed codebook biased toward zero). Per-block dequant:
//   weight[j]      = d * KVALUES[qs[j] & 0x0F]
//   weight[j+16]   = d * KVALUES[qs[j] >> 4]
//
// Topology: one work-item per output row (mirrors Q4_K_M template).
// Arithmetic is a line-for-line port of the CPU scalar reference at
// `kernels-cpu::matvec_iq4_nl_w_f32_a_scalar`; CPU runs the parity
// gate, GPU validation requires Intel Arc / Iris Xe hardware.
}  // extern "C" — temporary close for the IQ4_NL template

// IQ4_NL/IQ4_XS shared codebook — duplicated here so the SYCL TU
// has no header-only constant dependency on the kernels-cpu crate.
// Values must match `KVALUES_IQ4XS` in
// `kernels-cpu::matvec_iq4_xs::KVALUES_IQ4XS`; the parity test in
// the kernels-cpu crate pins both ends.
static constexpr int8_t KVALUES_IQ4XS_SYCL[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,  25,  38,  53,  69,  89, 113,
};

template <std::size_t LWS_T>
inline void matvec_iq4_nl_packed_f32_usm_impl(
    sycl::queue& q,
    const void* w_bytes_usm,
    const float* x_usm,
    float* out_usm,
    int M, int K) {
    constexpr std::size_t LWS = LWS_T;
    constexpr int BLOCK_BYTES = 18;
    constexpr int QK = 32;
    const int blocks_per_row = K / QK;
    const int bytes_per_row = blocks_per_row * BLOCK_BYTES;
    const uint8_t* w_bytes = static_cast<const uint8_t*>(w_bytes_usm);
    const std::size_t global =
        ((static_cast<std::size_t>(M) + LWS - 1) / LWS) * LWS;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(LWS)),
            [=](sycl::nd_item<1> it) {
                const int m = static_cast<int>(it.get_global_id(0));
                if (m >= M) return;
                const uint8_t* row = w_bytes + m * bytes_per_row;
                float acc = 0.0f;
                for (int b = 0; b < blocks_per_row; ++b) {
                    const uint8_t* blk = row + b * BLOCK_BYTES;
                    uint16_t d_bits = static_cast<uint16_t>(blk[0])
                                      | (static_cast<uint16_t>(blk[1]) << 8);
                    const float d = bits_to_f32(d_bits);
                    const uint8_t* qs = blk + 2;
                    const int x_base = b * QK;
                    for (int j = 0; j < 16; ++j) {
                        const uint8_t q = qs[j];
                        const int lo = static_cast<int>(q & 0x0F);
                        const int hi = static_cast<int>(q >> 4);
                        acc += d * static_cast<float>(KVALUES_IQ4XS_SYCL[lo])
                                 * x_usm[x_base + j];
                        acc += d * static_cast<float>(KVALUES_IQ4XS_SYCL[hi])
                                 * x_usm[x_base + j + 16];
                    }
                }
                out_usm[m] = acc;
            });
    }).wait();
}

extern "C" {

void rsl_matvec_iq4_nl_packed_f32_usm(rsl_stream* s,
                                      const void* w_bytes_usm,
                                      const float* x_usm,
                                      float* out_usm,
                                      int M, int K,
                                      int lws) RSL_FFI_BODY_VOID(
    "rsl_matvec_iq4_nl_packed_f32_usm", {
    if (s == nullptr || w_bytes_usm == nullptr
        || x_usm == nullptr || out_usm == nullptr) {
        return;
    }
    if (M <= 0 || K <= 0 || (K % 32) != 0) {
        return;
    }
    auto& q = s->q;
    const int eff_lws = (lws <= 0) ? static_cast<int>(RSL_LWS) : lws;
    switch (eff_lws) {
        case 16:  matvec_iq4_nl_packed_f32_usm_impl<16>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 32:  matvec_iq4_nl_packed_f32_usm_impl<32>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 64:  matvec_iq4_nl_packed_f32_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 128: matvec_iq4_nl_packed_f32_usm_impl<128>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 256: matvec_iq4_nl_packed_f32_usm_impl<256>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        default:  matvec_iq4_nl_packed_f32_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
    }
})

}  // extern "C" — close for the IQ4_NL gate+up fused template

// H4: IQ4_NL gate + up fused matvec — see q4_k_gate_up_fused for the
// fusion pattern. Per-block dequant arithmetic mirrors
// matvec_iq4_nl_packed_f32_usm_impl exactly.
template <std::size_t LWS_T>
inline void matvec_iq4_nl_gate_up_fused_usm_impl(
    sycl::queue& q,
    const void* gate_w_bytes_usm,
    const void* up_w_bytes_usm,
    const float* x_usm,
    float* gate_out_usm,
    float* up_out_usm,
    int M, int K) {
    constexpr std::size_t LWS = LWS_T;
    constexpr int BLOCK_BYTES = 18;
    constexpr int QK = 32;
    const int blocks_per_row = K / QK;
    const int bytes_per_row = blocks_per_row * BLOCK_BYTES;
    const uint8_t* gate_w_bytes = static_cast<const uint8_t*>(gate_w_bytes_usm);
    const uint8_t* up_w_bytes   = static_cast<const uint8_t*>(up_w_bytes_usm);
    const std::size_t global =
        ((static_cast<std::size_t>(M) + LWS - 1) / LWS) * LWS;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(LWS)),
            [=](sycl::nd_item<1> it) {
                const int m = static_cast<int>(it.get_global_id(0));
                if (m >= M) return;
                const uint8_t* g_row = gate_w_bytes + m * bytes_per_row;
                const uint8_t* u_row = up_w_bytes   + m * bytes_per_row;
                float gate_acc = 0.0f;
                float up_acc = 0.0f;
                for (int b = 0; b < blocks_per_row; ++b) {
                    const uint8_t* g_blk = g_row + b * BLOCK_BYTES;
                    const uint8_t* u_blk = u_row + b * BLOCK_BYTES;
                    const float g_d = bits_to_f32(static_cast<uint16_t>(g_blk[0])
                                                   | (static_cast<uint16_t>(g_blk[1]) << 8));
                    const float u_d = bits_to_f32(static_cast<uint16_t>(u_blk[0])
                                                   | (static_cast<uint16_t>(u_blk[1]) << 8));
                    const uint8_t* g_qs = g_blk + 2;
                    const uint8_t* u_qs = u_blk + 2;
                    const int x_base = b * QK;
                    for (int j = 0; j < 16; ++j) {
                        const float x0 = x_usm[x_base + j];
                        const float x1 = x_usm[x_base + j + 16];
                        const uint8_t gq = g_qs[j];
                        const uint8_t uq = u_qs[j];
                        const int g_lo = static_cast<int>(gq & 0x0F);
                        const int g_hi = static_cast<int>(gq >> 4);
                        const int u_lo = static_cast<int>(uq & 0x0F);
                        const int u_hi = static_cast<int>(uq >> 4);
                        gate_acc += g_d * static_cast<float>(KVALUES_IQ4XS_SYCL[g_lo]) * x0;
                        gate_acc += g_d * static_cast<float>(KVALUES_IQ4XS_SYCL[g_hi]) * x1;
                        up_acc   += u_d * static_cast<float>(KVALUES_IQ4XS_SYCL[u_lo]) * x0;
                        up_acc   += u_d * static_cast<float>(KVALUES_IQ4XS_SYCL[u_hi]) * x1;
                    }
                }
                gate_out_usm[m] = gate_acc;
                up_out_usm[m] = up_acc;
            });
    }).wait();
}
extern "C" {

void rsl_matvec_iq4_nl_gate_up_fused_usm(rsl_stream* s,
                                         const void* gate_w_bytes_usm,
                                         const void* up_w_bytes_usm,
                                         const float* x_usm,
                                         float* gate_out_usm,
                                         float* up_out_usm,
                                         int M, int K,
                                         int lws) RSL_FFI_BODY_VOID(
    "rsl_matvec_iq4_nl_gate_up_fused_usm", {
    if (s == nullptr || gate_w_bytes_usm == nullptr || up_w_bytes_usm == nullptr
        || x_usm == nullptr || gate_out_usm == nullptr || up_out_usm == nullptr) {
        return;
    }
    if (M <= 0 || K <= 0 || (K % 32) != 0) {
        return;
    }
    auto& q = s->q;
    const int eff_lws = (lws <= 0) ? static_cast<int>(RSL_LWS) : lws;
    switch (eff_lws) {
        case 16:  matvec_iq4_nl_gate_up_fused_usm_impl<16>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        case 32:  matvec_iq4_nl_gate_up_fused_usm_impl<32>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        case 64:  matvec_iq4_nl_gate_up_fused_usm_impl<64>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        case 128: matvec_iq4_nl_gate_up_fused_usm_impl<128>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        case 256: matvec_iq4_nl_gate_up_fused_usm_impl<256>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        default:  matvec_iq4_nl_gate_up_fused_usm_impl<64>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
    }
})

// F4: USM-resident IQ4_XS packed matvec. 136 bytes per 256-weight
// super-block:
//   bytes [0..2)    d:         f16 super-block scale
//   bytes [2..4)    scales_h:  u16 — high 2 bits per sub-block scale
//                              (8 sub-blocks; 2 bits each)
//   bytes [4..8)    scales_l:  [u8; 4] — low 4 bits per sub-block
//                              scale (2 nibbles per byte, 8 total)
//   bytes [8..136)  qs:        128 bytes of 4-bit nibbles
//
// Per sub-block (32 weights):
//   ls = ((lo_nibble | (hi_bits << 4)) as i8 - 32) as f32
//   sub_d = d * ls
//   for j in 0..16:
//       q = qs[ib*16 + j]
//       acc += sub_d * KVALUES[q & 0xF] * x[ib*32 + j]
//       acc += sub_d * KVALUES[q >> 4]  * x[ib*32 + 16 + j]
//
// Same one-work-item-per-row topology as IQ4_NL.
}  // extern "C" — temporary close for the IQ4_XS template

template <std::size_t LWS_T>
inline void matvec_iq4_xs_packed_f32_usm_impl(
    sycl::queue& q,
    const void* w_bytes_usm,
    const float* x_usm,
    float* out_usm,
    int M, int K) {
    constexpr std::size_t LWS = LWS_T;
    constexpr int BLOCK_BYTES = 136;
    constexpr int QK_K = 256;
    const int blocks_per_row = K / QK_K;
    const int bytes_per_row = blocks_per_row * BLOCK_BYTES;
    const uint8_t* w_bytes = static_cast<const uint8_t*>(w_bytes_usm);
    const std::size_t global =
        ((static_cast<std::size_t>(M) + LWS - 1) / LWS) * LWS;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(LWS)),
            [=](sycl::nd_item<1> it) {
                const int m = static_cast<int>(it.get_global_id(0));
                if (m >= M) return;
                const uint8_t* row = w_bytes + m * bytes_per_row;
                float acc = 0.0f;
                for (int b = 0; b < blocks_per_row; ++b) {
                    const uint8_t* blk = row + b * BLOCK_BYTES;
                    uint16_t d_bits = static_cast<uint16_t>(blk[0])
                                      | (static_cast<uint16_t>(blk[1]) << 8);
                    const float d = bits_to_f32(d_bits);
                    uint16_t scales_h = static_cast<uint16_t>(blk[2])
                                        | (static_cast<uint16_t>(blk[3]) << 8);
                    const uint8_t* scales_l = blk + 4;
                    const uint8_t* qs = blk + 8;
                    const int x_base = b * QK_K;
                    for (int ib = 0; ib < 8; ++ib) {
                        const uint8_t lo_nibble = (ib % 2 == 0)
                            ? (scales_l[ib / 2] & 0x0F)
                            : (scales_l[ib / 2] >> 4);
                        const uint8_t hi_bits =
                            static_cast<uint8_t>((scales_h >> (2 * ib)) & 0x03);
                        const int8_t ls_i =
                            static_cast<int8_t>(static_cast<uint8_t>(
                                lo_nibble | (hi_bits << 4))) - 32;
                        const float sub_d = d * static_cast<float>(ls_i);
                        const int q_off = ib * 16;
                        const int x_off = ib * 32;
                        for (int j = 0; j < 16; ++j) {
                            const uint8_t q = qs[q_off + j];
                            const int lo = static_cast<int>(q & 0x0F);
                            const int hi = static_cast<int>(q >> 4);
                            acc += sub_d
                                   * static_cast<float>(KVALUES_IQ4XS_SYCL[lo])
                                   * x_usm[x_base + x_off + j];
                            acc += sub_d
                                   * static_cast<float>(KVALUES_IQ4XS_SYCL[hi])
                                   * x_usm[x_base + x_off + 16 + j];
                        }
                    }
                }
                out_usm[m] = acc;
            });
    }).wait();
}

extern "C" {

void rsl_matvec_iq4_xs_packed_f32_usm(rsl_stream* s,
                                      const void* w_bytes_usm,
                                      const float* x_usm,
                                      float* out_usm,
                                      int M, int K,
                                      int lws) RSL_FFI_BODY_VOID(
    "rsl_matvec_iq4_xs_packed_f32_usm", {
    if (s == nullptr || w_bytes_usm == nullptr
        || x_usm == nullptr || out_usm == nullptr) {
        return;
    }
    if (M <= 0 || K <= 0 || (K % 256) != 0) {
        return;
    }
    auto& q = s->q;
    const int eff_lws = (lws <= 0) ? static_cast<int>(RSL_LWS) : lws;
    switch (eff_lws) {
        case 16:  matvec_iq4_xs_packed_f32_usm_impl<16>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 32:  matvec_iq4_xs_packed_f32_usm_impl<32>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 64:  matvec_iq4_xs_packed_f32_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 128: matvec_iq4_xs_packed_f32_usm_impl<128>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 256: matvec_iq4_xs_packed_f32_usm_impl<256>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        default:  matvec_iq4_xs_packed_f32_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
    }
})

}  // extern "C" — close for the IQ4_XS gate+up fused template

// H4: IQ4_XS gate + up fused matvec — see q4_k_gate_up_fused for the
// fusion pattern. Per-block dequant arithmetic mirrors
// matvec_iq4_xs_packed_f32_usm_impl exactly.
template <std::size_t LWS_T>
inline void matvec_iq4_xs_gate_up_fused_usm_impl(
    sycl::queue& q,
    const void* gate_w_bytes_usm,
    const void* up_w_bytes_usm,
    const float* x_usm,
    float* gate_out_usm,
    float* up_out_usm,
    int M, int K) {
    constexpr std::size_t LWS = LWS_T;
    constexpr int BLOCK_BYTES = 136;
    constexpr int QK_K = 256;
    const int blocks_per_row = K / QK_K;
    const int bytes_per_row = blocks_per_row * BLOCK_BYTES;
    const uint8_t* gate_w_bytes = static_cast<const uint8_t*>(gate_w_bytes_usm);
    const uint8_t* up_w_bytes   = static_cast<const uint8_t*>(up_w_bytes_usm);
    const std::size_t global =
        ((static_cast<std::size_t>(M) + LWS - 1) / LWS) * LWS;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(LWS)),
            [=](sycl::nd_item<1> it) {
                const int m = static_cast<int>(it.get_global_id(0));
                if (m >= M) return;
                const uint8_t* g_row = gate_w_bytes + m * bytes_per_row;
                const uint8_t* u_row = up_w_bytes   + m * bytes_per_row;
                float gate_acc = 0.0f;
                float up_acc = 0.0f;
                for (int b = 0; b < blocks_per_row; ++b) {
                    const uint8_t* g_blk = g_row + b * BLOCK_BYTES;
                    const uint8_t* u_blk = u_row + b * BLOCK_BYTES;
                    const float g_d = bits_to_f32(static_cast<uint16_t>(g_blk[0])
                                                   | (static_cast<uint16_t>(g_blk[1]) << 8));
                    const float u_d = bits_to_f32(static_cast<uint16_t>(u_blk[0])
                                                   | (static_cast<uint16_t>(u_blk[1]) << 8));
                    const uint16_t g_scales_h = static_cast<uint16_t>(g_blk[2])
                                                | (static_cast<uint16_t>(g_blk[3]) << 8);
                    const uint16_t u_scales_h = static_cast<uint16_t>(u_blk[2])
                                                | (static_cast<uint16_t>(u_blk[3]) << 8);
                    const uint8_t* g_scales_l = g_blk + 4;
                    const uint8_t* u_scales_l = u_blk + 4;
                    const uint8_t* g_qs = g_blk + 8;
                    const uint8_t* u_qs = u_blk + 8;
                    const int x_base = b * QK_K;
                    for (int ib = 0; ib < 8; ++ib) {
                        const uint8_t g_lo_nibble = (ib % 2 == 0)
                            ? (g_scales_l[ib / 2] & 0x0F)
                            : (g_scales_l[ib / 2] >> 4);
                        const uint8_t u_lo_nibble = (ib % 2 == 0)
                            ? (u_scales_l[ib / 2] & 0x0F)
                            : (u_scales_l[ib / 2] >> 4);
                        const uint8_t g_hi_bits =
                            static_cast<uint8_t>((g_scales_h >> (2 * ib)) & 0x03);
                        const uint8_t u_hi_bits =
                            static_cast<uint8_t>((u_scales_h >> (2 * ib)) & 0x03);
                        const int8_t g_ls = static_cast<int8_t>(static_cast<uint8_t>(
                            g_lo_nibble | (g_hi_bits << 4))) - 32;
                        const int8_t u_ls = static_cast<int8_t>(static_cast<uint8_t>(
                            u_lo_nibble | (u_hi_bits << 4))) - 32;
                        const float g_sub_d = g_d * static_cast<float>(g_ls);
                        const float u_sub_d = u_d * static_cast<float>(u_ls);
                        const int q_off = ib * 16;
                        const int x_off = ib * 32;
                        for (int j = 0; j < 16; ++j) {
                            const float x0 = x_usm[x_base + x_off + j];
                            const float x1 = x_usm[x_base + x_off + 16 + j];
                            const uint8_t gq = g_qs[q_off + j];
                            const uint8_t uq = u_qs[q_off + j];
                            const int g_lo = static_cast<int>(gq & 0x0F);
                            const int g_hi = static_cast<int>(gq >> 4);
                            const int u_lo = static_cast<int>(uq & 0x0F);
                            const int u_hi = static_cast<int>(uq >> 4);
                            gate_acc += g_sub_d
                                * static_cast<float>(KVALUES_IQ4XS_SYCL[g_lo]) * x0;
                            gate_acc += g_sub_d
                                * static_cast<float>(KVALUES_IQ4XS_SYCL[g_hi]) * x1;
                            up_acc   += u_sub_d
                                * static_cast<float>(KVALUES_IQ4XS_SYCL[u_lo]) * x0;
                            up_acc   += u_sub_d
                                * static_cast<float>(KVALUES_IQ4XS_SYCL[u_hi]) * x1;
                        }
                    }
                }
                gate_out_usm[m] = gate_acc;
                up_out_usm[m] = up_acc;
            });
    }).wait();
}
extern "C" {

void rsl_matvec_iq4_xs_gate_up_fused_usm(rsl_stream* s,
                                         const void* gate_w_bytes_usm,
                                         const void* up_w_bytes_usm,
                                         const float* x_usm,
                                         float* gate_out_usm,
                                         float* up_out_usm,
                                         int M, int K,
                                         int lws) RSL_FFI_BODY_VOID(
    "rsl_matvec_iq4_xs_gate_up_fused_usm", {
    if (s == nullptr || gate_w_bytes_usm == nullptr || up_w_bytes_usm == nullptr
        || x_usm == nullptr || gate_out_usm == nullptr || up_out_usm == nullptr) {
        return;
    }
    if (M <= 0 || K <= 0 || (K % 256) != 0) {
        return;
    }
    auto& q = s->q;
    const int eff_lws = (lws <= 0) ? static_cast<int>(RSL_LWS) : lws;
    switch (eff_lws) {
        case 16:  matvec_iq4_xs_gate_up_fused_usm_impl<16>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        case 32:  matvec_iq4_xs_gate_up_fused_usm_impl<32>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        case 64:  matvec_iq4_xs_gate_up_fused_usm_impl<64>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        case 128: matvec_iq4_xs_gate_up_fused_usm_impl<128>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        case 256: matvec_iq4_xs_gate_up_fused_usm_impl<256>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        default:  matvec_iq4_xs_gate_up_fused_usm_impl<64>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
    }
})

// F4 follow-up: batched IQ4_NL / IQ4_XS matvecs for the prefill path.
// Mirrors the existing per-row kernels (same block layout + arithmetic)
// but adds an outer N dimension so prefill multi-token batches don't
// fall back to CPU. Two new entries: `rsl_matvec_iq4_nl_packed_f32_batched_usm`
// and `rsl_matvec_iq4_xs_packed_f32_batched_usm`.
//
// Topology: 2D nd_range (M rows × N batch), one work-item per (m, n)
// output cell. Same shape as the existing batched Q4_K/Q5_K/Q6_K/Q8_0
// kernels; the per-cell math is the single-row IQ4 inner loop reading
// `x_usm + n*K` and writing `out_usm[n*M + m]`.
//
// Hardware validation pending — gated alongside the IQ encoder kernels.

}  // extern "C" — temporary close for the IQ4 batched template

template <std::size_t LWS_T>
inline void matvec_iq4_nl_packed_f32_batched_usm_impl(
    sycl::queue& q,
    const void* w_bytes_usm,
    const float* x_usm,
    float* out_usm,
    int M, int K, int N) {
    constexpr std::size_t LWS = LWS_T;
    constexpr int BLOCK_BYTES = 18;
    constexpr int QK = 32;
    const int blocks_per_row = K / QK;
    const int bytes_per_row = blocks_per_row * BLOCK_BYTES;
    const uint8_t* w_bytes = static_cast<const uint8_t*>(w_bytes_usm);
    const std::size_t global_m =
        ((static_cast<std::size_t>(M) + LWS - 1) / LWS) * LWS;
    sycl::range<2> global(global_m, static_cast<std::size_t>(N));
    sycl::range<2> local(LWS, 1);
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<2>(global, local),
            [=](sycl::nd_item<2> it) {
                const int m = static_cast<int>(it.get_global_id(0));
                const int n = static_cast<int>(it.get_global_id(1));
                if (m >= M || n >= N) return;
                const uint8_t* row = w_bytes + m * bytes_per_row;
                const float* x_row = x_usm + n * K;
                float acc = 0.0f;
                for (int b = 0; b < blocks_per_row; ++b) {
                    const uint8_t* blk = row + b * BLOCK_BYTES;
                    uint16_t d_bits = static_cast<uint16_t>(blk[0])
                                      | (static_cast<uint16_t>(blk[1]) << 8);
                    const float d = bits_to_f32(d_bits);
                    const uint8_t* qs = blk + 2;
                    const int x_base = b * QK;
                    for (int j = 0; j < 16; ++j) {
                        const uint8_t q = qs[j];
                        const int lo = static_cast<int>(q & 0x0F);
                        const int hi = static_cast<int>(q >> 4);
                        acc += d * static_cast<float>(KVALUES_IQ4XS_SYCL[lo])
                                 * x_row[x_base + j];
                        acc += d * static_cast<float>(KVALUES_IQ4XS_SYCL[hi])
                                 * x_row[x_base + j + 16];
                    }
                }
                out_usm[n * M + m] = acc;
            });
    }).wait();
}

template <std::size_t LWS_T>
inline void matvec_iq4_xs_packed_f32_batched_usm_impl(
    sycl::queue& q,
    const void* w_bytes_usm,
    const float* x_usm,
    float* out_usm,
    int M, int K, int N) {
    constexpr std::size_t LWS = LWS_T;
    constexpr int BLOCK_BYTES = 136;
    constexpr int QK_K = 256;
    const int blocks_per_row = K / QK_K;
    const int bytes_per_row = blocks_per_row * BLOCK_BYTES;
    const uint8_t* w_bytes = static_cast<const uint8_t*>(w_bytes_usm);
    const std::size_t global_m =
        ((static_cast<std::size_t>(M) + LWS - 1) / LWS) * LWS;
    sycl::range<2> global(global_m, static_cast<std::size_t>(N));
    sycl::range<2> local(LWS, 1);
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<2>(global, local),
            [=](sycl::nd_item<2> it) {
                const int m = static_cast<int>(it.get_global_id(0));
                const int n = static_cast<int>(it.get_global_id(1));
                if (m >= M || n >= N) return;
                const uint8_t* row = w_bytes + m * bytes_per_row;
                const float* x_row = x_usm + n * K;
                float acc = 0.0f;
                for (int b = 0; b < blocks_per_row; ++b) {
                    const uint8_t* blk = row + b * BLOCK_BYTES;
                    uint16_t d_bits = static_cast<uint16_t>(blk[0])
                                      | (static_cast<uint16_t>(blk[1]) << 8);
                    const float d = bits_to_f32(d_bits);
                    uint16_t scales_h = static_cast<uint16_t>(blk[2])
                                        | (static_cast<uint16_t>(blk[3]) << 8);
                    const uint8_t* scales_l = blk + 4;
                    const uint8_t* qs = blk + 8;
                    const int x_base = b * QK_K;
                    for (int ib = 0; ib < 8; ++ib) {
                        const uint8_t lo_nibble = (ib % 2 == 0)
                            ? (scales_l[ib / 2] & 0x0F)
                            : (scales_l[ib / 2] >> 4);
                        const uint8_t hi_bits =
                            static_cast<uint8_t>((scales_h >> (2 * ib)) & 0x03);
                        const int8_t ls_i =
                            static_cast<int8_t>(static_cast<uint8_t>(
                                lo_nibble | (hi_bits << 4))) - 32;
                        const float sub_d = d * static_cast<float>(ls_i);
                        const int q_off = ib * 16;
                        const int x_off = ib * 32;
                        for (int j = 0; j < 16; ++j) {
                            const uint8_t q = qs[q_off + j];
                            const int lo = static_cast<int>(q & 0x0F);
                            const int hi = static_cast<int>(q >> 4);
                            acc += sub_d
                                   * static_cast<float>(KVALUES_IQ4XS_SYCL[lo])
                                   * x_row[x_base + x_off + j];
                            acc += sub_d
                                   * static_cast<float>(KVALUES_IQ4XS_SYCL[hi])
                                   * x_row[x_base + x_off + 16 + j];
                        }
                    }
                }
                out_usm[n * M + m] = acc;
            });
    }).wait();
}

extern "C" {

void rsl_matvec_iq4_nl_packed_f32_batched_usm(rsl_stream* s,
                                              const void* w_bytes_usm,
                                              const float* x_usm,
                                              float* out_usm,
                                              int M, int K, int N,
                                              int lws) RSL_FFI_BODY_VOID(
    "rsl_matvec_iq4_nl_packed_f32_batched_usm", {
    if (s == nullptr || w_bytes_usm == nullptr
        || x_usm == nullptr || out_usm == nullptr) {
        return;
    }
    if (M <= 0 || K <= 0 || N <= 0 || (K % 32) != 0) {
        return;
    }
    auto& q = s->q;
    const int eff_lws = (lws <= 0) ? static_cast<int>(RSL_LWS) : lws;
    switch (eff_lws) {
        case 16:  matvec_iq4_nl_packed_f32_batched_usm_impl<16>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        case 32:  matvec_iq4_nl_packed_f32_batched_usm_impl<32>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        case 64:  matvec_iq4_nl_packed_f32_batched_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        case 128: matvec_iq4_nl_packed_f32_batched_usm_impl<128>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        case 256: matvec_iq4_nl_packed_f32_batched_usm_impl<256>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        default:  matvec_iq4_nl_packed_f32_batched_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
    }
})

void rsl_matvec_iq4_xs_packed_f32_batched_usm(rsl_stream* s,
                                              const void* w_bytes_usm,
                                              const float* x_usm,
                                              float* out_usm,
                                              int M, int K, int N,
                                              int lws) RSL_FFI_BODY_VOID(
    "rsl_matvec_iq4_xs_packed_f32_batched_usm", {
    if (s == nullptr || w_bytes_usm == nullptr
        || x_usm == nullptr || out_usm == nullptr) {
        return;
    }
    if (M <= 0 || K <= 0 || N <= 0 || (K % 256) != 0) {
        return;
    }
    auto& q = s->q;
    const int eff_lws = (lws <= 0) ? static_cast<int>(RSL_LWS) : lws;
    switch (eff_lws) {
        case 16:  matvec_iq4_xs_packed_f32_batched_usm_impl<16>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        case 32:  matvec_iq4_xs_packed_f32_batched_usm_impl<32>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        case 64:  matvec_iq4_xs_packed_f32_batched_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        case 128: matvec_iq4_xs_packed_f32_batched_usm_impl<128>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        case 256: matvec_iq4_xs_packed_f32_batched_usm_impl<256>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        default:  matvec_iq4_xs_packed_f32_batched_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
    }
})

// F4 inference: IQ1_S packed matvec. 50 bytes per 256-weight super-
// block: f16 d (2) + qs (32 — low 8 bits of grid index per chunk) +
// qh (16 — 8 × u16: high 3 bits of grid idx for 4 chunks + 3-bit
// sub-block scale + delta-sign bit). Mirrors CPU
// `matvec_iq1_s_w_f32_a_scalar`. Per-cell dequant:
//   d_super = f16(d) → f32
//   for each ib32 (8 sub-blocks of 32 weights):
//     qh = u16 at qh_bytes[ib32*2..ib32*2+2]
//     dl = d_super * (2 * ((qh >> 12) & 7) + 1)
//     delta = (qh & 0x8000) ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA
//     for each chunk l (4 per sub-block):
//       grid_idx = qs[ib32*4 + l] | (((qh >> (3*l)) & 7) << 8)
//       grid_bytes = IQ1S_GRID_SYCL[grid_idx]
//       for j in 0..8: acc += dl * (i8(grid_bytes[j]) + delta) * x[…]
//
// Same single-row dispatch shape as the Q4_K/Q5_K/Q6_K kernels.

}  // extern "C" — temporary close for the IQ1_S template

template <std::size_t LWS_T>
inline void matvec_iq1_s_packed_f32_usm_impl(
    sycl::queue& q,
    const void* w_bytes_usm,
    const float* x_usm,
    float* out_usm,
    int M, int K) {
    constexpr std::size_t LWS = LWS_T;
    constexpr int BLOCK_BYTES = 50;
    constexpr int QK_K = 256;
    // IQ1_S delta from rustllama-gguf::dequant::IQ1S_DELTA.
    constexpr float IQ1S_DELTA = 0.125f;
    const int blocks_per_row = K / QK_K;
    const int bytes_per_row = blocks_per_row * BLOCK_BYTES;
    const uint8_t* w_bytes = static_cast<const uint8_t*>(w_bytes_usm);
    const std::size_t global =
        ((static_cast<std::size_t>(M) + LWS - 1) / LWS) * LWS;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(LWS)),
            [=](sycl::nd_item<1> it) {
                const int m = static_cast<int>(it.get_global_id(0));
                if (m >= M) return;
                const uint8_t* row = w_bytes + m * bytes_per_row;
                float acc = 0.0f;
                for (int b = 0; b < blocks_per_row; ++b) {
                    const uint8_t* blk = row + b * BLOCK_BYTES;
                    uint16_t d_bits = static_cast<uint16_t>(blk[0])
                                      | (static_cast<uint16_t>(blk[1]) << 8);
                    const float d = bits_to_f32(d_bits);
                    const uint8_t* qs = blk + 2;       // 32 bytes
                    const uint8_t* qh_bytes = blk + 34; // 16 bytes
                    const int x_base = b * QK_K;
                    for (int ib32 = 0; ib32 < 8; ++ib32) {
                        const uint16_t qh =
                            static_cast<uint16_t>(qh_bytes[ib32 * 2])
                            | (static_cast<uint16_t>(qh_bytes[ib32 * 2 + 1]) << 8);
                        const float dl = d * (2.0f * static_cast<float>((qh >> 12) & 7) + 1.0f);
                        const float delta = (qh & 0x8000)
                            ? (-1.0f - IQ1S_DELTA)
                            : (-1.0f + IQ1S_DELTA);
                        const int x_off = x_base + ib32 * 32;
                        #pragma unroll
                        for (int l = 0; l < 4; ++l) {
                            const std::uint32_t idx =
                                static_cast<std::uint32_t>(qs[ib32 * 4 + l])
                                | ((static_cast<std::uint32_t>(qh >> (3 * l)) & 7u) << 8);
                            const std::uint64_t grid_bits =
                                rsl::IQ1S_GRID_SYCL[idx];
                            #pragma unroll
                            for (int j = 0; j < 8; ++j) {
                                const std::int8_t gi = static_cast<std::int8_t>(
                                    (grid_bits >> (j * 8)) & 0xFFu);
                                acc += dl
                                     * (static_cast<float>(gi) + delta)
                                     * x_usm[x_off + l * 8 + j];
                            }
                        }
                    }
                }
                out_usm[m] = acc;
            });
    }).wait();
}

extern "C" {

void rsl_matvec_iq1_s_packed_f32_usm(rsl_stream* s,
                                     const void* w_bytes_usm,
                                     const float* x_usm,
                                     float* out_usm,
                                     int M, int K,
                                     int lws) RSL_FFI_BODY_VOID(
    "rsl_matvec_iq1_s_packed_f32_usm", {
    if (s == nullptr || w_bytes_usm == nullptr
        || x_usm == nullptr || out_usm == nullptr) {
        return;
    }
    if (M <= 0 || K <= 0 || (K % 256) != 0) {
        return;
    }
    auto& q = s->q;
    const int eff_lws = (lws <= 0) ? static_cast<int>(RSL_LWS) : lws;
    switch (eff_lws) {
        case 16:  matvec_iq1_s_packed_f32_usm_impl<16>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 32:  matvec_iq1_s_packed_f32_usm_impl<32>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 64:  matvec_iq1_s_packed_f32_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 128: matvec_iq1_s_packed_f32_usm_impl<128>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 256: matvec_iq1_s_packed_f32_usm_impl<256>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        default:  matvec_iq1_s_packed_f32_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
    }
})

}  // extern "C" — close for the IQ1_S gate+up fused template

// H4: IQ1_S gate + up fused matvec — see q4_k_gate_up_fused for the
// fusion pattern. Per-block dequant arithmetic mirrors
// matvec_iq1_s_packed_f32_usm_impl exactly; the IQ1S_GRID lookup is
// shared between gate and up since each weight indexes independently.
template <std::size_t LWS_T>
inline void matvec_iq1_s_gate_up_fused_usm_impl(
    sycl::queue& q,
    const void* gate_w_bytes_usm,
    const void* up_w_bytes_usm,
    const float* x_usm,
    float* gate_out_usm,
    float* up_out_usm,
    int M, int K) {
    constexpr std::size_t LWS = LWS_T;
    constexpr int BLOCK_BYTES = 50;
    constexpr int QK_K = 256;
    constexpr float IQ1S_DELTA = 0.125f;
    const int blocks_per_row = K / QK_K;
    const int bytes_per_row = blocks_per_row * BLOCK_BYTES;
    const uint8_t* gate_w_bytes = static_cast<const uint8_t*>(gate_w_bytes_usm);
    const uint8_t* up_w_bytes   = static_cast<const uint8_t*>(up_w_bytes_usm);
    const std::size_t global =
        ((static_cast<std::size_t>(M) + LWS - 1) / LWS) * LWS;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(LWS)),
            [=](sycl::nd_item<1> it) {
                const int m = static_cast<int>(it.get_global_id(0));
                if (m >= M) return;
                const uint8_t* g_row = gate_w_bytes + m * bytes_per_row;
                const uint8_t* u_row = up_w_bytes   + m * bytes_per_row;
                float gate_acc = 0.0f;
                float up_acc = 0.0f;
                for (int b = 0; b < blocks_per_row; ++b) {
                    const uint8_t* g_blk = g_row + b * BLOCK_BYTES;
                    const uint8_t* u_blk = u_row + b * BLOCK_BYTES;
                    const float g_d = bits_to_f32(static_cast<uint16_t>(g_blk[0])
                                                   | (static_cast<uint16_t>(g_blk[1]) << 8));
                    const float u_d = bits_to_f32(static_cast<uint16_t>(u_blk[0])
                                                   | (static_cast<uint16_t>(u_blk[1]) << 8));
                    const uint8_t* g_qs = g_blk + 2;
                    const uint8_t* g_qh_bytes = g_blk + 34;
                    const uint8_t* u_qs = u_blk + 2;
                    const uint8_t* u_qh_bytes = u_blk + 34;
                    const int x_base = b * QK_K;
                    for (int ib32 = 0; ib32 < 8; ++ib32) {
                        const uint16_t g_qh =
                            static_cast<uint16_t>(g_qh_bytes[ib32 * 2])
                            | (static_cast<uint16_t>(g_qh_bytes[ib32 * 2 + 1]) << 8);
                        const uint16_t u_qh =
                            static_cast<uint16_t>(u_qh_bytes[ib32 * 2])
                            | (static_cast<uint16_t>(u_qh_bytes[ib32 * 2 + 1]) << 8);
                        const float g_dl = g_d * (2.0f * static_cast<float>((g_qh >> 12) & 7) + 1.0f);
                        const float u_dl = u_d * (2.0f * static_cast<float>((u_qh >> 12) & 7) + 1.0f);
                        const float g_delta = (g_qh & 0x8000)
                            ? (-1.0f - IQ1S_DELTA) : (-1.0f + IQ1S_DELTA);
                        const float u_delta = (u_qh & 0x8000)
                            ? (-1.0f - IQ1S_DELTA) : (-1.0f + IQ1S_DELTA);
                        const int x_off = x_base + ib32 * 32;
                        #pragma unroll
                        for (int l = 0; l < 4; ++l) {
                            const std::uint32_t g_idx =
                                static_cast<std::uint32_t>(g_qs[ib32 * 4 + l])
                                | ((static_cast<std::uint32_t>(g_qh >> (3 * l)) & 7u) << 8);
                            const std::uint32_t u_idx =
                                static_cast<std::uint32_t>(u_qs[ib32 * 4 + l])
                                | ((static_cast<std::uint32_t>(u_qh >> (3 * l)) & 7u) << 8);
                            const std::uint64_t g_grid_bits = rsl::IQ1S_GRID_SYCL[g_idx];
                            const std::uint64_t u_grid_bits = rsl::IQ1S_GRID_SYCL[u_idx];
                            #pragma unroll
                            for (int j = 0; j < 8; ++j) {
                                const float xv = x_usm[x_off + l * 8 + j];
                                const std::int8_t g_gi = static_cast<std::int8_t>(
                                    (g_grid_bits >> (j * 8)) & 0xFFu);
                                const std::int8_t u_gi = static_cast<std::int8_t>(
                                    (u_grid_bits >> (j * 8)) & 0xFFu);
                                gate_acc += g_dl * (static_cast<float>(g_gi) + g_delta) * xv;
                                up_acc   += u_dl * (static_cast<float>(u_gi) + u_delta) * xv;
                            }
                        }
                    }
                }
                gate_out_usm[m] = gate_acc;
                up_out_usm[m] = up_acc;
            });
    }).wait();
}
extern "C" {

void rsl_matvec_iq1_s_gate_up_fused_usm(rsl_stream* s,
                                        const void* gate_w_bytes_usm,
                                        const void* up_w_bytes_usm,
                                        const float* x_usm,
                                        float* gate_out_usm,
                                        float* up_out_usm,
                                        int M, int K,
                                        int lws) RSL_FFI_BODY_VOID(
    "rsl_matvec_iq1_s_gate_up_fused_usm", {
    if (s == nullptr || gate_w_bytes_usm == nullptr || up_w_bytes_usm == nullptr
        || x_usm == nullptr || gate_out_usm == nullptr || up_out_usm == nullptr) {
        return;
    }
    if (M <= 0 || K <= 0 || (K % 256) != 0) {
        return;
    }
    auto& q = s->q;
    const int eff_lws = (lws <= 0) ? static_cast<int>(RSL_LWS) : lws;
    switch (eff_lws) {
        case 16:  matvec_iq1_s_gate_up_fused_usm_impl<16>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        case 32:  matvec_iq1_s_gate_up_fused_usm_impl<32>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        case 64:  matvec_iq1_s_gate_up_fused_usm_impl<64>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        case 128: matvec_iq1_s_gate_up_fused_usm_impl<128>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        case 256: matvec_iq1_s_gate_up_fused_usm_impl<256>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        default:  matvec_iq1_s_gate_up_fused_usm_impl<64>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
    }
})

}  // extern "C" — close before the IQ1_S batched template

// Batched IQ1_S packed-USM matvec. Same per-block arithmetic as the
// single-row kernel above; the difference is iterating over N input
// activations (`x_usm` is `[N × K]`) and writing to `out_usm[n*M+m]`.
// Replaces the CPU-rayon fallback in `try_matvec_tensor_batched_usm_f32`
// for IQ1_S, closing the GPU-decode/CPU-prefill asymmetry.
template <std::size_t LWS_T>
inline void matvec_iq1_s_packed_f32_batched_usm_impl(
    sycl::queue& q,
    const void* w_bytes_usm,
    const float* x_usm,
    float* out_usm,
    int M, int K, int N) {
    constexpr std::size_t LWS = LWS_T;
    constexpr int BLOCK_BYTES = 50;
    constexpr int QK_K = 256;
    constexpr float IQ1S_DELTA = 0.125f;
    const int blocks_per_row = K / QK_K;
    const int bytes_per_row = blocks_per_row * BLOCK_BYTES;
    const uint8_t* w_bytes = static_cast<const uint8_t*>(w_bytes_usm);
    const std::size_t global_m =
        ((static_cast<std::size_t>(M) + LWS - 1) / LWS) * LWS;
    sycl::range<2> global(global_m, static_cast<std::size_t>(N));
    sycl::range<2> local(LWS, 1);
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<2>(global, local),
            [=](sycl::nd_item<2> it) {
                const int m = static_cast<int>(it.get_global_id(0));
                const int n = static_cast<int>(it.get_global_id(1));
                if (m >= M || n >= N) return;
                const uint8_t* row = w_bytes + m * bytes_per_row;
                const float* x_row = x_usm + n * K;
                float acc = 0.0f;
                for (int b = 0; b < blocks_per_row; ++b) {
                    const uint8_t* blk = row + b * BLOCK_BYTES;
                    uint16_t d_bits = static_cast<uint16_t>(blk[0])
                                      | (static_cast<uint16_t>(blk[1]) << 8);
                    const float d = bits_to_f32(d_bits);
                    const uint8_t* qs = blk + 2;
                    const uint8_t* qh_bytes = blk + 34;
                    const int x_base = b * QK_K;
                    for (int ib32 = 0; ib32 < 8; ++ib32) {
                        const uint16_t qh =
                            static_cast<uint16_t>(qh_bytes[ib32 * 2])
                            | (static_cast<uint16_t>(qh_bytes[ib32 * 2 + 1]) << 8);
                        const float dl = d * (2.0f * static_cast<float>((qh >> 12) & 7) + 1.0f);
                        const float delta = (qh & 0x8000)
                            ? (-1.0f - IQ1S_DELTA)
                            : (-1.0f + IQ1S_DELTA);
                        const int x_off = x_base + ib32 * 32;
                        #pragma unroll
                        for (int l = 0; l < 4; ++l) {
                            const std::uint32_t idx =
                                static_cast<std::uint32_t>(qs[ib32 * 4 + l])
                                | ((static_cast<std::uint32_t>(qh >> (3 * l)) & 7u) << 8);
                            const std::uint64_t grid_bits =
                                rsl::IQ1S_GRID_SYCL[idx];
                            #pragma unroll
                            for (int j = 0; j < 8; ++j) {
                                const std::int8_t gi = static_cast<std::int8_t>(
                                    (grid_bits >> (j * 8)) & 0xFFu);
                                acc += dl
                                     * (static_cast<float>(gi) + delta)
                                     * x_row[x_off + l * 8 + j];
                            }
                        }
                    }
                }
                out_usm[n * M + m] = acc;
            });
    }).wait();
}

extern "C" {

void rsl_matvec_iq1_s_packed_f32_batched_usm(rsl_stream* s,
                                              const void* w_bytes_usm,
                                              const float* x_usm,
                                              float* out_usm,
                                              int M, int K, int N,
                                              int lws) RSL_FFI_BODY_VOID(
    "rsl_matvec_iq1_s_packed_f32_batched_usm", {
    if (s == nullptr || w_bytes_usm == nullptr
        || x_usm == nullptr || out_usm == nullptr) {
        return;
    }
    if (M <= 0 || K <= 0 || N <= 0 || (K % 256) != 0) {
        return;
    }
    auto& q = s->q;
    const int eff_lws = (lws <= 0) ? static_cast<int>(RSL_LWS) : lws;
    switch (eff_lws) {
        case 16:  matvec_iq1_s_packed_f32_batched_usm_impl<16>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        case 32:  matvec_iq1_s_packed_f32_batched_usm_impl<32>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        case 64:  matvec_iq1_s_packed_f32_batched_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        case 128: matvec_iq1_s_packed_f32_batched_usm_impl<128>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        case 256: matvec_iq1_s_packed_f32_batched_usm_impl<256>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        default:  matvec_iq1_s_packed_f32_batched_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
    }
})

}  // extern "C" — temporary close for the IQ2_XXS template

// =====================================================================
// IQ2_XXS packed-USM single-row matvec.
// =====================================================================
//
// Block layout (66 bytes per 256-weight super-block, per the GGML
// IQ2_XXS spec):
//   { d: f16, qs: [u8; 64] }
// where `qs` is split into 8 sub-blocks of 8 bytes each. Each
// sub-block packs two little-endian u32 words:
//   aux0 : 4 × 8-bit grid indices into `IQ2XXS_GRID` (one byte per chunk)
//   aux1 : 4 × 7-bit sign-table indices in the low 28 bits +
//          a 4-bit sub-block scale shift in the top 4 bits
// Sub-block scale: `db = d * (0.5 + (aux1 >> 28)) * 0.25`.
// Per 8-weight chunk: `grid_bytes = IQ2XXS_GRID[idx]` (8 packed u8
// codebook entries), `signs = KSIGNS_IQ2XS[(aux1 >> 7l) & 127]`,
// then `acc += db * grid_bytes[j] * (signs & KMASK[j] ? -1 : 1) * x[…]`.
//
// One thread per output row — matches the IQ1_S kernel topology.

template <std::size_t LWS_T>
inline void matvec_iq2_xxs_packed_f32_usm_impl(
    sycl::queue& q,
    const void* w_bytes_usm,
    const float* x_usm,
    float* out_usm,
    int M, int K) {
    constexpr std::size_t LWS = LWS_T;
    constexpr int BLOCK_BYTES = 66;
    constexpr int QK_K = 256;
    const int blocks_per_row = K / QK_K;
    const int bytes_per_row = blocks_per_row * BLOCK_BYTES;
    const uint8_t* w_bytes = static_cast<const uint8_t*>(w_bytes_usm);
    const std::size_t global =
        ((static_cast<std::size_t>(M) + LWS - 1) / LWS) * LWS;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(LWS)),
            [=](sycl::nd_item<1> it) {
                const int m = static_cast<int>(it.get_global_id(0));
                if (m >= M) return;
                const uint8_t* row = w_bytes + m * bytes_per_row;
                float acc = 0.0f;
                for (int b = 0; b < blocks_per_row; ++b) {
                    const uint8_t* blk = row + b * BLOCK_BYTES;
                    uint16_t d_bits = static_cast<uint16_t>(blk[0])
                                      | (static_cast<uint16_t>(blk[1]) << 8);
                    const float d = bits_to_f32(d_bits);
                    const uint8_t* qs = blk + 2;  // 64 bytes
                    const int x_base = b * QK_K;
                    for (int ib32 = 0; ib32 < 8; ++ib32) {
                        const std::uint32_t aux0 =
                            static_cast<std::uint32_t>(qs[8 * ib32])
                            | (static_cast<std::uint32_t>(qs[8 * ib32 + 1]) << 8)
                            | (static_cast<std::uint32_t>(qs[8 * ib32 + 2]) << 16)
                            | (static_cast<std::uint32_t>(qs[8 * ib32 + 3]) << 24);
                        const std::uint32_t aux1 =
                            static_cast<std::uint32_t>(qs[8 * ib32 + 4])
                            | (static_cast<std::uint32_t>(qs[8 * ib32 + 5]) << 8)
                            | (static_cast<std::uint32_t>(qs[8 * ib32 + 6]) << 16)
                            | (static_cast<std::uint32_t>(qs[8 * ib32 + 7]) << 24);
                        const float db =
                            d * (0.5f + static_cast<float>(aux1 >> 28)) * 0.25f;
                        #pragma unroll
                        for (int l = 0; l < 4; ++l) {
                            const std::size_t grid_idx =
                                (aux0 >> (8 * l)) & 0xFFu;
                            const std::uint64_t grid_bits =
                                rsl::IQ2XXS_GRID_SYCL[grid_idx];
                            const std::uint8_t signs =
                                rsl::KSIGNS_IQ2XS_SYCL[(aux1 >> (7 * l)) & 127u];
                            const int x_off = x_base + ib32 * 32 + l * 8;
                            #pragma unroll
                            for (int j = 0; j < 8; ++j) {
                                const std::uint8_t gi =
                                    static_cast<std::uint8_t>(
                                        (grid_bits >> (j * 8)) & 0xFFu);
                                const float s = (signs & rsl::KMASK_IQ2XS_SYCL[j])
                                    ? -1.0f : 1.0f;
                                acc += db
                                     * static_cast<float>(gi)
                                     * s
                                     * x_usm[x_off + j];
                            }
                        }
                    }
                }
                out_usm[m] = acc;
            });
    }).wait();
}

extern "C" {

void rsl_matvec_iq2_xxs_packed_f32_usm(rsl_stream* s,
                                       const void* w_bytes_usm,
                                       const float* x_usm,
                                       float* out_usm,
                                       int M, int K,
                                       int lws) RSL_FFI_BODY_VOID(
    "rsl_matvec_iq2_xxs_packed_f32_usm", {
    if (s == nullptr || w_bytes_usm == nullptr
        || x_usm == nullptr || out_usm == nullptr) {
        return;
    }
    if (M <= 0 || K <= 0 || (K % 256) != 0) {
        return;
    }
    auto& q = s->q;
    const int eff_lws = (lws <= 0) ? static_cast<int>(RSL_LWS) : lws;
    switch (eff_lws) {
        case 16:  matvec_iq2_xxs_packed_f32_usm_impl<16>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 32:  matvec_iq2_xxs_packed_f32_usm_impl<32>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 64:  matvec_iq2_xxs_packed_f32_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 128: matvec_iq2_xxs_packed_f32_usm_impl<128>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 256: matvec_iq2_xxs_packed_f32_usm_impl<256>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        default:  matvec_iq2_xxs_packed_f32_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
    }
})

}  // extern "C" — close before IQ2_XXS gate+up fused template

// H4: IQ2_XXS gate + up fused matvec — see q4_k_gate_up_fused for the
// fusion pattern. Per-block dequant arithmetic mirrors
// matvec_iq2_xxs_packed_f32_usm_impl exactly.
template <std::size_t LWS_T>
inline void matvec_iq2_xxs_gate_up_fused_usm_impl(
    sycl::queue& q,
    const void* gate_w_bytes_usm,
    const void* up_w_bytes_usm,
    const float* x_usm,
    float* gate_out_usm,
    float* up_out_usm,
    int M, int K) {
    constexpr std::size_t LWS = LWS_T;
    constexpr int BLOCK_BYTES = 66;
    constexpr int QK_K = 256;
    const int blocks_per_row = K / QK_K;
    const int bytes_per_row = blocks_per_row * BLOCK_BYTES;
    const uint8_t* gate_w_bytes = static_cast<const uint8_t*>(gate_w_bytes_usm);
    const uint8_t* up_w_bytes   = static_cast<const uint8_t*>(up_w_bytes_usm);
    const std::size_t global =
        ((static_cast<std::size_t>(M) + LWS - 1) / LWS) * LWS;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(LWS)),
            [=](sycl::nd_item<1> it) {
                const int m = static_cast<int>(it.get_global_id(0));
                if (m >= M) return;
                const uint8_t* g_row = gate_w_bytes + m * bytes_per_row;
                const uint8_t* u_row = up_w_bytes   + m * bytes_per_row;
                float gate_acc = 0.0f;
                float up_acc = 0.0f;
                for (int b = 0; b < blocks_per_row; ++b) {
                    const uint8_t* g_blk = g_row + b * BLOCK_BYTES;
                    const uint8_t* u_blk = u_row + b * BLOCK_BYTES;
                    const float g_d = bits_to_f32(static_cast<uint16_t>(g_blk[0])
                                                   | (static_cast<uint16_t>(g_blk[1]) << 8));
                    const float u_d = bits_to_f32(static_cast<uint16_t>(u_blk[0])
                                                   | (static_cast<uint16_t>(u_blk[1]) << 8));
                    const uint8_t* g_qs = g_blk + 2;
                    const uint8_t* u_qs = u_blk + 2;
                    const int x_base = b * QK_K;
                    for (int ib32 = 0; ib32 < 8; ++ib32) {
                        const std::uint32_t g_aux0 =
                            static_cast<std::uint32_t>(g_qs[8 * ib32])
                            | (static_cast<std::uint32_t>(g_qs[8 * ib32 + 1]) << 8)
                            | (static_cast<std::uint32_t>(g_qs[8 * ib32 + 2]) << 16)
                            | (static_cast<std::uint32_t>(g_qs[8 * ib32 + 3]) << 24);
                        const std::uint32_t g_aux1 =
                            static_cast<std::uint32_t>(g_qs[8 * ib32 + 4])
                            | (static_cast<std::uint32_t>(g_qs[8 * ib32 + 5]) << 8)
                            | (static_cast<std::uint32_t>(g_qs[8 * ib32 + 6]) << 16)
                            | (static_cast<std::uint32_t>(g_qs[8 * ib32 + 7]) << 24);
                        const std::uint32_t u_aux0 =
                            static_cast<std::uint32_t>(u_qs[8 * ib32])
                            | (static_cast<std::uint32_t>(u_qs[8 * ib32 + 1]) << 8)
                            | (static_cast<std::uint32_t>(u_qs[8 * ib32 + 2]) << 16)
                            | (static_cast<std::uint32_t>(u_qs[8 * ib32 + 3]) << 24);
                        const std::uint32_t u_aux1 =
                            static_cast<std::uint32_t>(u_qs[8 * ib32 + 4])
                            | (static_cast<std::uint32_t>(u_qs[8 * ib32 + 5]) << 8)
                            | (static_cast<std::uint32_t>(u_qs[8 * ib32 + 6]) << 16)
                            | (static_cast<std::uint32_t>(u_qs[8 * ib32 + 7]) << 24);
                        const float g_db = g_d * (0.5f + static_cast<float>(g_aux1 >> 28)) * 0.25f;
                        const float u_db = u_d * (0.5f + static_cast<float>(u_aux1 >> 28)) * 0.25f;
                        #pragma unroll
                        for (int l = 0; l < 4; ++l) {
                            const std::size_t g_grid_idx = (g_aux0 >> (8 * l)) & 0xFFu;
                            const std::size_t u_grid_idx = (u_aux0 >> (8 * l)) & 0xFFu;
                            const std::uint64_t g_grid_bits = rsl::IQ2XXS_GRID_SYCL[g_grid_idx];
                            const std::uint64_t u_grid_bits = rsl::IQ2XXS_GRID_SYCL[u_grid_idx];
                            const std::uint8_t g_signs = rsl::KSIGNS_IQ2XS_SYCL[(g_aux1 >> (7 * l)) & 127u];
                            const std::uint8_t u_signs = rsl::KSIGNS_IQ2XS_SYCL[(u_aux1 >> (7 * l)) & 127u];
                            const int x_off = x_base + ib32 * 32 + l * 8;
                            #pragma unroll
                            for (int j = 0; j < 8; ++j) {
                                const float xv = x_usm[x_off + j];
                                const std::uint8_t g_gi =
                                    static_cast<std::uint8_t>((g_grid_bits >> (j * 8)) & 0xFFu);
                                const std::uint8_t u_gi =
                                    static_cast<std::uint8_t>((u_grid_bits >> (j * 8)) & 0xFFu);
                                const float g_s = (g_signs & rsl::KMASK_IQ2XS_SYCL[j]) ? -1.0f : 1.0f;
                                const float u_s = (u_signs & rsl::KMASK_IQ2XS_SYCL[j]) ? -1.0f : 1.0f;
                                gate_acc += g_db * static_cast<float>(g_gi) * g_s * xv;
                                up_acc   += u_db * static_cast<float>(u_gi) * u_s * xv;
                            }
                        }
                    }
                }
                gate_out_usm[m] = gate_acc;
                up_out_usm[m] = up_acc;
            });
    }).wait();
}
extern "C" {

void rsl_matvec_iq2_xxs_gate_up_fused_usm(rsl_stream* s,
                                          const void* gate_w_bytes_usm,
                                          const void* up_w_bytes_usm,
                                          const float* x_usm,
                                          float* gate_out_usm,
                                          float* up_out_usm,
                                          int M, int K,
                                          int lws) RSL_FFI_BODY_VOID(
    "rsl_matvec_iq2_xxs_gate_up_fused_usm", {
    if (s == nullptr || gate_w_bytes_usm == nullptr || up_w_bytes_usm == nullptr
        || x_usm == nullptr || gate_out_usm == nullptr || up_out_usm == nullptr) {
        return;
    }
    if (M <= 0 || K <= 0 || (K % 256) != 0) {
        return;
    }
    auto& q = s->q;
    const int eff_lws = (lws <= 0) ? static_cast<int>(RSL_LWS) : lws;
    switch (eff_lws) {
        case 16:  matvec_iq2_xxs_gate_up_fused_usm_impl<16>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        case 32:  matvec_iq2_xxs_gate_up_fused_usm_impl<32>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        case 64:  matvec_iq2_xxs_gate_up_fused_usm_impl<64>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        case 128: matvec_iq2_xxs_gate_up_fused_usm_impl<128>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        case 256: matvec_iq2_xxs_gate_up_fused_usm_impl<256>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        default:  matvec_iq2_xxs_gate_up_fused_usm_impl<64>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
    }
})

}  // extern "C" — close before the IQ2_XXS batched template

// Batched IQ2_XXS packed-USM matvec. See `matvec_iq2_xxs_packed_f32_usm_impl`
// for the per-block math; this variant iterates over N input rows.
template <std::size_t LWS_T>
inline void matvec_iq2_xxs_packed_f32_batched_usm_impl(
    sycl::queue& q,
    const void* w_bytes_usm,
    const float* x_usm,
    float* out_usm,
    int M, int K, int N) {
    constexpr std::size_t LWS = LWS_T;
    constexpr int BLOCK_BYTES = 66;
    constexpr int QK_K = 256;
    const int blocks_per_row = K / QK_K;
    const int bytes_per_row = blocks_per_row * BLOCK_BYTES;
    const uint8_t* w_bytes = static_cast<const uint8_t*>(w_bytes_usm);
    const std::size_t global_m =
        ((static_cast<std::size_t>(M) + LWS - 1) / LWS) * LWS;
    sycl::range<2> global(global_m, static_cast<std::size_t>(N));
    sycl::range<2> local(LWS, 1);
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<2>(global, local),
            [=](sycl::nd_item<2> it) {
                const int m = static_cast<int>(it.get_global_id(0));
                const int n = static_cast<int>(it.get_global_id(1));
                if (m >= M || n >= N) return;
                const uint8_t* row = w_bytes + m * bytes_per_row;
                const float* x_row = x_usm + n * K;
                float acc = 0.0f;
                for (int b = 0; b < blocks_per_row; ++b) {
                    const uint8_t* blk = row + b * BLOCK_BYTES;
                    uint16_t d_bits = static_cast<uint16_t>(blk[0])
                                      | (static_cast<uint16_t>(blk[1]) << 8);
                    const float d = bits_to_f32(d_bits);
                    const uint8_t* qs = blk + 2;
                    const int x_base = b * QK_K;
                    for (int ib32 = 0; ib32 < 8; ++ib32) {
                        const std::uint32_t aux0 =
                            static_cast<std::uint32_t>(qs[8 * ib32])
                            | (static_cast<std::uint32_t>(qs[8 * ib32 + 1]) << 8)
                            | (static_cast<std::uint32_t>(qs[8 * ib32 + 2]) << 16)
                            | (static_cast<std::uint32_t>(qs[8 * ib32 + 3]) << 24);
                        const std::uint32_t aux1 =
                            static_cast<std::uint32_t>(qs[8 * ib32 + 4])
                            | (static_cast<std::uint32_t>(qs[8 * ib32 + 5]) << 8)
                            | (static_cast<std::uint32_t>(qs[8 * ib32 + 6]) << 16)
                            | (static_cast<std::uint32_t>(qs[8 * ib32 + 7]) << 24);
                        const float db =
                            d * (0.5f + static_cast<float>(aux1 >> 28)) * 0.25f;
                        #pragma unroll
                        for (int l = 0; l < 4; ++l) {
                            const std::size_t grid_idx =
                                (aux0 >> (8 * l)) & 0xFFu;
                            const std::uint64_t grid_bits =
                                rsl::IQ2XXS_GRID_SYCL[grid_idx];
                            const std::uint8_t signs =
                                rsl::KSIGNS_IQ2XS_SYCL[(aux1 >> (7 * l)) & 127u];
                            const int x_off = x_base + ib32 * 32 + l * 8;
                            #pragma unroll
                            for (int j = 0; j < 8; ++j) {
                                const std::uint8_t gi =
                                    static_cast<std::uint8_t>(
                                        (grid_bits >> (j * 8)) & 0xFFu);
                                const float s = (signs & rsl::KMASK_IQ2XS_SYCL[j])
                                    ? -1.0f : 1.0f;
                                acc += db
                                     * static_cast<float>(gi)
                                     * s
                                     * x_row[x_off + j];
                            }
                        }
                    }
                }
                out_usm[n * M + m] = acc;
            });
    }).wait();
}

extern "C" {

void rsl_matvec_iq2_xxs_packed_f32_batched_usm(rsl_stream* s,
                                                const void* w_bytes_usm,
                                                const float* x_usm,
                                                float* out_usm,
                                                int M, int K, int N,
                                                int lws) RSL_FFI_BODY_VOID(
    "rsl_matvec_iq2_xxs_packed_f32_batched_usm", {
    if (s == nullptr || w_bytes_usm == nullptr
        || x_usm == nullptr || out_usm == nullptr) {
        return;
    }
    if (M <= 0 || K <= 0 || N <= 0 || (K % 256) != 0) {
        return;
    }
    auto& q = s->q;
    const int eff_lws = (lws <= 0) ? static_cast<int>(RSL_LWS) : lws;
    switch (eff_lws) {
        case 16:  matvec_iq2_xxs_packed_f32_batched_usm_impl<16>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        case 32:  matvec_iq2_xxs_packed_f32_batched_usm_impl<32>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        case 64:  matvec_iq2_xxs_packed_f32_batched_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        case 128: matvec_iq2_xxs_packed_f32_batched_usm_impl<128>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        case 256: matvec_iq2_xxs_packed_f32_batched_usm_impl<256>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        default:  matvec_iq2_xxs_packed_f32_batched_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
    }
})

}  // extern "C" — temporary close for the IQ1_M template

// =====================================================================
// IQ1_M packed-USM single-row matvec.
// =====================================================================
//
// Block layout (56 bytes per 256-weight super-block, per the GGML
// IQ1_M spec):
//   { qs: [u8; 32], qh: [u8; 16], scales: [u8; 8] }
// The 8-byte `scales` is read as 4 × u16 little-endian words. The
// f16 super-block scale `d` is reassembled from nibbles spread
// across all four scale words:
//   d_bits = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0)
//          | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000)
// Each super-block has 8 sub-blocks of 32 weights. Each sub-block
// has two per-half scales (`dl1`, `dl2`) extracted as 3-bit fields
// from `sc[ib/2]` at offsets `6*(ib%2)` and `6*(ib%2)+3`. Each of
// the 4 lanes in a sub-block (8 weights each) uses dl1 for lanes
// 0-1 and dl2 for lanes 2-3; the delta sign flips per lane on
// `qh[ib*2]` / `qh[ib*2+1]` bits 0x08 and 0x80. Grid index is an
// 11-bit value: low 8 bits from `qs[ib*4 + l]`, high 3 bits from
// `qh` nibbles. Reuses the IQ1S 2048-entry codebook.
//
// One thread per output row — matches the IQ1_S kernel topology.

template <std::size_t LWS_T>
inline void matvec_iq1_m_packed_f32_usm_impl(
    sycl::queue& q,
    const void* w_bytes_usm,
    const float* x_usm,
    float* out_usm,
    int M, int K) {
    constexpr std::size_t LWS = LWS_T;
    constexpr int BLOCK_BYTES = 56;
    constexpr int QK_K = 256;
    constexpr float IQ1S_DELTA = 0.125f;
    const int blocks_per_row = K / QK_K;
    const int bytes_per_row = blocks_per_row * BLOCK_BYTES;
    const uint8_t* w_bytes = static_cast<const uint8_t*>(w_bytes_usm);
    const std::size_t global =
        ((static_cast<std::size_t>(M) + LWS - 1) / LWS) * LWS;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(LWS)),
            [=](sycl::nd_item<1> it) {
                const int m = static_cast<int>(it.get_global_id(0));
                if (m >= M) return;
                const uint8_t* row = w_bytes + m * bytes_per_row;
                float acc = 0.0f;
                for (int b = 0; b < blocks_per_row; ++b) {
                    const uint8_t* blk = row + b * BLOCK_BYTES;
                    const uint8_t* qs = blk;             // 32 bytes
                    const uint8_t* qh = blk + 32;        // 16 bytes
                    const uint8_t* sc_bytes = blk + 48;  // 8 bytes
                    std::uint16_t sc[4];
                    for (int ii = 0; ii < 4; ++ii) {
                        sc[ii] = static_cast<std::uint16_t>(sc_bytes[ii * 2])
                            | (static_cast<std::uint16_t>(sc_bytes[ii * 2 + 1]) << 8);
                    }
                    const std::uint16_t d_bits =
                        (std::uint16_t)((sc[0] >> 12)
                        | ((sc[1] >> 8) & 0x00F0u)
                        | ((sc[2] >> 4) & 0x0F00u)
                        | (sc[3] & 0xF000u));
                    const float d = bits_to_f32(d_bits);
                    const int x_base = b * QK_K;
                    int x_off = 0;
                    for (int ib = 0; ib < 8; ++ib) {
                        const std::uint16_t s_word = sc[ib / 2];
                        const int shift0 = 6 * (ib % 2);
                        const int shift1 = 6 * (ib % 2) + 3;
                        const float dl1 =
                            d * (2.0f * static_cast<float>((s_word >> shift0) & 0x7u) + 1.0f);
                        const float dl2 =
                            d * (2.0f * static_cast<float>((s_word >> shift1) & 0x7u) + 1.0f);
                        const std::uint8_t qh0 = qh[ib * 2];
                        const std::uint8_t qh1 = qh[ib * 2 + 1];
                        const float delta_l[4] = {
                            (qh0 & 0x08u) ? (-1.0f - IQ1S_DELTA) : (-1.0f + IQ1S_DELTA),
                            (qh0 & 0x80u) ? (-1.0f - IQ1S_DELTA) : (-1.0f + IQ1S_DELTA),
                            (qh1 & 0x08u) ? (-1.0f - IQ1S_DELTA) : (-1.0f + IQ1S_DELTA),
                            (qh1 & 0x80u) ? (-1.0f - IQ1S_DELTA) : (-1.0f + IQ1S_DELTA),
                        };
                        const std::size_t idx_l[4] = {
                            static_cast<std::size_t>(qs[ib * 4 + 0])
                                | (static_cast<std::size_t>(qh0 & 0x07u) << 8),
                            static_cast<std::size_t>(qs[ib * 4 + 1])
                                | (static_cast<std::size_t>((qh0 >> 4) & 0x07u) << 8),
                            static_cast<std::size_t>(qs[ib * 4 + 2])
                                | (static_cast<std::size_t>(qh1 & 0x07u) << 8),
                            static_cast<std::size_t>(qs[ib * 4 + 3])
                                | (static_cast<std::size_t>((qh1 >> 4) & 0x07u) << 8),
                        };
                        const float dl_l[4] = { dl1, dl1, dl2, dl2 };
                        #pragma unroll
                        for (int l = 0; l < 4; ++l) {
                            const std::uint64_t grid_bits = rsl::IQ1S_GRID_SYCL[idx_l[l]];
                            const float dl = dl_l[l];
                            const float delta_val = delta_l[l];
                            // Match the CPU AVX-512 reduction shape:
                            //   val = dl * gi + dl_delta  (FMA-friendly)
                            //   acc = acc + val * x       (compiler-FMA-able)
                            // Pre-computing dl_delta outside the j loop and
                            // splitting val from the x-multiply ensures the
                            // compiler can emit `mad.f32` and matches CPU FP
                            // precision. The prior `dl*(gi+delta)*x` shape
                            // accumulated ~0.02 error vs CPU AVX-512.
                            const float dl_delta = dl * delta_val;
                            #pragma unroll
                            for (int j = 0; j < 8; ++j) {
                                const std::int8_t gi = static_cast<std::int8_t>(
                                    (grid_bits >> (j * 8)) & 0xFFu);
                                const float val = dl * static_cast<float>(gi) + dl_delta;
                                acc = acc + val * x_usm[x_base + x_off + 8 * l + j];
                            }
                        }
                        x_off += 32;
                    }
                }
                out_usm[m] = acc;
            });
    }).wait();
}

extern "C" {

void rsl_matvec_iq1_m_packed_f32_usm(rsl_stream* s,
                                     const void* w_bytes_usm,
                                     const float* x_usm,
                                     float* out_usm,
                                     int M, int K,
                                     int lws) RSL_FFI_BODY_VOID(
    "rsl_matvec_iq1_m_packed_f32_usm", {
    if (s == nullptr || w_bytes_usm == nullptr
        || x_usm == nullptr || out_usm == nullptr) {
        return;
    }
    if (M <= 0 || K <= 0 || (K % 256) != 0) {
        return;
    }
    auto& q = s->q;
    const int eff_lws = (lws <= 0) ? static_cast<int>(RSL_LWS) : lws;
    switch (eff_lws) {
        case 16:  matvec_iq1_m_packed_f32_usm_impl<16>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 32:  matvec_iq1_m_packed_f32_usm_impl<32>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 64:  matvec_iq1_m_packed_f32_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 128: matvec_iq1_m_packed_f32_usm_impl<128>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 256: matvec_iq1_m_packed_f32_usm_impl<256>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        default:  matvec_iq1_m_packed_f32_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
    }
})

}  // extern "C" — close before IQ1_M gate+up fused template

// H4: IQ1_M gate + up fused matvec — mirrors matvec_iq1_m_packed_f32_usm_impl
// with paired gate/up accumulators sharing x loads.
template <std::size_t LWS_T>
inline void matvec_iq1_m_gate_up_fused_usm_impl(
    sycl::queue& q,
    const void* gate_w_bytes_usm,
    const void* up_w_bytes_usm,
    const float* x_usm,
    float* gate_out_usm,
    float* up_out_usm,
    int M, int K) {
    constexpr std::size_t LWS = LWS_T;
    constexpr int BLOCK_BYTES = 56;
    constexpr int QK_K = 256;
    constexpr float IQ1S_DELTA = 0.125f;
    const int blocks_per_row = K / QK_K;
    const int bytes_per_row = blocks_per_row * BLOCK_BYTES;
    const uint8_t* gate_w_bytes = static_cast<const uint8_t*>(gate_w_bytes_usm);
    const uint8_t* up_w_bytes   = static_cast<const uint8_t*>(up_w_bytes_usm);
    const std::size_t global =
        ((static_cast<std::size_t>(M) + LWS - 1) / LWS) * LWS;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(LWS)),
            [=](sycl::nd_item<1> it) {
                const int m = static_cast<int>(it.get_global_id(0));
                if (m >= M) return;
                const uint8_t* g_row = gate_w_bytes + m * bytes_per_row;
                const uint8_t* u_row = up_w_bytes   + m * bytes_per_row;
                float gate_acc = 0.0f;
                float up_acc = 0.0f;
                for (int b = 0; b < blocks_per_row; ++b) {
                    const uint8_t* g_blk = g_row + b * BLOCK_BYTES;
                    const uint8_t* u_blk = u_row + b * BLOCK_BYTES;
                    const uint8_t* g_qs = g_blk;
                    const uint8_t* g_qh = g_blk + 32;
                    const uint8_t* g_sc_bytes = g_blk + 48;
                    const uint8_t* u_qs = u_blk;
                    const uint8_t* u_qh = u_blk + 32;
                    const uint8_t* u_sc_bytes = u_blk + 48;
                    std::uint16_t g_sc[4], u_sc[4];
                    for (int ii = 0; ii < 4; ++ii) {
                        g_sc[ii] = static_cast<std::uint16_t>(g_sc_bytes[ii * 2])
                            | (static_cast<std::uint16_t>(g_sc_bytes[ii * 2 + 1]) << 8);
                        u_sc[ii] = static_cast<std::uint16_t>(u_sc_bytes[ii * 2])
                            | (static_cast<std::uint16_t>(u_sc_bytes[ii * 2 + 1]) << 8);
                    }
                    const std::uint16_t g_d_bits =
                        (std::uint16_t)((g_sc[0] >> 12)
                        | ((g_sc[1] >> 8) & 0x00F0u)
                        | ((g_sc[2] >> 4) & 0x0F00u)
                        | (g_sc[3] & 0xF000u));
                    const std::uint16_t u_d_bits =
                        (std::uint16_t)((u_sc[0] >> 12)
                        | ((u_sc[1] >> 8) & 0x00F0u)
                        | ((u_sc[2] >> 4) & 0x0F00u)
                        | (u_sc[3] & 0xF000u));
                    const float g_d = bits_to_f32(g_d_bits);
                    const float u_d = bits_to_f32(u_d_bits);
                    const int x_base = b * QK_K;
                    int x_off = 0;
                    for (int ib = 0; ib < 8; ++ib) {
                        const std::uint16_t g_s_word = g_sc[ib / 2];
                        const std::uint16_t u_s_word = u_sc[ib / 2];
                        const int shift0 = 6 * (ib % 2);
                        const int shift1 = 6 * (ib % 2) + 3;
                        const float g_dl1 =
                            g_d * (2.0f * static_cast<float>((g_s_word >> shift0) & 0x7u) + 1.0f);
                        const float g_dl2 =
                            g_d * (2.0f * static_cast<float>((g_s_word >> shift1) & 0x7u) + 1.0f);
                        const float u_dl1 =
                            u_d * (2.0f * static_cast<float>((u_s_word >> shift0) & 0x7u) + 1.0f);
                        const float u_dl2 =
                            u_d * (2.0f * static_cast<float>((u_s_word >> shift1) & 0x7u) + 1.0f);
                        const std::uint8_t g_qh0 = g_qh[ib * 2];
                        const std::uint8_t g_qh1 = g_qh[ib * 2 + 1];
                        const std::uint8_t u_qh0 = u_qh[ib * 2];
                        const std::uint8_t u_qh1 = u_qh[ib * 2 + 1];
                        const float g_delta_l[4] = {
                            (g_qh0 & 0x08u) ? (-1.0f - IQ1S_DELTA) : (-1.0f + IQ1S_DELTA),
                            (g_qh0 & 0x80u) ? (-1.0f - IQ1S_DELTA) : (-1.0f + IQ1S_DELTA),
                            (g_qh1 & 0x08u) ? (-1.0f - IQ1S_DELTA) : (-1.0f + IQ1S_DELTA),
                            (g_qh1 & 0x80u) ? (-1.0f - IQ1S_DELTA) : (-1.0f + IQ1S_DELTA),
                        };
                        const float u_delta_l[4] = {
                            (u_qh0 & 0x08u) ? (-1.0f - IQ1S_DELTA) : (-1.0f + IQ1S_DELTA),
                            (u_qh0 & 0x80u) ? (-1.0f - IQ1S_DELTA) : (-1.0f + IQ1S_DELTA),
                            (u_qh1 & 0x08u) ? (-1.0f - IQ1S_DELTA) : (-1.0f + IQ1S_DELTA),
                            (u_qh1 & 0x80u) ? (-1.0f - IQ1S_DELTA) : (-1.0f + IQ1S_DELTA),
                        };
                        const std::size_t g_idx_l[4] = {
                            static_cast<std::size_t>(g_qs[ib * 4 + 0])
                                | (static_cast<std::size_t>(g_qh0 & 0x07u) << 8),
                            static_cast<std::size_t>(g_qs[ib * 4 + 1])
                                | (static_cast<std::size_t>((g_qh0 >> 4) & 0x07u) << 8),
                            static_cast<std::size_t>(g_qs[ib * 4 + 2])
                                | (static_cast<std::size_t>(g_qh1 & 0x07u) << 8),
                            static_cast<std::size_t>(g_qs[ib * 4 + 3])
                                | (static_cast<std::size_t>((g_qh1 >> 4) & 0x07u) << 8),
                        };
                        const std::size_t u_idx_l[4] = {
                            static_cast<std::size_t>(u_qs[ib * 4 + 0])
                                | (static_cast<std::size_t>(u_qh0 & 0x07u) << 8),
                            static_cast<std::size_t>(u_qs[ib * 4 + 1])
                                | (static_cast<std::size_t>((u_qh0 >> 4) & 0x07u) << 8),
                            static_cast<std::size_t>(u_qs[ib * 4 + 2])
                                | (static_cast<std::size_t>(u_qh1 & 0x07u) << 8),
                            static_cast<std::size_t>(u_qs[ib * 4 + 3])
                                | (static_cast<std::size_t>((u_qh1 >> 4) & 0x07u) << 8),
                        };
                        const float g_dl_l[4] = { g_dl1, g_dl1, g_dl2, g_dl2 };
                        const float u_dl_l[4] = { u_dl1, u_dl1, u_dl2, u_dl2 };
                        #pragma unroll
                        for (int l = 0; l < 4; ++l) {
                            const std::uint64_t g_grid_bits = rsl::IQ1S_GRID_SYCL[g_idx_l[l]];
                            const std::uint64_t u_grid_bits = rsl::IQ1S_GRID_SYCL[u_idx_l[l]];
                            const float g_dl = g_dl_l[l];
                            const float u_dl = u_dl_l[l];
                            const float g_dl_delta = g_dl * g_delta_l[l];
                            const float u_dl_delta = u_dl * u_delta_l[l];
                            #pragma unroll
                            for (int j = 0; j < 8; ++j) {
                                const float xv = x_usm[x_base + x_off + 8 * l + j];
                                const std::int8_t g_gi = static_cast<std::int8_t>(
                                    (g_grid_bits >> (j * 8)) & 0xFFu);
                                const std::int8_t u_gi = static_cast<std::int8_t>(
                                    (u_grid_bits >> (j * 8)) & 0xFFu);
                                const float g_val = g_dl * static_cast<float>(g_gi) + g_dl_delta;
                                const float u_val = u_dl * static_cast<float>(u_gi) + u_dl_delta;
                                gate_acc = gate_acc + g_val * xv;
                                up_acc   = up_acc   + u_val * xv;
                            }
                        }
                        x_off += 32;
                    }
                }
                gate_out_usm[m] = gate_acc;
                up_out_usm[m] = up_acc;
            });
    }).wait();
}
extern "C" {

void rsl_matvec_iq1_m_gate_up_fused_usm(rsl_stream* s,
                                        const void* gate_w_bytes_usm,
                                        const void* up_w_bytes_usm,
                                        const float* x_usm,
                                        float* gate_out_usm,
                                        float* up_out_usm,
                                        int M, int K,
                                        int lws) RSL_FFI_BODY_VOID(
    "rsl_matvec_iq1_m_gate_up_fused_usm", {
    if (s == nullptr || gate_w_bytes_usm == nullptr || up_w_bytes_usm == nullptr
        || x_usm == nullptr || gate_out_usm == nullptr || up_out_usm == nullptr) {
        return;
    }
    if (M <= 0 || K <= 0 || (K % 256) != 0) {
        return;
    }
    auto& q = s->q;
    const int eff_lws = (lws <= 0) ? static_cast<int>(RSL_LWS) : lws;
    switch (eff_lws) {
        case 16:  matvec_iq1_m_gate_up_fused_usm_impl<16>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        case 32:  matvec_iq1_m_gate_up_fused_usm_impl<32>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        case 64:  matvec_iq1_m_gate_up_fused_usm_impl<64>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        case 128: matvec_iq1_m_gate_up_fused_usm_impl<128>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        case 256: matvec_iq1_m_gate_up_fused_usm_impl<256>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        default:  matvec_iq1_m_gate_up_fused_usm_impl<64>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
    }
})

}  // extern "C" — close before the IQ1_M batched template

// Batched IQ1_M packed-USM matvec. Same per-block math as the single-row
// kernel above; iterates over N input rows.
template <std::size_t LWS_T>
inline void matvec_iq1_m_packed_f32_batched_usm_impl(
    sycl::queue& q,
    const void* w_bytes_usm,
    const float* x_usm,
    float* out_usm,
    int M, int K, int N) {
    constexpr std::size_t LWS = LWS_T;
    constexpr int BLOCK_BYTES = 56;
    constexpr int QK_K = 256;
    constexpr float IQ1S_DELTA = 0.125f;
    const int blocks_per_row = K / QK_K;
    const int bytes_per_row = blocks_per_row * BLOCK_BYTES;
    const uint8_t* w_bytes = static_cast<const uint8_t*>(w_bytes_usm);
    const std::size_t global_m =
        ((static_cast<std::size_t>(M) + LWS - 1) / LWS) * LWS;
    sycl::range<2> global(global_m, static_cast<std::size_t>(N));
    sycl::range<2> local(LWS, 1);
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<2>(global, local),
            [=](sycl::nd_item<2> it) {
                const int m = static_cast<int>(it.get_global_id(0));
                const int n = static_cast<int>(it.get_global_id(1));
                if (m >= M || n >= N) return;
                const uint8_t* row = w_bytes + m * bytes_per_row;
                const float* x_row = x_usm + n * K;
                float acc = 0.0f;
                for (int b = 0; b < blocks_per_row; ++b) {
                    const uint8_t* blk = row + b * BLOCK_BYTES;
                    const uint8_t* qs = blk;
                    const uint8_t* qh = blk + 32;
                    const uint8_t* sc_bytes = blk + 48;
                    std::uint16_t sc[4];
                    for (int ii = 0; ii < 4; ++ii) {
                        sc[ii] = static_cast<std::uint16_t>(sc_bytes[ii * 2])
                            | (static_cast<std::uint16_t>(sc_bytes[ii * 2 + 1]) << 8);
                    }
                    const std::uint16_t d_bits =
                        (std::uint16_t)((sc[0] >> 12)
                        | ((sc[1] >> 8) & 0x00F0u)
                        | ((sc[2] >> 4) & 0x0F00u)
                        | (sc[3] & 0xF000u));
                    const float d = bits_to_f32(d_bits);
                    const int x_base = b * QK_K;
                    int x_off = 0;
                    for (int ib = 0; ib < 8; ++ib) {
                        const std::uint16_t s_word = sc[ib / 2];
                        const int shift0 = 6 * (ib % 2);
                        const int shift1 = 6 * (ib % 2) + 3;
                        const float dl1 =
                            d * (2.0f * static_cast<float>((s_word >> shift0) & 0x7u) + 1.0f);
                        const float dl2 =
                            d * (2.0f * static_cast<float>((s_word >> shift1) & 0x7u) + 1.0f);
                        const std::uint8_t qh0 = qh[ib * 2];
                        const std::uint8_t qh1 = qh[ib * 2 + 1];
                        const float delta_l[4] = {
                            (qh0 & 0x08u) ? (-1.0f - IQ1S_DELTA) : (-1.0f + IQ1S_DELTA),
                            (qh0 & 0x80u) ? (-1.0f - IQ1S_DELTA) : (-1.0f + IQ1S_DELTA),
                            (qh1 & 0x08u) ? (-1.0f - IQ1S_DELTA) : (-1.0f + IQ1S_DELTA),
                            (qh1 & 0x80u) ? (-1.0f - IQ1S_DELTA) : (-1.0f + IQ1S_DELTA),
                        };
                        const std::size_t idx_l[4] = {
                            static_cast<std::size_t>(qs[ib * 4 + 0])
                                | (static_cast<std::size_t>(qh0 & 0x07u) << 8),
                            static_cast<std::size_t>(qs[ib * 4 + 1])
                                | (static_cast<std::size_t>((qh0 >> 4) & 0x07u) << 8),
                            static_cast<std::size_t>(qs[ib * 4 + 2])
                                | (static_cast<std::size_t>(qh1 & 0x07u) << 8),
                            static_cast<std::size_t>(qs[ib * 4 + 3])
                                | (static_cast<std::size_t>((qh1 >> 4) & 0x07u) << 8),
                        };
                        const float dl_l[4] = { dl1, dl1, dl2, dl2 };
                        #pragma unroll
                        for (int l = 0; l < 4; ++l) {
                            const std::uint64_t grid_bits = rsl::IQ1S_GRID_SYCL[idx_l[l]];
                            const float dl = dl_l[l];
                            const float delta_val = delta_l[l];
                            // See single-row kernel for FMA-shape rationale.
                            const float dl_delta = dl * delta_val;
                            #pragma unroll
                            for (int j = 0; j < 8; ++j) {
                                const std::int8_t gi = static_cast<std::int8_t>(
                                    (grid_bits >> (j * 8)) & 0xFFu);
                                const float val = dl * static_cast<float>(gi) + dl_delta;
                                acc = acc + val * x_row[x_base + x_off + 8 * l + j];
                            }
                        }
                        x_off += 32;
                    }
                }
                out_usm[n * M + m] = acc;
            });
    }).wait();
}

extern "C" {

void rsl_matvec_iq1_m_packed_f32_batched_usm(rsl_stream* s,
                                              const void* w_bytes_usm,
                                              const float* x_usm,
                                              float* out_usm,
                                              int M, int K, int N,
                                              int lws) RSL_FFI_BODY_VOID(
    "rsl_matvec_iq1_m_packed_f32_batched_usm", {
    if (s == nullptr || w_bytes_usm == nullptr
        || x_usm == nullptr || out_usm == nullptr) {
        return;
    }
    if (M <= 0 || K <= 0 || N <= 0 || (K % 256) != 0) {
        return;
    }
    auto& q = s->q;
    const int eff_lws = (lws <= 0) ? static_cast<int>(RSL_LWS) : lws;
    switch (eff_lws) {
        case 16:  matvec_iq1_m_packed_f32_batched_usm_impl<16>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        case 32:  matvec_iq1_m_packed_f32_batched_usm_impl<32>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        case 64:  matvec_iq1_m_packed_f32_batched_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        case 128: matvec_iq1_m_packed_f32_batched_usm_impl<128>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        case 256: matvec_iq1_m_packed_f32_batched_usm_impl<256>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        default:  matvec_iq1_m_packed_f32_batched_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
    }
})

}  // extern "C" — temporary close for the IQ2_XS template

// =====================================================================
// IQ2_XS packed-USM single-row matvec.
// =====================================================================
//
// Block layout (74 bytes per 256-weight super-block, per the GGML
// IQ2_XS spec):
//   { d: f16, qs: [u16; 32], scales: [u8; 8] }
// `qs` is stored as 64 bytes interpreted as 32 little-endian u16
// words. Each sub-block (ib32 in 0..8) has 4 such u16 words; for
// each word, the low 9 bits are the grid index into the 512-entry
// IQ2XS codebook and the high 7 bits are the sign-table index
// into KSIGNS_IQ2XS. Each scale byte packs two 4-bit sub-block
// scales: `db_lo` for chunks 0-1, `db_hi` for chunks 2-3.
//
// One thread per output row — matches the IQ1_S/IQ2_XXS topology.

template <std::size_t LWS_T>
inline void matvec_iq2_xs_packed_f32_usm_impl(
    sycl::queue& q,
    const void* w_bytes_usm,
    const float* x_usm,
    float* out_usm,
    int M, int K) {
    constexpr std::size_t LWS = LWS_T;
    constexpr int BLOCK_BYTES = 74;
    constexpr int QK_K = 256;
    const int blocks_per_row = K / QK_K;
    const int bytes_per_row = blocks_per_row * BLOCK_BYTES;
    const uint8_t* w_bytes = static_cast<const uint8_t*>(w_bytes_usm);
    const std::size_t global =
        ((static_cast<std::size_t>(M) + LWS - 1) / LWS) * LWS;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(LWS)),
            [=](sycl::nd_item<1> it) {
                const int m = static_cast<int>(it.get_global_id(0));
                if (m >= M) return;
                const uint8_t* row = w_bytes + m * bytes_per_row;
                float acc = 0.0f;
                for (int b = 0; b < blocks_per_row; ++b) {
                    const uint8_t* blk = row + b * BLOCK_BYTES;
                    uint16_t d_bits = static_cast<uint16_t>(blk[0])
                                      | (static_cast<uint16_t>(blk[1]) << 8);
                    const float d = bits_to_f32(d_bits);
                    const uint8_t* qs = blk + 2;          // 64 bytes
                    const uint8_t* scales = blk + 2 + 64; // 8 bytes
                    const int x_base = b * QK_K;
                    for (int ib32 = 0; ib32 < 8; ++ib32) {
                        const std::uint8_t scale_byte = scales[ib32];
                        const float db_lo =
                            d * (0.5f + static_cast<float>(scale_byte & 0x0Fu)) * 0.25f;
                        const float db_hi =
                            d * (0.5f + static_cast<float>(scale_byte >> 4)) * 0.25f;
                        const int base = 8 * ib32;
                        #pragma unroll
                        for (int l = 0; l < 4; ++l) {
                            const std::uint16_t q =
                                static_cast<std::uint16_t>(qs[base + 2 * l])
                                | (static_cast<std::uint16_t>(qs[base + 2 * l + 1]) << 8);
                            const std::size_t grid_idx =
                                static_cast<std::size_t>(q & 0x1FFu);
                            const std::size_t sign_idx =
                                static_cast<std::size_t>(q >> 9);
                            const std::uint64_t grid_bits =
                                rsl::IQ2XS_GRID_SYCL[grid_idx];
                            const std::uint8_t signs =
                                rsl::KSIGNS_IQ2XS_SYCL[sign_idx];
                            const float db = (l < 2) ? db_lo : db_hi;
                            const int x_off = x_base + ib32 * 32 + l * 8;
                            #pragma unroll
                            for (int j = 0; j < 8; ++j) {
                                const std::uint8_t gi =
                                    static_cast<std::uint8_t>(
                                        (grid_bits >> (j * 8)) & 0xFFu);
                                const float s = (signs & rsl::KMASK_IQ2XS_SYCL[j])
                                    ? -1.0f : 1.0f;
                                acc += db
                                     * static_cast<float>(gi)
                                     * s
                                     * x_usm[x_off + j];
                            }
                        }
                    }
                }
                out_usm[m] = acc;
            });
    }).wait();
}

extern "C" {

void rsl_matvec_iq2_xs_packed_f32_usm(rsl_stream* s,
                                      const void* w_bytes_usm,
                                      const float* x_usm,
                                      float* out_usm,
                                      int M, int K,
                                      int lws) RSL_FFI_BODY_VOID(
    "rsl_matvec_iq2_xs_packed_f32_usm", {
    if (s == nullptr || w_bytes_usm == nullptr
        || x_usm == nullptr || out_usm == nullptr) {
        return;
    }
    if (M <= 0 || K <= 0 || (K % 256) != 0) {
        return;
    }
    auto& q = s->q;
    const int eff_lws = (lws <= 0) ? static_cast<int>(RSL_LWS) : lws;
    switch (eff_lws) {
        case 16:  matvec_iq2_xs_packed_f32_usm_impl<16>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 32:  matvec_iq2_xs_packed_f32_usm_impl<32>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 64:  matvec_iq2_xs_packed_f32_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 128: matvec_iq2_xs_packed_f32_usm_impl<128>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 256: matvec_iq2_xs_packed_f32_usm_impl<256>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        default:  matvec_iq2_xs_packed_f32_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
    }
})

}  // extern "C" — close before IQ2_XS gate+up fused template

// H4: IQ2_XS gate + up fused matvec — mirrors matvec_iq2_xs_packed_f32_usm_impl
// with paired accumulators.
template <std::size_t LWS_T>
inline void matvec_iq2_xs_gate_up_fused_usm_impl(
    sycl::queue& q,
    const void* gate_w_bytes_usm,
    const void* up_w_bytes_usm,
    const float* x_usm,
    float* gate_out_usm,
    float* up_out_usm,
    int M, int K) {
    constexpr std::size_t LWS = LWS_T;
    constexpr int BLOCK_BYTES = 74;
    constexpr int QK_K = 256;
    const int blocks_per_row = K / QK_K;
    const int bytes_per_row = blocks_per_row * BLOCK_BYTES;
    const uint8_t* gate_w_bytes = static_cast<const uint8_t*>(gate_w_bytes_usm);
    const uint8_t* up_w_bytes   = static_cast<const uint8_t*>(up_w_bytes_usm);
    const std::size_t global =
        ((static_cast<std::size_t>(M) + LWS - 1) / LWS) * LWS;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(LWS)),
            [=](sycl::nd_item<1> it) {
                const int m = static_cast<int>(it.get_global_id(0));
                if (m >= M) return;
                const uint8_t* g_row = gate_w_bytes + m * bytes_per_row;
                const uint8_t* u_row = up_w_bytes   + m * bytes_per_row;
                float gate_acc = 0.0f;
                float up_acc = 0.0f;
                for (int b = 0; b < blocks_per_row; ++b) {
                    const uint8_t* g_blk = g_row + b * BLOCK_BYTES;
                    const uint8_t* u_blk = u_row + b * BLOCK_BYTES;
                    const float g_d = bits_to_f32(static_cast<uint16_t>(g_blk[0])
                                                   | (static_cast<uint16_t>(g_blk[1]) << 8));
                    const float u_d = bits_to_f32(static_cast<uint16_t>(u_blk[0])
                                                   | (static_cast<uint16_t>(u_blk[1]) << 8));
                    const uint8_t* g_qs = g_blk + 2;
                    const uint8_t* g_scales = g_blk + 2 + 64;
                    const uint8_t* u_qs = u_blk + 2;
                    const uint8_t* u_scales = u_blk + 2 + 64;
                    const int x_base = b * QK_K;
                    for (int ib32 = 0; ib32 < 8; ++ib32) {
                        const std::uint8_t g_scale_byte = g_scales[ib32];
                        const std::uint8_t u_scale_byte = u_scales[ib32];
                        const float g_db_lo = g_d * (0.5f + static_cast<float>(g_scale_byte & 0x0Fu)) * 0.25f;
                        const float g_db_hi = g_d * (0.5f + static_cast<float>(g_scale_byte >> 4)) * 0.25f;
                        const float u_db_lo = u_d * (0.5f + static_cast<float>(u_scale_byte & 0x0Fu)) * 0.25f;
                        const float u_db_hi = u_d * (0.5f + static_cast<float>(u_scale_byte >> 4)) * 0.25f;
                        const int base = 8 * ib32;
                        #pragma unroll
                        for (int l = 0; l < 4; ++l) {
                            const std::uint16_t g_q =
                                static_cast<std::uint16_t>(g_qs[base + 2 * l])
                                | (static_cast<std::uint16_t>(g_qs[base + 2 * l + 1]) << 8);
                            const std::uint16_t u_q =
                                static_cast<std::uint16_t>(u_qs[base + 2 * l])
                                | (static_cast<std::uint16_t>(u_qs[base + 2 * l + 1]) << 8);
                            const std::size_t g_grid_idx = static_cast<std::size_t>(g_q & 0x1FFu);
                            const std::size_t g_sign_idx = static_cast<std::size_t>(g_q >> 9);
                            const std::size_t u_grid_idx = static_cast<std::size_t>(u_q & 0x1FFu);
                            const std::size_t u_sign_idx = static_cast<std::size_t>(u_q >> 9);
                            const std::uint64_t g_grid_bits = rsl::IQ2XS_GRID_SYCL[g_grid_idx];
                            const std::uint64_t u_grid_bits = rsl::IQ2XS_GRID_SYCL[u_grid_idx];
                            const std::uint8_t g_signs = rsl::KSIGNS_IQ2XS_SYCL[g_sign_idx];
                            const std::uint8_t u_signs = rsl::KSIGNS_IQ2XS_SYCL[u_sign_idx];
                            const float g_db = (l < 2) ? g_db_lo : g_db_hi;
                            const float u_db = (l < 2) ? u_db_lo : u_db_hi;
                            const int x_off = x_base + ib32 * 32 + l * 8;
                            #pragma unroll
                            for (int j = 0; j < 8; ++j) {
                                const float xv = x_usm[x_off + j];
                                const std::uint8_t g_gi =
                                    static_cast<std::uint8_t>((g_grid_bits >> (j * 8)) & 0xFFu);
                                const std::uint8_t u_gi =
                                    static_cast<std::uint8_t>((u_grid_bits >> (j * 8)) & 0xFFu);
                                const float g_s = (g_signs & rsl::KMASK_IQ2XS_SYCL[j]) ? -1.0f : 1.0f;
                                const float u_s = (u_signs & rsl::KMASK_IQ2XS_SYCL[j]) ? -1.0f : 1.0f;
                                gate_acc += g_db * static_cast<float>(g_gi) * g_s * xv;
                                up_acc   += u_db * static_cast<float>(u_gi) * u_s * xv;
                            }
                        }
                    }
                }
                gate_out_usm[m] = gate_acc;
                up_out_usm[m] = up_acc;
            });
    }).wait();
}
extern "C" {

void rsl_matvec_iq2_xs_gate_up_fused_usm(rsl_stream* s,
                                         const void* gate_w_bytes_usm,
                                         const void* up_w_bytes_usm,
                                         const float* x_usm,
                                         float* gate_out_usm,
                                         float* up_out_usm,
                                         int M, int K,
                                         int lws) RSL_FFI_BODY_VOID(
    "rsl_matvec_iq2_xs_gate_up_fused_usm", {
    if (s == nullptr || gate_w_bytes_usm == nullptr || up_w_bytes_usm == nullptr
        || x_usm == nullptr || gate_out_usm == nullptr || up_out_usm == nullptr) {
        return;
    }
    if (M <= 0 || K <= 0 || (K % 256) != 0) {
        return;
    }
    auto& q = s->q;
    const int eff_lws = (lws <= 0) ? static_cast<int>(RSL_LWS) : lws;
    switch (eff_lws) {
        case 16:  matvec_iq2_xs_gate_up_fused_usm_impl<16>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        case 32:  matvec_iq2_xs_gate_up_fused_usm_impl<32>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        case 64:  matvec_iq2_xs_gate_up_fused_usm_impl<64>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        case 128: matvec_iq2_xs_gate_up_fused_usm_impl<128>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        case 256: matvec_iq2_xs_gate_up_fused_usm_impl<256>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        default:  matvec_iq2_xs_gate_up_fused_usm_impl<64>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
    }
})

}  // extern "C" — close before the IQ2_XS batched template

// Batched IQ2_XS packed-USM matvec. Iterates over N input rows.
template <std::size_t LWS_T>
inline void matvec_iq2_xs_packed_f32_batched_usm_impl(
    sycl::queue& q,
    const void* w_bytes_usm,
    const float* x_usm,
    float* out_usm,
    int M, int K, int N) {
    constexpr std::size_t LWS = LWS_T;
    constexpr int BLOCK_BYTES = 74;
    constexpr int QK_K = 256;
    const int blocks_per_row = K / QK_K;
    const int bytes_per_row = blocks_per_row * BLOCK_BYTES;
    const uint8_t* w_bytes = static_cast<const uint8_t*>(w_bytes_usm);
    const std::size_t global_m =
        ((static_cast<std::size_t>(M) + LWS - 1) / LWS) * LWS;
    sycl::range<2> global(global_m, static_cast<std::size_t>(N));
    sycl::range<2> local(LWS, 1);
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<2>(global, local),
            [=](sycl::nd_item<2> it) {
                const int m = static_cast<int>(it.get_global_id(0));
                const int n = static_cast<int>(it.get_global_id(1));
                if (m >= M || n >= N) return;
                const uint8_t* row = w_bytes + m * bytes_per_row;
                const float* x_row = x_usm + n * K;
                float acc = 0.0f;
                for (int b = 0; b < blocks_per_row; ++b) {
                    const uint8_t* blk = row + b * BLOCK_BYTES;
                    uint16_t d_bits = static_cast<uint16_t>(blk[0])
                                      | (static_cast<uint16_t>(blk[1]) << 8);
                    const float d = bits_to_f32(d_bits);
                    const uint8_t* qs = blk + 2;
                    const uint8_t* scales = blk + 2 + 64;
                    const int x_base = b * QK_K;
                    for (int ib32 = 0; ib32 < 8; ++ib32) {
                        const std::uint8_t scale_byte = scales[ib32];
                        const float db_lo =
                            d * (0.5f + static_cast<float>(scale_byte & 0x0Fu)) * 0.25f;
                        const float db_hi =
                            d * (0.5f + static_cast<float>(scale_byte >> 4)) * 0.25f;
                        const int base = 8 * ib32;
                        #pragma unroll
                        for (int l = 0; l < 4; ++l) {
                            const std::uint16_t q =
                                static_cast<std::uint16_t>(qs[base + 2 * l])
                                | (static_cast<std::uint16_t>(qs[base + 2 * l + 1]) << 8);
                            const std::size_t grid_idx =
                                static_cast<std::size_t>(q & 0x1FFu);
                            const std::size_t sign_idx =
                                static_cast<std::size_t>(q >> 9);
                            const std::uint64_t grid_bits =
                                rsl::IQ2XS_GRID_SYCL[grid_idx];
                            const std::uint8_t signs =
                                rsl::KSIGNS_IQ2XS_SYCL[sign_idx];
                            const float db = (l < 2) ? db_lo : db_hi;
                            const int x_off = x_base + ib32 * 32 + l * 8;
                            #pragma unroll
                            for (int j = 0; j < 8; ++j) {
                                const std::uint8_t gi =
                                    static_cast<std::uint8_t>(
                                        (grid_bits >> (j * 8)) & 0xFFu);
                                const float s = (signs & rsl::KMASK_IQ2XS_SYCL[j])
                                    ? -1.0f : 1.0f;
                                acc += db
                                     * static_cast<float>(gi)
                                     * s
                                     * x_row[x_off + j];
                            }
                        }
                    }
                }
                out_usm[n * M + m] = acc;
            });
    }).wait();
}

extern "C" {

void rsl_matvec_iq2_xs_packed_f32_batched_usm(rsl_stream* s,
                                               const void* w_bytes_usm,
                                               const float* x_usm,
                                               float* out_usm,
                                               int M, int K, int N,
                                               int lws) RSL_FFI_BODY_VOID(
    "rsl_matvec_iq2_xs_packed_f32_batched_usm", {
    if (s == nullptr || w_bytes_usm == nullptr
        || x_usm == nullptr || out_usm == nullptr) {
        return;
    }
    if (M <= 0 || K <= 0 || N <= 0 || (K % 256) != 0) {
        return;
    }
    auto& q = s->q;
    const int eff_lws = (lws <= 0) ? static_cast<int>(RSL_LWS) : lws;
    switch (eff_lws) {
        case 16:  matvec_iq2_xs_packed_f32_batched_usm_impl<16>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        case 32:  matvec_iq2_xs_packed_f32_batched_usm_impl<32>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        case 64:  matvec_iq2_xs_packed_f32_batched_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        case 128: matvec_iq2_xs_packed_f32_batched_usm_impl<128>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        case 256: matvec_iq2_xs_packed_f32_batched_usm_impl<256>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        default:  matvec_iq2_xs_packed_f32_batched_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
    }
})

}  // extern "C" — temporary close for the IQ2_S template

// =====================================================================
// IQ2_S packed-USM single-row matvec.
// =====================================================================
//
// Block layout (82 bytes per 256-weight super-block, per the GGML
// IQ2_S spec):
//   { d: f16, qs_lo: [u8; 32], signs: [u8; 32], qh: [u8; 8], scales: [u8; 8] }
// Unlike IQ2_XS, the sign mask is stored *directly* (one byte per
// 8-weight chunk) instead of going through a KSIGNS lookup. Grid
// index is 10 bits: low 8 from `qs_lo`, high 2 packed in `qh`
// (4 chunks per qh byte, each chunk's 2 bits live at offset
// `8 - 2*l` to be shifted into bits 8-9). Codebook is 1024 entries.
// Sub-block scales are 4-bit nibbles per ib32, same db_lo/db_hi
// shape as IQ2_XS.
//
// One thread per output row.

template <std::size_t LWS_T>
inline void matvec_iq2_s_packed_f32_usm_impl(
    sycl::queue& q,
    const void* w_bytes_usm,
    const float* x_usm,
    float* out_usm,
    int M, int K) {
    constexpr std::size_t LWS = LWS_T;
    constexpr int BLOCK_BYTES = 82;
    constexpr int QK_K = 256;
    const int blocks_per_row = K / QK_K;
    const int bytes_per_row = blocks_per_row * BLOCK_BYTES;
    const uint8_t* w_bytes = static_cast<const uint8_t*>(w_bytes_usm);
    const std::size_t global =
        ((static_cast<std::size_t>(M) + LWS - 1) / LWS) * LWS;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(LWS)),
            [=](sycl::nd_item<1> it) {
                const int m = static_cast<int>(it.get_global_id(0));
                if (m >= M) return;
                const uint8_t* row = w_bytes + m * bytes_per_row;
                float acc = 0.0f;
                for (int b = 0; b < blocks_per_row; ++b) {
                    const uint8_t* blk = row + b * BLOCK_BYTES;
                    uint16_t d_bits = static_cast<uint16_t>(blk[0])
                                      | (static_cast<uint16_t>(blk[1]) << 8);
                    const float d = bits_to_f32(d_bits);
                    const uint8_t* qs_lo = blk + 2;            // 32 bytes
                    const uint8_t* signs = blk + 2 + 32;       // 32 bytes
                    const uint8_t* qh     = blk + 2 + 64;      // 8 bytes
                    const uint8_t* scales = blk + 2 + 64 + 8;  // 8 bytes
                    const int x_base = b * QK_K;
                    for (int ib32 = 0; ib32 < 8; ++ib32) {
                        const std::uint8_t scale_byte = scales[ib32];
                        const float db_lo =
                            d * (0.5f + static_cast<float>(scale_byte & 0x0Fu)) * 0.25f;
                        const float db_hi =
                            d * (0.5f + static_cast<float>(scale_byte >> 4)) * 0.25f;
                        const int qs_off = ib32 * 4;
                        const std::uint8_t qh_byte = qh[ib32];
                        #pragma unroll
                        for (int l = 0; l < 4; ++l) {
                            const std::size_t high_bits =
                                (static_cast<std::size_t>(qh_byte) << (8 - 2 * l)) & 0x300u;
                            const std::size_t grid_idx =
                                static_cast<std::size_t>(qs_lo[qs_off + l]) | high_bits;
                            const std::uint8_t sign_byte = signs[qs_off + l];
                            const std::uint64_t grid_bits =
                                rsl::IQ2S_GRID_SYCL[grid_idx];
                            const float db = (l < 2) ? db_lo : db_hi;
                            const int x_off = x_base + ib32 * 32 + l * 8;
                            #pragma unroll
                            for (int j = 0; j < 8; ++j) {
                                const std::uint8_t gi =
                                    static_cast<std::uint8_t>(
                                        (grid_bits >> (j * 8)) & 0xFFu);
                                const float s = (sign_byte & rsl::KMASK_IQ2XS_SYCL[j])
                                    ? -1.0f : 1.0f;
                                acc += db
                                     * static_cast<float>(gi)
                                     * s
                                     * x_usm[x_off + j];
                            }
                        }
                    }
                }
                out_usm[m] = acc;
            });
    }).wait();
}

extern "C" {

void rsl_matvec_iq2_s_packed_f32_usm(rsl_stream* s,
                                     const void* w_bytes_usm,
                                     const float* x_usm,
                                     float* out_usm,
                                     int M, int K,
                                     int lws) RSL_FFI_BODY_VOID(
    "rsl_matvec_iq2_s_packed_f32_usm", {
    if (s == nullptr || w_bytes_usm == nullptr
        || x_usm == nullptr || out_usm == nullptr) {
        return;
    }
    if (M <= 0 || K <= 0 || (K % 256) != 0) {
        return;
    }
    auto& q = s->q;
    const int eff_lws = (lws <= 0) ? static_cast<int>(RSL_LWS) : lws;
    switch (eff_lws) {
        case 16:  matvec_iq2_s_packed_f32_usm_impl<16>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 32:  matvec_iq2_s_packed_f32_usm_impl<32>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 64:  matvec_iq2_s_packed_f32_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 128: matvec_iq2_s_packed_f32_usm_impl<128>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 256: matvec_iq2_s_packed_f32_usm_impl<256>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        default:  matvec_iq2_s_packed_f32_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
    }
})

}  // extern "C" — close before IQ2_S gate+up fused template

// H4: IQ2_S gate + up fused matvec — mirrors matvec_iq2_s_packed_f32_usm_impl
// with paired accumulators.
template <std::size_t LWS_T>
inline void matvec_iq2_s_gate_up_fused_usm_impl(
    sycl::queue& q,
    const void* gate_w_bytes_usm,
    const void* up_w_bytes_usm,
    const float* x_usm,
    float* gate_out_usm,
    float* up_out_usm,
    int M, int K) {
    constexpr std::size_t LWS = LWS_T;
    constexpr int BLOCK_BYTES = 82;
    constexpr int QK_K = 256;
    const int blocks_per_row = K / QK_K;
    const int bytes_per_row = blocks_per_row * BLOCK_BYTES;
    const uint8_t* gate_w_bytes = static_cast<const uint8_t*>(gate_w_bytes_usm);
    const uint8_t* up_w_bytes   = static_cast<const uint8_t*>(up_w_bytes_usm);
    const std::size_t global =
        ((static_cast<std::size_t>(M) + LWS - 1) / LWS) * LWS;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(LWS)),
            [=](sycl::nd_item<1> it) {
                const int m = static_cast<int>(it.get_global_id(0));
                if (m >= M) return;
                const uint8_t* g_row = gate_w_bytes + m * bytes_per_row;
                const uint8_t* u_row = up_w_bytes   + m * bytes_per_row;
                float gate_acc = 0.0f;
                float up_acc = 0.0f;
                for (int b = 0; b < blocks_per_row; ++b) {
                    const uint8_t* g_blk = g_row + b * BLOCK_BYTES;
                    const uint8_t* u_blk = u_row + b * BLOCK_BYTES;
                    const float g_d = bits_to_f32(static_cast<uint16_t>(g_blk[0])
                                                   | (static_cast<uint16_t>(g_blk[1]) << 8));
                    const float u_d = bits_to_f32(static_cast<uint16_t>(u_blk[0])
                                                   | (static_cast<uint16_t>(u_blk[1]) << 8));
                    const uint8_t* g_qs_lo  = g_blk + 2;
                    const uint8_t* g_signs  = g_blk + 2 + 32;
                    const uint8_t* g_qh     = g_blk + 2 + 64;
                    const uint8_t* g_scales = g_blk + 2 + 64 + 8;
                    const uint8_t* u_qs_lo  = u_blk + 2;
                    const uint8_t* u_signs  = u_blk + 2 + 32;
                    const uint8_t* u_qh     = u_blk + 2 + 64;
                    const uint8_t* u_scales = u_blk + 2 + 64 + 8;
                    const int x_base = b * QK_K;
                    for (int ib32 = 0; ib32 < 8; ++ib32) {
                        const std::uint8_t g_scale_byte = g_scales[ib32];
                        const std::uint8_t u_scale_byte = u_scales[ib32];
                        const float g_db_lo = g_d * (0.5f + static_cast<float>(g_scale_byte & 0x0Fu)) * 0.25f;
                        const float g_db_hi = g_d * (0.5f + static_cast<float>(g_scale_byte >> 4)) * 0.25f;
                        const float u_db_lo = u_d * (0.5f + static_cast<float>(u_scale_byte & 0x0Fu)) * 0.25f;
                        const float u_db_hi = u_d * (0.5f + static_cast<float>(u_scale_byte >> 4)) * 0.25f;
                        const int qs_off = ib32 * 4;
                        const std::uint8_t g_qh_byte = g_qh[ib32];
                        const std::uint8_t u_qh_byte = u_qh[ib32];
                        #pragma unroll
                        for (int l = 0; l < 4; ++l) {
                            const std::size_t g_high_bits =
                                (static_cast<std::size_t>(g_qh_byte) << (8 - 2 * l)) & 0x300u;
                            const std::size_t u_high_bits =
                                (static_cast<std::size_t>(u_qh_byte) << (8 - 2 * l)) & 0x300u;
                            const std::size_t g_grid_idx =
                                static_cast<std::size_t>(g_qs_lo[qs_off + l]) | g_high_bits;
                            const std::size_t u_grid_idx =
                                static_cast<std::size_t>(u_qs_lo[qs_off + l]) | u_high_bits;
                            const std::uint8_t g_sign_byte = g_signs[qs_off + l];
                            const std::uint8_t u_sign_byte = u_signs[qs_off + l];
                            const std::uint64_t g_grid_bits = rsl::IQ2S_GRID_SYCL[g_grid_idx];
                            const std::uint64_t u_grid_bits = rsl::IQ2S_GRID_SYCL[u_grid_idx];
                            const float g_db = (l < 2) ? g_db_lo : g_db_hi;
                            const float u_db = (l < 2) ? u_db_lo : u_db_hi;
                            const int x_off = x_base + ib32 * 32 + l * 8;
                            #pragma unroll
                            for (int j = 0; j < 8; ++j) {
                                const float xv = x_usm[x_off + j];
                                const std::uint8_t g_gi =
                                    static_cast<std::uint8_t>((g_grid_bits >> (j * 8)) & 0xFFu);
                                const std::uint8_t u_gi =
                                    static_cast<std::uint8_t>((u_grid_bits >> (j * 8)) & 0xFFu);
                                const float g_s = (g_sign_byte & rsl::KMASK_IQ2XS_SYCL[j]) ? -1.0f : 1.0f;
                                const float u_s = (u_sign_byte & rsl::KMASK_IQ2XS_SYCL[j]) ? -1.0f : 1.0f;
                                gate_acc += g_db * static_cast<float>(g_gi) * g_s * xv;
                                up_acc   += u_db * static_cast<float>(u_gi) * u_s * xv;
                            }
                        }
                    }
                }
                gate_out_usm[m] = gate_acc;
                up_out_usm[m] = up_acc;
            });
    }).wait();
}
extern "C" {

void rsl_matvec_iq2_s_gate_up_fused_usm(rsl_stream* s,
                                        const void* gate_w_bytes_usm,
                                        const void* up_w_bytes_usm,
                                        const float* x_usm,
                                        float* gate_out_usm,
                                        float* up_out_usm,
                                        int M, int K,
                                        int lws) RSL_FFI_BODY_VOID(
    "rsl_matvec_iq2_s_gate_up_fused_usm", {
    if (s == nullptr || gate_w_bytes_usm == nullptr || up_w_bytes_usm == nullptr
        || x_usm == nullptr || gate_out_usm == nullptr || up_out_usm == nullptr) {
        return;
    }
    if (M <= 0 || K <= 0 || (K % 256) != 0) {
        return;
    }
    auto& q = s->q;
    const int eff_lws = (lws <= 0) ? static_cast<int>(RSL_LWS) : lws;
    switch (eff_lws) {
        case 16:  matvec_iq2_s_gate_up_fused_usm_impl<16>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        case 32:  matvec_iq2_s_gate_up_fused_usm_impl<32>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        case 64:  matvec_iq2_s_gate_up_fused_usm_impl<64>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        case 128: matvec_iq2_s_gate_up_fused_usm_impl<128>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        case 256: matvec_iq2_s_gate_up_fused_usm_impl<256>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        default:  matvec_iq2_s_gate_up_fused_usm_impl<64>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
    }
})

}  // extern "C" — close before the IQ2_S batched template

// Batched IQ2_S packed-USM matvec. Iterates over N input rows.
template <std::size_t LWS_T>
inline void matvec_iq2_s_packed_f32_batched_usm_impl(
    sycl::queue& q,
    const void* w_bytes_usm,
    const float* x_usm,
    float* out_usm,
    int M, int K, int N) {
    constexpr std::size_t LWS = LWS_T;
    constexpr int BLOCK_BYTES = 82;
    constexpr int QK_K = 256;
    const int blocks_per_row = K / QK_K;
    const int bytes_per_row = blocks_per_row * BLOCK_BYTES;
    const uint8_t* w_bytes = static_cast<const uint8_t*>(w_bytes_usm);
    const std::size_t global_m =
        ((static_cast<std::size_t>(M) + LWS - 1) / LWS) * LWS;
    sycl::range<2> global(global_m, static_cast<std::size_t>(N));
    sycl::range<2> local(LWS, 1);
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<2>(global, local),
            [=](sycl::nd_item<2> it) {
                const int m = static_cast<int>(it.get_global_id(0));
                const int n = static_cast<int>(it.get_global_id(1));
                if (m >= M || n >= N) return;
                const uint8_t* row = w_bytes + m * bytes_per_row;
                const float* x_row = x_usm + n * K;
                float acc = 0.0f;
                for (int b = 0; b < blocks_per_row; ++b) {
                    const uint8_t* blk = row + b * BLOCK_BYTES;
                    uint16_t d_bits = static_cast<uint16_t>(blk[0])
                                      | (static_cast<uint16_t>(blk[1]) << 8);
                    const float d = bits_to_f32(d_bits);
                    const uint8_t* qs_lo = blk + 2;
                    const uint8_t* signs = blk + 2 + 32;
                    const uint8_t* qh     = blk + 2 + 64;
                    const uint8_t* scales = blk + 2 + 64 + 8;
                    const int x_base = b * QK_K;
                    for (int ib32 = 0; ib32 < 8; ++ib32) {
                        const std::uint8_t scale_byte = scales[ib32];
                        const float db_lo =
                            d * (0.5f + static_cast<float>(scale_byte & 0x0Fu)) * 0.25f;
                        const float db_hi =
                            d * (0.5f + static_cast<float>(scale_byte >> 4)) * 0.25f;
                        const int qs_off = ib32 * 4;
                        const std::uint8_t qh_byte = qh[ib32];
                        #pragma unroll
                        for (int l = 0; l < 4; ++l) {
                            const std::size_t high_bits =
                                (static_cast<std::size_t>(qh_byte) << (8 - 2 * l)) & 0x300u;
                            const std::size_t grid_idx =
                                static_cast<std::size_t>(qs_lo[qs_off + l]) | high_bits;
                            const std::uint8_t sign_byte = signs[qs_off + l];
                            const std::uint64_t grid_bits =
                                rsl::IQ2S_GRID_SYCL[grid_idx];
                            const float db = (l < 2) ? db_lo : db_hi;
                            const int x_off = x_base + ib32 * 32 + l * 8;
                            #pragma unroll
                            for (int j = 0; j < 8; ++j) {
                                const std::uint8_t gi =
                                    static_cast<std::uint8_t>(
                                        (grid_bits >> (j * 8)) & 0xFFu);
                                const float s = (sign_byte & rsl::KMASK_IQ2XS_SYCL[j])
                                    ? -1.0f : 1.0f;
                                acc += db
                                     * static_cast<float>(gi)
                                     * s
                                     * x_row[x_off + j];
                            }
                        }
                    }
                }
                out_usm[n * M + m] = acc;
            });
    }).wait();
}

extern "C" {

void rsl_matvec_iq2_s_packed_f32_batched_usm(rsl_stream* s,
                                              const void* w_bytes_usm,
                                              const float* x_usm,
                                              float* out_usm,
                                              int M, int K, int N,
                                              int lws) RSL_FFI_BODY_VOID(
    "rsl_matvec_iq2_s_packed_f32_batched_usm", {
    if (s == nullptr || w_bytes_usm == nullptr
        || x_usm == nullptr || out_usm == nullptr) {
        return;
    }
    if (M <= 0 || K <= 0 || N <= 0 || (K % 256) != 0) {
        return;
    }
    auto& q = s->q;
    const int eff_lws = (lws <= 0) ? static_cast<int>(RSL_LWS) : lws;
    switch (eff_lws) {
        case 16:  matvec_iq2_s_packed_f32_batched_usm_impl<16>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        case 32:  matvec_iq2_s_packed_f32_batched_usm_impl<32>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        case 64:  matvec_iq2_s_packed_f32_batched_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        case 128: matvec_iq2_s_packed_f32_batched_usm_impl<128>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        case 256: matvec_iq2_s_packed_f32_batched_usm_impl<256>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        default:  matvec_iq2_s_packed_f32_batched_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
    }
})

}  // extern "C" — temporary close for the IQ3_XXS template

// =====================================================================
// IQ3_XXS packed-USM single-row matvec.
// =====================================================================
//
// Block layout (98 bytes per 256-weight super-block, per the GGML
// IQ3_XXS spec):
//   { d: f16, qs_grid: [u8; 64], qs_sas: [u32; 8] }
// `qs_grid` provides 64 × u8 indices into the 256-entry IQ3XXS
// codebook (each codebook entry is `u32` = 4 packed u8 grid
// coordinates). `qs_sas` packs the per-sub-block scale (4-bit,
// top nibble) + four 7-bit sign-table indices (low 28 bits).
//
// Sub-block scale: `db = d * (0.5 + (aux32 >> 28)) * 0.5`. Each
// 8-weight chunk uses two grid lookups — one for the low 4
// weights and one for the high 4 — with signs from the same
// KSIGNS lookup split into low/high nibbles via KMASK[0..4] vs
// KMASK[4..8].
//
// One thread per output row.

template <std::size_t LWS_T>
inline void matvec_iq3_xxs_packed_f32_usm_impl(
    sycl::queue& q,
    const void* w_bytes_usm,
    const float* x_usm,
    float* out_usm,
    int M, int K) {
    constexpr std::size_t LWS = LWS_T;
    constexpr int BLOCK_BYTES = 98;
    constexpr int QK_K = 256;
    const int blocks_per_row = K / QK_K;
    const int bytes_per_row = blocks_per_row * BLOCK_BYTES;
    const uint8_t* w_bytes = static_cast<const uint8_t*>(w_bytes_usm);
    const std::size_t global =
        ((static_cast<std::size_t>(M) + LWS - 1) / LWS) * LWS;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(LWS)),
            [=](sycl::nd_item<1> it) {
                const int m = static_cast<int>(it.get_global_id(0));
                if (m >= M) return;
                const uint8_t* row = w_bytes + m * bytes_per_row;
                float acc = 0.0f;
                for (int b = 0; b < blocks_per_row; ++b) {
                    const uint8_t* blk = row + b * BLOCK_BYTES;
                    uint16_t d_bits = static_cast<uint16_t>(blk[0])
                                      | (static_cast<uint16_t>(blk[1]) << 8);
                    const float d = bits_to_f32(d_bits);
                    const uint8_t* qs_grid = blk + 2;       // 64 bytes
                    const uint8_t* qs_sas  = blk + 2 + 64;  // 32 bytes
                    const int x_base = b * QK_K;
                    for (int ib32 = 0; ib32 < 8; ++ib32) {
                        const std::uint32_t aux32 =
                            static_cast<std::uint32_t>(qs_sas[4 * ib32])
                            | (static_cast<std::uint32_t>(qs_sas[4 * ib32 + 1]) << 8)
                            | (static_cast<std::uint32_t>(qs_sas[4 * ib32 + 2]) << 16)
                            | (static_cast<std::uint32_t>(qs_sas[4 * ib32 + 3]) << 24);
                        const float db =
                            d * (0.5f + static_cast<float>(aux32 >> 28)) * 0.5f;
                        const int qs_off = 8 * ib32;
                        #pragma unroll
                        for (int l = 0; l < 4; ++l) {
                            const std::size_t g1_idx =
                                static_cast<std::size_t>(qs_grid[qs_off + 2 * l]);
                            const std::size_t g2_idx =
                                static_cast<std::size_t>(qs_grid[qs_off + 2 * l + 1]);
                            const std::uint32_t grid1_bits =
                                rsl::IQ3XXS_GRID_SYCL[g1_idx];
                            const std::uint32_t grid2_bits =
                                rsl::IQ3XXS_GRID_SYCL[g2_idx];
                            const std::uint8_t signs =
                                rsl::KSIGNS_IQ2XS_SYCL[(aux32 >> (7 * l)) & 127u];
                            const int x_off = x_base + ib32 * 32 + l * 8;
                            #pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                const std::uint8_t g1 =
                                    static_cast<std::uint8_t>(
                                        (grid1_bits >> (j * 8)) & 0xFFu);
                                const std::uint8_t g2 =
                                    static_cast<std::uint8_t>(
                                        (grid2_bits >> (j * 8)) & 0xFFu);
                                const float s_lo = (signs & rsl::KMASK_IQ2XS_SYCL[j])
                                    ? -1.0f : 1.0f;
                                const float s_hi = (signs & rsl::KMASK_IQ2XS_SYCL[j + 4])
                                    ? -1.0f : 1.0f;
                                acc += db
                                     * static_cast<float>(g1)
                                     * s_lo
                                     * x_usm[x_off + j];
                                acc += db
                                     * static_cast<float>(g2)
                                     * s_hi
                                     * x_usm[x_off + j + 4];
                            }
                        }
                    }
                }
                out_usm[m] = acc;
            });
    }).wait();
}

extern "C" {

void rsl_matvec_iq3_xxs_packed_f32_usm(rsl_stream* s,
                                       const void* w_bytes_usm,
                                       const float* x_usm,
                                       float* out_usm,
                                       int M, int K,
                                       int lws) RSL_FFI_BODY_VOID(
    "rsl_matvec_iq3_xxs_packed_f32_usm", {
    if (s == nullptr || w_bytes_usm == nullptr
        || x_usm == nullptr || out_usm == nullptr) {
        return;
    }
    if (M <= 0 || K <= 0 || (K % 256) != 0) {
        return;
    }
    auto& q = s->q;
    const int eff_lws = (lws <= 0) ? static_cast<int>(RSL_LWS) : lws;
    switch (eff_lws) {
        case 16:  matvec_iq3_xxs_packed_f32_usm_impl<16>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 32:  matvec_iq3_xxs_packed_f32_usm_impl<32>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 64:  matvec_iq3_xxs_packed_f32_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 128: matvec_iq3_xxs_packed_f32_usm_impl<128>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 256: matvec_iq3_xxs_packed_f32_usm_impl<256>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        default:  matvec_iq3_xxs_packed_f32_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
    }
})

}  // extern "C" — close before IQ3_XXS gate+up fused template

// H4: IQ3_XXS gate + up fused matvec — mirrors matvec_iq3_xxs_packed_f32_usm_impl
// with paired accumulators.
template <std::size_t LWS_T>
inline void matvec_iq3_xxs_gate_up_fused_usm_impl(
    sycl::queue& q,
    const void* gate_w_bytes_usm,
    const void* up_w_bytes_usm,
    const float* x_usm,
    float* gate_out_usm,
    float* up_out_usm,
    int M, int K) {
    constexpr std::size_t LWS = LWS_T;
    constexpr int BLOCK_BYTES = 98;
    constexpr int QK_K = 256;
    const int blocks_per_row = K / QK_K;
    const int bytes_per_row = blocks_per_row * BLOCK_BYTES;
    const uint8_t* gate_w_bytes = static_cast<const uint8_t*>(gate_w_bytes_usm);
    const uint8_t* up_w_bytes   = static_cast<const uint8_t*>(up_w_bytes_usm);
    const std::size_t global =
        ((static_cast<std::size_t>(M) + LWS - 1) / LWS) * LWS;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(LWS)),
            [=](sycl::nd_item<1> it) {
                const int m = static_cast<int>(it.get_global_id(0));
                if (m >= M) return;
                const uint8_t* g_row = gate_w_bytes + m * bytes_per_row;
                const uint8_t* u_row = up_w_bytes   + m * bytes_per_row;
                float gate_acc = 0.0f;
                float up_acc = 0.0f;
                for (int b = 0; b < blocks_per_row; ++b) {
                    const uint8_t* g_blk = g_row + b * BLOCK_BYTES;
                    const uint8_t* u_blk = u_row + b * BLOCK_BYTES;
                    const float g_d = bits_to_f32(static_cast<uint16_t>(g_blk[0])
                                                   | (static_cast<uint16_t>(g_blk[1]) << 8));
                    const float u_d = bits_to_f32(static_cast<uint16_t>(u_blk[0])
                                                   | (static_cast<uint16_t>(u_blk[1]) << 8));
                    const uint8_t* g_qs_grid = g_blk + 2;
                    const uint8_t* g_qs_sas  = g_blk + 2 + 64;
                    const uint8_t* u_qs_grid = u_blk + 2;
                    const uint8_t* u_qs_sas  = u_blk + 2 + 64;
                    const int x_base = b * QK_K;
                    for (int ib32 = 0; ib32 < 8; ++ib32) {
                        const std::uint32_t g_aux32 =
                            static_cast<std::uint32_t>(g_qs_sas[4 * ib32])
                            | (static_cast<std::uint32_t>(g_qs_sas[4 * ib32 + 1]) << 8)
                            | (static_cast<std::uint32_t>(g_qs_sas[4 * ib32 + 2]) << 16)
                            | (static_cast<std::uint32_t>(g_qs_sas[4 * ib32 + 3]) << 24);
                        const std::uint32_t u_aux32 =
                            static_cast<std::uint32_t>(u_qs_sas[4 * ib32])
                            | (static_cast<std::uint32_t>(u_qs_sas[4 * ib32 + 1]) << 8)
                            | (static_cast<std::uint32_t>(u_qs_sas[4 * ib32 + 2]) << 16)
                            | (static_cast<std::uint32_t>(u_qs_sas[4 * ib32 + 3]) << 24);
                        const float g_db = g_d * (0.5f + static_cast<float>(g_aux32 >> 28)) * 0.5f;
                        const float u_db = u_d * (0.5f + static_cast<float>(u_aux32 >> 28)) * 0.5f;
                        const int qs_off = 8 * ib32;
                        #pragma unroll
                        for (int l = 0; l < 4; ++l) {
                            const std::size_t g_g1_idx =
                                static_cast<std::size_t>(g_qs_grid[qs_off + 2 * l]);
                            const std::size_t g_g2_idx =
                                static_cast<std::size_t>(g_qs_grid[qs_off + 2 * l + 1]);
                            const std::size_t u_g1_idx =
                                static_cast<std::size_t>(u_qs_grid[qs_off + 2 * l]);
                            const std::size_t u_g2_idx =
                                static_cast<std::size_t>(u_qs_grid[qs_off + 2 * l + 1]);
                            const std::uint32_t g_grid1 = rsl::IQ3XXS_GRID_SYCL[g_g1_idx];
                            const std::uint32_t g_grid2 = rsl::IQ3XXS_GRID_SYCL[g_g2_idx];
                            const std::uint32_t u_grid1 = rsl::IQ3XXS_GRID_SYCL[u_g1_idx];
                            const std::uint32_t u_grid2 = rsl::IQ3XXS_GRID_SYCL[u_g2_idx];
                            const std::uint8_t g_signs = rsl::KSIGNS_IQ2XS_SYCL[(g_aux32 >> (7 * l)) & 127u];
                            const std::uint8_t u_signs = rsl::KSIGNS_IQ2XS_SYCL[(u_aux32 >> (7 * l)) & 127u];
                            const int x_off = x_base + ib32 * 32 + l * 8;
                            #pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                const float xv_lo = x_usm[x_off + j];
                                const float xv_hi = x_usm[x_off + j + 4];
                                const std::uint8_t g_g1b =
                                    static_cast<std::uint8_t>((g_grid1 >> (j * 8)) & 0xFFu);
                                const std::uint8_t g_g2b =
                                    static_cast<std::uint8_t>((g_grid2 >> (j * 8)) & 0xFFu);
                                const std::uint8_t u_g1b =
                                    static_cast<std::uint8_t>((u_grid1 >> (j * 8)) & 0xFFu);
                                const std::uint8_t u_g2b =
                                    static_cast<std::uint8_t>((u_grid2 >> (j * 8)) & 0xFFu);
                                const float g_s_lo = (g_signs & rsl::KMASK_IQ2XS_SYCL[j]) ? -1.0f : 1.0f;
                                const float g_s_hi = (g_signs & rsl::KMASK_IQ2XS_SYCL[j + 4]) ? -1.0f : 1.0f;
                                const float u_s_lo = (u_signs & rsl::KMASK_IQ2XS_SYCL[j]) ? -1.0f : 1.0f;
                                const float u_s_hi = (u_signs & rsl::KMASK_IQ2XS_SYCL[j + 4]) ? -1.0f : 1.0f;
                                gate_acc += g_db * static_cast<float>(g_g1b) * g_s_lo * xv_lo;
                                gate_acc += g_db * static_cast<float>(g_g2b) * g_s_hi * xv_hi;
                                up_acc   += u_db * static_cast<float>(u_g1b) * u_s_lo * xv_lo;
                                up_acc   += u_db * static_cast<float>(u_g2b) * u_s_hi * xv_hi;
                            }
                        }
                    }
                }
                gate_out_usm[m] = gate_acc;
                up_out_usm[m] = up_acc;
            });
    }).wait();
}
extern "C" {

void rsl_matvec_iq3_xxs_gate_up_fused_usm(rsl_stream* s,
                                          const void* gate_w_bytes_usm,
                                          const void* up_w_bytes_usm,
                                          const float* x_usm,
                                          float* gate_out_usm,
                                          float* up_out_usm,
                                          int M, int K,
                                          int lws) RSL_FFI_BODY_VOID(
    "rsl_matvec_iq3_xxs_gate_up_fused_usm", {
    if (s == nullptr || gate_w_bytes_usm == nullptr || up_w_bytes_usm == nullptr
        || x_usm == nullptr || gate_out_usm == nullptr || up_out_usm == nullptr) {
        return;
    }
    if (M <= 0 || K <= 0 || (K % 256) != 0) {
        return;
    }
    auto& q = s->q;
    const int eff_lws = (lws <= 0) ? static_cast<int>(RSL_LWS) : lws;
    switch (eff_lws) {
        case 16:  matvec_iq3_xxs_gate_up_fused_usm_impl<16>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        case 32:  matvec_iq3_xxs_gate_up_fused_usm_impl<32>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        case 64:  matvec_iq3_xxs_gate_up_fused_usm_impl<64>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        case 128: matvec_iq3_xxs_gate_up_fused_usm_impl<128>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        case 256: matvec_iq3_xxs_gate_up_fused_usm_impl<256>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        default:  matvec_iq3_xxs_gate_up_fused_usm_impl<64>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
    }
})

}  // extern "C" — close before the IQ3_XXS batched template

// Batched IQ3_XXS packed-USM matvec. Iterates over N input rows.
template <std::size_t LWS_T>
inline void matvec_iq3_xxs_packed_f32_batched_usm_impl(
    sycl::queue& q,
    const void* w_bytes_usm,
    const float* x_usm,
    float* out_usm,
    int M, int K, int N) {
    constexpr std::size_t LWS = LWS_T;
    constexpr int BLOCK_BYTES = 98;
    constexpr int QK_K = 256;
    const int blocks_per_row = K / QK_K;
    const int bytes_per_row = blocks_per_row * BLOCK_BYTES;
    const uint8_t* w_bytes = static_cast<const uint8_t*>(w_bytes_usm);
    const std::size_t global_m =
        ((static_cast<std::size_t>(M) + LWS - 1) / LWS) * LWS;
    sycl::range<2> global(global_m, static_cast<std::size_t>(N));
    sycl::range<2> local(LWS, 1);
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<2>(global, local),
            [=](sycl::nd_item<2> it) {
                const int m = static_cast<int>(it.get_global_id(0));
                const int n = static_cast<int>(it.get_global_id(1));
                if (m >= M || n >= N) return;
                const uint8_t* row = w_bytes + m * bytes_per_row;
                const float* x_row = x_usm + n * K;
                float acc = 0.0f;
                for (int b = 0; b < blocks_per_row; ++b) {
                    const uint8_t* blk = row + b * BLOCK_BYTES;
                    uint16_t d_bits = static_cast<uint16_t>(blk[0])
                                      | (static_cast<uint16_t>(blk[1]) << 8);
                    const float d = bits_to_f32(d_bits);
                    const uint8_t* qs_grid = blk + 2;
                    const uint8_t* qs_sas  = blk + 2 + 64;
                    const int x_base = b * QK_K;
                    for (int ib32 = 0; ib32 < 8; ++ib32) {
                        const std::uint32_t aux32 =
                            static_cast<std::uint32_t>(qs_sas[4 * ib32])
                            | (static_cast<std::uint32_t>(qs_sas[4 * ib32 + 1]) << 8)
                            | (static_cast<std::uint32_t>(qs_sas[4 * ib32 + 2]) << 16)
                            | (static_cast<std::uint32_t>(qs_sas[4 * ib32 + 3]) << 24);
                        const float db =
                            d * (0.5f + static_cast<float>(aux32 >> 28)) * 0.5f;
                        const int qs_off = 8 * ib32;
                        #pragma unroll
                        for (int l = 0; l < 4; ++l) {
                            const std::size_t g1_idx =
                                static_cast<std::size_t>(qs_grid[qs_off + 2 * l]);
                            const std::size_t g2_idx =
                                static_cast<std::size_t>(qs_grid[qs_off + 2 * l + 1]);
                            const std::uint32_t grid1_bits =
                                rsl::IQ3XXS_GRID_SYCL[g1_idx];
                            const std::uint32_t grid2_bits =
                                rsl::IQ3XXS_GRID_SYCL[g2_idx];
                            const std::uint8_t signs =
                                rsl::KSIGNS_IQ2XS_SYCL[(aux32 >> (7 * l)) & 127u];
                            const int x_off = x_base + ib32 * 32 + l * 8;
                            #pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                const std::uint8_t g1 =
                                    static_cast<std::uint8_t>(
                                        (grid1_bits >> (j * 8)) & 0xFFu);
                                const std::uint8_t g2 =
                                    static_cast<std::uint8_t>(
                                        (grid2_bits >> (j * 8)) & 0xFFu);
                                const float s_lo = (signs & rsl::KMASK_IQ2XS_SYCL[j])
                                    ? -1.0f : 1.0f;
                                const float s_hi = (signs & rsl::KMASK_IQ2XS_SYCL[j + 4])
                                    ? -1.0f : 1.0f;
                                acc += db
                                     * static_cast<float>(g1)
                                     * s_lo
                                     * x_row[x_off + j];
                                acc += db
                                     * static_cast<float>(g2)
                                     * s_hi
                                     * x_row[x_off + j + 4];
                            }
                        }
                    }
                }
                out_usm[n * M + m] = acc;
            });
    }).wait();
}

extern "C" {

void rsl_matvec_iq3_xxs_packed_f32_batched_usm(rsl_stream* s,
                                                const void* w_bytes_usm,
                                                const float* x_usm,
                                                float* out_usm,
                                                int M, int K, int N,
                                                int lws) RSL_FFI_BODY_VOID(
    "rsl_matvec_iq3_xxs_packed_f32_batched_usm", {
    if (s == nullptr || w_bytes_usm == nullptr
        || x_usm == nullptr || out_usm == nullptr) {
        return;
    }
    if (M <= 0 || K <= 0 || N <= 0 || (K % 256) != 0) {
        return;
    }
    auto& q = s->q;
    const int eff_lws = (lws <= 0) ? static_cast<int>(RSL_LWS) : lws;
    switch (eff_lws) {
        case 16:  matvec_iq3_xxs_packed_f32_batched_usm_impl<16>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        case 32:  matvec_iq3_xxs_packed_f32_batched_usm_impl<32>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        case 64:  matvec_iq3_xxs_packed_f32_batched_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        case 128: matvec_iq3_xxs_packed_f32_batched_usm_impl<128>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        case 256: matvec_iq3_xxs_packed_f32_batched_usm_impl<256>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        default:  matvec_iq3_xxs_packed_f32_batched_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
    }
})

}  // extern "C" — temporary close for the IQ3_S template

// =====================================================================
// IQ3_S packed-USM single-row matvec.
// =====================================================================
//
// Block layout (110 bytes per 256-weight super-block, per the GGML
// IQ3_S spec):
//   { d: f16, qs: [u8; 64], qh: [u8; 8], signs: [u8; 32], scales: [u8; 4] }
// 9-bit grid index per chunk: low 8 from `qs`, high 1 packed in
// `qh` (one chunk per bit, alternating odd/even via the `<< (8 - 2l)`
// / `<< (7 - 2l)` mask trick). Codebook is 512 × u32 (4 packed u8
// grid coords each). Signs stored inline (32 bytes for 32 chunks).
// Scales are 4 bytes total — 8 sub-block scales packed two-per-byte
// as 4-bit nibbles. Sub-block scale: `db = d * (1 + 2 * nibble)`.
//
// Sub-blocks pair-up: pair p ∈ [0,4) covers ib32 = 2p (using nibble
// low) and ib32 = 2p+1 (using nibble high). Each ib32 walks 8 qs
// bytes (4 chunks × 2 grids) + 4 sign bytes. Each 8-weight chunk
// does two grid lookups for low-4 and high-4 weights, with signs
// split via KMASK[0..4] / KMASK[4..8].
//
// One thread per output row.

template <std::size_t LWS_T>
inline void matvec_iq3_s_packed_f32_usm_impl(
    sycl::queue& q,
    const void* w_bytes_usm,
    const float* x_usm,
    float* out_usm,
    int M, int K) {
    constexpr std::size_t LWS = LWS_T;
    constexpr int BLOCK_BYTES = 110;
    constexpr int QK_K = 256;
    const int blocks_per_row = K / QK_K;
    const int bytes_per_row = blocks_per_row * BLOCK_BYTES;
    const uint8_t* w_bytes = static_cast<const uint8_t*>(w_bytes_usm);
    const std::size_t global =
        ((static_cast<std::size_t>(M) + LWS - 1) / LWS) * LWS;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(LWS)),
            [=](sycl::nd_item<1> it) {
                const int m = static_cast<int>(it.get_global_id(0));
                if (m >= M) return;
                const uint8_t* row = w_bytes + m * bytes_per_row;
                float acc = 0.0f;
                for (int b = 0; b < blocks_per_row; ++b) {
                    const uint8_t* blk = row + b * BLOCK_BYTES;
                    uint16_t d_bits = static_cast<uint16_t>(blk[0])
                                      | (static_cast<uint16_t>(blk[1]) << 8);
                    const float d = bits_to_f32(d_bits);
                    const uint8_t* qs     = blk + 2;             // 64 bytes
                    const uint8_t* qh     = blk + 2 + 64;        // 8 bytes
                    const uint8_t* signs  = blk + 2 + 64 + 8;    // 32 bytes
                    const uint8_t* scales = blk + 2 + 64 + 8 + 32; // 4 bytes
                    const int x_base = b * QK_K;
                    for (int ib32 = 0; ib32 < 8; ++ib32) {
                        const int pair = ib32 >> 1;
                        const std::uint8_t scale_byte = scales[pair];
                        const float db = (ib32 & 1)
                            ? d * (1.0f + 2.0f * static_cast<float>(scale_byte >> 4))
                            : d * (1.0f + 2.0f * static_cast<float>(scale_byte & 0x0Fu));
                        const int qs_off = ib32 * 8;
                        const int signs_off = ib32 * 4;
                        const std::uint8_t qh_byte = qh[ib32];
                        const int x_off_block = ib32 * 32;
                        #pragma unroll
                        for (int l = 0; l < 4; ++l) {
                            const std::size_t g1_idx =
                                static_cast<std::size_t>(qs[qs_off + 2 * l])
                                | ((static_cast<std::size_t>(qh_byte) << (8 - 2 * l)) & 0x100u);
                            const std::size_t g2_idx =
                                static_cast<std::size_t>(qs[qs_off + 2 * l + 1])
                                | ((static_cast<std::size_t>(qh_byte) << (7 - 2 * l)) & 0x100u);
                            const std::uint32_t grid1_bits =
                                rsl::IQ3S_GRID_SYCL[g1_idx];
                            const std::uint32_t grid2_bits =
                                rsl::IQ3S_GRID_SYCL[g2_idx];
                            const std::uint8_t sign_byte = signs[signs_off + l];
                            const int x_off = x_base + x_off_block + l * 8;
                            #pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                const std::uint8_t g1 =
                                    static_cast<std::uint8_t>(
                                        (grid1_bits >> (j * 8)) & 0xFFu);
                                const std::uint8_t g2 =
                                    static_cast<std::uint8_t>(
                                        (grid2_bits >> (j * 8)) & 0xFFu);
                                const float s_lo = (sign_byte & rsl::KMASK_IQ2XS_SYCL[j])
                                    ? -1.0f : 1.0f;
                                const float s_hi = (sign_byte & rsl::KMASK_IQ2XS_SYCL[j + 4])
                                    ? -1.0f : 1.0f;
                                acc += db
                                     * static_cast<float>(g1)
                                     * s_lo
                                     * x_usm[x_off + j];
                                acc += db
                                     * static_cast<float>(g2)
                                     * s_hi
                                     * x_usm[x_off + j + 4];
                            }
                        }
                    }
                }
                out_usm[m] = acc;
            });
    }).wait();
}

extern "C" {

void rsl_matvec_iq3_s_packed_f32_usm(rsl_stream* s,
                                     const void* w_bytes_usm,
                                     const float* x_usm,
                                     float* out_usm,
                                     int M, int K,
                                     int lws) RSL_FFI_BODY_VOID(
    "rsl_matvec_iq3_s_packed_f32_usm", {
    if (s == nullptr || w_bytes_usm == nullptr
        || x_usm == nullptr || out_usm == nullptr) {
        return;
    }
    if (M <= 0 || K <= 0 || (K % 256) != 0) {
        return;
    }
    auto& q = s->q;
    const int eff_lws = (lws <= 0) ? static_cast<int>(RSL_LWS) : lws;
    switch (eff_lws) {
        case 16:  matvec_iq3_s_packed_f32_usm_impl<16>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 32:  matvec_iq3_s_packed_f32_usm_impl<32>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 64:  matvec_iq3_s_packed_f32_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 128: matvec_iq3_s_packed_f32_usm_impl<128>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 256: matvec_iq3_s_packed_f32_usm_impl<256>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        default:  matvec_iq3_s_packed_f32_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
    }
})

}  // extern "C" — close before IQ3_S gate+up fused template

// H4: IQ3_S gate + up fused matvec — mirrors matvec_iq3_s_packed_f32_usm_impl
// with paired accumulators.
template <std::size_t LWS_T>
inline void matvec_iq3_s_gate_up_fused_usm_impl(
    sycl::queue& q,
    const void* gate_w_bytes_usm,
    const void* up_w_bytes_usm,
    const float* x_usm,
    float* gate_out_usm,
    float* up_out_usm,
    int M, int K) {
    constexpr std::size_t LWS = LWS_T;
    constexpr int BLOCK_BYTES = 110;
    constexpr int QK_K = 256;
    const int blocks_per_row = K / QK_K;
    const int bytes_per_row = blocks_per_row * BLOCK_BYTES;
    const uint8_t* gate_w_bytes = static_cast<const uint8_t*>(gate_w_bytes_usm);
    const uint8_t* up_w_bytes   = static_cast<const uint8_t*>(up_w_bytes_usm);
    const std::size_t global =
        ((static_cast<std::size_t>(M) + LWS - 1) / LWS) * LWS;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(LWS)),
            [=](sycl::nd_item<1> it) {
                const int m = static_cast<int>(it.get_global_id(0));
                if (m >= M) return;
                const uint8_t* g_row = gate_w_bytes + m * bytes_per_row;
                const uint8_t* u_row = up_w_bytes   + m * bytes_per_row;
                float gate_acc = 0.0f;
                float up_acc = 0.0f;
                for (int b = 0; b < blocks_per_row; ++b) {
                    const uint8_t* g_blk = g_row + b * BLOCK_BYTES;
                    const uint8_t* u_blk = u_row + b * BLOCK_BYTES;
                    const float g_d = bits_to_f32(static_cast<uint16_t>(g_blk[0])
                                                   | (static_cast<uint16_t>(g_blk[1]) << 8));
                    const float u_d = bits_to_f32(static_cast<uint16_t>(u_blk[0])
                                                   | (static_cast<uint16_t>(u_blk[1]) << 8));
                    const uint8_t* g_qs     = g_blk + 2;
                    const uint8_t* g_qh     = g_blk + 2 + 64;
                    const uint8_t* g_signs  = g_blk + 2 + 64 + 8;
                    const uint8_t* g_scales = g_blk + 2 + 64 + 8 + 32;
                    const uint8_t* u_qs     = u_blk + 2;
                    const uint8_t* u_qh     = u_blk + 2 + 64;
                    const uint8_t* u_signs  = u_blk + 2 + 64 + 8;
                    const uint8_t* u_scales = u_blk + 2 + 64 + 8 + 32;
                    const int x_base = b * QK_K;
                    for (int ib32 = 0; ib32 < 8; ++ib32) {
                        const int pair = ib32 >> 1;
                        const std::uint8_t g_scale_byte = g_scales[pair];
                        const std::uint8_t u_scale_byte = u_scales[pair];
                        const float g_db = (ib32 & 1)
                            ? g_d * (1.0f + 2.0f * static_cast<float>(g_scale_byte >> 4))
                            : g_d * (1.0f + 2.0f * static_cast<float>(g_scale_byte & 0x0Fu));
                        const float u_db = (ib32 & 1)
                            ? u_d * (1.0f + 2.0f * static_cast<float>(u_scale_byte >> 4))
                            : u_d * (1.0f + 2.0f * static_cast<float>(u_scale_byte & 0x0Fu));
                        const int qs_off = ib32 * 8;
                        const int signs_off = ib32 * 4;
                        const std::uint8_t g_qh_byte = g_qh[ib32];
                        const std::uint8_t u_qh_byte = u_qh[ib32];
                        const int x_off_block = ib32 * 32;
                        #pragma unroll
                        for (int l = 0; l < 4; ++l) {
                            const std::size_t g_g1_idx =
                                static_cast<std::size_t>(g_qs[qs_off + 2 * l])
                                | ((static_cast<std::size_t>(g_qh_byte) << (8 - 2 * l)) & 0x100u);
                            const std::size_t g_g2_idx =
                                static_cast<std::size_t>(g_qs[qs_off + 2 * l + 1])
                                | ((static_cast<std::size_t>(g_qh_byte) << (7 - 2 * l)) & 0x100u);
                            const std::size_t u_g1_idx =
                                static_cast<std::size_t>(u_qs[qs_off + 2 * l])
                                | ((static_cast<std::size_t>(u_qh_byte) << (8 - 2 * l)) & 0x100u);
                            const std::size_t u_g2_idx =
                                static_cast<std::size_t>(u_qs[qs_off + 2 * l + 1])
                                | ((static_cast<std::size_t>(u_qh_byte) << (7 - 2 * l)) & 0x100u);
                            const std::uint32_t g_grid1 = rsl::IQ3S_GRID_SYCL[g_g1_idx];
                            const std::uint32_t g_grid2 = rsl::IQ3S_GRID_SYCL[g_g2_idx];
                            const std::uint32_t u_grid1 = rsl::IQ3S_GRID_SYCL[u_g1_idx];
                            const std::uint32_t u_grid2 = rsl::IQ3S_GRID_SYCL[u_g2_idx];
                            const std::uint8_t g_sign_byte = g_signs[signs_off + l];
                            const std::uint8_t u_sign_byte = u_signs[signs_off + l];
                            const int x_off = x_base + x_off_block + l * 8;
                            #pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                const float xv_lo = x_usm[x_off + j];
                                const float xv_hi = x_usm[x_off + j + 4];
                                const std::uint8_t g_g1b =
                                    static_cast<std::uint8_t>((g_grid1 >> (j * 8)) & 0xFFu);
                                const std::uint8_t g_g2b =
                                    static_cast<std::uint8_t>((g_grid2 >> (j * 8)) & 0xFFu);
                                const std::uint8_t u_g1b =
                                    static_cast<std::uint8_t>((u_grid1 >> (j * 8)) & 0xFFu);
                                const std::uint8_t u_g2b =
                                    static_cast<std::uint8_t>((u_grid2 >> (j * 8)) & 0xFFu);
                                const float g_s_lo = (g_sign_byte & rsl::KMASK_IQ2XS_SYCL[j]) ? -1.0f : 1.0f;
                                const float g_s_hi = (g_sign_byte & rsl::KMASK_IQ2XS_SYCL[j + 4]) ? -1.0f : 1.0f;
                                const float u_s_lo = (u_sign_byte & rsl::KMASK_IQ2XS_SYCL[j]) ? -1.0f : 1.0f;
                                const float u_s_hi = (u_sign_byte & rsl::KMASK_IQ2XS_SYCL[j + 4]) ? -1.0f : 1.0f;
                                gate_acc += g_db * static_cast<float>(g_g1b) * g_s_lo * xv_lo;
                                gate_acc += g_db * static_cast<float>(g_g2b) * g_s_hi * xv_hi;
                                up_acc   += u_db * static_cast<float>(u_g1b) * u_s_lo * xv_lo;
                                up_acc   += u_db * static_cast<float>(u_g2b) * u_s_hi * xv_hi;
                            }
                        }
                    }
                }
                gate_out_usm[m] = gate_acc;
                up_out_usm[m] = up_acc;
            });
    }).wait();
}
extern "C" {

void rsl_matvec_iq3_s_gate_up_fused_usm(rsl_stream* s,
                                        const void* gate_w_bytes_usm,
                                        const void* up_w_bytes_usm,
                                        const float* x_usm,
                                        float* gate_out_usm,
                                        float* up_out_usm,
                                        int M, int K,
                                        int lws) RSL_FFI_BODY_VOID(
    "rsl_matvec_iq3_s_gate_up_fused_usm", {
    if (s == nullptr || gate_w_bytes_usm == nullptr || up_w_bytes_usm == nullptr
        || x_usm == nullptr || gate_out_usm == nullptr || up_out_usm == nullptr) {
        return;
    }
    if (M <= 0 || K <= 0 || (K % 256) != 0) {
        return;
    }
    auto& q = s->q;
    const int eff_lws = (lws <= 0) ? static_cast<int>(RSL_LWS) : lws;
    switch (eff_lws) {
        case 16:  matvec_iq3_s_gate_up_fused_usm_impl<16>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        case 32:  matvec_iq3_s_gate_up_fused_usm_impl<32>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        case 64:  matvec_iq3_s_gate_up_fused_usm_impl<64>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        case 128: matvec_iq3_s_gate_up_fused_usm_impl<128>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        case 256: matvec_iq3_s_gate_up_fused_usm_impl<256>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        default:  matvec_iq3_s_gate_up_fused_usm_impl<64>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
    }
})

}  // extern "C" — close before the IQ3_S batched template

// Batched IQ3_S packed-USM matvec. Iterates over N input rows.
template <std::size_t LWS_T>
inline void matvec_iq3_s_packed_f32_batched_usm_impl(
    sycl::queue& q,
    const void* w_bytes_usm,
    const float* x_usm,
    float* out_usm,
    int M, int K, int N) {
    constexpr std::size_t LWS = LWS_T;
    constexpr int BLOCK_BYTES = 110;
    constexpr int QK_K = 256;
    const int blocks_per_row = K / QK_K;
    const int bytes_per_row = blocks_per_row * BLOCK_BYTES;
    const uint8_t* w_bytes = static_cast<const uint8_t*>(w_bytes_usm);
    const std::size_t global_m =
        ((static_cast<std::size_t>(M) + LWS - 1) / LWS) * LWS;
    sycl::range<2> global(global_m, static_cast<std::size_t>(N));
    sycl::range<2> local(LWS, 1);
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<2>(global, local),
            [=](sycl::nd_item<2> it) {
                const int m = static_cast<int>(it.get_global_id(0));
                const int n = static_cast<int>(it.get_global_id(1));
                if (m >= M || n >= N) return;
                const uint8_t* row = w_bytes + m * bytes_per_row;
                const float* x_row = x_usm + n * K;
                float acc = 0.0f;
                for (int b = 0; b < blocks_per_row; ++b) {
                    const uint8_t* blk = row + b * BLOCK_BYTES;
                    uint16_t d_bits = static_cast<uint16_t>(blk[0])
                                      | (static_cast<uint16_t>(blk[1]) << 8);
                    const float d = bits_to_f32(d_bits);
                    const uint8_t* qs     = blk + 2;
                    const uint8_t* qh     = blk + 2 + 64;
                    const uint8_t* signs  = blk + 2 + 64 + 8;
                    const uint8_t* scales = blk + 2 + 64 + 8 + 32;
                    const int x_base = b * QK_K;
                    for (int ib32 = 0; ib32 < 8; ++ib32) {
                        const int pair = ib32 >> 1;
                        const std::uint8_t scale_byte = scales[pair];
                        const float db = (ib32 & 1)
                            ? d * (1.0f + 2.0f * static_cast<float>(scale_byte >> 4))
                            : d * (1.0f + 2.0f * static_cast<float>(scale_byte & 0x0Fu));
                        const int qs_off = ib32 * 8;
                        const int signs_off = ib32 * 4;
                        const std::uint8_t qh_byte = qh[ib32];
                        const int x_off_block = ib32 * 32;
                        #pragma unroll
                        for (int l = 0; l < 4; ++l) {
                            const std::size_t g1_idx =
                                static_cast<std::size_t>(qs[qs_off + 2 * l])
                                | ((static_cast<std::size_t>(qh_byte) << (8 - 2 * l)) & 0x100u);
                            const std::size_t g2_idx =
                                static_cast<std::size_t>(qs[qs_off + 2 * l + 1])
                                | ((static_cast<std::size_t>(qh_byte) << (7 - 2 * l)) & 0x100u);
                            const std::uint32_t grid1_bits =
                                rsl::IQ3S_GRID_SYCL[g1_idx];
                            const std::uint32_t grid2_bits =
                                rsl::IQ3S_GRID_SYCL[g2_idx];
                            const std::uint8_t sign_byte = signs[signs_off + l];
                            const int x_off = x_base + x_off_block + l * 8;
                            #pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                const std::uint8_t g1 =
                                    static_cast<std::uint8_t>(
                                        (grid1_bits >> (j * 8)) & 0xFFu);
                                const std::uint8_t g2 =
                                    static_cast<std::uint8_t>(
                                        (grid2_bits >> (j * 8)) & 0xFFu);
                                const float s_lo = (sign_byte & rsl::KMASK_IQ2XS_SYCL[j])
                                    ? -1.0f : 1.0f;
                                const float s_hi = (sign_byte & rsl::KMASK_IQ2XS_SYCL[j + 4])
                                    ? -1.0f : 1.0f;
                                acc += db
                                     * static_cast<float>(g1)
                                     * s_lo
                                     * x_row[x_off + j];
                                acc += db
                                     * static_cast<float>(g2)
                                     * s_hi
                                     * x_row[x_off + j + 4];
                            }
                        }
                    }
                }
                out_usm[n * M + m] = acc;
            });
    }).wait();
}

extern "C" {

void rsl_matvec_iq3_s_packed_f32_batched_usm(rsl_stream* s,
                                              const void* w_bytes_usm,
                                              const float* x_usm,
                                              float* out_usm,
                                              int M, int K, int N,
                                              int lws) RSL_FFI_BODY_VOID(
    "rsl_matvec_iq3_s_packed_f32_batched_usm", {
    if (s == nullptr || w_bytes_usm == nullptr
        || x_usm == nullptr || out_usm == nullptr) {
        return;
    }
    if (M <= 0 || K <= 0 || N <= 0 || (K % 256) != 0) {
        return;
    }
    auto& q = s->q;
    const int eff_lws = (lws <= 0) ? static_cast<int>(RSL_LWS) : lws;
    switch (eff_lws) {
        case 16:  matvec_iq3_s_packed_f32_batched_usm_impl<16>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        case 32:  matvec_iq3_s_packed_f32_batched_usm_impl<32>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        case 64:  matvec_iq3_s_packed_f32_batched_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        case 128: matvec_iq3_s_packed_f32_batched_usm_impl<128>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        case 256: matvec_iq3_s_packed_f32_batched_usm_impl<256>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        default:  matvec_iq3_s_packed_f32_batched_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
    }
})

// =====================================================================
// IQ-encoder codebook-search GPU offload.
// =====================================================================
//
// The CPU encode-side of the IQ-quant family (IQ1_S/M, IQ2_XXS/XS/S,
// IQ3_XXS/S) is dominated by exhaustive grid search per 8-weight chunk
// — a 2048-entry × 8-element FMA loop for IQ1_S, similar shape for the
// IQ2/IQ3 family. The search is embarrassingly parallel across chunks;
// GPU offload of the inner loop turns multi-hour quantization runs
// into minutes on a capable Intel iGPU / Arc card.
//
// This first kernel ships the IQ1_S/IQ1_M path (8-element grid +
// per-batch delta sign, no sign mask). IQ2/IQ3 variants follow the
// same shape with a sign-mask additional output; landing them is a
// follow-up session once the IQ1_S kernel has been hardware-validated.
//
// **Status**: Experimental. The kernel arithmetic is a line-by-line
// port of the CPU `best_iq1s_grid_for_chunk_scalar` reference; the
// SYCL plumbing has not yet been validated on real hardware. Expect
// 1-2 iteration cycles on an Iris Xe / Arc box before defaulting to
// GPU for production quantize runs.
//
// Kernel topology:
//   - One subgroup per input chunk. Intel iGPU SG = 16 by default.
//   - Each lane evaluates GRID_SIZE / 16 = 128 grid candidates serially
//     (for IQ1_S's GRID_SIZE = 2048).
//   - Subgroup reduction picks the winning lane by max of
//     `dot² / norm`; argmin/argmax via custom select_from_group pattern.
//   - Lane 0 writes (grid_idx, signed_score, norm_sq) per chunk.

}  // extern "C" — temporary close for the IQ encoder template

// #3: Variant returning all THREE picks per chunk (max-|score|,
// max-positive-score, max-negative-score) in one kernel dispatch.
// Eliminates the CPU-side AVX2 pos_neg sweep that previously dominated
// the encode_iq1_s_with_encoder bit-pack stage.
template <int GRID_SIZE, int LWS>
inline void iq_search_8elt_delta_all3_impl(
    sycl::queue& q,
    const float* targets,
    float delta,
    const float* grid_f32,
    uint16_t* out_grid_idx_abs,
    float* out_signed_score_abs,
    float* out_norm_sq_abs,
    uint16_t* out_grid_idx_pos,
    float* out_signed_score_pos,
    float* out_norm_sq_pos,
    uint16_t* out_grid_idx_neg,
    float* out_signed_score_neg,
    float* out_norm_sq_neg,
    int n_chunks) {
    static_assert(GRID_SIZE % LWS == 0,
        "GRID_SIZE must be divisible by LWS");
    constexpr int CANDIDATES_PER_LANE = GRID_SIZE / LWS;
    const std::size_t global =
        static_cast<std::size_t>(n_chunks) * LWS;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(LWS)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(LWS)]] {
                auto sg = it.get_sub_group();
                const int chunk = static_cast<int>(it.get_group(0));
                const int lane = static_cast<int>(it.get_local_id(0));

                float target_arr[8];
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    target_arr[j] = targets[chunk * 8 + j];
                }

                // Three independent best-trackers per lane.
                float best_abs_s2n = -1.0f;
                int   best_abs_idx = lane * CANDIDATES_PER_LANE;
                float best_abs_dot = 0.0f;
                float best_abs_norm = 1.0f;

                float best_pos_dot = -INFINITY;
                int   best_pos_idx = lane * CANDIDATES_PER_LANE;
                float best_pos_norm = 1.0f;

                float best_neg_dot = INFINITY;
                int   best_neg_idx = lane * CANDIDATES_PER_LANE;
                float best_neg_norm = 1.0f;

                #pragma unroll
                for (int c = 0; c < CANDIDATES_PER_LANE; ++c) {
                    const int idx = lane + c * LWS;
                    float dot = 0.0f;
                    float norm = 0.0f;
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const float g = grid_f32[idx * 8 + j] + delta;
                        dot += target_arr[j] * g;
                        norm += g * g;
                    }
                    if (norm > 0.0f) {
                        const float s2n = dot * dot / norm;
                        if (s2n > best_abs_s2n) {
                            best_abs_s2n = s2n;
                            best_abs_idx = idx;
                            best_abs_dot = dot;
                            best_abs_norm = norm;
                        }
                        if (dot > best_pos_dot) {
                            best_pos_dot = dot;
                            best_pos_idx = idx;
                            best_pos_norm = norm;
                        }
                        if (dot < best_neg_dot) {
                            best_neg_dot = dot;
                            best_neg_idx = idx;
                            best_neg_norm = norm;
                        }
                    }
                }

                // Three sub-group reductions.
                // abs (by s2n max):
                const float gw_abs_max = sycl::reduce_over_group(
                    sg, best_abs_s2n, sycl::maximum<float>());
                int my_abs_lane = (best_abs_s2n == gw_abs_max) ? lane : LWS;
                const int abs_lane = sycl::reduce_over_group(
                    sg, my_abs_lane, sycl::minimum<int>());
                const int abs_idx = sycl::select_from_group(sg, best_abs_idx, abs_lane);
                const float abs_dot = sycl::select_from_group(sg, best_abs_dot, abs_lane);
                const float abs_norm = sycl::select_from_group(sg, best_abs_norm, abs_lane);

                // pos (by signed dot max):
                const float gw_pos_max = sycl::reduce_over_group(
                    sg, best_pos_dot, sycl::maximum<float>());
                int my_pos_lane = (best_pos_dot == gw_pos_max) ? lane : LWS;
                const int pos_lane = sycl::reduce_over_group(
                    sg, my_pos_lane, sycl::minimum<int>());
                const int pos_idx = sycl::select_from_group(sg, best_pos_idx, pos_lane);
                const float pos_dot = sycl::select_from_group(sg, best_pos_dot, pos_lane);
                const float pos_norm = sycl::select_from_group(sg, best_pos_norm, pos_lane);

                // neg (by signed dot min):
                const float gw_neg_min = sycl::reduce_over_group(
                    sg, best_neg_dot, sycl::minimum<float>());
                int my_neg_lane = (best_neg_dot == gw_neg_min) ? lane : LWS;
                const int neg_lane = sycl::reduce_over_group(
                    sg, my_neg_lane, sycl::minimum<int>());
                const int neg_idx = sycl::select_from_group(sg, best_neg_idx, neg_lane);
                const float neg_dot = sycl::select_from_group(sg, best_neg_dot, neg_lane);
                const float neg_norm = sycl::select_from_group(sg, best_neg_norm, neg_lane);

                if (lane == 0) {
                    out_grid_idx_abs[chunk] = static_cast<uint16_t>(abs_idx);
                    out_signed_score_abs[chunk] = abs_dot;
                    out_norm_sq_abs[chunk] = abs_norm;
                    out_grid_idx_pos[chunk] = static_cast<uint16_t>(pos_idx);
                    out_signed_score_pos[chunk] = pos_dot;
                    out_norm_sq_pos[chunk] = pos_norm;
                    out_grid_idx_neg[chunk] = static_cast<uint16_t>(neg_idx);
                    out_signed_score_neg[chunk] = neg_dot;
                    out_norm_sq_neg[chunk] = neg_norm;
                }
            });
    }).wait();
}

// Imatrix-weighted sibling of iq_search_8elt_delta_all3_impl. Adds a
// per-chunk `weights` array ([n_chunks × 8] f32 in USM); the dot and
// norm accumulations are weighted per-element so the grid search
// minimizes the importance-weighted reconstruction error. The signed
// score returned is the weighted dot (Σ w·t·g); the norm is the
// weighted grid norm (Σ w·g²). This matches the CPU/AVX2 weighted
// reference `best_iq1s_grid_all3_scalar_w` term-for-term.
template <int GRID_SIZE, int LWS>
inline void iq_search_8elt_delta_all3_w_impl(
    sycl::queue& q,
    const float* targets,
    const float* weights,
    float delta,
    const float* grid_f32,
    uint16_t* out_grid_idx_abs,
    float* out_signed_score_abs,
    float* out_norm_sq_abs,
    uint16_t* out_grid_idx_pos,
    float* out_signed_score_pos,
    float* out_norm_sq_pos,
    uint16_t* out_grid_idx_neg,
    float* out_signed_score_neg,
    float* out_norm_sq_neg,
    int n_chunks) {
    static_assert(GRID_SIZE % LWS == 0,
        "GRID_SIZE must be divisible by LWS");
    constexpr int CANDIDATES_PER_LANE = GRID_SIZE / LWS;
    const std::size_t global =
        static_cast<std::size_t>(n_chunks) * LWS;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(LWS)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(LWS)]] {
                auto sg = it.get_sub_group();
                const int chunk = static_cast<int>(it.get_group(0));
                const int lane = static_cast<int>(it.get_local_id(0));

                float target_arr[8];
                float weight_arr[8];
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    target_arr[j] = targets[chunk * 8 + j];
                    weight_arr[j] = weights[chunk * 8 + j];
                }

                float best_abs_s2n = -1.0f;
                int   best_abs_idx = lane * CANDIDATES_PER_LANE;
                float best_abs_dot = 0.0f;
                float best_abs_norm = 1.0f;

                float best_pos_dot = -INFINITY;
                int   best_pos_idx = lane * CANDIDATES_PER_LANE;
                float best_pos_norm = 1.0f;

                float best_neg_dot = INFINITY;
                int   best_neg_idx = lane * CANDIDATES_PER_LANE;
                float best_neg_norm = 1.0f;

                #pragma unroll
                for (int c = 0; c < CANDIDATES_PER_LANE; ++c) {
                    const int idx = lane + c * LWS;
                    float dot = 0.0f;
                    float norm = 0.0f;
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const float g = grid_f32[idx * 8 + j] + delta;
                        dot += weight_arr[j] * target_arr[j] * g;
                        norm += weight_arr[j] * g * g;
                    }
                    if (norm > 0.0f) {
                        const float s2n = dot * dot / norm;
                        if (s2n > best_abs_s2n) {
                            best_abs_s2n = s2n;
                            best_abs_idx = idx;
                            best_abs_dot = dot;
                            best_abs_norm = norm;
                        }
                        if (dot > best_pos_dot) {
                            best_pos_dot = dot;
                            best_pos_idx = idx;
                            best_pos_norm = norm;
                        }
                        if (dot < best_neg_dot) {
                            best_neg_dot = dot;
                            best_neg_idx = idx;
                            best_neg_norm = norm;
                        }
                    }
                }

                const float gw_abs_max = sycl::reduce_over_group(
                    sg, best_abs_s2n, sycl::maximum<float>());
                int my_abs_lane = (best_abs_s2n == gw_abs_max) ? lane : LWS;
                const int abs_lane = sycl::reduce_over_group(
                    sg, my_abs_lane, sycl::minimum<int>());
                const int abs_idx = sycl::select_from_group(sg, best_abs_idx, abs_lane);
                const float abs_dot = sycl::select_from_group(sg, best_abs_dot, abs_lane);
                const float abs_norm = sycl::select_from_group(sg, best_abs_norm, abs_lane);

                const float gw_pos_max = sycl::reduce_over_group(
                    sg, best_pos_dot, sycl::maximum<float>());
                int my_pos_lane = (best_pos_dot == gw_pos_max) ? lane : LWS;
                const int pos_lane = sycl::reduce_over_group(
                    sg, my_pos_lane, sycl::minimum<int>());
                const int pos_idx = sycl::select_from_group(sg, best_pos_idx, pos_lane);
                const float pos_dot = sycl::select_from_group(sg, best_pos_dot, pos_lane);
                const float pos_norm = sycl::select_from_group(sg, best_pos_norm, pos_lane);

                const float gw_neg_min = sycl::reduce_over_group(
                    sg, best_neg_dot, sycl::minimum<float>());
                int my_neg_lane = (best_neg_dot == gw_neg_min) ? lane : LWS;
                const int neg_lane = sycl::reduce_over_group(
                    sg, my_neg_lane, sycl::minimum<int>());
                const int neg_idx = sycl::select_from_group(sg, best_neg_idx, neg_lane);
                const float neg_dot = sycl::select_from_group(sg, best_neg_dot, neg_lane);
                const float neg_norm = sycl::select_from_group(sg, best_neg_norm, neg_lane);

                if (lane == 0) {
                    out_grid_idx_abs[chunk] = static_cast<uint16_t>(abs_idx);
                    out_signed_score_abs[chunk] = abs_dot;
                    out_norm_sq_abs[chunk] = abs_norm;
                    out_grid_idx_pos[chunk] = static_cast<uint16_t>(pos_idx);
                    out_signed_score_pos[chunk] = pos_dot;
                    out_norm_sq_pos[chunk] = pos_norm;
                    out_grid_idx_neg[chunk] = static_cast<uint16_t>(neg_idx);
                    out_signed_score_neg[chunk] = neg_dot;
                    out_norm_sq_neg[chunk] = neg_norm;
                }
            });
    }).wait();
}

template <int GRID_SIZE, int LWS>
inline void iq_search_8elt_delta_impl(
    sycl::queue& q,
    const float* targets,
    float delta,
    const float* grid_f32,
    uint16_t* out_grid_idx,
    float* out_signed_score,
    float* out_norm_sq,
    int n_chunks) {
    // Match the SG-sized work-group convention: one chunk per WG,
    // LWS work-items per WG. Each lane evaluates GRID_SIZE/LWS
    // grid candidates serially.
    static_assert(GRID_SIZE % LWS == 0,
        "GRID_SIZE must be divisible by LWS for clean per-lane work assignment");
    constexpr int CANDIDATES_PER_LANE = GRID_SIZE / LWS;

    const std::size_t global =
        static_cast<std::size_t>(n_chunks) * LWS;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(LWS)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(LWS)]] {
                auto sg = it.get_sub_group();
                const int chunk = static_cast<int>(it.get_group(0));
                const int lane = static_cast<int>(it.get_local_id(0));

                // Load target into per-lane private regs — each
                // lane holds the same 8-element target. Costs 8
                // loads per lane but eliminates SLM coordination.
                float target_arr[8];
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    target_arr[j] = targets[chunk * 8 + j];
                }

                // Per-lane best across CANDIDATES_PER_LANE entries.
                // Tracking score_sq_div_norm = dot² / norm so the
                // subgroup reduction is a single max-of-floats.
                float best_score_sq_div_norm = -1.0f;
                int best_idx = lane * CANDIDATES_PER_LANE;
                float best_dot = 0.0f;
                float best_norm = 1.0f;

                #pragma unroll
                for (int c = 0; c < CANDIDATES_PER_LANE; ++c) {
                    const int idx = lane + c * LWS;
                    float dot = 0.0f;
                    float norm = 0.0f;
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const float g = grid_f32[idx * 8 + j] + delta;
                        dot += target_arr[j] * g;
                        norm += g * g;
                    }
                    if (norm > 0.0f) {
                        const float s2n = dot * dot / norm;
                        if (s2n > best_score_sq_div_norm) {
                            best_score_sq_div_norm = s2n;
                            best_idx = idx;
                            best_dot = dot;
                            best_norm = norm;
                        }
                    }
                }

                // Subgroup reduction: find the max score across
                // all LWS lanes and broadcast the winning lane's
                // (idx, dot, norm) tuple back via select_from_group.
                const float gw_max = sycl::reduce_over_group(
                    sg, best_score_sq_div_norm, sycl::maximum<float>());

                // Determine winner lane (lowest-lane tiebreak).
                int my_lane = (best_score_sq_div_norm == gw_max) ? lane : LWS;
                const int winner_lane = sycl::reduce_over_group(
                    sg, my_lane, sycl::minimum<int>());

                const int winning_idx = sycl::select_from_group(
                    sg, best_idx, winner_lane);
                const float winning_dot = sycl::select_from_group(
                    sg, best_dot, winner_lane);
                const float winning_norm = sycl::select_from_group(
                    sg, best_norm, winner_lane);

                if (lane == 0) {
                    out_grid_idx[chunk] =
                        static_cast<uint16_t>(winning_idx);
                    out_signed_score[chunk] = winning_dot;
                    out_norm_sq[chunk] = winning_norm;
                }
            });
    }).wait();
}

extern "C" {

// IQ1_S / IQ1_M batched grid search. `targets` is `n_chunks × 8`
// f32 in USM; `grid_f32` is the precomputed 2048 × 8 f32 grid;
// outputs are `n_chunks`-sized per-chunk picks. Caller (Rust side)
// must keep all USM pointers alive for the duration of the call;
// the kernel `.wait()`s before returning.
void rsl_iq_search_8elt_delta_iq1s(rsl_stream* s,
                                   const float* targets,
                                   float delta,
                                   const float* grid_f32,
                                   uint16_t* out_grid_idx,
                                   float* out_signed_score,
                                   float* out_norm_sq,
                                   int n_chunks) RSL_FFI_BODY_VOID(
    "rsl_iq_search_8elt_delta_iq1s", {
    if (s == nullptr || targets == nullptr || grid_f32 == nullptr
        || out_grid_idx == nullptr || out_signed_score == nullptr
        || out_norm_sq == nullptr) {
        return;
    }
    if (n_chunks <= 0) return;
    // IQ1_S grid is 2048 entries; LWS=16 matches Intel iGPU's
    // default sub-group size, so 2048/16 = 128 candidates per lane.
    iq_search_8elt_delta_impl<2048, 16>(
        s->q, targets, delta, grid_f32,
        out_grid_idx, out_signed_score, out_norm_sq, n_chunks);
})

// #3: Companion that returns all THREE picks (max-|score|, max-positive,
// max-negative) per chunk in one GPU dispatch.
void rsl_iq_search_8elt_delta_iq1s_all3(rsl_stream* s,
                                        const float* targets,
                                        float delta,
                                        const float* grid_f32,
                                        uint16_t* out_grid_idx_abs,
                                        float* out_signed_score_abs,
                                        float* out_norm_sq_abs,
                                        uint16_t* out_grid_idx_pos,
                                        float* out_signed_score_pos,
                                        float* out_norm_sq_pos,
                                        uint16_t* out_grid_idx_neg,
                                        float* out_signed_score_neg,
                                        float* out_norm_sq_neg,
                                        int n_chunks) RSL_FFI_BODY_VOID(
    "rsl_iq_search_8elt_delta_iq1s_all3", {
    if (s == nullptr || targets == nullptr || grid_f32 == nullptr) return;
    if (out_grid_idx_abs == nullptr || out_signed_score_abs == nullptr
        || out_norm_sq_abs == nullptr || out_grid_idx_pos == nullptr
        || out_signed_score_pos == nullptr || out_norm_sq_pos == nullptr
        || out_grid_idx_neg == nullptr || out_signed_score_neg == nullptr
        || out_norm_sq_neg == nullptr) {
        return;
    }
    if (n_chunks <= 0) return;
    iq_search_8elt_delta_all3_impl<2048, 16>(
        s->q, targets, delta, grid_f32,
        out_grid_idx_abs, out_signed_score_abs, out_norm_sq_abs,
        out_grid_idx_pos, out_signed_score_pos, out_norm_sq_pos,
        out_grid_idx_neg, out_signed_score_neg, out_norm_sq_neg,
        n_chunks);
})

// Imatrix-weighted companion to rsl_iq_search_8elt_delta_iq1s_all3.
// `weights` is [n_chunks × 8] f32 in USM (per-element importance for
// each chunk's 8 targets). Routes to the weighted grid-search kernel.
void rsl_iq_search_8elt_delta_iq1s_all3_w(rsl_stream* s,
                                          const float* targets,
                                          const float* weights,
                                          float delta,
                                          const float* grid_f32,
                                          uint16_t* out_grid_idx_abs,
                                          float* out_signed_score_abs,
                                          float* out_norm_sq_abs,
                                          uint16_t* out_grid_idx_pos,
                                          float* out_signed_score_pos,
                                          float* out_norm_sq_pos,
                                          uint16_t* out_grid_idx_neg,
                                          float* out_signed_score_neg,
                                          float* out_norm_sq_neg,
                                          int n_chunks) RSL_FFI_BODY_VOID(
    "rsl_iq_search_8elt_delta_iq1s_all3_w", {
    if (s == nullptr || targets == nullptr || weights == nullptr
        || grid_f32 == nullptr) return;
    if (out_grid_idx_abs == nullptr || out_signed_score_abs == nullptr
        || out_norm_sq_abs == nullptr || out_grid_idx_pos == nullptr
        || out_signed_score_pos == nullptr || out_norm_sq_pos == nullptr
        || out_grid_idx_neg == nullptr || out_signed_score_neg == nullptr
        || out_norm_sq_neg == nullptr) {
        return;
    }
    if (n_chunks <= 0) return;
    iq_search_8elt_delta_all3_w_impl<2048, 16>(
        s->q, targets, weights, delta, grid_f32,
        out_grid_idx_abs, out_signed_score_abs, out_norm_sq_abs,
        out_grid_idx_pos, out_signed_score_pos, out_norm_sq_pos,
        out_grid_idx_neg, out_signed_score_neg, out_norm_sq_neg,
        n_chunks);
})

// F1 (sampler GPU offload, first piece): USM-resident argmax over
// a logits buffer. Used by the greedy decode short-circuit
// (`temperature == 0`) so the chosen token id can be computed
// without a logits buffer round-trip to host memory.
//
// Tie-break: lowest-index lane wins (matches CPU `argmax_scalar` and
// the AVX2/AVX-512 paths, all of which use strict `>` comparison).
//
// Topology: one work-group of LWS lanes; each lane sweeps a stride
// of LWS across the vocab tracking its local (max, idx); subgroup
// reduction picks the global max + the lowest-index lane that
// achieved it. Sized for vocabs in the 32K-256K range typical of
// LLMs; the loop trip count is `vocab / LWS` per lane (~2K trips
// for 32K vocab + LWS=16). Bigger vocabs scale linearly, still in
// the millisecond-per-call budget.

}  // extern "C" — temporary close for the argmax template

template <int LWS>
inline void sampler_argmax_impl(
    sycl::queue& q,
    const float* logits,
    int vocab,
    int* out_idx) {
    // Single-task linear scan. The earlier parallel_for(LWS, LWS) +
    // subgroup-reduce variant produced silent dispatch failures on
    // Iris Xe (oneAPI 2026.0 + driver 32.0.101.7076) — out_idx stayed
    // at the host-init value of 0 despite the kernel reporting no
    // exception. Switching to single_task with a serial scan keeps
    // the kernel small and deterministic; for typical vocab sizes
    // (≤ 200K) the scan is bandwidth-bound and runs in single-digit
    // microseconds — well within the per-sample budget. The point of
    // the GPU argmax is to avoid a 200K-f32 host round-trip, not to
    // outpace x86 AVX-512 argmax.
    (void)0; // suppress unused LWS template parameter warning
    q.submit([&](sycl::handler& h) {
        h.single_task([=]() {
            float best_v = -3.4028235e38f;
            int best_i = 0;
            for (int i = 0; i < vocab; ++i) {
                const float v = logits[i];
                if (v > best_v) {
                    best_v = v;
                    best_i = i;
                }
            }
            *out_idx = best_i;
        });
    }).wait();
}

extern "C" {

void rsl_sampler_argmax_usm(rsl_stream* s,
                            const float* logits_usm,
                            int vocab,
                            int* out_idx_usm) RSL_FFI_BODY_VOID(
    "rsl_sampler_argmax_usm", {
    if (s == nullptr || logits_usm == nullptr || out_idx_usm == nullptr) {
        return;
    }
    if (vocab <= 0) {
        // Empty vocab — write -1 to signal no valid pick (the Rust
        // wrapper rejects 0-vocab before calling; this is just a
        // defensive write).
        *out_idx_usm = -1;
        return;
    }
    sampler_argmax_impl<16>(s->q, logits_usm, vocab, out_idx_usm);
})

// F1 (sampler GPU offload, second piece): fused temperature scale +
// numerically-stable softmax, in-place over a USM logits buffer.
// Mirrors `fused_temp_softmax_inplace_scalar` in kernels-cpu:
//   pass 1: x[i] *= inv_temp; max = reduce_max(x)
//   pass 2: x[i] = exp(x[i] - max); sum = reduce_sum(x)
//   pass 3: x[i] /= sum
// One work-group of LWS lanes; each lane sweeps a stride of LWS
// across the vocab; subgroup reductions for max and sum. Vocab is
// expected to fit a single SG-sized work-group sweep (128K vocab /
// LWS=16 = 8K trips per lane, ~µs on Intel iGPU).

}  // extern "C" — temporary close for the softmax template

template <int LWS>
inline void sampler_temp_softmax_impl(
    sycl::queue& q,
    float* logits,
    int vocab,
    float inv_temp) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(LWS), sycl::range<1>(LWS)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(LWS)]] {
                auto sg = it.get_sub_group();
                const int lane = static_cast<int>(it.get_local_id(0));

                // Pass 1: scale by inv_temp + find local max.
                // Write scaled values back so passes 2 + 3 don't
                // need to repeat the multiply.
                float local_max = -3.4028235e38f;
                for (int i = lane; i < vocab; i += LWS) {
                    const float v = logits[i] * inv_temp;
                    logits[i] = v;
                    if (v > local_max) local_max = v;
                }
                const float gmax = sycl::reduce_over_group(
                    sg, local_max, sycl::maximum<float>());

                // Pass 2: x[i] = exp(x[i] - max); track local sum.
                // `sycl::exp` (IEEE-compliant) over `sycl::native::exp`
                // (~3-4 ULP error) — softmax amplifies exp's error onto
                // the small-probability tail. Hardware validation on
                // Iris Xe showed `native::exp` produced max_err ~1.7e-3
                // vs CPU `f32::exp`, far above the 5e-5 tolerance the
                // engine parity test requires.
                float local_sum = 0.0f;
                for (int i = lane; i < vocab; i += LWS) {
                    const float e = sycl::exp(logits[i] - gmax);
                    logits[i] = e;
                    local_sum += e;
                }
                const float gsum = sycl::reduce_over_group(
                    sg, local_sum, sycl::plus<float>());
                // Guard against the degenerate gsum == 0 case
                // (every logit was -inf, which can't actually happen
                // post-exp — but float underflow on extreme inputs
                // could in theory produce zero). Caller's CPU
                // reference would divide by 1.0/sum=inf and emit NaN;
                // we match that behavior for parity.
                const float inv = (gsum > 0.0f) ? (1.0f / gsum) : 0.0f;

                // Pass 3: divide each prob by sum.
                for (int i = lane; i < vocab; i += LWS) {
                    logits[i] *= inv;
                }
            });
    }).wait();
}

extern "C" {

void rsl_sampler_temp_softmax_usm(rsl_stream* s,
                                  float* logits_usm,
                                  int vocab,
                                  float inv_temp) RSL_FFI_BODY_VOID(
    "rsl_sampler_temp_softmax_usm", {
    if (s == nullptr || logits_usm == nullptr) return;
    if (vocab <= 0) return;
    sampler_temp_softmax_impl<16>(s->q, logits_usm, vocab, inv_temp);
})

// F1 (sampler GPU offload, third piece): multinomial draw from a
// normalized probability distribution in USM. Mirrors the CPU
// `multinomial(probs, rng)` reference in `rustllama-engine`:
//   u = next_u64(seed) >> 40 / 2^24  (uniform [0, 1))
//   walk left-to-right: return first i where u < cum_sum(probs[0..=i])
//   fallback: last index if u didn't cross any boundary (FP edge).
//
// Determinism: takes the caller's `rng_state` (SplitMix64 state) as
// input and writes the **updated** state to `rng_state_out`. Both
// pointers are USM so the engine can keep RNG state device-resident
// across many sample calls without a CPU sync. The mixing constants
// and step order match `rustllama-engine::sampling::Rng::next_u64`
// bit-for-bit so seeded outputs are reproducible.
//
// Topology: single work-item — vocab is at most ~200K f32 (~800KB),
// linear scan is bandwidth-bound, fits the iGPU L2/L3. Parallelism
// here would require an exact-order parallel cumsum, which CPU FP
// ordering doesn't grant; single-lane is the only bit-parity option.

}  // extern "C" — temporary close for the multinomial template

inline void sampler_multinomial_impl(
    sycl::queue& q,
    const float* probs,
    int vocab,
    uint64_t* rng_state,
    int* out_idx) {
    q.submit([&](sycl::handler& h) {
        h.single_task([=]() {
            // SplitMix64 next_u64: state += Weyl; mix; output.
            uint64_t state = *rng_state;
            state += 0x9E3779B97F4A7C15ULL;
            uint64_t z = state;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
            z = z ^ (z >> 31);
            *rng_state = state;  // persist updated state
            // u = (top 24 bits) / 2^24, ∈ [0, 1).
            const float u =
                static_cast<float>(static_cast<uint32_t>(z >> 40))
                / static_cast<float>(1u << 24);

            // Linear cumsum walk — same FP order as CPU multinomial.
            float cum = 0.0f;
            int pick = vocab - 1;
            for (int i = 0; i < vocab; ++i) {
                cum += probs[i];
                if (u < cum) {
                    pick = i;
                    break;
                }
            }
            *out_idx = pick;
        });
    }).wait();
}

extern "C" {

void rsl_sampler_multinomial_usm(rsl_stream* s,
                                 const float* probs_usm,
                                 int vocab,
                                 uint64_t* rng_state_usm,
                                 int* out_idx_usm) RSL_FFI_BODY_VOID(
    "rsl_sampler_multinomial_usm", {
    if (s == nullptr || probs_usm == nullptr
        || rng_state_usm == nullptr || out_idx_usm == nullptr) {
        return;
    }
    if (vocab <= 0) {
        *out_idx_usm = -1;
        return;
    }
    sampler_multinomial_impl(s->q, probs_usm, vocab, rng_state_usm, out_idx_usm);
})

// F1 (sampler GPU offload, fourth piece): penalty pass. Applies
// repetition + frequency + presence penalties to logits before
// softmax. Mirrors the CPU `apply_all_penalties` reference:
//   - repetition (multiplicative): for each tok in `recent`, divide
//     `logits[tok]` by `repeat` if logit ≥ 0, else multiply.
//   - frequency + presence (additive): for each unique tok in
//     `recent`, subtract `frequency * count(tok) + presence`.
// O(n²) in `recent.len()` because we sort-free dedup by scanning
// for first-occurrence per token. With typical `recent.len()` ≤ 256
// that's <= 64K compares — trivial.
//
// Single-task kernel. The point is keeping `logits` in USM
// throughout the sampler pipeline (penalty → softmax → multinomial),
// not raw compute speed. CPU dispatch overhead for the equivalent
// op is comparable.

}  // extern "C" — temporary close for the penalty template

inline void sampler_penalty_impl(
    sycl::queue& q,
    float* logits,
    int vocab,
    const uint32_t* recent,
    int recent_n,
    float repeat,
    float frequency,
    float presence) {
    q.submit([&](sycl::handler& h) {
        h.single_task([=]() {
            // Pass 1: repetition penalty (multiplicative). Applied
            // PER OCCURRENCE — if a token appears N times in
            // `recent`, the divide/multiply runs N times, matching
            // the CPU loop.
            if (repeat != 1.0f) {
                for (int i = 0; i < recent_n; ++i) {
                    const uint32_t tok = recent[i];
                    if (tok < static_cast<uint32_t>(vocab)) {
                        const float v = logits[tok];
                        logits[tok] = (v >= 0.0f) ? (v / repeat) : (v * repeat);
                    }
                }
            }
            // Pass 2: frequency + presence. Per UNIQUE token —
            // first-occurrence scan + count.
            if (frequency != 0.0f || presence != 0.0f) {
                for (int i = 0; i < recent_n; ++i) {
                    const uint32_t tok = recent[i];
                    if (tok >= static_cast<uint32_t>(vocab)) continue;
                    // First-occurrence check (skip if already counted).
                    bool first = true;
                    for (int j = 0; j < i; ++j) {
                        if (recent[j] == tok) { first = false; break; }
                    }
                    if (!first) continue;
                    int count = 0;
                    for (int j = i; j < recent_n; ++j) {
                        if (recent[j] == tok) count++;
                    }
                    const float penalty =
                        frequency * static_cast<float>(count) + presence;
                    logits[tok] -= penalty;
                }
            }
        });
    }).wait();
}

extern "C" {

void rsl_sampler_penalty_usm(rsl_stream* s,
                             float* logits_usm,
                             int vocab,
                             const uint32_t* recent_usm,
                             int recent_n,
                             float repeat,
                             float frequency,
                             float presence) RSL_FFI_BODY_VOID(
    "rsl_sampler_penalty_usm", {
    if (s == nullptr || logits_usm == nullptr) return;
    if (vocab <= 0) return;
    // recent_n == 0 or null recent_usm means "no penalties to apply";
    // the conditions inside the kernel handle the no-op cleanly, but
    // skip the dispatch if recent_n is zero.
    if (recent_n <= 0 || recent_usm == nullptr) return;
    sampler_penalty_impl(s->q, logits_usm, vocab,
                         recent_usm, recent_n,
                         repeat, frequency, presence);
})

// F1 (sampler GPU offload, fifth piece): top-k mask + renormalize.
// Mirrors the CPU `apply_top_k` reference:
//   1. Find threshold = (k-1)-th-largest probability.
//   2. For each prob: if `prob < threshold`, zero it; else add to sum.
//   3. Divide every prob by sum (so survivors sum to 1.0).
//
// The threshold-selection step is the only delicate part. CPU uses
// `select_nth_unstable_by` (O(n) average); GPU uses an O(n·k)
// insertion-into-sorted-top-k-array scan because that's the simplest
// single-pass algorithm with no dynamic memory. For typical k ≤ 100,
// the inner loop runs ~vocab·100 = 12.8M ops on a 128K vocab — a few
// ms at iGPU clock, fine for the per-token budget.
//
// GPU-side k is capped at MAX_TOP_K_GPU = 256. The host wrapper
// falls back to CPU for larger k (rare in practice; documented).

constexpr int MAX_TOP_K_GPU = 256;

}  // extern "C" — temporary close for the top_k template

inline void sampler_top_k_impl(
    sycl::queue& q,
    float* probs,
    int vocab,
    int k) {
    q.submit([&](sycl::handler& h) {
        h.single_task([=]() {
            if (k <= 0 || k >= vocab) return;  // no-op (caller short-circuits)
            const int kk = (k > MAX_TOP_K_GPU) ? MAX_TOP_K_GPU : k;

            // Top-kk array (descending). Seeded to -inf so any real
            // prob displaces the tail.
            float top_k[MAX_TOP_K_GPU];
            #pragma unroll
            for (int i = 0; i < MAX_TOP_K_GPU; ++i) {
                top_k[i] = -3.4028235e38f;
            }

            // Pass 1: find the kk-th largest probability by streaming
            // insertion-sort. For each prob, if larger than the
            // current smallest of the top-kk, shift it into the
            // sorted descending array.
            for (int i = 0; i < vocab; ++i) {
                const float v = probs[i];
                if (v > top_k[kk - 1]) {
                    int j = kk - 1;
                    while (j > 0 && top_k[j - 1] < v) {
                        top_k[j] = top_k[j - 1];
                        j--;
                    }
                    top_k[j] = v;
                }
            }
            const float threshold = top_k[kk - 1];

            // Pass 2: mask + sum. Cut at `< threshold` (matches CPU
            // tie-semantics: ties at the threshold are kept).
            float sum = 0.0f;
            for (int i = 0; i < vocab; ++i) {
                if (probs[i] < threshold) {
                    probs[i] = 0.0f;
                } else {
                    sum += probs[i];
                }
            }

            // Pass 3: renormalize.
            const float inv = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
            for (int i = 0; i < vocab; ++i) {
                probs[i] *= inv;
            }
        });
    }).wait();
}

extern "C" {

void rsl_sampler_top_k_usm(rsl_stream* s,
                           float* probs_usm,
                           int vocab,
                           int k) RSL_FFI_BODY_VOID(
    "rsl_sampler_top_k_usm", {
    if (s == nullptr || probs_usm == nullptr) return;
    if (vocab <= 0) return;
    if (k <= 0 || k >= vocab) return;  // no-op gates match CPU
    sampler_top_k_impl(s->q, probs_usm, vocab, k);
})

// F1 (sampler GPU offload, sixth piece): top-p (nucleus) mask +
// renormalize. Mirrors the CPU `apply_top_p` semantics:
//   1. Find the smallest top-N prefix (by descending probability)
//      whose cumulative sum ≥ p. Threshold = the smallest prob in
//      that prefix.
//   2. Zero probabilities strictly below the threshold.
//   3. Renormalize survivors to sum to 1.0.
//
// Algorithm:
//   Phase 1: streaming min-heap (size MAX_TOP_P_GPU=1024) collects
//            the top-MAX-most-probable values in O(vocab · log MAX).
//   Phase 2: heap-sort the result descending (O(MAX log MAX)).
//   Phase 3: cumsum walk to find the cutoff rank.
//   Phase 4 (skipped on fallback): mask + renormalize.
//
// **Bounded coverage**: caps at the top-MAX entries. If the
// cumulative sum at rank MAX still hasn't crossed p, the kernel
// writes `*needs_fallback_usm = 1` and leaves `probs_usm` UNCHANGED
// — the host wrapper then runs CPU `apply_top_p` on the original
// probs. On success it writes 0 and the mask + renormalize have
// completed.
//
// Phase ordering is critical: phases 1–3 only READ probs; the only
// mutation is in phase 4, which is gated on the cum ≥ p check. So
// the "leave probs untouched on fallback" invariant is structurally
// guaranteed.

constexpr int MAX_TOP_P_GPU = 1024;

}  // extern "C" — temporary close for the top_p template

inline void sampler_top_p_impl(
    sycl::queue& q,
    float* probs,
    int vocab,
    float p,
    int* needs_fallback) {
    q.submit([&](sycl::handler& h) {
        h.single_task([=]() {
            constexpr int MAX_P = MAX_TOP_P_GPU;

            // Phase 1: streaming min-heap of size min(MAX_P, vocab).
            // Seeded with -inf so the first MAX_P inserts fill.
            float heap[MAX_P];
            const int max_p = (vocab < MAX_P) ? vocab : MAX_P;
            for (int i = 0; i < max_p; ++i) heap[i] = -3.4028235e38f;

            for (int i = 0; i < vocab; ++i) {
                const float v = probs[i];
                if (v > heap[0]) {
                    heap[0] = v;
                    // Sift-down to restore min-heap.
                    int j = 0;
                    while (true) {
                        const int l = 2 * j + 1;
                        const int r = 2 * j + 2;
                        int smallest = j;
                        if (l < max_p && heap[l] < heap[smallest]) smallest = l;
                        if (r < max_p && heap[r] < heap[smallest]) smallest = r;
                        if (smallest == j) break;
                        const float t = heap[j];
                        heap[j] = heap[smallest];
                        heap[smallest] = t;
                        j = smallest;
                    }
                }
            }

            // Phase 2: heapsort to ascending, then reverse to descending.
            for (int sz = max_p; sz > 1; --sz) {
                const float t = heap[0];
                heap[0] = heap[sz - 1];
                heap[sz - 1] = t;
                // Sift down on heap[0..sz-1].
                int j = 0;
                const int end = sz - 1;
                while (true) {
                    const int l = 2 * j + 1;
                    const int r = 2 * j + 2;
                    int smallest = j;
                    if (l < end && heap[l] < heap[smallest]) smallest = l;
                    if (r < end && heap[r] < heap[smallest]) smallest = r;
                    if (smallest == j) break;
                    const float u = heap[j];
                    heap[j] = heap[smallest];
                    heap[smallest] = u;
                    j = smallest;
                }
            }
            // Now heap[0..max_p] is ascending. Reverse → descending.
            for (int i = 0; i < max_p / 2; ++i) {
                const float t = heap[i];
                heap[i] = heap[max_p - 1 - i];
                heap[max_p - 1 - i] = t;
            }

            // Phase 3: cumsum walk to find cutoff rank.
            float cum = 0.0f;
            int cutoff = 0;
            for (int i = 0; i < max_p; ++i) {
                cum += heap[i];
                if (cum >= p) {
                    cutoff = i + 1;
                    break;
                }
            }
            if (cutoff == 0) {
                // Top-MAX didn't reach p. Caller must fall back to
                // CPU. We have NOT touched probs yet — invariant.
                *needs_fallback = 1;
                return;
            }
            *needs_fallback = 0;
            const float threshold = heap[cutoff - 1];

            // Phase 4: mask + sum.
            float sum = 0.0f;
            for (int i = 0; i < vocab; ++i) {
                if (probs[i] < threshold) {
                    probs[i] = 0.0f;
                } else {
                    sum += probs[i];
                }
            }
            // Phase 5: renormalize.
            const float inv = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
            for (int i = 0; i < vocab; ++i) {
                probs[i] *= inv;
            }
        });
    }).wait();
}

extern "C" {

void rsl_sampler_top_p_usm(rsl_stream* s,
                           float* probs_usm,
                           int vocab,
                           float p,
                           int* needs_fallback_usm) RSL_FFI_BODY_VOID(
    "rsl_sampler_top_p_usm", {
    if (s == nullptr || probs_usm == nullptr
        || needs_fallback_usm == nullptr) {
        return;
    }
    if (vocab <= 0) {
        *needs_fallback_usm = 0;
        return;
    }
    // p ≤ 0 or p ≥ 1 are CPU no-ops; kernel leaves probs untouched.
    if (p <= 0.0f || p >= 1.0f) {
        *needs_fallback_usm = 0;
        return;
    }
    sampler_top_p_impl(s->q, probs_usm, vocab, p, needs_fallback_usm);
})

// IQ2-family batched grid search. Handles IQ2_XXS / IQ2_XS / IQ2_S
// (all 8-element grid; n_grid differs). Per-chunk search: for each
// grid candidate, compute the greedy sign mask (negate each lane
// whose `target[j] * grid[j]` is negative), parity-fix if odd
// popcount by flipping the smallest-|contrib| lane, then compare
// `score²·best_norm vs best_score²·norm` (Cauchy-Schwarz form to
// avoid a divide per candidate). Skip grids whose sign mask isn't
// one of the 128 representable patterns (`ksigns_rev[mask] == 0xFF`).
//
// Inputs in USM:
//   targets   : n_chunks × 8 f32, row-major
//   grid_f32  : n_grid × 8 f32, row-major
//   grid_norm : n_grid f32 — precomputed |grid[g]|²
//   ksigns_rev: 256 u8 — inverse of KSIGNS_IQ2XS; 0xFF for unmapped
// Outputs in USM:
//   out_grid_idx, out_sign_idx, out_signed_score, out_grid_norm_sq
// Status: kernel arithmetic mirrors `search_chunk_8_scalar` in
// `encode_iq_vec`. Hardware validation pending.

}  // extern "C" — temporary close for the IQ2 template

// **One work-item per chunk** — no subgroup ops. Three earlier
// attempts at the subgroup-parallel pattern all failed on Iris Xe
// at higher chunk indices or higher CPL counts (subgroup-op count
// scales with the kernel's complexity, and IGC's SPIR-V codegen
// appears to have a bug interacting with the IQ2 control flow).
// The serial-per-chunk pattern avoids subgroup ops entirely;
// parallelism still comes from running many chunks in parallel.
// For typical IQ2 use the dispatch is bandwidth-bound on grid +
// targets reads, so the per-chunk serial sweep is fine.
inline void iq_search_8elt_signed_impl(
    sycl::queue& q,
    const float* targets,
    const float* grid_f32,
    const float* grid_norm_sq_table,
    const uint8_t* ksigns_rev,  // 256 entries
    int n_grid,
    uint16_t* out_grid_idx,
    uint8_t* out_sign_idx,
    float* out_signed_score,
    float* out_grid_norm_sq,
    int n_chunks) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(n_chunks), [=](sycl::id<1> id) {
            const int chunk = static_cast<int>(id[0]);

            float target_arr[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                target_arr[j] = targets[chunk * 8 + j];
            }

            // **Direct-divide form** of the comparison. The earlier
            // Cauchy-Schwarz cross-multiply form
            // (`score² * best_norm > best_score² * norm`) produced
            // wrong results on Iris Xe for certain chunks: hardware
            // diagnostic showed the kernel could correctly compute
            // the winning grid's score on demand, but the running-
            // best update via cross-multiply was being mis-codegen'd
            // by IGC. The direct ratio form costs one FP divide per
            // candidate but avoids the IGC bug.
            //
            // Initial metric=0.0 — any real candidate with a non-
            // zero dot product wins. The earlier baseline of 1.0
            // (mirroring the CPU sentinel `best_score=-1,
            // best_norm=1`) silently zeroed out small-magnitude
            // inputs: an embedding-row chunk with rms ~0.01 has
            // score² ~ 1e-4 against grid_norm ~64–256, never
            // beating metric=1, so every chunk picked the sentinel
            // → sub_scale=0 → super-block d=0 → all output bytes 0.
            float best_metric = 0.0f;
            float best_score = 0.0f;
            float best_norm = 1.0f;
            int best_idx = -1;
            int best_sign = 0;

            for (int idx = 0; idx < n_grid; ++idx) {
                uint8_t mask = 0;
                float score = 0.0f;
                float min_abs = 3.4028235e38f;
                int min_j = 0;
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float cv = target_arr[j] * grid_f32[idx * 8 + j];
                    const float a = sycl::fabs(cv);
                    if (a < min_abs) {
                        min_abs = a;
                        min_j = j;
                    }
                    if (cv >= 0.0f) {
                        score += cv;
                    } else {
                        score -= cv;
                        mask |= static_cast<uint8_t>(1u << j);
                    }
                }
                if (sycl::popcount(static_cast<unsigned int>(mask)) & 1u) {
                    mask ^= static_cast<uint8_t>(1u << min_j);
                    score -= 2.0f * min_abs;
                }
                const uint8_t sign_idx = ksigns_rev[mask];
                if (sign_idx == 0xFF) continue;
                const float norm = grid_norm_sq_table[idx];
                const float metric = (norm > 0.0f) ? (score * score / norm) : 0.0f;
                if (metric > best_metric) {
                    best_metric = metric;
                    best_score = score;
                    best_norm = norm;
                    best_idx = idx;
                    best_sign = static_cast<int>(sign_idx);
                }
            }

            if (best_idx < 0) {
                out_grid_idx[chunk] = 0;
                out_sign_idx[chunk] = 0;
                out_signed_score[chunk] = 0.0f;
                out_grid_norm_sq[chunk] = 1.0f;
            } else {
                out_grid_idx[chunk] = static_cast<uint16_t>(best_idx);
                out_sign_idx[chunk] = static_cast<uint8_t>(best_sign);
                out_signed_score[chunk] = best_score;
                out_grid_norm_sq[chunk] = best_norm;
            }
        });
    }).wait();
}

extern "C" {

void rsl_iq_search_8elt_signed(rsl_stream* s,
                               const float* targets,
                               const float* grid_f32,
                               const float* grid_norm_sq_table,
                               const uint8_t* ksigns_rev,
                               int n_grid,
                               uint16_t* out_grid_idx,
                               uint8_t* out_sign_idx,
                               float* out_signed_score,
                               float* out_grid_norm_sq,
                               int n_chunks) RSL_FFI_BODY_VOID(
    "rsl_iq_search_8elt_signed", {
    if (s == nullptr || targets == nullptr || grid_f32 == nullptr
        || grid_norm_sq_table == nullptr || ksigns_rev == nullptr
        || out_grid_idx == nullptr || out_sign_idx == nullptr
        || out_signed_score == nullptr || out_grid_norm_sq == nullptr) {
        return;
    }
    if (n_chunks <= 0 || n_grid <= 0) return;
    iq_search_8elt_signed_impl(
        s->q, targets, grid_f32, grid_norm_sq_table, ksigns_rev, n_grid,
        out_grid_idx, out_sign_idx, out_signed_score, out_grid_norm_sq,
        n_chunks);
})

// IQ3-family batched grid search. Handles IQ3_XXS (n_grid=256) and
// IQ3_S (n_grid=512). Per-chunk algorithm mirrors the CPU
// `search_chunk_iq3xxs` reference:
//   1. Split the 8-element target into lo (j=0..3) and hi (j=4..7).
//   2. For each half independently, find the best 4-element grid
//      entry (Cauchy-Schwarz `score²/norm`). Each search uses a
//      greedy 4-bit sign mask local to its half (bit j ↔ lane j).
//   3. Combine: build the 8-bit global mask via
//      `KMASK_IQ2XS[j]` (lo: j=0..3) and `KMASK_IQ2XS[j+4]` (hi).
//   4. Parity-fix: if odd popcount, flip the smallest-|contrib|
//      lane across all 8 positions (score -= 2 * min_abs).
//   5. Reverse-lookup `sign_idx = ksigns_rev[mask]` (0xFF unmapped).
//
// Inputs in USM:
//   targets   : [n_chunks, 8] f32
//   grid_f32  : [n_grid, 4] f32 (4-element entries)
//   grid_norm : [n_grid] f32 — |grid|² per entry
//   kmask     : [8] u8 — KMASK_IQ2XS (caller's responsibility)
//   ksigns_rev: [256] u8 — inverse KSIGNS_IQ2XS
// Outputs in USM:
//   out_grid1_idx, out_grid2_idx, out_sign_idx,
//   out_signed_score, out_grid_norm_sq

}  // extern "C" — temporary close for the IQ3 template

// **One work-item per chunk** — same rationale as the IQ2 kernel
// above. Subgroup-parallel variants failed on Iris Xe regardless of
// CPL, control-flow shape, or sub-group size attribute. Serial
// per-chunk keeps the kernel small and avoids the SPIR-V codegen
// bug; many chunks still run in parallel via the GPU's work-item
// scheduler.
inline void iq_search_4elt_paired_signed_impl(
    sycl::queue& q,
    const float* targets,
    const float* grid_f32,
    const float* grid_norm_sq_table,
    const uint8_t* kmask,        // KMASK_IQ2XS[8]
    const uint8_t* ksigns_rev,   // 256 entries
    int n_grid,
    uint16_t* out_grid1_idx,
    uint16_t* out_grid2_idx,
    uint8_t* out_sign_idx,
    float* out_signed_score,
    float* out_grid_norm_sq,
    int n_chunks) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(n_chunks), [=](sycl::id<1> id) {
            const int chunk = static_cast<int>(id[0]);

            float target_arr[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                target_arr[j] = targets[chunk * 8 + j];
            }

            // --- Inlined lo-half search ---
            // Matches CPU `best_grid_4_scalar`: Cauchy-Schwarz
            // cross-multiply with best_score=-1, best_norm=1 initial,
            // best_g=0 default. The `score² > norm` quality floor
            // rejects poor fits (small-magnitude inputs stay at
            // grid_idx=0); IQ3 round-trip quality depends on this.
            int g1 = 0;
            uint8_t mask_lo = 0;
            float score_lo = -1.0f;
            float norm_lo = 1.0f;
            for (int idx = 0; idx < n_grid; ++idx) {
                uint8_t mask = 0;
                float score = 0.0f;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float cv = target_arr[j] * grid_f32[idx * 4 + j];
                    if (cv >= 0.0f) {
                        score += cv;
                    } else {
                        score -= cv;
                        mask |= static_cast<uint8_t>(1u << j);
                    }
                }
                const float norm = grid_norm_sq_table[idx];
                const float lhs = score * score * norm_lo;
                const float rhs = score_lo * score_lo * norm;
                if (lhs > rhs) {
                    g1 = idx;
                    mask_lo = mask;
                    score_lo = score;
                    norm_lo = norm;
                }
            }

            // --- Inlined hi-half search (target_arr + 4) ---
            // Same `score² > norm` quality-floor sentinel as lo-half.
            int g2 = 0;
            uint8_t mask_hi = 0;
            float score_hi = -1.0f;
            float norm_hi = 1.0f;
            for (int idx = 0; idx < n_grid; ++idx) {
                uint8_t mask = 0;
                float score = 0.0f;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float cv = target_arr[j + 4] * grid_f32[idx * 4 + j];
                    if (cv >= 0.0f) {
                        score += cv;
                    } else {
                        score -= cv;
                        mask |= static_cast<uint8_t>(1u << j);
                    }
                }
                const float norm = grid_norm_sq_table[idx];
                const float lhs = score * score * norm_hi;
                const float rhs = score_hi * score_hi * norm;
                if (lhs > rhs) {
                    g2 = idx;
                    mask_hi = mask;
                    score_hi = score;
                    norm_hi = norm;
                }
            }

            // No "no winner" sentinel — best_g defaults to 0 matching
            // CPU `best_grid_4_scalar` semantics. The combine step
            // below handles parity-fix even if no candidate beat the
            // initial sentinel (signed_score stays 0 for that case,
            // matching CPU behavior).
            if (false) {
                out_grid1_idx[chunk] = 0;
                out_grid2_idx[chunk] = 0;
                out_sign_idx[chunk] = 0;
                out_signed_score[chunk] = 0.0f;
                out_grid_norm_sq[chunk] = 1.0f;
                return;
            }

            // Build combined 8-bit mask.
            uint8_t mask = 0;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                if (mask_lo & (1u << j)) mask |= kmask[j];
                if (mask_hi & (1u << j)) mask |= kmask[j + 4];
            }
            float signed_score = score_lo + score_hi;
            if (sycl::popcount(static_cast<unsigned int>(mask)) & 1u) {
                float min_abs = 3.4028235e38f;
                int min_j = 0;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float c = sycl::fabs(target_arr[j] * grid_f32[g1 * 4 + j]);
                    if (c < min_abs) { min_abs = c; min_j = j; }
                }
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float c = sycl::fabs(target_arr[j + 4] * grid_f32[g2 * 4 + j]);
                    if (c < min_abs) { min_abs = c; min_j = j + 4; }
                }
                mask ^= kmask[min_j];
                signed_score -= 2.0f * min_abs;
            }
            const uint8_t sign_idx = ksigns_rev[mask];
            out_grid1_idx[chunk] = static_cast<uint16_t>(g1);
            out_grid2_idx[chunk] = static_cast<uint16_t>(g2);
            out_sign_idx[chunk] = sign_idx;
            out_signed_score[chunk] = signed_score;
            out_grid_norm_sq[chunk] = norm_lo + norm_hi;
        });
    }).wait();
}

extern "C" {

void rsl_iq_search_4elt_paired_signed(rsl_stream* s,
                                       const float* targets,
                                       const float* grid_f32,
                                       const float* grid_norm_sq_table,
                                       const uint8_t* kmask,
                                       const uint8_t* ksigns_rev,
                                       int n_grid,
                                       uint16_t* out_grid1_idx,
                                       uint16_t* out_grid2_idx,
                                       uint8_t* out_sign_idx,
                                       float* out_signed_score,
                                       float* out_grid_norm_sq,
                                       int n_chunks) RSL_FFI_BODY_VOID(
    "rsl_iq_search_4elt_paired_signed", {
    if (s == nullptr || targets == nullptr || grid_f32 == nullptr
        || grid_norm_sq_table == nullptr || kmask == nullptr
        || ksigns_rev == nullptr || out_grid1_idx == nullptr
        || out_grid2_idx == nullptr || out_sign_idx == nullptr
        || out_signed_score == nullptr || out_grid_norm_sq == nullptr) {
        return;
    }
    if (n_chunks <= 0 || n_grid <= 0) return;
    iq_search_4elt_paired_signed_impl(
        s->q, targets, grid_f32, grid_norm_sq_table, kmask, ksigns_rev, n_grid,
        out_grid1_idx, out_grid2_idx, out_sign_idx,
        out_signed_score, out_grid_norm_sq, n_chunks);
})

// USM-resident Q8_0 packed matvec — batched over N input rows. Same
// per-row math as `rsl_matvec_q8_0_packed_f32_usm` but with one
// kernel launch covering all N rows. Inputs are contiguous row-major:
//   x_usm:    [N, K] f32      (x[n,k] = x_usm[n*K + k])
//   out_usm:  [N, M] f32      (out[n,m] = out_usm[n*M + m])
// Output: out[n, m] = sum_k W[m, k] * x[n, k]. One work-item per
// (n, m) pair. The big win over launching N times is amortizing the
// fixed per-launch overhead; the per-work-item cost stays the same.
}  // extern "C" — temporary close for the Q8_0 batched template

template <std::size_t LWS_T>
inline void matvec_q8_0_packed_f32_batched_usm_impl(
    sycl::queue& q,
    const void* w_bytes_usm,
    const float* x_usm,
    float* out_usm,
    int M, int K, int N) {
    constexpr std::size_t LWS = LWS_T;
    const int blocks_per_row = K / 32;
    const int bytes_per_row = blocks_per_row * 34;
    const uint8_t* w_bytes = static_cast<const uint8_t*>(w_bytes_usm);
    const std::size_t global_m =
        ((static_cast<std::size_t>(M) + LWS - 1) / LWS) * LWS;
    sycl::range<2> global(global_m, static_cast<std::size_t>(N));
    sycl::range<2> local(LWS, 1);
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<2>(global, local),
            [=](sycl::nd_item<2> it) {
                const int m = static_cast<int>(it.get_global_id(0));
                const int n = static_cast<int>(it.get_global_id(1));
                if (m >= M || n >= N) return;
                const uint8_t* row = w_bytes + m * bytes_per_row;
                const float* x_row = x_usm + n * K;
                float acc = 0.0f;
                for (int b = 0; b < blocks_per_row; ++b) {
                    const uint8_t* blk = row + b * 34;
                    uint16_t scale_bits =
                        static_cast<uint16_t>(blk[0]) |
                        (static_cast<uint16_t>(blk[1]) << 8);
                    const float scale = bits_to_f32(scale_bits);
                    const int x_off = b * 32;
                    float block_dot = 0.0f;
                    for (int d = 0; d < 32; ++d) {
                        const int8_t w_i8 = static_cast<int8_t>(blk[2 + d]);
                        block_dot += static_cast<float>(w_i8) * x_row[x_off + d];
                    }
                    acc += scale * block_dot;
                }
                out_usm[n * M + m] = acc;
            });
    }).wait();
}

extern "C" {

void rsl_matvec_q8_0_packed_f32_batched_usm(rsl_stream* s,
                                            const void* w_bytes_usm,
                                            const float* x_usm,
                                            float* out_usm,
                                            int M, int K, int N,
                                            int lws) RSL_FFI_BODY_VOID(
    "rsl_matvec_q8_0_packed_f32_batched_usm", {
    if (s == nullptr || w_bytes_usm == nullptr
        || x_usm == nullptr || out_usm == nullptr) {
        return;
    }
    if (M <= 0 || K <= 0 || N <= 0 || (K % 32) != 0) {
        return;
    }
    auto& q = s->q;
    const int eff_lws = (lws <= 0) ? static_cast<int>(RSL_LWS) : lws;
    switch (eff_lws) {
        case 16:  matvec_q8_0_packed_f32_batched_usm_impl<16>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        case 32:  matvec_q8_0_packed_f32_batched_usm_impl<32>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        case 64:  matvec_q8_0_packed_f32_batched_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        case 128: matvec_q8_0_packed_f32_batched_usm_impl<128>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        case 256: matvec_q8_0_packed_f32_batched_usm_impl<256>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        default:  matvec_q8_0_packed_f32_batched_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
    }
})

// USM-resident Q4_K_M packed matvec — batched over N input rows.
// See `rsl_matvec_q4_k_packed_f32_usm` for the super-block layout.
}  // extern "C" — temporary close for the Q4_K batched template

template <std::size_t LWS_T>
inline void matvec_q4_k_packed_f32_batched_usm_impl(
    sycl::queue& q,
    const void* w_bytes_usm,
    const float* x_usm,
    float* out_usm,
    int M, int K, int N) {
    constexpr std::size_t LWS = LWS_T;
    const int blocks_per_row = K / 256;
    const int bytes_per_row = blocks_per_row * 144;
    const uint8_t* w_bytes = static_cast<const uint8_t*>(w_bytes_usm);
    const std::size_t global_m =
        ((static_cast<std::size_t>(M) + LWS - 1) / LWS) * LWS;
    sycl::range<2> global(global_m, static_cast<std::size_t>(N));
    sycl::range<2> local(LWS, 1);
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<2>(global, local),
            [=](sycl::nd_item<2> it) {
                const int m = static_cast<int>(it.get_global_id(0));
                const int n = static_cast<int>(it.get_global_id(1));
                if (m >= M || n >= N) return;
                const uint8_t* row = w_bytes + m * bytes_per_row;
                const float* x_row = x_usm + n * K;
                float acc = 0.0f;
                for (int b = 0; b < blocks_per_row; ++b) {
                    const uint8_t* blk = row + b * 144;
                    uint16_t d_bits = static_cast<uint16_t>(blk[0])
                                      | (static_cast<uint16_t>(blk[1]) << 8);
                    uint16_t dmin_bits = static_cast<uint16_t>(blk[2])
                                         | (static_cast<uint16_t>(blk[3]) << 8);
                    const float d = bits_to_f32(d_bits);
                    const float dmin = bits_to_f32(dmin_bits);
                    const uint8_t* sb = blk + 4;
                    uint8_t sc[8];
                    uint8_t mn[8];
                    for (int j = 0; j < 8; ++j) {
                        if (j < 4) {
                            sc[j] = sb[j] & 0x3F;
                            mn[j] = sb[j + 4] & 0x3F;
                        } else {
                            sc[j] = (sb[j + 4] & 0x0F) | ((sb[j - 4] >> 6) << 4);
                            mn[j] = (sb[j + 4] >> 4) | ((sb[j] >> 6) << 4);
                        }
                    }
                    const uint8_t* qs = blk + 16;
                    const int x_base = b * 256;
                    for (int group = 0; group < 4; ++group) {
                        const uint8_t* qc = qs + group * 32;
                        const float d_lo = d * static_cast<float>(sc[group * 2]);
                        const float m_lo = dmin * static_cast<float>(mn[group * 2]);
                        const float d_hi = d * static_cast<float>(sc[group * 2 + 1]);
                        const float m_hi = dmin * static_cast<float>(mn[group * 2 + 1]);
                        const int x_lo_off = x_base + group * 64;
                        const int x_hi_off = x_lo_off + 32;
                        for (int l = 0; l < 32; ++l) {
                            const uint8_t qb = qc[l];
                            const float q_lo = static_cast<float>(qb & 0x0F);
                            const float q_hi = static_cast<float>(qb >> 4);
                            acc += (d_lo * q_lo - m_lo) * x_row[x_lo_off + l];
                            acc += (d_hi * q_hi - m_hi) * x_row[x_hi_off + l];
                        }
                    }
                }
                out_usm[n * M + m] = acc;
            });
    }).wait();
}

extern "C" {

void rsl_matvec_q4_k_packed_f32_batched_usm(rsl_stream* s,
                                            const void* w_bytes_usm,
                                            const float* x_usm,
                                            float* out_usm,
                                            int M, int K, int N,
                                            int lws) RSL_FFI_BODY_VOID(
    "rsl_matvec_q4_k_packed_f32_batched_usm", {
    if (s == nullptr || w_bytes_usm == nullptr
        || x_usm == nullptr || out_usm == nullptr) {
        return;
    }
    if (M <= 0 || K <= 0 || N <= 0 || (K % 256) != 0) {
        return;
    }
    auto& q = s->q;
    const int eff_lws = (lws <= 0) ? static_cast<int>(RSL_LWS) : lws;
    switch (eff_lws) {
        case 16:  matvec_q4_k_packed_f32_batched_usm_impl<16>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        case 32:  matvec_q4_k_packed_f32_batched_usm_impl<32>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        case 64:  matvec_q4_k_packed_f32_batched_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        case 128: matvec_q4_k_packed_f32_batched_usm_impl<128>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        case 256: matvec_q4_k_packed_f32_batched_usm_impl<256>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        default:  matvec_q4_k_packed_f32_batched_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
    }
})

// PrismML PTQ1_0 (Bonsai ternary) packed matvec. Block layout
// (28 B / 128 weights): { qs: [u8;24], qh: [u8;2], d: f16 }. qs packs
// 5 base-3 digits per byte in ceiling fixed point, chunk-staged
// {16, 8} bytes with digit-major element order; qh packs 4 digits
// per byte (first digit pre-shifted to the top). Extraction is the
// same 8-bit trick as the CPU kernel: q' = q * 3^n (wrapping), then
// trit = ((q' * 3) >> 8) in 0..=2, value = (trit - 1) * d.
}  // extern "C" — temporary close for the PTQ1_0 templates

template <std::size_t LWS_T>
inline void matvec_ptq1_0_packed_f32_usm_impl(
    sycl::queue& q,
    const void* w_bytes_usm,
    const float* x_usm,
    float* out_usm,
    int M, int K) {
    constexpr std::size_t LWS = LWS_T;
    const int blocks_per_row = K / 128;
    const int bytes_per_row = blocks_per_row * 28;
    const uint8_t* w_bytes = static_cast<const uint8_t*>(w_bytes_usm);
    const std::size_t global =
        ((static_cast<std::size_t>(M) + LWS - 1) / LWS) * LWS;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(LWS)),
            [=](sycl::nd_item<1> it) {
                const int m = static_cast<int>(it.get_global_id(0));
                if (m >= M) return;
                const uint8_t pow3[5] = {1, 3, 9, 27, 81};
                const uint8_t* row = w_bytes + m * bytes_per_row;
                float acc = 0.0f;
                for (int b = 0; b < blocks_per_row; ++b) {
                    const uint8_t* blk = row + b * 28;
                    const uint8_t* qs = blk;
                    const uint8_t* qh = blk + 24;
                    uint16_t d_bits = static_cast<uint16_t>(blk[26])
                                      | (static_cast<uint16_t>(blk[27]) << 8);
                    const float d = bits_to_f32(d_bits);
                    const int x_base = b * 128;
                    float sum = 0.0f;
                    // Chunk 1: qs[0..16], 5 digit stages x 16 lanes -> e 0..80.
                    for (int n = 0; n < 5; ++n) {
                        const uint8_t p3 = pow3[n];
                        const int e0 = x_base + n * 16;
                        for (int mm = 0; mm < 16; ++mm) {
                            const uint8_t qv =
                                static_cast<uint8_t>(qs[mm] * p3);
                            const int trit =
                                ((static_cast<int>(qv) * 3) >> 8) - 1;
                            sum += static_cast<float>(trit) * x_usm[e0 + mm];
                        }
                    }
                    // Chunk 2: qs[16..24], 5 digit stages x 8 lanes -> e 80..120.
                    for (int n = 0; n < 5; ++n) {
                        const uint8_t p3 = pow3[n];
                        const int e0 = x_base + 80 + n * 8;
                        for (int mm = 0; mm < 8; ++mm) {
                            const uint8_t qv =
                                static_cast<uint8_t>(qs[16 + mm] * p3);
                            const int trit =
                                ((static_cast<int>(qv) * 3) >> 8) - 1;
                            sum += static_cast<float>(trit) * x_usm[e0 + mm];
                        }
                    }
                    // qh: 2 bytes x 4 digit stages -> e 120..128.
                    for (int n = 0; n < 4; ++n) {
                        const uint8_t p3 = pow3[n];
                        const int e0 = x_base + 120 + n * 2;
                        for (int hh = 0; hh < 2; ++hh) {
                            const uint8_t qv =
                                static_cast<uint8_t>(qh[hh] * p3);
                            const int trit =
                                ((static_cast<int>(qv) * 3) >> 8) - 1;
                            sum += static_cast<float>(trit) * x_usm[e0 + hh];
                        }
                    }
                    acc += d * sum;
                }
                out_usm[m] = acc;
            });
    }).wait();
}

template <std::size_t LWS_T>
inline void matvec_ptq1_0_packed_f32_batched_usm_impl(
    sycl::queue& q,
    const void* w_bytes_usm,
    const float* x_usm,
    float* out_usm,
    int M, int K, int N) {
    constexpr std::size_t LWS = LWS_T;
    const int blocks_per_row = K / 128;
    const int bytes_per_row = blocks_per_row * 28;
    const uint8_t* w_bytes = static_cast<const uint8_t*>(w_bytes_usm);
    const std::size_t global_m =
        ((static_cast<std::size_t>(M) + LWS - 1) / LWS) * LWS;
    sycl::range<2> global(global_m, static_cast<std::size_t>(N));
    sycl::range<2> local(LWS, 1);
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<2>(global, local),
            [=](sycl::nd_item<2> it) {
                const int m = static_cast<int>(it.get_global_id(0));
                const int n_row = static_cast<int>(it.get_global_id(1));
                if (m >= M || n_row >= N) return;
                const uint8_t pow3[5] = {1, 3, 9, 27, 81};
                const uint8_t* row = w_bytes + m * bytes_per_row;
                const float* x_row = x_usm + static_cast<std::size_t>(n_row) * K;
                float acc = 0.0f;
                for (int b = 0; b < blocks_per_row; ++b) {
                    const uint8_t* blk = row + b * 28;
                    const uint8_t* qs = blk;
                    const uint8_t* qh = blk + 24;
                    uint16_t d_bits = static_cast<uint16_t>(blk[26])
                                      | (static_cast<uint16_t>(blk[27]) << 8);
                    const float d = bits_to_f32(d_bits);
                    const int x_base = b * 128;
                    float sum = 0.0f;
                    for (int n = 0; n < 5; ++n) {
                        const uint8_t p3 = pow3[n];
                        const int e0 = x_base + n * 16;
                        for (int mm = 0; mm < 16; ++mm) {
                            const uint8_t qv =
                                static_cast<uint8_t>(qs[mm] * p3);
                            const int trit =
                                ((static_cast<int>(qv) * 3) >> 8) - 1;
                            sum += static_cast<float>(trit) * x_row[e0 + mm];
                        }
                    }
                    for (int n = 0; n < 5; ++n) {
                        const uint8_t p3 = pow3[n];
                        const int e0 = x_base + 80 + n * 8;
                        for (int mm = 0; mm < 8; ++mm) {
                            const uint8_t qv =
                                static_cast<uint8_t>(qs[16 + mm] * p3);
                            const int trit =
                                ((static_cast<int>(qv) * 3) >> 8) - 1;
                            sum += static_cast<float>(trit) * x_row[e0 + mm];
                        }
                    }
                    for (int n = 0; n < 4; ++n) {
                        const uint8_t p3 = pow3[n];
                        const int e0 = x_base + 120 + n * 2;
                        for (int hh = 0; hh < 2; ++hh) {
                            const uint8_t qv =
                                static_cast<uint8_t>(qh[hh] * p3);
                            const int trit =
                                ((static_cast<int>(qv) * 3) >> 8) - 1;
                            sum += static_cast<float>(trit) * x_row[e0 + hh];
                        }
                    }
                    acc += d * sum;
                }
                out_usm[static_cast<std::size_t>(n_row) * M + m] = acc;
            });
    }).wait();
}

// Blockwise Prism Hadamard rotation on the GPU: out = (1/sqrt(block))
// * WHT(signs (*) x) per `block`-sized span. One work-group per span,
// butterfly staged in SLM with a barrier per stage — the USM-chain
// companion of the PTQ1_0 matvec (rotate once in device memory, then
// every folded matvec reads the rotated activation without a host
// round-trip).
inline void hadamard_forward_usm_impl(
    sycl::queue& q,
    const float* x_usm,
    const float* signs_usm,
    float* out_usm,
    int n_elems, int block) {
    const int n_blocks = n_elems / block;
    const int lws = block < 256 ? block : 256;
    const int per_thread = block / lws;
    const float inv_sqrt = 1.0f / sycl::sqrt(static_cast<float>(block));
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> slm(sycl::range<1>(block), h);
        h.parallel_for(
            sycl::nd_range<1>(
                sycl::range<1>(static_cast<std::size_t>(n_blocks) * lws),
                sycl::range<1>(lws)),
            [=](sycl::nd_item<1> it) {
                const int blk = static_cast<int>(it.get_group(0));
                const int tid = static_cast<int>(it.get_local_id(0));
                const int base = blk * block;
                for (int i = 0; i < per_thread; ++i) {
                    const int e = tid * per_thread + i;
                    slm[e] = x_usm[base + e] * signs_usm[base + e];
                }
                it.barrier(sycl::access::fence_space::local_space);
                for (int half = 1; half < block; half <<= 1) {
                    for (int i = 0; i < per_thread; ++i) {
                        const int e = tid * per_thread + i;
                        const int lo = (e / half) * (half << 1) + (e % half);
                        if (e < block / 2) {
                            const int hi = lo + half;
                            const float a = slm[lo];
                            const float bqv = slm[hi];
                            slm[lo] = a + bqv;
                            slm[hi] = a - bqv;
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                }
                for (int i = 0; i < per_thread; ++i) {
                    const int e = tid * per_thread + i;
                    out_usm[base + e] = slm[e] * inv_sqrt;
                }
            });
    }).wait();
}

extern "C" {

void rsl_matvec_ptq1_0_packed_f32_usm(rsl_stream* s,
                                      const void* w_bytes_usm,
                                      const float* x_usm,
                                      float* out_usm,
                                      int M, int K,
                                      int lws) RSL_FFI_BODY_VOID(
    "rsl_matvec_ptq1_0_packed_f32_usm", {
    if (s == nullptr || w_bytes_usm == nullptr
        || x_usm == nullptr || out_usm == nullptr) {
        return;
    }
    if (M <= 0 || K <= 0 || (K % 128) != 0) {
        return;
    }
    auto& q = s->q;
    const int eff_lws = (lws <= 0) ? static_cast<int>(RSL_LWS) : lws;
    switch (eff_lws) {
        case 16:  matvec_ptq1_0_packed_f32_usm_impl<16>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 32:  matvec_ptq1_0_packed_f32_usm_impl<32>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 64:  matvec_ptq1_0_packed_f32_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 128: matvec_ptq1_0_packed_f32_usm_impl<128>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 256: matvec_ptq1_0_packed_f32_usm_impl<256>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        default:  matvec_ptq1_0_packed_f32_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
    }
})

void rsl_matvec_ptq1_0_packed_f32_batched_usm(rsl_stream* s,
                                              const void* w_bytes_usm,
                                              const float* x_usm,
                                              float* out_usm,
                                              int M, int K, int N,
                                              int lws) RSL_FFI_BODY_VOID(
    "rsl_matvec_ptq1_0_packed_f32_batched_usm", {
    if (s == nullptr || w_bytes_usm == nullptr
        || x_usm == nullptr || out_usm == nullptr) {
        return;
    }
    if (M <= 0 || K <= 0 || N <= 0 || (K % 128) != 0) {
        return;
    }
    auto& q = s->q;
    const int eff_lws = (lws <= 0) ? static_cast<int>(RSL_LWS) : lws;
    switch (eff_lws) {
        case 16:  matvec_ptq1_0_packed_f32_batched_usm_impl<16>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        case 32:  matvec_ptq1_0_packed_f32_batched_usm_impl<32>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        case 64:  matvec_ptq1_0_packed_f32_batched_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        case 128: matvec_ptq1_0_packed_f32_batched_usm_impl<128>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        case 256: matvec_ptq1_0_packed_f32_batched_usm_impl<256>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        default:  matvec_ptq1_0_packed_f32_batched_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
    }
})

void rsl_hadamard_forward_usm(rsl_stream* s,
                              const float* x_usm,
                              const float* signs_usm,
                              float* out_usm,
                              int n_elems, int block) RSL_FFI_BODY_VOID(
    "rsl_hadamard_forward_usm", {
    if (s == nullptr || x_usm == nullptr || signs_usm == nullptr
        || out_usm == nullptr) {
        return;
    }
    if (n_elems <= 0 || block <= 0 || (n_elems % block) != 0
        || (block & (block - 1)) != 0 || block > 4096) {
        return;
    }
    auto& q = s->q;
    hadamard_forward_usm_impl(q, x_usm, signs_usm, out_usm, n_elems, block);
})

// =========================================================================
// CPU-parity packed matvecs: legacy Q4_0/Q5_0/Q4_1/Q5_1 (block 32), K-quant
// Q2_K/Q3_K/Q8_K (block 256), PrismML PQ2_0 (block 128). Each mirrors the
// CPU scalar reference in `rustllama-kernels-cpu` byte-for-byte (same block
// layout, same per-element decode + accumulation order) so the GPU output
// matches the CPU reference under `doctor --gpu-parity`. One work-item per
// output row; templated on LWS like the other packed matvecs.
// =========================================================================
}  // extern "C" — close so we can declare the new-format templates

// ---- Q4_0 (18 bytes / 32 weights: f16 d + 16 packed-nibble bytes) ----
// weight = d * (nibble - 8); low nibble -> x[j], high nibble -> x[j+16].
template <std::size_t LWS_T>
inline void matvec_q4_0_packed_f32_usm_impl(
    sycl::queue& q,
    const void* w_bytes_usm,
    const float* x_usm,
    float* out_usm,
    int M, int K) {
    constexpr std::size_t LWS = LWS_T;
    const int blocks_per_row = K / 32;
    const int bytes_per_row = blocks_per_row * 18;
    const uint8_t* w_bytes = static_cast<const uint8_t*>(w_bytes_usm);
    const std::size_t global =
        ((static_cast<std::size_t>(M) + LWS - 1) / LWS) * LWS;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(LWS)),
            [=](sycl::nd_item<1> it) {
                const int m = static_cast<int>(it.get_global_id(0));
                if (m >= M) return;
                const uint8_t* row = w_bytes + m * bytes_per_row;
                float acc = 0.0f;
                for (int b = 0; b < blocks_per_row; ++b) {
                    const uint8_t* blk = row + b * 18;
                    const float d = bits_to_f32(static_cast<uint16_t>(blk[0])
                                                | (static_cast<uint16_t>(blk[1]) << 8));
                    const uint8_t* qs = blk + 2;
                    const int x_off = b * 32;
                    for (int j = 0; j < 16; ++j) {
                        const int x0 = static_cast<int>(qs[j] & 0x0F) - 8;
                        const int x1 = static_cast<int>(qs[j] >> 4) - 8;
                        acc += d * static_cast<float>(x0) * x_usm[x_off + j];
                        acc += d * static_cast<float>(x1) * x_usm[x_off + j + 16];
                    }
                }
                out_usm[m] = acc;
            });
    }).wait();
}

// ---- Q5_0 (22 bytes / 32: f16 d + u32 qh + 16 nibble bytes) ----
// 5-bit signed: value = (nibble | 5th_bit<<4) - 16. The 5th bit for the
// low-nibble weight j is qh bit j; for the high-nibble weight it is bit
// j+16 (matches the CPU `((qh>>j)<<4)&0x10` / `(qh>>(j+12))&0x10` trick).
template <std::size_t LWS_T>
inline void matvec_q5_0_packed_f32_usm_impl(
    sycl::queue& q,
    const void* w_bytes_usm,
    const float* x_usm,
    float* out_usm,
    int M, int K) {
    constexpr std::size_t LWS = LWS_T;
    const int blocks_per_row = K / 32;
    const int bytes_per_row = blocks_per_row * 22;
    const uint8_t* w_bytes = static_cast<const uint8_t*>(w_bytes_usm);
    const std::size_t global =
        ((static_cast<std::size_t>(M) + LWS - 1) / LWS) * LWS;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(LWS)),
            [=](sycl::nd_item<1> it) {
                const int m = static_cast<int>(it.get_global_id(0));
                if (m >= M) return;
                const uint8_t* row = w_bytes + m * bytes_per_row;
                float acc = 0.0f;
                for (int b = 0; b < blocks_per_row; ++b) {
                    const uint8_t* blk = row + b * 22;
                    const float d = bits_to_f32(static_cast<uint16_t>(blk[0])
                                                | (static_cast<uint16_t>(blk[1]) << 8));
                    const uint32_t qh =
                        static_cast<uint32_t>(blk[2])
                        | (static_cast<uint32_t>(blk[3]) << 8)
                        | (static_cast<uint32_t>(blk[4]) << 16)
                        | (static_cast<uint32_t>(blk[5]) << 24);
                    const uint8_t* qs = blk + 6;
                    const int x_off = b * 32;
                    for (int j = 0; j < 16; ++j) {
                        const uint32_t xh_0 = ((qh >> j) << 4) & 0x10u;
                        const uint32_t xh_1 = (qh >> (j + 12)) & 0x10u;
                        const int x0 = (static_cast<int>(qs[j] & 0x0F)
                                        | static_cast<int>(xh_0)) - 16;
                        const int x1 = (static_cast<int>(qs[j] >> 4)
                                        | static_cast<int>(xh_1)) - 16;
                        acc += d * static_cast<float>(x0) * x_usm[x_off + j];
                        acc += d * static_cast<float>(x1) * x_usm[x_off + j + 16];
                    }
                }
                out_usm[m] = acc;
            });
    }).wait();
}

// ---- Q4_1 (20 bytes / 32: f16 d + f16 min + 16 nibble bytes) ----
// weight = d * nibble + min (no -8 zero-point; min carries the offset).
template <std::size_t LWS_T>
inline void matvec_q4_1_packed_f32_usm_impl(
    sycl::queue& q,
    const void* w_bytes_usm,
    const float* x_usm,
    float* out_usm,
    int M, int K) {
    constexpr std::size_t LWS = LWS_T;
    const int blocks_per_row = K / 32;
    const int bytes_per_row = blocks_per_row * 20;
    const uint8_t* w_bytes = static_cast<const uint8_t*>(w_bytes_usm);
    const std::size_t global =
        ((static_cast<std::size_t>(M) + LWS - 1) / LWS) * LWS;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(LWS)),
            [=](sycl::nd_item<1> it) {
                const int m = static_cast<int>(it.get_global_id(0));
                if (m >= M) return;
                const uint8_t* row = w_bytes + m * bytes_per_row;
                float acc = 0.0f;
                for (int b = 0; b < blocks_per_row; ++b) {
                    const uint8_t* blk = row + b * 20;
                    const float d = bits_to_f32(static_cast<uint16_t>(blk[0])
                                                | (static_cast<uint16_t>(blk[1]) << 8));
                    const float mn = bits_to_f32(static_cast<uint16_t>(blk[2])
                                                 | (static_cast<uint16_t>(blk[3]) << 8));
                    const uint8_t* qs = blk + 4;
                    const int x_off = b * 32;
                    for (int j = 0; j < 16; ++j) {
                        const int q0 = static_cast<int>(qs[j] & 0x0F);
                        const int q1 = static_cast<int>(qs[j] >> 4);
                        acc += (d * static_cast<float>(q0) + mn) * x_usm[x_off + j];
                        acc += (d * static_cast<float>(q1) + mn) * x_usm[x_off + j + 16];
                    }
                }
                out_usm[m] = acc;
            });
    }).wait();
}

// ---- Q5_1 (24 bytes / 32: f16 d + f16 min + u32 qh + 16 nibble bytes) ----
// weight = d * (nibble | 5th_bit<<4) + min. Same qh bit-selection as Q5_0.
template <std::size_t LWS_T>
inline void matvec_q5_1_packed_f32_usm_impl(
    sycl::queue& q,
    const void* w_bytes_usm,
    const float* x_usm,
    float* out_usm,
    int M, int K) {
    constexpr std::size_t LWS = LWS_T;
    const int blocks_per_row = K / 32;
    const int bytes_per_row = blocks_per_row * 24;
    const uint8_t* w_bytes = static_cast<const uint8_t*>(w_bytes_usm);
    const std::size_t global =
        ((static_cast<std::size_t>(M) + LWS - 1) / LWS) * LWS;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(LWS)),
            [=](sycl::nd_item<1> it) {
                const int m = static_cast<int>(it.get_global_id(0));
                if (m >= M) return;
                const uint8_t* row = w_bytes + m * bytes_per_row;
                float acc = 0.0f;
                for (int b = 0; b < blocks_per_row; ++b) {
                    const uint8_t* blk = row + b * 24;
                    const float d = bits_to_f32(static_cast<uint16_t>(blk[0])
                                                | (static_cast<uint16_t>(blk[1]) << 8));
                    const float mn = bits_to_f32(static_cast<uint16_t>(blk[2])
                                                 | (static_cast<uint16_t>(blk[3]) << 8));
                    const uint32_t qh =
                        static_cast<uint32_t>(blk[4])
                        | (static_cast<uint32_t>(blk[5]) << 8)
                        | (static_cast<uint32_t>(blk[6]) << 16)
                        | (static_cast<uint32_t>(blk[7]) << 24);
                    const uint8_t* qs = blk + 8;
                    const int x_off = b * 32;
                    for (int j = 0; j < 16; ++j) {
                        const uint32_t xh_0 = ((qh >> j) << 4) & 0x10u;
                        const uint32_t xh_1 = (qh >> (j + 12)) & 0x10u;
                        const int q0 = static_cast<int>(qs[j] & 0x0F)
                                       | static_cast<int>(xh_0);
                        const int q1 = static_cast<int>(qs[j] >> 4)
                                       | static_cast<int>(xh_1);
                        acc += (d * static_cast<float>(q0) + mn) * x_usm[x_off + j];
                        acc += (d * static_cast<float>(q1) + mn) * x_usm[x_off + j + 16];
                    }
                }
                out_usm[m] = acc;
            });
    }).wait();
}

// ---- Q2_K (84 bytes / 256: 16 scale bytes + 64 qs + f16 d + f16 dmin) ----
// 8 sub-blocks of 32 (two 16-wide halves) traversed as 2 chunks x 4 shifts;
// each 6-bit scale byte packs dl=(sc&0xF) and ml=(sc>>4). weight = dl*q - ml.
template <std::size_t LWS_T>
inline void matvec_q2_k_packed_f32_usm_impl(
    sycl::queue& q,
    const void* w_bytes_usm,
    const float* x_usm,
    float* out_usm,
    int M, int K) {
    constexpr std::size_t LWS = LWS_T;
    const int blocks_per_row = K / 256;
    const int bytes_per_row = blocks_per_row * 84;
    const uint8_t* w_bytes = static_cast<const uint8_t*>(w_bytes_usm);
    const std::size_t global =
        ((static_cast<std::size_t>(M) + LWS - 1) / LWS) * LWS;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(LWS)),
            [=](sycl::nd_item<1> it) {
                const int m = static_cast<int>(it.get_global_id(0));
                if (m >= M) return;
                const uint8_t* row = w_bytes + m * bytes_per_row;
                float acc = 0.0f;
                for (int b = 0; b < blocks_per_row; ++b) {
                    const uint8_t* blk = row + b * 84;
                    const uint8_t* scales = blk;          // [0..16)
                    const uint8_t* qs = blk + 16;         // [16..80)
                    const float d = bits_to_f32(static_cast<uint16_t>(blk[80])
                                                | (static_cast<uint16_t>(blk[81]) << 8));
                    const float dmin = bits_to_f32(static_cast<uint16_t>(blk[82])
                                                 | (static_cast<uint16_t>(blk[83]) << 8));
                    const int x_base = b * 256;
                    int x_off = 0;
                    int is = 0;
                    for (int chunk = 0; chunk < 2; ++chunk) {
                        const uint8_t* qc = qs + chunk * 32;
                        for (int sh = 0; sh < 4; ++sh) {
                            const uint32_t shift = static_cast<uint32_t>(sh * 2);
                            // Sub-block A: qc[0..16] >> shift
                            const uint8_t sc_a = scales[is];
                            const float dl_a = d * static_cast<float>(sc_a & 0x0F);
                            const float ml_a = dmin * static_cast<float>(sc_a >> 4);
                            for (int l = 0; l < 16; ++l) {
                                const float qv = static_cast<float>((qc[l] >> shift) & 3);
                                acc += (dl_a * qv - ml_a) * x_usm[x_base + x_off + l];
                            }
                            x_off += 16;
                            ++is;
                            // Sub-block B: qc[16..32] >> shift
                            const uint8_t sc_b = scales[is];
                            const float dl_b = d * static_cast<float>(sc_b & 0x0F);
                            const float ml_b = dmin * static_cast<float>(sc_b >> 4);
                            for (int l = 0; l < 16; ++l) {
                                const float qv = static_cast<float>((qc[l + 16] >> shift) & 3);
                                acc += (dl_b * qv - ml_b) * x_usm[x_base + x_off + l];
                            }
                            x_off += 16;
                            ++is;
                        }
                    }
                }
                out_usm[m] = acc;
            });
    }).wait();
}

// ---- Q3_K (110 bytes / 256: 32 hmask + 64 qs + 12 scales + f16 d) ----
// 6-bit scales are unpacked via the KMASK1/KMASK2 shuffle into 16 signed
// scales (biased -32). Each weight = (low2 - (hmask_bit ? 0 : 4)); the
// m_bit walks 1..128 across the two chunks. Mirrors the CPU scalar +
// the on-device `rsl_dequant_q3_k_to_f32_usm` unpack.
template <std::size_t LWS_T>
inline void matvec_q3_k_packed_f32_usm_impl(
    sycl::queue& q,
    const void* w_bytes_usm,
    const float* x_usm,
    float* out_usm,
    int M, int K) {
    constexpr std::size_t LWS = LWS_T;
    const int blocks_per_row = K / 256;
    const int bytes_per_row = blocks_per_row * 110;
    const uint8_t* w_bytes = static_cast<const uint8_t*>(w_bytes_usm);
    const std::size_t global =
        ((static_cast<std::size_t>(M) + LWS - 1) / LWS) * LWS;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(LWS)),
            [=](sycl::nd_item<1> it) {
                const int m = static_cast<int>(it.get_global_id(0));
                if (m >= M) return;
                const uint32_t KMASK1 = 0x03030303u;
                const uint32_t KMASK2 = 0x0f0f0f0fu;
                const uint8_t* row = w_bytes + m * bytes_per_row;
                float acc = 0.0f;
                for (int b = 0; b < blocks_per_row; ++b) {
                    const uint8_t* blk = row + b * 110;
                    const uint8_t* hmask = blk;            // [0..32)
                    const uint8_t* qs = blk + 32;          // [32..96)
                    const uint8_t* sc_raw = blk + 96;      // [96..108)
                    const float d_all = bits_to_f32(static_cast<uint16_t>(blk[108])
                                                 | (static_cast<uint16_t>(blk[109]) << 8));
                    uint32_t aux[4];
                    aux[0] = static_cast<uint32_t>(sc_raw[0])
                             | (static_cast<uint32_t>(sc_raw[1]) << 8)
                             | (static_cast<uint32_t>(sc_raw[2]) << 16)
                             | (static_cast<uint32_t>(sc_raw[3]) << 24);
                    aux[1] = static_cast<uint32_t>(sc_raw[4])
                             | (static_cast<uint32_t>(sc_raw[5]) << 8)
                             | (static_cast<uint32_t>(sc_raw[6]) << 16)
                             | (static_cast<uint32_t>(sc_raw[7]) << 24);
                    aux[2] = static_cast<uint32_t>(sc_raw[8])
                             | (static_cast<uint32_t>(sc_raw[9]) << 8)
                             | (static_cast<uint32_t>(sc_raw[10]) << 16)
                             | (static_cast<uint32_t>(sc_raw[11]) << 24);
                    const uint32_t tmp = aux[2];
                    aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
                    aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
                    aux[0] = (aux[0] & KMASK2) | ((tmp & KMASK1) << 4);
                    aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
                    int8_t scales[16];
                    for (int j = 0; j < 16; ++j) {
                        scales[j] = static_cast<int8_t>(
                            static_cast<uint8_t>((aux[j >> 2] >> ((j & 3) * 8)) & 0xFF));
                    }
                    const int x_base = b * 256;
                    int q_cursor = 0;
                    int x_off = 0;
                    uint8_t m_bit = 1;
                    int is = 0;
                    for (int chunk = 0; chunk < 2; ++chunk) {
                        uint32_t shift = 0;
                        for (int j4 = 0; j4 < 4; ++j4) {
                            float dl = d_all * (static_cast<float>(scales[is]) - 32.0f);
                            ++is;
                            for (int l = 0; l < 16; ++l) {
                                const int lo = static_cast<int>((qs[q_cursor + l] >> shift) & 3);
                                const int hi_sub = (hmask[l] & m_bit) != 0 ? 0 : 4;
                                acc += dl * static_cast<float>(lo - hi_sub)
                                          * x_usm[x_base + x_off + l];
                            }
                            x_off += 16;
                            dl = d_all * (static_cast<float>(scales[is]) - 32.0f);
                            ++is;
                            for (int l = 0; l < 16; ++l) {
                                const int lo = static_cast<int>((qs[q_cursor + l + 16] >> shift) & 3);
                                const int hi_sub = (hmask[l + 16] & m_bit) != 0 ? 0 : 4;
                                acc += dl * static_cast<float>(lo - hi_sub)
                                          * x_usm[x_base + x_off + l];
                            }
                            x_off += 16;
                            shift += 2;
                            m_bit = static_cast<uint8_t>(m_bit << 1);
                        }
                        q_cursor += 32;
                    }
                }
                out_usm[m] = acc;
            });
    }).wait();
}

// ---- Q8_K (292 bytes / 256: f32 d + 256 i8 qs + 16 i16 bsums) ----
// weight = d * i8. The f32 scale is little-endian (GGUF); bsums unused by
// matvec. NOTE: d is F32 here, not the f16 used by the other formats.
template <std::size_t LWS_T>
inline void matvec_q8_k_packed_f32_usm_impl(
    sycl::queue& q,
    const void* w_bytes_usm,
    const float* x_usm,
    float* out_usm,
    int M, int K) {
    constexpr std::size_t LWS = LWS_T;
    const int blocks_per_row = K / 256;
    const int bytes_per_row = blocks_per_row * 292;
    const uint8_t* w_bytes = static_cast<const uint8_t*>(w_bytes_usm);
    const std::size_t global =
        ((static_cast<std::size_t>(M) + LWS - 1) / LWS) * LWS;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(LWS)),
            [=](sycl::nd_item<1> it) {
                const int m = static_cast<int>(it.get_global_id(0));
                if (m >= M) return;
                const uint8_t* row = w_bytes + m * bytes_per_row;
                float acc = 0.0f;
                for (int b = 0; b < blocks_per_row; ++b) {
                    const uint8_t* blk = row + b * 292;
                    const uint32_t d_bits =
                        static_cast<uint32_t>(blk[0])
                        | (static_cast<uint32_t>(blk[1]) << 8)
                        | (static_cast<uint32_t>(blk[2]) << 16)
                        | (static_cast<uint32_t>(blk[3]) << 24);
                    float d;
                    std::memcpy(&d, &d_bits, sizeof(float));
                    const uint8_t* qs = blk + 4;
                    const int x_off = b * 256;
                    for (int j = 0; j < 256; ++j) {
                        const int8_t w_i8 = static_cast<int8_t>(qs[j]);
                        acc += d * static_cast<float>(w_i8) * x_usm[x_off + j];
                    }
                }
                out_usm[m] = acc;
            });
    }).wait();
}

// ---- PQ2_0 (PrismML Bonsai, 34 bytes / 128: f16 d + 32 packed 2-bit) ----
// weight = d * (code - 1), code in [0,3] read low-to-high 2 bits per byte.
template <std::size_t LWS_T>
inline void matvec_pq2_0_packed_f32_usm_impl(
    sycl::queue& q,
    const void* w_bytes_usm,
    const float* x_usm,
    float* out_usm,
    int M, int K) {
    constexpr std::size_t LWS = LWS_T;
    const int blocks_per_row = K / 128;
    const int bytes_per_row = blocks_per_row * 34;
    const uint8_t* w_bytes = static_cast<const uint8_t*>(w_bytes_usm);
    const std::size_t global =
        ((static_cast<std::size_t>(M) + LWS - 1) / LWS) * LWS;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(LWS)),
            [=](sycl::nd_item<1> it) {
                const int m = static_cast<int>(it.get_global_id(0));
                if (m >= M) return;
                const uint8_t* row = w_bytes + m * bytes_per_row;
                float acc = 0.0f;
                for (int b = 0; b < blocks_per_row; ++b) {
                    const uint8_t* blk = row + b * 34;
                    const float d = bits_to_f32(static_cast<uint16_t>(blk[0])
                                                | (static_cast<uint16_t>(blk[1]) << 8));
                    const uint8_t* qs = blk + 2;
                    const int x_off = b * 128;
                    // Accumulate in the integer-code domain then scale once
                    // per block: acc += d * sum((code-1) * x).
                    float sum = 0.0f;
                    for (int j = 0; j < 128; ++j) {
                        const int qv = static_cast<int>((qs[j / 4] >> ((j % 4) * 2)) & 0x3) - 1;
                        sum += static_cast<float>(qv) * x_usm[x_off + j];
                    }
                    acc += d * sum;
                }
                out_usm[m] = acc;
            });
    }).wait();
}

// PTQ1_0 gate+up FUSED matvec — one launch, shares x_usm across gate + up.
// Same trit decode as `rsl_matvec_ptq1_0_packed_f32_usm`; d folds per block.
template <std::size_t LWS_T>
inline void matvec_ptq1_0_gate_up_fused_usm_impl(
    sycl::queue& q,
    const void* gate_w_bytes_usm,
    const void* up_w_bytes_usm,
    const float* x_usm,
    float* gate_out_usm,
    float* up_out_usm,
    int M, int K) {
    constexpr std::size_t LWS = LWS_T;
    const int blocks_per_row = K / 128;
    const int bytes_per_row = blocks_per_row * 28;
    const uint8_t* g_bytes = static_cast<const uint8_t*>(gate_w_bytes_usm);
    const uint8_t* u_bytes = static_cast<const uint8_t*>(up_w_bytes_usm);
    const std::size_t global =
        ((static_cast<std::size_t>(M) + LWS - 1) / LWS) * LWS;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(LWS)),
            [=](sycl::nd_item<1> it) {
                const int m = static_cast<int>(it.get_global_id(0));
                if (m >= M) return;
                const uint8_t pow3[5] = {1, 3, 9, 27, 81};
                const uint8_t* g_row = g_bytes + m * bytes_per_row;
                const uint8_t* u_row = u_bytes + m * bytes_per_row;
                float gate_acc = 0.0f;
                float up_acc = 0.0f;
                for (int b = 0; b < blocks_per_row; ++b) {
                    const uint8_t* g_blk = g_row + b * 28;
                    const uint8_t* u_blk = u_row + b * 28;
                    const float g_d = bits_to_f32(static_cast<uint16_t>(g_blk[26])
                                                  | (static_cast<uint16_t>(g_blk[27]) << 8));
                    const float u_d = bits_to_f32(static_cast<uint16_t>(u_blk[26])
                                                  | (static_cast<uint16_t>(u_blk[27]) << 8));
                    const int x_base = b * 128;
                    float g_sum = 0.0f;
                    float u_sum = 0.0f;
                    // Chunk 1: qs[0..16], 5 digit stages x 16 lanes.
                    for (int n = 0; n < 5; ++n) {
                        const uint8_t p3 = pow3[n];
                        const int e0 = x_base + n * 16;
                        for (int mm = 0; mm < 16; ++mm) {
                            const float xv = x_usm[e0 + mm];
                            const int gt = ((static_cast<int>(static_cast<uint8_t>(g_blk[mm] * p3)) * 3) >> 8) - 1;
                            const int ut = ((static_cast<int>(static_cast<uint8_t>(u_blk[mm] * p3)) * 3) >> 8) - 1;
                            g_sum += static_cast<float>(gt) * xv;
                            u_sum += static_cast<float>(ut) * xv;
                        }
                    }
                    // Chunk 2: qs[16..24], 5 digit stages x 8 lanes.
                    for (int n = 0; n < 5; ++n) {
                        const uint8_t p3 = pow3[n];
                        const int e0 = x_base + 80 + n * 8;
                        for (int mm = 0; mm < 8; ++mm) {
                            const float xv = x_usm[e0 + mm];
                            const int gt = ((static_cast<int>(static_cast<uint8_t>(g_blk[16 + mm] * p3)) * 3) >> 8) - 1;
                            const int ut = ((static_cast<int>(static_cast<uint8_t>(u_blk[16 + mm] * p3)) * 3) >> 8) - 1;
                            g_sum += static_cast<float>(gt) * xv;
                            u_sum += static_cast<float>(ut) * xv;
                        }
                    }
                    // qh: 2 bytes x 4 digit stages.
                    for (int n = 0; n < 4; ++n) {
                        const uint8_t p3 = pow3[n];
                        const int e0 = x_base + 120 + n * 2;
                        for (int hh = 0; hh < 2; ++hh) {
                            const float xv = x_usm[e0 + hh];
                            const int gt = ((static_cast<int>(static_cast<uint8_t>(g_blk[24 + hh] * p3)) * 3) >> 8) - 1;
                            const int ut = ((static_cast<int>(static_cast<uint8_t>(u_blk[24 + hh] * p3)) * 3) >> 8) - 1;
                            g_sum += static_cast<float>(gt) * xv;
                            u_sum += static_cast<float>(ut) * xv;
                        }
                    }
                    gate_acc += g_d * g_sum;
                    up_acc += u_d * u_sum;
                }
                gate_out_usm[m] = gate_acc;
                up_out_usm[m] = up_acc;
            });
    }).wait();
}

extern "C" {

void rsl_matvec_q4_0_packed_f32_usm(rsl_stream* s,
                                    const void* w_bytes_usm,
                                    const float* x_usm,
                                    float* out_usm,
                                    int M, int K,
                                    int lws) RSL_FFI_BODY_VOID(
    "rsl_matvec_q4_0_packed_f32_usm", {
    if (s == nullptr || w_bytes_usm == nullptr
        || x_usm == nullptr || out_usm == nullptr) {
        return;
    }
    if (M <= 0 || K <= 0 || (K % 32) != 0) {
        return;
    }
    auto& q = s->q;
    const int eff_lws = (lws <= 0) ? static_cast<int>(RSL_LWS) : lws;
    switch (eff_lws) {
        case 16:  matvec_q4_0_packed_f32_usm_impl<16>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 32:  matvec_q4_0_packed_f32_usm_impl<32>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 64:  matvec_q4_0_packed_f32_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 128: matvec_q4_0_packed_f32_usm_impl<128>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 256: matvec_q4_0_packed_f32_usm_impl<256>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        default:  matvec_q4_0_packed_f32_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
    }
})

void rsl_matvec_q5_0_packed_f32_usm(rsl_stream* s,
                                    const void* w_bytes_usm,
                                    const float* x_usm,
                                    float* out_usm,
                                    int M, int K,
                                    int lws) RSL_FFI_BODY_VOID(
    "rsl_matvec_q5_0_packed_f32_usm", {
    if (s == nullptr || w_bytes_usm == nullptr
        || x_usm == nullptr || out_usm == nullptr) {
        return;
    }
    if (M <= 0 || K <= 0 || (K % 32) != 0) {
        return;
    }
    auto& q = s->q;
    const int eff_lws = (lws <= 0) ? static_cast<int>(RSL_LWS) : lws;
    switch (eff_lws) {
        case 16:  matvec_q5_0_packed_f32_usm_impl<16>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 32:  matvec_q5_0_packed_f32_usm_impl<32>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 64:  matvec_q5_0_packed_f32_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 128: matvec_q5_0_packed_f32_usm_impl<128>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 256: matvec_q5_0_packed_f32_usm_impl<256>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        default:  matvec_q5_0_packed_f32_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
    }
})

void rsl_matvec_q4_1_packed_f32_usm(rsl_stream* s,
                                    const void* w_bytes_usm,
                                    const float* x_usm,
                                    float* out_usm,
                                    int M, int K,
                                    int lws) RSL_FFI_BODY_VOID(
    "rsl_matvec_q4_1_packed_f32_usm", {
    if (s == nullptr || w_bytes_usm == nullptr
        || x_usm == nullptr || out_usm == nullptr) {
        return;
    }
    if (M <= 0 || K <= 0 || (K % 32) != 0) {
        return;
    }
    auto& q = s->q;
    const int eff_lws = (lws <= 0) ? static_cast<int>(RSL_LWS) : lws;
    switch (eff_lws) {
        case 16:  matvec_q4_1_packed_f32_usm_impl<16>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 32:  matvec_q4_1_packed_f32_usm_impl<32>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 64:  matvec_q4_1_packed_f32_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 128: matvec_q4_1_packed_f32_usm_impl<128>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 256: matvec_q4_1_packed_f32_usm_impl<256>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        default:  matvec_q4_1_packed_f32_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
    }
})

void rsl_matvec_q5_1_packed_f32_usm(rsl_stream* s,
                                    const void* w_bytes_usm,
                                    const float* x_usm,
                                    float* out_usm,
                                    int M, int K,
                                    int lws) RSL_FFI_BODY_VOID(
    "rsl_matvec_q5_1_packed_f32_usm", {
    if (s == nullptr || w_bytes_usm == nullptr
        || x_usm == nullptr || out_usm == nullptr) {
        return;
    }
    if (M <= 0 || K <= 0 || (K % 32) != 0) {
        return;
    }
    auto& q = s->q;
    const int eff_lws = (lws <= 0) ? static_cast<int>(RSL_LWS) : lws;
    switch (eff_lws) {
        case 16:  matvec_q5_1_packed_f32_usm_impl<16>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 32:  matvec_q5_1_packed_f32_usm_impl<32>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 64:  matvec_q5_1_packed_f32_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 128: matvec_q5_1_packed_f32_usm_impl<128>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 256: matvec_q5_1_packed_f32_usm_impl<256>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        default:  matvec_q5_1_packed_f32_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
    }
})

void rsl_matvec_q2_k_packed_f32_usm(rsl_stream* s,
                                    const void* w_bytes_usm,
                                    const float* x_usm,
                                    float* out_usm,
                                    int M, int K,
                                    int lws) RSL_FFI_BODY_VOID(
    "rsl_matvec_q2_k_packed_f32_usm", {
    if (s == nullptr || w_bytes_usm == nullptr
        || x_usm == nullptr || out_usm == nullptr) {
        return;
    }
    if (M <= 0 || K <= 0 || (K % 256) != 0) {
        return;
    }
    auto& q = s->q;
    const int eff_lws = (lws <= 0) ? static_cast<int>(RSL_LWS) : lws;
    switch (eff_lws) {
        case 16:  matvec_q2_k_packed_f32_usm_impl<16>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 32:  matvec_q2_k_packed_f32_usm_impl<32>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 64:  matvec_q2_k_packed_f32_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 128: matvec_q2_k_packed_f32_usm_impl<128>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 256: matvec_q2_k_packed_f32_usm_impl<256>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        default:  matvec_q2_k_packed_f32_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
    }
})

void rsl_matvec_q3_k_packed_f32_usm(rsl_stream* s,
                                    const void* w_bytes_usm,
                                    const float* x_usm,
                                    float* out_usm,
                                    int M, int K,
                                    int lws) RSL_FFI_BODY_VOID(
    "rsl_matvec_q3_k_packed_f32_usm", {
    if (s == nullptr || w_bytes_usm == nullptr
        || x_usm == nullptr || out_usm == nullptr) {
        return;
    }
    if (M <= 0 || K <= 0 || (K % 256) != 0) {
        return;
    }
    auto& q = s->q;
    const int eff_lws = (lws <= 0) ? static_cast<int>(RSL_LWS) : lws;
    switch (eff_lws) {
        case 16:  matvec_q3_k_packed_f32_usm_impl<16>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 32:  matvec_q3_k_packed_f32_usm_impl<32>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 64:  matvec_q3_k_packed_f32_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 128: matvec_q3_k_packed_f32_usm_impl<128>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 256: matvec_q3_k_packed_f32_usm_impl<256>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        default:  matvec_q3_k_packed_f32_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
    }
})

void rsl_matvec_q8_k_packed_f32_usm(rsl_stream* s,
                                    const void* w_bytes_usm,
                                    const float* x_usm,
                                    float* out_usm,
                                    int M, int K,
                                    int lws) RSL_FFI_BODY_VOID(
    "rsl_matvec_q8_k_packed_f32_usm", {
    if (s == nullptr || w_bytes_usm == nullptr
        || x_usm == nullptr || out_usm == nullptr) {
        return;
    }
    if (M <= 0 || K <= 0 || (K % 256) != 0) {
        return;
    }
    auto& q = s->q;
    const int eff_lws = (lws <= 0) ? static_cast<int>(RSL_LWS) : lws;
    switch (eff_lws) {
        case 16:  matvec_q8_k_packed_f32_usm_impl<16>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 32:  matvec_q8_k_packed_f32_usm_impl<32>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 64:  matvec_q8_k_packed_f32_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 128: matvec_q8_k_packed_f32_usm_impl<128>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 256: matvec_q8_k_packed_f32_usm_impl<256>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        default:  matvec_q8_k_packed_f32_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
    }
})

void rsl_matvec_pq2_0_packed_f32_usm(rsl_stream* s,
                                     const void* w_bytes_usm,
                                     const float* x_usm,
                                     float* out_usm,
                                     int M, int K,
                                     int lws) RSL_FFI_BODY_VOID(
    "rsl_matvec_pq2_0_packed_f32_usm", {
    if (s == nullptr || w_bytes_usm == nullptr
        || x_usm == nullptr || out_usm == nullptr) {
        return;
    }
    if (M <= 0 || K <= 0 || (K % 128) != 0) {
        return;
    }
    auto& q = s->q;
    const int eff_lws = (lws <= 0) ? static_cast<int>(RSL_LWS) : lws;
    switch (eff_lws) {
        case 16:  matvec_pq2_0_packed_f32_usm_impl<16>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 32:  matvec_pq2_0_packed_f32_usm_impl<32>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 64:  matvec_pq2_0_packed_f32_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 128: matvec_pq2_0_packed_f32_usm_impl<128>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        case 256: matvec_pq2_0_packed_f32_usm_impl<256>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
        default:  matvec_pq2_0_packed_f32_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K); break;
    }
})

void rsl_matvec_ptq1_0_gate_up_fused_usm(rsl_stream* s,
                                         const void* gate_w_bytes_usm,
                                         const void* up_w_bytes_usm,
                                         const float* x_usm,
                                         float* gate_out_usm,
                                         float* up_out_usm,
                                         int M, int K,
                                         int lws) RSL_FFI_BODY_VOID(
    "rsl_matvec_ptq1_0_gate_up_fused_usm", {
    if (s == nullptr || gate_w_bytes_usm == nullptr || up_w_bytes_usm == nullptr
        || x_usm == nullptr || gate_out_usm == nullptr || up_out_usm == nullptr) {
        return;
    }
    if (M <= 0 || K <= 0 || (K % 128) != 0) {
        return;
    }
    auto& q = s->q;
    const int eff_lws = (lws <= 0) ? static_cast<int>(RSL_LWS) : lws;
    switch (eff_lws) {
        case 16:  matvec_ptq1_0_gate_up_fused_usm_impl<16>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        case 32:  matvec_ptq1_0_gate_up_fused_usm_impl<32>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        case 64:  matvec_ptq1_0_gate_up_fused_usm_impl<64>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        case 128: matvec_ptq1_0_gate_up_fused_usm_impl<128>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        case 256: matvec_ptq1_0_gate_up_fused_usm_impl<256>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
        default:  matvec_ptq1_0_gate_up_fused_usm_impl<64>(q, gate_w_bytes_usm, up_w_bytes_usm, x_usm, gate_out_usm, up_out_usm, M, K); break;
    }
})

// USM-resident Q5_K_M packed matvec — batched over N input rows.
// See `rsl_matvec_q5_k_packed_f32_usm` for the super-block layout.
}  // extern "C" — temporary close for the Q5_K batched template

template <std::size_t LWS_T>
inline void matvec_q5_k_packed_f32_batched_usm_impl(
    sycl::queue& q,
    const void* w_bytes_usm,
    const float* x_usm,
    float* out_usm,
    int M, int K, int N) {
    constexpr std::size_t LWS = LWS_T;
    const int blocks_per_row = K / 256;
    const int bytes_per_row = blocks_per_row * 176;
    const uint8_t* w_bytes = static_cast<const uint8_t*>(w_bytes_usm);
    const std::size_t global_m =
        ((static_cast<std::size_t>(M) + LWS - 1) / LWS) * LWS;
    sycl::range<2> global(global_m, static_cast<std::size_t>(N));
    sycl::range<2> local(LWS, 1);
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<2>(global, local),
            [=](sycl::nd_item<2> it) {
                const int m = static_cast<int>(it.get_global_id(0));
                const int n = static_cast<int>(it.get_global_id(1));
                if (m >= M || n >= N) return;
                const uint8_t* row = w_bytes + m * bytes_per_row;
                const float* x_row = x_usm + n * K;
                float acc = 0.0f;
                for (int b = 0; b < blocks_per_row; ++b) {
                    const uint8_t* blk = row + b * 176;
                    uint16_t d_bits = static_cast<uint16_t>(blk[0])
                                      | (static_cast<uint16_t>(blk[1]) << 8);
                    uint16_t dmin_bits = static_cast<uint16_t>(blk[2])
                                         | (static_cast<uint16_t>(blk[3]) << 8);
                    const float d = bits_to_f32(d_bits);
                    const float dmin = bits_to_f32(dmin_bits);
                    const uint8_t* sb = blk + 4;
                    uint8_t sc[8];
                    uint8_t mn[8];
                    for (int j = 0; j < 8; ++j) {
                        if (j < 4) {
                            sc[j] = sb[j] & 0x3F;
                            mn[j] = sb[j + 4] & 0x3F;
                        } else {
                            sc[j] = (sb[j + 4] & 0x0F) | ((sb[j - 4] >> 6) << 4);
                            mn[j] = (sb[j + 4] >> 4) | ((sb[j] >> 6) << 4);
                        }
                    }
                    const uint8_t* qh = blk + 16;
                    const uint8_t* qs = blk + 48;
                    const int x_base = b * 256;
                    for (int group = 0; group < 4; ++group) {
                        const uint8_t* qc = qs + group * 32;
                        const float d_lo = d * static_cast<float>(sc[group * 2]);
                        const float m_lo = dmin * static_cast<float>(mn[group * 2]);
                        const float d_hi = d * static_cast<float>(sc[group * 2 + 1]);
                        const float m_hi = dmin * static_cast<float>(mn[group * 2 + 1]);
                        const int bit_lo = group * 2;
                        const int bit_hi = group * 2 + 1;
                        const int x_lo_off = x_base + group * 64;
                        const int x_hi_off = x_lo_off + 32;
                        for (int l = 0; l < 32; ++l) {
                            const uint8_t qb = qc[l];
                            const uint8_t qhb = qh[l];
                            const uint32_t lo = static_cast<uint32_t>(qb & 0x0F)
                                | (static_cast<uint32_t>((qhb >> bit_lo) & 1) << 4);
                            const uint32_t hi = static_cast<uint32_t>(qb >> 4)
                                | (static_cast<uint32_t>((qhb >> bit_hi) & 1) << 4);
                            acc += (d_lo * static_cast<float>(lo) - m_lo) * x_row[x_lo_off + l];
                            acc += (d_hi * static_cast<float>(hi) - m_hi) * x_row[x_hi_off + l];
                        }
                    }
                }
                out_usm[n * M + m] = acc;
            });
    }).wait();
}

extern "C" {

void rsl_matvec_q5_k_packed_f32_batched_usm(rsl_stream* s,
                                            const void* w_bytes_usm,
                                            const float* x_usm,
                                            float* out_usm,
                                            int M, int K, int N,
                                            int lws) RSL_FFI_BODY_VOID(
    "rsl_matvec_q5_k_packed_f32_batched_usm", {
    if (s == nullptr || w_bytes_usm == nullptr
        || x_usm == nullptr || out_usm == nullptr) {
        return;
    }
    if (M <= 0 || K <= 0 || N <= 0 || (K % 256) != 0) {
        return;
    }
    auto& q = s->q;
    const int eff_lws = (lws <= 0) ? static_cast<int>(RSL_LWS) : lws;
    switch (eff_lws) {
        case 16:  matvec_q5_k_packed_f32_batched_usm_impl<16>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        case 32:  matvec_q5_k_packed_f32_batched_usm_impl<32>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        case 64:  matvec_q5_k_packed_f32_batched_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        case 128: matvec_q5_k_packed_f32_batched_usm_impl<128>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        case 256: matvec_q5_k_packed_f32_batched_usm_impl<256>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        default:  matvec_q5_k_packed_f32_batched_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
    }
})

// USM-resident Q6_K packed matvec — batched over N input rows.
// See `rsl_matvec_q6_k_packed_f32_usm` for the super-block layout.
}  // extern "C" — temporary close for the Q6_K batched template

template <std::size_t LWS_T>
inline void matvec_q6_k_packed_f32_batched_usm_impl(
    sycl::queue& q,
    const void* w_bytes_usm,
    const float* x_usm,
    float* out_usm,
    int M, int K, int N) {
    constexpr std::size_t LWS = LWS_T;
    const int blocks_per_row = K / 256;
    const int bytes_per_row = blocks_per_row * 210;
    const uint8_t* w_bytes = static_cast<const uint8_t*>(w_bytes_usm);
    const std::size_t global_m =
        ((static_cast<std::size_t>(M) + LWS - 1) / LWS) * LWS;
    sycl::range<2> global(global_m, static_cast<std::size_t>(N));
    sycl::range<2> local(LWS, 1);
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<2>(global, local),
            [=](sycl::nd_item<2> it) {
                const int m = static_cast<int>(it.get_global_id(0));
                const int n = static_cast<int>(it.get_global_id(1));
                if (m >= M || n >= N) return;
                const uint8_t* row = w_bytes + m * bytes_per_row;
                const float* x_row = x_usm + n * K;
                float acc = 0.0f;
                for (int b = 0; b < blocks_per_row; ++b) {
                    const uint8_t* blk = row + b * 210;
                    const uint8_t* ql = blk;
                    const uint8_t* qh = blk + 128;
                    const int8_t* scales = reinterpret_cast<const int8_t*>(blk + 192);
                    uint16_t d_bits = static_cast<uint16_t>(blk[208])
                                      | (static_cast<uint16_t>(blk[209]) << 8);
                    const float d = bits_to_f32(d_bits);
                    const int x_base = b * 256;
                    for (int nh = 0; nh < 2; ++nh) {
                        for (int l = 0; l < 32; ++l) {
                            const int is = l / 16 + nh * 8;
                            const int qh_byte = qh[32 * nh + l];
                            const int q1 = (static_cast<int>(ql[64 * nh + l] & 0x0F)
                                | ((qh_byte >> 0) & 0x03) << 4) - 32;
                            const int q2 = (static_cast<int>(ql[64 * nh + l + 32] & 0x0F)
                                | ((qh_byte >> 2) & 0x03) << 4) - 32;
                            const int q3 = (static_cast<int>(ql[64 * nh + l] >> 4)
                                | ((qh_byte >> 4) & 0x03) << 4) - 32;
                            const int q4 = (static_cast<int>(ql[64 * nh + l + 32] >> 4)
                                | ((qh_byte >> 6) & 0x03) << 4) - 32;
                            const float s0 = static_cast<float>(scales[is]);
                            const float s1 = static_cast<float>(scales[is + 2]);
                            const float s2 = static_cast<float>(scales[is + 4]);
                            const float s3 = static_cast<float>(scales[is + 6]);
                            const int base = nh * 128 + l;
                            acc += d * s0 * static_cast<float>(q1) * x_row[x_base + base];
                            acc += d * s1 * static_cast<float>(q2) * x_row[x_base + base + 32];
                            acc += d * s2 * static_cast<float>(q3) * x_row[x_base + base + 64];
                            acc += d * s3 * static_cast<float>(q4) * x_row[x_base + base + 96];
                        }
                    }
                }
                out_usm[n * M + m] = acc;
            });
    }).wait();
}

extern "C" {

void rsl_matvec_q6_k_packed_f32_batched_usm(rsl_stream* s,
                                            const void* w_bytes_usm,
                                            const float* x_usm,
                                            float* out_usm,
                                            int M, int K, int N,
                                            int lws) RSL_FFI_BODY_VOID(
    "rsl_matvec_q6_k_packed_f32_batched_usm", {
    if (s == nullptr || w_bytes_usm == nullptr
        || x_usm == nullptr || out_usm == nullptr) {
        return;
    }
    if (M <= 0 || K <= 0 || N <= 0 || (K % 256) != 0) {
        return;
    }
    auto& q = s->q;
    const int eff_lws = (lws <= 0) ? static_cast<int>(RSL_LWS) : lws;
    switch (eff_lws) {
        case 16:  matvec_q6_k_packed_f32_batched_usm_impl<16>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        case 32:  matvec_q6_k_packed_f32_batched_usm_impl<32>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        case 64:  matvec_q6_k_packed_f32_batched_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        case 128: matvec_q6_k_packed_f32_batched_usm_impl<128>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        case 256: matvec_q6_k_packed_f32_batched_usm_impl<256>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
        default:  matvec_q6_k_packed_f32_batched_usm_impl<64>(q, w_bytes_usm, x_usm, out_usm, M, K, N); break;
    }
})

// USM-resident F16 GEMM. Same naive tiled algorithm as
// `rsl_gemm_f16` but operating directly on USM pointers — no
// internal alloc/copy. Use M=1 (row × matrix) or N=1 (matvec)
// for the engine's projection call sites.
void rsl_gemm_f16_usm(rsl_stream* s,
                      const uint16_t* a_usm,
                      const uint16_t* b_usm,
                      uint16_t* c_usm,
                      int M, int N, int K,
                      int lda, int ldb, int ldc) RSL_FFI_BODY_VOID("rsl_gemm_f16_usm", {
    if (s == nullptr || a_usm == nullptr || b_usm == nullptr || c_usm == nullptr) {
        return;
    }
    if (M <= 0 || N <= 0 || K <= 0) {
        return;
    }
    constexpr int TILE = 16;
    auto& q = s->q;
    const int M_tiles = (M + TILE - 1) / TILE;
    const int N_tiles = (N + TILE - 1) / TILE;
    sycl::range<2> global(M_tiles * TILE, N_tiles * TILE);
    sycl::range<2> local(TILE, TILE);
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<2>(global, local),
            [=](sycl::nd_item<2> it) {
                const int row = static_cast<int>(it.get_global_id(0));
                const int col = static_cast<int>(it.get_global_id(1));
                if (row >= M || col >= N) {
                    return;
                }
                float acc = 0.0f;
                for (int p = 0; p < K; ++p) {
                    const uint16_t a_bits = a_usm[row * lda + p];
                    const uint16_t b_bits = b_usm[p * ldb + col];
                    acc += bits_to_f32(a_bits) * bits_to_f32(b_bits);
                }
                c_usm[row * ldc + col] = f32_to_bits(acc);
            });
    }).wait();
})

// USM-resident half-split RoPE. In-place on `qk_usm` —
// `inv_freq_usm` is read-only. Each work-item handles one (head, j)
// pair where j ∈ [0, head_dim/2); writes both `qk[base + j]` and
// `qk[base + j + half]`.
void rsl_rope_usm(rsl_stream* s,
                  uint16_t* qk_usm, int n_heads, int head_dim, int pos,
                  const uint16_t* inv_freq_usm) RSL_FFI_BODY_VOID("rsl_rope_usm", {
    if (s == nullptr || qk_usm == nullptr || inv_freq_usm == nullptr) return;
    if (n_heads <= 0 || head_dim <= 0 || (head_dim % 2) != 0) return;
    auto& q = s->q;
    const int half = head_dim / 2;
    const int total = n_heads * half;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(round_up_to_lws(total)),
                              sycl::range<1>(RSL_LWS)),
            [=](sycl::nd_item<1> it) {
            const int flat = static_cast<int>(it.get_global_id(0));
            if (flat >= total) return;
            const int head = flat / half;
            const int j = flat % half;
            const float freq = bits_to_f32(inv_freq_usm[j]);
            const float angle = static_cast<float>(pos) * freq;
            const float c = sycl::cos(angle);
            const float si = sycl::sin(angle);
            const int base = head * head_dim;
            const float x0 = bits_to_f32(qk_usm[base + j]);
            const float x1 = bits_to_f32(qk_usm[base + j + half]);
            qk_usm[base + j] = f32_to_bits(x0 * c - x1 * si);
            qk_usm[base + j + half] = f32_to_bits(x0 * si + x1 * c);
        });
    }).wait();
})

// USM-resident SwiGLU. One work-item per element; reads x[i] and
// y[i] from USM, writes silu(x) * y to out[i].
void rsl_silu_mul_usm(rsl_stream* s,
                      const uint16_t* x_usm, const uint16_t* y_usm,
                      uint16_t* out_usm, int n) RSL_FFI_BODY_VOID("rsl_silu_mul_usm", {
    if (s == nullptr || x_usm == nullptr || y_usm == nullptr || out_usm == nullptr) return;
    if (n <= 0) return;
    auto& q = s->q;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(round_up_to_lws(static_cast<std::size_t>(n))),
                              sycl::range<1>(RSL_LWS)),
            [=](sycl::nd_item<1> it) {
            const int i = static_cast<int>(it.get_global_id(0));
            if (i >= n) return;
            const float xv = bits_to_f32(x_usm[i]);
            const float yv = bits_to_f32(y_usm[i]);
            const float silu = xv / (1.0f + sycl::exp(-xv));
            out_usm[i] = f32_to_bits(silu * yv);
        });
    }).wait();
})

// G2: KV-cache Q8_0 quantize-on-store. Reads f32 activations laid out
// as `[n_new][n_kv_heads][head_dim]` contiguous, writes i8 quantized
// bytes + per-row f32 scales into the strided KV-cache layout
// `[n_kv_heads][max_ctx][head_dim]`. Replaces the CPU
// `quantize_row_q8_0` loop in the prefill/decode paths when KV cache
// is USM-resident (`RUSTLLAMA_USM_KV=1`).
//
// Per-row algorithm matches `quantize_row_q8_0` exactly:
//   max_abs = max(|src[i]|)
//   if max_abs == 0: q[i] = 0 for all i, scale = 1.0
//   else: scale = max_abs/127, inv = 1/scale
//         q[i] = clamp(round(src[i] * inv), -128, 127)
//
// 2D dispatch: (n_new, n_kv_heads). Each work-item handles one
// (i, h) pair → one full row of `head_dim` weights.
void rsl_kv_quantize_q8_0_store_usm(
    rsl_stream* s,
    const float* src_usm,      // [n_new × n_kv_heads × head_dim] f32
    int8_t* q_dst_usm,         // [n_kv_heads × max_ctx × head_dim] i8 KV cache
    float* scales_dst_usm,     // [n_kv_heads × max_ctx] f32 KV scales
    int n_new,
    int n_kv_heads,
    int head_dim,
    int max_ctx,
    int kv_len_base
) RSL_FFI_BODY_VOID("rsl_kv_quantize_q8_0_store_usm", {
    if (s == nullptr || src_usm == nullptr || q_dst_usm == nullptr || scales_dst_usm == nullptr) return;
    if (n_new <= 0 || n_kv_heads <= 0 || head_dim <= 0 || max_ctx <= 0) return;
    auto& q = s->q;
    sycl::range<2> global(
        static_cast<std::size_t>(n_new),
        static_cast<std::size_t>(n_kv_heads));
    q.submit([&](sycl::handler& h) {
        h.parallel_for(global, [=](sycl::id<2> id) {
            const int i = static_cast<int>(id[0]);
            const int hd_h = static_cast<int>(id[1]);
            const int pos_i = kv_len_base + i;
            const int row_idx = hd_h * max_ctx + pos_i;
            const float* src = src_usm + (i * n_kv_heads + hd_h) * head_dim;
            int8_t* qdst = q_dst_usm + row_idx * head_dim;
            float max_abs = 0.0f;
            for (int j = 0; j < head_dim; ++j) {
                const float a = sycl::fabs(src[j]);
                if (a > max_abs) max_abs = a;
            }
            if (max_abs == 0.0f) {
                for (int j = 0; j < head_dim; ++j) qdst[j] = 0;
                scales_dst_usm[row_idx] = 1.0f;
                return;
            }
            const float scale = max_abs / 127.0f;
            const float inv = 1.0f / scale;
            for (int j = 0; j < head_dim; ++j) {
                const float r = sycl::round(src[j] * inv);
                const float c = sycl::clamp(r, -128.0f, 127.0f);
                qdst[j] = static_cast<int8_t>(c);
            }
            scales_dst_usm[row_idx] = scale;
        });
    }).wait();
})

// G6: Q6_K block encoder. One work-item per 256-weight super-block.
// Port of `encode_q6_k` from rustllama-gguf/src/encode_k.rs. Block
// layout (210 bytes): ql[128] + qh[64] + scales[16] + d_f16[2].
// Algorithm is fully analytical (no iterative refinement), so the
// per-block work is bounded and deterministic.
void rsl_encode_q6_k_blocks_usm(
    rsl_stream* s,
    const float* src_usm,   // [n_blocks × 256] f32 weights
    uint8_t* dst_usm,       // [n_blocks × 210] encoded bytes
    int n_blocks
) RSL_FFI_BODY_VOID("rsl_encode_q6_k_blocks_usm", {
    if (s == nullptr || src_usm == nullptr || dst_usm == nullptr || n_blocks <= 0) return;
    auto& q = s->q;
    constexpr int QK_K = 256;
    constexpr int BLOCK_BYTES = 210;
    constexpr int N_SUB = 16;  // 16 sub-blocks of 16 weights each
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(
                sycl::range<1>(round_up_to_lws(static_cast<std::size_t>(n_blocks))),
                sycl::range<1>(RSL_LWS)),
            [=](sycl::nd_item<1> it) {
                const int b = static_cast<int>(it.get_global_id(0));
                if (b >= n_blocks) return;
                const float* xs = src_usm + b * QK_K;
                uint8_t* dst = dst_usm + b * BLOCK_BYTES;
                // Stage 1: per-sub-block ideal signed scale = max_val / -32.
                float sub_scales[N_SUB];
                for (int k = 0; k < N_SUB; ++k) {
                    float amax = 0.0f;
                    float max_val = 0.0f;
                    for (int j = 0; j < 16; ++j) {
                        const float x = xs[k * 16 + j];
                        const float a = sycl::fabs(x);
                        if (a > amax) { amax = a; max_val = x; }
                    }
                    sub_scales[k] = (amax == 0.0f) ? 0.0f : max_val / -32.0f;
                }
                // Stage 2: super-block d.
                float max_abs_scale = 0.0f;
                float max_signed_scale = 0.0f;
                for (int k = 0; k < N_SUB; ++k) {
                    const float a = sycl::fabs(sub_scales[k]);
                    if (a > max_abs_scale) { max_abs_scale = a; max_signed_scale = sub_scales[k]; }
                }
                const float iscale = (max_abs_scale > 0.0f) ? (-128.0f / max_signed_scale) : 0.0f;
                const float d_super = (iscale != 0.0f) ? (1.0f / iscale) : 0.0f;
                // Quantize per-sub-block scales as i8.
                int8_t scales_q[N_SUB];
                for (int k = 0; k < N_SUB; ++k) {
                    const float r = sycl::round(iscale * sub_scales[k]);
                    const float c = sycl::clamp(r, -128.0f, 127.0f);
                    scales_q[k] = static_cast<int8_t>(c);
                }
                // Stage 3: quantize 6-bit weights.
                int32_t q_signed[QK_K];
                for (int k = 0; k < N_SUB; ++k) {
                    const float s_q = static_cast<float>(scales_q[k]);
                    const float dl = d_super * s_q;
                    const float idl = (dl != 0.0f) ? (1.0f / dl) : 0.0f;
                    const int base = k * 16;
                    for (int l = 0; l < 16; ++l) {
                        const float r = sycl::round(xs[base + l] * idl);
                        const float c = sycl::clamp(r, -32.0f, 31.0f);
                        q_signed[base + l] = static_cast<int32_t>(c);
                    }
                }
                // Pack ql + qh.
                uint8_t* ql = dst;          // 128 bytes
                uint8_t* qh = dst + 128;    // 64 bytes
                for (int i = 0; i < 128; ++i) ql[i] = 0;
                for (int i = 0; i < 64; ++i) qh[i] = 0;
                for (int n = 0; n < 2; ++n) {
                    for (int l = 0; l < 32; ++l) {
                        const uint32_t q1 = static_cast<uint32_t>(q_signed[n * 128 + l] + 32);
                        const uint32_t q2 = static_cast<uint32_t>(q_signed[n * 128 + 32 + l] + 32);
                        const uint32_t q3 = static_cast<uint32_t>(q_signed[n * 128 + 64 + l] + 32);
                        const uint32_t q4 = static_cast<uint32_t>(q_signed[n * 128 + 96 + l] + 32);
                        ql[64 * n + l]      = static_cast<uint8_t>((q1 & 0x0F) | ((q3 & 0x0F) << 4));
                        ql[64 * n + l + 32] = static_cast<uint8_t>((q2 & 0x0F) | ((q4 & 0x0F) << 4));
                        const uint32_t qh_byte =
                            ((q1 >> 4) & 0x03)
                            | (((q2 >> 4) & 0x03) << 2)
                            | (((q3 >> 4) & 0x03) << 4)
                            | (((q4 >> 4) & 0x03) << 6);
                        qh[32 * n + l] = static_cast<uint8_t>(qh_byte);
                    }
                }
                // Store i8 sub-scales at offset 128 + 64 = 192.
                uint8_t* scales_out = dst + 192;
                for (int k = 0; k < N_SUB; ++k) {
                    scales_out[k] = static_cast<uint8_t>(scales_q[k]);
                }
                // f16 super-block d at offset 208.
                const uint16_t d_bits = f32_to_bits(d_super);
                dst[208] = static_cast<uint8_t>(d_bits & 0xFF);
                dst[209] = static_cast<uint8_t>((d_bits >> 8) & 0xFF);
            });
    }).wait();
})

} // close extern "C" — templates below need C++ linkage

// G6: iterative asymmetric sub-block quantizer for Q4_K/Q5_K.
// Mirrors `make_qkx2_quants_asym<NMAX>` in encode_k.rs: 20-step
// perturbation walk, weighted-LS refit, accept on improved L2.
// Templated on NMAX (15 for Q4_K, 31 for Q5_K) and N (sub-block
// size = 32 weights). Stack-allocates the candidate arrays so it
// runs entirely on the GPU work-item without heap traffic.
template <int NMAX, int N>
inline void make_qkx2_quants_asym_sycl(
    const float* xs,
    uint8_t* out_q,
    float& out_d,
    float& out_min
) {
    constexpr float nmax_f = static_cast<float>(NMAX);
    // Find min/max.
    float mn = xs[0];
    float mx = xs[0];
    for (int i = 1; i < N; ++i) {
        if (xs[i] < mn) mn = xs[i];
        if (xs[i] > mx) mx = xs[i];
    }
    if (mx == mn) {
        const float m = (mn < 0.0f) ? -mn : 0.0f;
        for (int i = 0; i < N; ++i) out_q[i] = 0;
        out_d = 0.0f;
        out_min = m;
        return;
    }
    float min_used = (mn <= 0.0f) ? -mn : 0.0f;
    float d = (mn <= 0.0f) ? ((mx - mn) / nmax_f) : (mx / nmax_f);
    float id = 1.0f / d;
    uint8_t best_l[N];
    for (int i = 0; i < N; ++i) {
        const float r = sycl::round((xs[i] + min_used) * id);
        const float c = sycl::clamp(r, 0.0f, nmax_f);
        best_l[i] = static_cast<uint8_t>(c);
    }
    float best_d = d;
    float best_min = min_used;
    float best_err = 0.0f;
    for (int i = 0; i < N; ++i) {
        const float v = best_d * static_cast<float>(best_l[i]) - best_min;
        const float diff = v - xs[i];
        best_err += diff * diff;
    }
    constexpr int NSTEP = 20;
    constexpr float RMIN = -1.0f;
    constexpr float RDELTA = 0.1f;
    for (int step = 0; step < NSTEP; ++step) {
        const float factor = RMIN + static_cast<float>(step) * RDELTA;
        if (factor == 0.0f) continue;
        const float inv_scale_try = id * factor;
        uint8_t l_try[N];
        for (int i = 0; i < N; ++i) {
            const float r = sycl::round((xs[i] + min_used) * inv_scale_try);
            l_try[i] = static_cast<uint8_t>(sycl::clamp(r, 0.0f, nmax_f));
        }
        // Refit (d, min) via weighted LS, weights = 1.
        float sum_l = 0.0f, sum_l2 = 0.0f, sum_x = 0.0f, sum_lx = 0.0f;
        for (int i = 0; i < N; ++i) {
            const float li = static_cast<float>(l_try[i]);
            sum_l  += li;
            sum_l2 += li * li;
            sum_x  += xs[i];
            sum_lx += li * xs[i];
        }
        constexpr float n_f = static_cast<float>(N);
        const float denom = n_f * sum_l2 - sum_l * sum_l;
        if (sycl::fabs(denom) < 1e-12f) continue;
        const float refit_d = (n_f * sum_lx - sum_l * sum_x) / denom;
        const float b_val   = (sum_l2 * sum_x - sum_l * sum_lx) / denom;
        const float refit_min_raw = -b_val;
        if (refit_d <= 0.0f) continue;
        const float m_clamped = sycl::fmax(refit_min_raw, 0.0f);
        const float refit_inv = 1.0f / refit_d;
        uint8_t l_refit[N];
        for (int i = 0; i < N; ++i) {
            const float r = sycl::round((xs[i] + m_clamped) * refit_inv);
            l_refit[i] = static_cast<uint8_t>(sycl::clamp(r, 0.0f, nmax_f));
        }
        float err = 0.0f;
        for (int i = 0; i < N; ++i) {
            const float v = refit_d * static_cast<float>(l_refit[i]) - m_clamped;
            const float diff = v - xs[i];
            err += diff * diff;
        }
        if (err < best_err) {
            best_err = err;
            best_d = refit_d;
            best_min = m_clamped;
            for (int i = 0; i < N; ++i) best_l[i] = l_refit[i];
            id = refit_inv;
            d = refit_d;
            min_used = m_clamped;
        }
    }
    for (int i = 0; i < N; ++i) out_q[i] = best_l[i];
    out_d = best_d;
    out_min = best_min;
}

// G6: shared scale-packing helper for Q4_K and Q5_K. Mirror of
// `pack_q4k_q5k_scales` in encode_k.rs. `sc` + `mn` are 8-element
// 6-bit unsigned arrays; `out` is 12 bytes.
inline void pack_q4k_q5k_scales_sycl(
    const uint8_t* sc,
    const uint8_t* mn,
    uint8_t* out
) {
    for (int i = 0; i < 12; ++i) out[i] = 0;
    for (int j = 0; j < 8; ++j) {
        if (j < 4) {
            out[j]     = sc[j] & 0x3Fu;
            out[j + 4] = mn[j] & 0x3Fu;
        } else {
            out[j + 4] |= sc[j] & 0x0Fu;
            out[j - 4] |= static_cast<uint8_t>((sc[j] >> 4) << 6);
            out[j + 4] |= static_cast<uint8_t>((mn[j] & 0x0Fu) << 4);
            out[j]     |= static_cast<uint8_t>((mn[j] >> 4) << 6);
        }
    }
}

extern "C" {

// G6: Q4_K block encoder. One work-item per 256-weight super-block.
// Block layout (144 bytes): d_f16[2] + dmin_f16[2] + scales[12] + qs[128].
void rsl_encode_q4_k_blocks_usm(
    rsl_stream* s,
    const float* src_usm,
    uint8_t* dst_usm,
    int n_blocks
) RSL_FFI_BODY_VOID("rsl_encode_q4_k_blocks_usm", {
    if (s == nullptr || src_usm == nullptr || dst_usm == nullptr || n_blocks <= 0) return;
    auto& q = s->q;
    constexpr int QK_K = 256;
    constexpr int BLOCK_BYTES = 144;
    constexpr int N_SUB = 8;  // 8 sub-blocks of 32 weights
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(
                sycl::range<1>(round_up_to_lws(static_cast<std::size_t>(n_blocks))),
                sycl::range<1>(RSL_LWS)),
            [=](sycl::nd_item<1> it) {
                const int b = static_cast<int>(it.get_global_id(0));
                if (b >= n_blocks) return;
                const float* xs = src_usm + b * QK_K;
                uint8_t* dst = dst_usm + b * BLOCK_BYTES;
                // Stage 1: per-sub-block (d_sub, m_sub) + 4-bit q.
                float sub_d[N_SUB];
                float sub_m[N_SUB];
                uint8_t q_all[QK_K];
                for (int k = 0; k < N_SUB; ++k) {
                    make_qkx2_quants_asym_sycl<15, 32>(
                        xs + k * 32, q_all + k * 32, sub_d[k], sub_m[k]);
                }
                // Stage 2: super-block d / dmin.
                float max_d = 0.0f;
                float max_m = 0.0f;
                for (int k = 0; k < N_SUB; ++k) {
                    if (sub_d[k] > max_d) max_d = sub_d[k];
                    if (sub_m[k] > max_m) max_m = sub_m[k];
                }
                const float d_super = max_d / 63.0f;
                const float dmin_super = max_m / 63.0f;
                const float id_super = (d_super > 0.0f) ? (1.0f / d_super) : 0.0f;
                const float im_super = (dmin_super > 0.0f) ? (1.0f / dmin_super) : 0.0f;
                uint8_t sc[8], mn[8];
                for (int k = 0; k < N_SUB; ++k) {
                    const float r_sc = sycl::round(sub_d[k] * id_super);
                    sc[k] = static_cast<uint8_t>(sycl::clamp(r_sc, 0.0f, 63.0f));
                    const float r_mn = sycl::round(sub_m[k] * im_super);
                    mn[k] = static_cast<uint8_t>(sycl::clamp(r_mn, 0.0f, 63.0f));
                }
                // Stage 3: header (d_f16, dmin_f16) + packed scales.
                const uint16_t d_bits = f32_to_bits(d_super);
                const uint16_t dmin_bits = f32_to_bits(dmin_super);
                dst[0] = static_cast<uint8_t>(d_bits & 0xFF);
                dst[1] = static_cast<uint8_t>((d_bits >> 8) & 0xFF);
                dst[2] = static_cast<uint8_t>(dmin_bits & 0xFF);
                dst[3] = static_cast<uint8_t>((dmin_bits >> 8) & 0xFF);
                pack_q4k_q5k_scales_sycl(sc, mn, dst + 4);
                // Stage 4: pack 256 4-bit q values into 128 qs bytes
                // (group-of-64: 32 lows + 32 highs).
                uint8_t* qs = dst + 16;
                for (int group = 0; group < 4; ++group) {
                    uint8_t* q_chunk = qs + group * 32;
                    const int g_base = group * 64;
                    for (int l = 0; l < 32; ++l) {
                        const uint8_t lo = q_all[g_base + l] & 0x0Fu;
                        const uint8_t hi = q_all[g_base + 32 + l] & 0x0Fu;
                        q_chunk[l] = static_cast<uint8_t>(lo | (hi << 4));
                    }
                }
            });
    }).wait();
})

// G6: Q5_K block encoder. Same shape as Q4_K but with NMAX=31 (5-bit
// q values) and an extra `qh` byte array for the 5th bit per weight.
// Block layout (176 bytes): d_f16[2] + dmin_f16[2] + scales[12] +
// qh[32] + qs[128].
void rsl_encode_q5_k_blocks_usm(
    rsl_stream* s,
    const float* src_usm,
    uint8_t* dst_usm,
    int n_blocks
) RSL_FFI_BODY_VOID("rsl_encode_q5_k_blocks_usm", {
    if (s == nullptr || src_usm == nullptr || dst_usm == nullptr || n_blocks <= 0) return;
    auto& q = s->q;
    constexpr int QK_K = 256;
    constexpr int BLOCK_BYTES = 176;
    constexpr int N_SUB = 8;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(
                sycl::range<1>(round_up_to_lws(static_cast<std::size_t>(n_blocks))),
                sycl::range<1>(RSL_LWS)),
            [=](sycl::nd_item<1> it) {
                const int b = static_cast<int>(it.get_global_id(0));
                if (b >= n_blocks) return;
                const float* xs = src_usm + b * QK_K;
                uint8_t* dst = dst_usm + b * BLOCK_BYTES;
                float sub_d[N_SUB];
                float sub_m[N_SUB];
                uint8_t q_all[QK_K];
                for (int k = 0; k < N_SUB; ++k) {
                    make_qkx2_quants_asym_sycl<31, 32>(
                        xs + k * 32, q_all + k * 32, sub_d[k], sub_m[k]);
                }
                float max_d = 0.0f;
                float max_m = 0.0f;
                for (int k = 0; k < N_SUB; ++k) {
                    if (sub_d[k] > max_d) max_d = sub_d[k];
                    if (sub_m[k] > max_m) max_m = sub_m[k];
                }
                const float d_super = max_d / 63.0f;
                const float dmin_super = max_m / 63.0f;
                const float id_super = (d_super > 0.0f) ? (1.0f / d_super) : 0.0f;
                const float im_super = (dmin_super > 0.0f) ? (1.0f / dmin_super) : 0.0f;
                uint8_t sc[8], mn[8];
                for (int k = 0; k < N_SUB; ++k) {
                    const float r_sc = sycl::round(sub_d[k] * id_super);
                    sc[k] = static_cast<uint8_t>(sycl::clamp(r_sc, 0.0f, 63.0f));
                    const float r_mn = sycl::round(sub_m[k] * im_super);
                    mn[k] = static_cast<uint8_t>(sycl::clamp(r_mn, 0.0f, 63.0f));
                }
                const uint16_t d_bits = f32_to_bits(d_super);
                const uint16_t dmin_bits = f32_to_bits(dmin_super);
                dst[0] = static_cast<uint8_t>(d_bits & 0xFF);
                dst[1] = static_cast<uint8_t>((d_bits >> 8) & 0xFF);
                dst[2] = static_cast<uint8_t>(dmin_bits & 0xFF);
                dst[3] = static_cast<uint8_t>((dmin_bits >> 8) & 0xFF);
                pack_q4k_q5k_scales_sycl(sc, mn, dst + 4);
                // qh + qs at offsets 16 / 48.
                uint8_t* qh = dst + 16;        // 32 bytes
                uint8_t* qs = dst + 16 + 32;   // 128 bytes
                for (int i = 0; i < 32; ++i) qh[i] = 0;
                for (int group = 0; group < 4; ++group) {
                    uint8_t* q_chunk = qs + group * 32;
                    const int g_base = group * 64;
                    for (int l = 0; l < 32; ++l) {
                        const uint8_t lo_full = q_all[g_base + l];
                        const uint8_t hi_full = q_all[g_base + 32 + l];
                        q_chunk[l] = static_cast<uint8_t>((lo_full & 0x0Fu) | ((hi_full & 0x0Fu) << 4));
                        if ((lo_full & 0x10u) != 0) qh[l] |= static_cast<uint8_t>(1u << (group * 2));
                        if ((hi_full & 0x10u) != 0) qh[l] |= static_cast<uint8_t>(1u << (group * 2 + 1));
                    }
                }
            });
    }).wait();
})

// G6: Q3_K block encoder. One work-item per 256-weight super-block.
// Port of `encode_q3_k` + `pack_q3k_scales` from encode_k.rs. Block
// layout (110 bytes): hmask[32] + qs[64] + scales[12] + d_f16[2].
// Analytical (no iterative refinement) so per-block work is bounded.
void rsl_encode_q3_k_blocks_usm(
    rsl_stream* s,
    const float* src_usm,   // [n_blocks × 256] f32 weights
    uint8_t* dst_usm,       // [n_blocks × 110] encoded bytes
    int n_blocks
) RSL_FFI_BODY_VOID("rsl_encode_q3_k_blocks_usm", {
    if (s == nullptr || src_usm == nullptr || dst_usm == nullptr || n_blocks <= 0) return;
    auto& q = s->q;
    constexpr int QK_K = 256;
    constexpr int BLOCK_BYTES = 110;
    constexpr int N_SUB = 16;  // 16 sub-blocks of 16 weights each
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(
                sycl::range<1>(round_up_to_lws(static_cast<std::size_t>(n_blocks))),
                sycl::range<1>(RSL_LWS)),
            [=](sycl::nd_item<1> it) {
                const int b = static_cast<int>(it.get_global_id(0));
                if (b >= n_blocks) return;
                const float* xs = src_usm + b * QK_K;
                uint8_t* dst = dst_usm + b * BLOCK_BYTES;
                // Stage 1: per-sub-block analytical signed scale = max_val / -4.
                float sub_scales[N_SUB];
                for (int k = 0; k < N_SUB; ++k) {
                    float amax = 0.0f;
                    float max_val = 0.0f;
                    for (int j = 0; j < 16; ++j) {
                        const float x = xs[k * 16 + j];
                        const float a = sycl::fabs(x);
                        if (a > amax) { amax = a; max_val = x; }
                    }
                    sub_scales[k] = (amax == 0.0f) ? 0.0f : max_val / -4.0f;
                }
                // Stage 2: super-block d so sub-scales fit in i8 [-32, 31].
                float signed_max = 0.0f;
                for (int k = 0; k < N_SUB; ++k) {
                    if (sycl::fabs(sub_scales[k]) > sycl::fabs(signed_max)) {
                        signed_max = sub_scales[k];
                    }
                }
                const float iscale = (signed_max != 0.0f) ? (-32.0f / signed_max) : 0.0f;
                const float d_super = (iscale != 0.0f) ? (1.0f / iscale) : 0.0f;
                int8_t scales_q[N_SUB];
                for (int k = 0; k < N_SUB; ++k) {
                    const float r = sycl::round(iscale * sub_scales[k]);
                    const float c = sycl::clamp(r, -32.0f, 31.0f);
                    scales_q[k] = static_cast<int8_t>(c);
                }
                // Stage 3: per-weight signed 3-bit q in [-4, 3].
                int32_t q_signed[QK_K];
                for (int k = 0; k < N_SUB; ++k) {
                    const float s_q = static_cast<float>(scales_q[k]);
                    const float dl = d_super * s_q;
                    const float idl = (dl != 0.0f) ? (1.0f / dl) : 0.0f;
                    const int base = k * 16;
                    for (int l = 0; l < 16; ++l) {
                        const float r = sycl::round(xs[base + l] * idl);
                        const float c = sycl::clamp(r, -4.0f, 3.0f);
                        q_signed[base + l] = static_cast<int32_t>(c);
                    }
                }
                // Stage 4: pack hmask + qs in the dequant-matching layout.
                uint8_t* hmask = dst;          // 32 bytes
                uint8_t* qs    = dst + 32;     // 64 bytes
                for (int i = 0; i < 32; ++i) hmask[i] = 0;
                for (int i = 0; i < 64; ++i) qs[i] = 0;
                int q_cursor = 0;
                uint8_t m = 1;
                for (int chunk = 0; chunk < 2; ++chunk) {
                    int shift = 0;
                    for (int jj = 0; jj < 4; ++jj) {
                        const int y_base_a = chunk * 128 + jj * 32;
                        for (int l = 0; l < 16; ++l) {
                            const uint32_t u3 = static_cast<uint32_t>(q_signed[y_base_a + l] + 4);
                            const uint32_t lo2 = u3 & 0x03u;
                            const uint32_t hi1 = (u3 >> 2) & 0x01u;
                            qs[q_cursor + l] |= static_cast<uint8_t>(lo2 << shift);
                            if (hi1 != 0) hmask[l] |= m;
                        }
                        const int y_base_b = chunk * 128 + jj * 32 + 16;
                        for (int l = 0; l < 16; ++l) {
                            const uint32_t u3 = static_cast<uint32_t>(q_signed[y_base_b + l] + 4);
                            const uint32_t lo2 = u3 & 0x03u;
                            const uint32_t hi1 = (u3 >> 2) & 0x01u;
                            qs[q_cursor + l + 16] |= static_cast<uint8_t>(lo2 << shift);
                            if (hi1 != 0) hmask[l + 16] |= m;
                        }
                        shift += 2;
                        m = static_cast<uint8_t>(m << 1);
                    }
                    q_cursor += 32;
                }
                // Stage 5: pack scales via pack_q3k_scales recipe.
                // s6[i] = clamp(scales_q[i] + 32, 0, 63).
                uint8_t s6[16];
                for (int i = 0; i < N_SUB; ++i) {
                    const int raw = static_cast<int>(scales_q[i]) + 32;
                    s6[i] = static_cast<uint8_t>(sycl::clamp(raw, 0, 63));
                }
                uint8_t* scales_out = dst + 96;  // 12 bytes
                for (int jj = 0; jj < 4; ++jj) {
                    const uint32_t v0  = s6[jj];
                    const uint32_t v4  = s6[jj + 4];
                    const uint32_t v8  = s6[jj + 8];
                    const uint32_t v12 = s6[jj + 12];
                    scales_out[jj]     = static_cast<uint8_t>((v0 & 0x0F) | ((v8 & 0x0F) << 4));
                    scales_out[jj + 4] = static_cast<uint8_t>((v4 & 0x0F) | ((v12 & 0x0F) << 4));
                    scales_out[jj + 8] = static_cast<uint8_t>(
                        ((v0  >> 4) & 0x3)
                        | (((v4  >> 4) & 0x3) << 2)
                        | (((v8  >> 4) & 0x3) << 4)
                        | (((v12 >> 4) & 0x3) << 6));
                }
                // f16 super-block d at offset 108.
                const uint16_t d_bits = f32_to_bits(d_super);
                dst[108] = static_cast<uint8_t>(d_bits & 0xFF);
                dst[109] = static_cast<uint8_t>((d_bits >> 8) & 0xFF);
            });
    }).wait();
})

// USM-resident embedding gather. The `ids` array can be host-side
// since it's small (n_ids per call, typically 1 for decode). The
// kernel reads ids by value (captured by lambda copy) so no extra
// USM allocation is needed for the index list.
void rsl_embedding_lookup_usm(rsl_stream* s,
                              const uint16_t* table_usm,
                              const int32_t* ids,
                              uint16_t* out_usm,
                              int n_ids, int d) RSL_FFI_BODY_VOID("rsl_embedding_lookup_usm", {
    if (s == nullptr || table_usm == nullptr || ids == nullptr || out_usm == nullptr) return;
    if (n_ids <= 0 || d <= 0) return;
    auto& q = s->q;
    // Reuse the stream-cached device-USM ID scratch. On decode this
    // is hit once (n_ids=1) at engine load and the same allocation
    // services every subsequent token. Grows monotonically: if a
    // prefill chunk asks for a larger n_ids than we've seen, we
    // resize and keep the new capacity.
    const std::size_t need = static_cast<std::size_t>(n_ids);
    if (s->embed_ids_capacity < need) {
        if (s->embed_ids_scratch != nullptr) {
            sycl::free(s->embed_ids_scratch, q);
            s->embed_ids_scratch = nullptr;
            s->embed_ids_capacity = 0;
        }
        s->embed_ids_scratch = sycl::malloc_device<int32_t>(need, q);
        if (s->embed_ids_scratch == nullptr) return;
        s->embed_ids_capacity = need;
    }
    int32_t* ids_d = s->embed_ids_scratch;
    q.memcpy(ids_d, ids, need * sizeof(int32_t));
    q.wait();
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(round_up_to_lws(static_cast<std::size_t>(n_ids))),
                              sycl::range<1>(RSL_LWS)),
            [=](sycl::nd_item<1> it) {
            const int i = static_cast<int>(it.get_global_id(0));
            if (i >= n_ids) return;
            const int32_t row = ids_d[i];
            const std::size_t out_off = static_cast<std::size_t>(i) * d;
            if (row < 0) {
                for (int j = 0; j < d; ++j) {
                    out_usm[out_off + j] = 0;
                }
            } else {
                const std::size_t src_off = static_cast<std::size_t>(row) * d;
                for (int j = 0; j < d; ++j) {
                    out_usm[out_off + j] = table_usm[src_off + j];
                }
            }
        });
    }).wait();
})

// USM-resident rmsnorm. Pointers MUST be USM allocations (e.g. from
// rsl_usm_alloc_shared) backed by the same context as `s->q`. No
// internal alloc/copy — the kernel reads/writes the supplied
// pointers directly. On integrated GPUs (shared LPDDR) this means
// zero host↔device data movement for the call. The submit + wait
// is the only synchronization; the runtime handles page migration
// on discrete GPUs.
void rsl_rmsnorm_usm(rsl_stream* s,
                     const uint16_t* x_usm, const uint16_t* w_usm,
                     uint16_t* y_usm,
                     int n_rows, int d, float eps) RSL_FFI_BODY_VOID("rsl_rmsnorm_usm", {
    if (s == nullptr || x_usm == nullptr || w_usm == nullptr || y_usm == nullptr) return;
    if (n_rows <= 0 || d <= 0) return;
    auto& q = s->q;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(round_up_to_lws(n_rows)),
                              sycl::range<1>(RSL_LWS)),
            [=](sycl::nd_item<1> it) {
            const int r = static_cast<int>(it.get_global_id(0));
            if (r >= n_rows) return;
            const int base = r * d;
            float sum_sq = 0.0f;
            for (int i = 0; i < d; ++i) {
                float v = bits_to_f32(x_usm[base + i]);
                sum_sq += v * v;
            }
            float scale = 1.0f / sycl::sqrt(sum_sq / static_cast<float>(d) + eps);
            for (int i = 0; i < d; ++i) {
                float v = bits_to_f32(x_usm[base + i]);
                float wv = bits_to_f32(w_usm[i]);
                y_usm[base + i] = f32_to_bits(v * scale * wv);
            }
        });
    }).wait();
})

// Fused RMSNorm + residual add. Computes
// `y_usm[i] = (x_usm[i] / norm(x_row)) * w_usm[i] + residual_usm[i]`
// in one kernel pass. Useful for architectures that do `norm(x) +
// residual` (some post-norm variants); Llama-family pre-norm uses
// the `rsl_add_rmsnorm_usm` variant below instead.
//
// All four pointers MUST be USM allocations on the same context.
// `residual_usm` is read-only; `y_usm` is the only write target.
// `residual_usm` MAY alias `x_usm` when the caller wants in-place
// behavior (the kernel reads each value before writing).
void rsl_rmsnorm_residual_usm(rsl_stream* s,
                              const uint16_t* x_usm, const uint16_t* w_usm,
                              const uint16_t* residual_usm,
                              uint16_t* y_usm,
                              int n_rows, int d, float eps)
    RSL_FFI_BODY_VOID("rsl_rmsnorm_residual_usm", {
    if (s == nullptr || x_usm == nullptr || w_usm == nullptr
        || residual_usm == nullptr || y_usm == nullptr) return;
    if (n_rows <= 0 || d <= 0) return;
    auto& q = s->q;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(round_up_to_lws(n_rows)),
                              sycl::range<1>(RSL_LWS)),
            [=](sycl::nd_item<1> it) {
            const int r = static_cast<int>(it.get_global_id(0));
            if (r >= n_rows) return;
            const int base = r * d;
            float sum_sq = 0.0f;
            for (int i = 0; i < d; ++i) {
                float v = bits_to_f32(x_usm[base + i]);
                sum_sq += v * v;
            }
            float scale = 1.0f / sycl::sqrt(sum_sq / static_cast<float>(d) + eps);
            for (int i = 0; i < d; ++i) {
                float v = bits_to_f32(x_usm[base + i]);
                float wv = bits_to_f32(w_usm[i]);
                float rv = bits_to_f32(residual_usm[base + i]);
                y_usm[base + i] = f32_to_bits(v * scale * wv + rv);
            }
        });
    }).wait();
})

// Fused "add residual + RMSNorm" — the Llama-family pre-norm pattern.
// Computes:
//   hidden_usm[i] = hidden_usm[i] + branch_usm[i]              (residual)
//   y_norm_usm[i] = rmsnorm(hidden_usm_row)[i] * w_usm[i]      (pre-norm)
//
// `hidden_usm` is read AND written: holds the running residual
// stream. `branch_usm` is the attention or FFN projection output.
// `y_norm_usm` is the pre-norm input for the NEXT block's QKV /
// gate-up matvec. One kernel pass replaces the historical
// (add_inplace → kernel barrier → rmsnorm) pair; on a 32-layer
// decode this eliminates ~64 kernel launches + barriers per token
// (post-attn and post-FFN norms in every block).
//
// All four pointers MUST be USM allocations on the same context.
void rsl_add_rmsnorm_usm(rsl_stream* s,
                         uint16_t* hidden_usm,
                         const uint16_t* branch_usm,
                         const uint16_t* w_usm,
                         uint16_t* y_norm_usm,
                         int n_rows, int d, float eps)
    RSL_FFI_BODY_VOID("rsl_add_rmsnorm_usm", {
    if (s == nullptr || hidden_usm == nullptr || branch_usm == nullptr
        || w_usm == nullptr || y_norm_usm == nullptr) return;
    if (n_rows <= 0 || d <= 0) return;
    auto& q = s->q;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(round_up_to_lws(n_rows)),
                              sycl::range<1>(RSL_LWS)),
            [=](sycl::nd_item<1> it) {
            const int r = static_cast<int>(it.get_global_id(0));
            if (r >= n_rows) return;
            const int base = r * d;
            // Pass 1: residual add + sum of squares (one walk).
            float sum_sq = 0.0f;
            for (int i = 0; i < d; ++i) {
                float a = bits_to_f32(hidden_usm[base + i]);
                float b = bits_to_f32(branch_usm[base + i]);
                float s_val = a + b;
                hidden_usm[base + i] = f32_to_bits(s_val);
                sum_sq += s_val * s_val;
            }
            // Pass 2: rmsnorm scale-and-mul. Reads the residual sum
            // back from `hidden_usm` (just written above) — no
            // need to re-add or temp-buffer.
            float scale = 1.0f / sycl::sqrt(sum_sq / static_cast<float>(d) + eps);
            for (int i = 0; i < d; ++i) {
                float s_val = bits_to_f32(hidden_usm[base + i]);
                float wv = bits_to_f32(w_usm[i]);
                y_norm_usm[base + i] = f32_to_bits(s_val * scale * wv);
            }
        });
    }).wait();
})

// F32-precision variant of `rsl_add_rmsnorm_usm`. Same algorithm
// (residual add + rmsnorm in one pass), but inputs/outputs are F32
// instead of F16-bits — so callers feeding F32 USM scratch (e.g. the
// H6 chained dispatcher) skip the F32→F16→F32 round-trip that the
// F16 variant requires.
//
// Topology mirrors the F16 variant exactly (`round_up_to_lws(n_rows)`
// global, RSL_LWS local, one row per work-item) — proven on the
// post-attn / post-FFN norm sites in decode. `hidden_usm` is read AND
// written (holds running residual stream); `branch_usm` is read-only
// (matvec output, e.g. attn projection). All four pointers MUST be
// USM allocations on the same context.
void rsl_add_rmsnorm_f32_usm(rsl_stream* s,
                              float* hidden_usm,
                              const float* branch_usm,
                              const float* w_usm,
                              float* y_norm_usm,
                              int n_rows, int d, float eps)
    RSL_FFI_BODY_VOID("rsl_add_rmsnorm_f32_usm", {
    if (s == nullptr || hidden_usm == nullptr || branch_usm == nullptr
        || w_usm == nullptr || y_norm_usm == nullptr) return;
    if (n_rows <= 0 || d <= 0) return;
    auto& q = s->q;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(round_up_to_lws(n_rows)),
                              sycl::range<1>(RSL_LWS)),
            [=](sycl::nd_item<1> it) {
            const int r = static_cast<int>(it.get_global_id(0));
            if (r >= n_rows) return;
            const int base = r * d;
            // Pass 1: residual add + sum of squares (one walk).
            float sum_sq = 0.0f;
            for (int i = 0; i < d; ++i) {
                const float a = hidden_usm[base + i];
                const float b = branch_usm[base + i];
                const float s_val = a + b;
                hidden_usm[base + i] = s_val;
                sum_sq += s_val * s_val;
            }
            // Pass 2: rmsnorm scale-and-mul. Reads residual sum back
            // from hidden_usm (just written above).
            const float scale = 1.0f / sycl::sqrt(sum_sq / static_cast<float>(d) + eps);
            for (int i = 0; i < d; ++i) {
                y_norm_usm[base + i] = hidden_usm[base + i] * scale * w_usm[i];
            }
        });
    }).wait();
})

// =========================================================================
// H6: Fused matvec + residual-add + rmsnorm kernels (per packed-quant dtype)
// =========================================================================
//
// Replaces the (matvec_X_packed_f32_usm → add_rmsnorm_usm) two-kernel
// sequence at post-attention norm sites in llama_arch.rs. One SYCL
// submission does:
//   1. matvec_dot[m] = dequant(W[m, :]) · attn[:]             (per output m)
//   2. fused_val[m]  = matvec_dot[m] + residual[m]            (residual add)
//   3. sum_sq = Σ_m fused_val[m]²                             (workgroup reduce)
//   4. inv_rms = 1 / sqrt(sum_sq / M + eps)
//   5. y_norm[m] = fused_val[m] * inv_rms * w_norm[m]         (per output m)
//
// Topology: 1 workgroup of `LWS` work-items per token. Each work-item
// handles `ceil(M / LWS)` output rows. Phase-1 result is staged in
// SLM (`local_accessor<float, 1>` sized to M). sycl::reduce_over_group
// collapses the local sum_sq across the workgroup. Phase-3 reads the
// staged SLM values + applies the normalize-and-scale.
//
// Decode hot path: M = d_model = 4096-8192. SLM cost = 16-32 KB per
// workgroup — well within Iris Xe's 64 KB-per-workgroup budget. LWS
// of 128 gives ~32-64 elements per work-item in Phase 1, matching the
// matvec_X_packed_f32_usm shape closely.
//
// Caller contract: residual is read-only; y_norm is the only write
// target. The matvec output is NOT exposed (callers that need it
// should call the unfused matvec + add_rmsnorm pair instead).
// `K` must be a multiple of the dtype's block size; same alignment
// requirement as the unfused matvec.

}  // extern "C" — close before Q4_K matvec+add+rmsnorm template

// `hidden_usm` is read (running residual stream) AND written (the
// post-attn residual sum, needed by the next FFN-residual add).
// `y_norm_usm` receives the normalized + scaled output (the next
// block's gate/up matvec input). Mirror of the F32-path
// `add_rmsnorm` writeback contract.
template <std::size_t LWS_T>
inline void matvec_q4_k_add_rmsnorm_usm_impl(
    sycl::queue& q,
    const void* w_bytes_usm,
    const float* attn_usm,
    float* hidden_usm,
    const float* w_norm_usm,
    float* y_norm_usm,
    int M, int K, float eps) {
    constexpr std::size_t LWS = LWS_T;
    const int blocks_per_row = K / 256;
    const int bytes_per_row = blocks_per_row * 144;
    const uint8_t* w_bytes = static_cast<const uint8_t*>(w_bytes_usm);
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> slm(sycl::range<1>(static_cast<std::size_t>(M)), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(LWS), sycl::range<1>(LWS)),
            [=](sycl::nd_item<1> it) {
                const int tid = static_cast<int>(it.get_local_id(0));
                const int per_thread = (M + static_cast<int>(LWS) - 1) / static_cast<int>(LWS);
                // Phase 1: per-row Q4_K matvec dot + residual add → SLM
                // (also written back to hidden_usm for the FFN residual).
                float my_sum_sq = 0.0f;
                for (int p = 0; p < per_thread; ++p) {
                    const int m = tid * per_thread + p;
                    if (m >= M) break;
                    const uint8_t* row = w_bytes + m * bytes_per_row;
                    float dot = 0.0f;
                    for (int b = 0; b < blocks_per_row; ++b) {
                        const uint8_t* blk = row + b * 144;
                        const float d_scale = bits_to_f32(static_cast<uint16_t>(blk[0])
                                                          | (static_cast<uint16_t>(blk[1]) << 8));
                        const float d_min = bits_to_f32(static_cast<uint16_t>(blk[2])
                                                         | (static_cast<uint16_t>(blk[3]) << 8));
                        uint8_t sc[8], mn[8];
                        {
                            const uint8_t* sb = blk + 4;
                            for (int j = 0; j < 8; ++j) {
                                if (j < 4) {
                                    sc[j] = sb[j] & 0x3F;
                                    mn[j] = sb[j + 4] & 0x3F;
                                } else {
                                    sc[j] = (sb[j + 4] & 0x0F) | ((sb[j - 4] >> 6) << 4);
                                    mn[j] = (sb[j + 4] >> 4) | ((sb[j] >> 6) << 4);
                                }
                            }
                        }
                        const uint8_t* qs = blk + 16;
                        const int x_base = b * 256;
                        for (int group = 0; group < 4; ++group) {
                            const float d_lo = d_scale * static_cast<float>(sc[group * 2]);
                            const float m_lo = d_min * static_cast<float>(mn[group * 2]);
                            const float d_hi = d_scale * static_cast<float>(sc[group * 2 + 1]);
                            const float m_hi = d_min * static_cast<float>(mn[group * 2 + 1]);
                            const uint8_t* qc = qs + group * 32;
                            const int x_lo_off = x_base + group * 64;
                            const int x_hi_off = x_lo_off + 32;
                            for (int l = 0; l < 32; ++l) {
                                const uint8_t qb = qc[l];
                                dot += (d_lo * static_cast<float>(qb & 0x0F) - m_lo) * attn_usm[x_lo_off + l];
                                dot += (d_hi * static_cast<float>(qb >> 4)   - m_hi) * attn_usm[x_hi_off + l];
                            }
                        }
                    }
                    const float val = dot + hidden_usm[m];
                    slm[m] = val;
                    hidden_usm[m] = val;
                    my_sum_sq += val * val;
                }
                // Phase 2: workgroup-wide sum_sq reduction
                it.barrier(sycl::access::fence_space::local_space);
                const float total_sum_sq = sycl::reduce_over_group(
                    it.get_group(), my_sum_sq, sycl::plus<float>());
                const float inv_rms = 1.0f / sycl::sqrt(total_sum_sq / static_cast<float>(M) + eps);
                // Phase 3: normalize + scale → y_norm
                for (int p = 0; p < per_thread; ++p) {
                    const int m = tid * per_thread + p;
                    if (m >= M) break;
                    y_norm_usm[m] = slm[m] * inv_rms * w_norm_usm[m];
                }
            });
    }).wait();
}
extern "C" {

void rsl_matvec_q4_k_add_rmsnorm_usm(rsl_stream* s,
                                     const void* w_bytes_usm,
                                     const float* attn_usm,
                                     float* hidden_usm,
                                     const float* w_norm_usm,
                                     float* y_norm_usm,
                                     int M, int K, float eps,
                                     int lws) RSL_FFI_BODY_VOID(
    "rsl_matvec_q4_k_add_rmsnorm_usm", {
    if (s == nullptr || w_bytes_usm == nullptr || attn_usm == nullptr
        || hidden_usm == nullptr || w_norm_usm == nullptr || y_norm_usm == nullptr) {
        return;
    }
    if (M <= 0 || K <= 0 || (K % 256) != 0) {
        return;
    }
    auto& q = s->q;
    const int eff_lws = (lws <= 0) ? 128 : lws;
    switch (eff_lws) {
        case 32:  matvec_q4_k_add_rmsnorm_usm_impl<32>(q, w_bytes_usm, attn_usm, hidden_usm, w_norm_usm, y_norm_usm, M, K, eps); break;
        case 64:  matvec_q4_k_add_rmsnorm_usm_impl<64>(q, w_bytes_usm, attn_usm, hidden_usm, w_norm_usm, y_norm_usm, M, K, eps); break;
        case 128: matvec_q4_k_add_rmsnorm_usm_impl<128>(q, w_bytes_usm, attn_usm, hidden_usm, w_norm_usm, y_norm_usm, M, K, eps); break;
        case 256: matvec_q4_k_add_rmsnorm_usm_impl<256>(q, w_bytes_usm, attn_usm, hidden_usm, w_norm_usm, y_norm_usm, M, K, eps); break;
        default:  matvec_q4_k_add_rmsnorm_usm_impl<128>(q, w_bytes_usm, attn_usm, hidden_usm, w_norm_usm, y_norm_usm, M, K, eps); break;
    }
})

}  // extern "C" — close before the remaining H6 templates

// H6 scaffold: factor out the SLM-staged workgroup-reduction rmsnorm
// shell shared by all packed-quant dtypes. Each dtype's kernel sets
// up `blocks_per_row` / `bytes_per_row` / `w_bytes`, then the BEGIN
// macro opens the per-row loop (declaring `m` and `dot`), the dtype
// body accumulates `dot`, and END does residual-add + writeback +
// workgroup sum_sq reduce + normalize. See
// matvec_q4_k_add_rmsnorm_usm_impl for the expanded reference form.
#define RSL_H6_BEGIN(LWS) \
    q.submit([&](sycl::handler& h) { \
        sycl::local_accessor<float, 1> slm(sycl::range<1>(static_cast<std::size_t>(M)), h); \
        h.parallel_for( \
            sycl::nd_range<1>(sycl::range<1>(LWS), sycl::range<1>(LWS)), \
            [=](sycl::nd_item<1> it) { \
                const int tid = static_cast<int>(it.get_local_id(0)); \
                const int per_thread = (M + static_cast<int>(LWS) - 1) / static_cast<int>(LWS); \
                float my_sum_sq = 0.0f; \
                for (int p = 0; p < per_thread; ++p) { \
                    const int m = tid * per_thread + p; \
                    if (m >= M) break; \
                    float dot = 0.0f;

#define RSL_H6_END \
                    const float val = dot + hidden_usm[m]; \
                    slm[m] = val; \
                    hidden_usm[m] = val; \
                    my_sum_sq += val * val; \
                } \
                it.barrier(sycl::access::fence_space::local_space); \
                const float total_sum_sq = sycl::reduce_over_group( \
                    it.get_group(), my_sum_sq, sycl::plus<float>()); \
                const float inv_rms = 1.0f / sycl::sqrt(total_sum_sq / static_cast<float>(M) + eps); \
                for (int p = 0; p < per_thread; ++p) { \
                    const int m = tid * per_thread + p; \
                    if (m >= M) break; \
                    y_norm_usm[m] = slm[m] * inv_rms * w_norm_usm[m]; \
                } \
            }); \
    }).wait();

// ---- Q8_0 (block 32, 34 bytes) ----
template <std::size_t LWS_T>
inline void matvec_q8_0_add_rmsnorm_usm_impl(
    sycl::queue& q, const void* w_bytes_usm, const float* attn_usm,
    float* hidden_usm, const float* w_norm_usm, float* y_norm_usm,
    int M, int K, float eps) {
    constexpr std::size_t LWS = LWS_T;
    const int blocks_per_row = K / 32;
    const int bytes_per_row = blocks_per_row * 34;
    const uint8_t* w_bytes = static_cast<const uint8_t*>(w_bytes_usm);
    RSL_H6_BEGIN(LWS)
        const uint8_t* row = w_bytes + m * bytes_per_row;
        for (int blk_i = 0; blk_i < blocks_per_row; ++blk_i) {
            const uint8_t* blk = row + blk_i * 34;
            const float scale = bits_to_f32(static_cast<uint16_t>(blk[0])
                                            | (static_cast<uint16_t>(blk[1]) << 8));
            const int x_off = blk_i * 32;
            for (int dd = 0; dd < 32; ++dd) {
                dot += scale * static_cast<float>(static_cast<int8_t>(blk[2 + dd]))
                             * attn_usm[x_off + dd];
            }
        }
    RSL_H6_END
}

// ---- Q5_K (block 256, 176 bytes) ----
template <std::size_t LWS_T>
inline void matvec_q5_k_add_rmsnorm_usm_impl(
    sycl::queue& q, const void* w_bytes_usm, const float* attn_usm,
    float* hidden_usm, const float* w_norm_usm, float* y_norm_usm,
    int M, int K, float eps) {
    constexpr std::size_t LWS = LWS_T;
    const int blocks_per_row = K / 256;
    const int bytes_per_row = blocks_per_row * 176;
    const uint8_t* w_bytes = static_cast<const uint8_t*>(w_bytes_usm);
    RSL_H6_BEGIN(LWS)
        const uint8_t* row = w_bytes + m * bytes_per_row;
        for (int b = 0; b < blocks_per_row; ++b) {
            const uint8_t* blk = row + b * 176;
            const float d = bits_to_f32(static_cast<uint16_t>(blk[0]) | (static_cast<uint16_t>(blk[1]) << 8));
            const float dmin = bits_to_f32(static_cast<uint16_t>(blk[2]) | (static_cast<uint16_t>(blk[3]) << 8));
            const uint8_t* sb = blk + 4;
            uint8_t sc[8], mn[8];
            for (int j = 0; j < 8; ++j) {
                if (j < 4) { sc[j] = sb[j] & 0x3F; mn[j] = sb[j + 4] & 0x3F; }
                else { sc[j] = (sb[j + 4] & 0x0F) | ((sb[j - 4] >> 6) << 4); mn[j] = (sb[j + 4] >> 4) | ((sb[j] >> 6) << 4); }
            }
            const uint8_t* qh = blk + 16;
            const uint8_t* qs = blk + 48;
            const int x_base = b * 256;
            for (int group = 0; group < 4; ++group) {
                const uint8_t* qc = qs + group * 32;
                const float d_lo = d * static_cast<float>(sc[group * 2]);
                const float m_lo = dmin * static_cast<float>(mn[group * 2]);
                const float d_hi = d * static_cast<float>(sc[group * 2 + 1]);
                const float m_hi = dmin * static_cast<float>(mn[group * 2 + 1]);
                const int bit_lo = group * 2, bit_hi = group * 2 + 1;
                const int x_lo_off = x_base + group * 64, x_hi_off = x_lo_off + 32;
                for (int l = 0; l < 32; ++l) {
                    const uint8_t qb = qc[l], qhb = qh[l];
                    const uint32_t lo = static_cast<uint32_t>(qb & 0x0F) | (static_cast<uint32_t>((qhb >> bit_lo) & 1) << 4);
                    const uint32_t hi = static_cast<uint32_t>(qb >> 4) | (static_cast<uint32_t>((qhb >> bit_hi) & 1) << 4);
                    dot += (d_lo * static_cast<float>(lo) - m_lo) * attn_usm[x_lo_off + l];
                    dot += (d_hi * static_cast<float>(hi) - m_hi) * attn_usm[x_hi_off + l];
                }
            }
        }
    RSL_H6_END
}

// ---- Q6_K (block 256, 210 bytes) ----
template <std::size_t LWS_T>
inline void matvec_q6_k_add_rmsnorm_usm_impl(
    sycl::queue& q, const void* w_bytes_usm, const float* attn_usm,
    float* hidden_usm, const float* w_norm_usm, float* y_norm_usm,
    int M, int K, float eps) {
    constexpr std::size_t LWS = LWS_T;
    const int blocks_per_row = K / 256;
    const int bytes_per_row = blocks_per_row * 210;
    const uint8_t* w_bytes = static_cast<const uint8_t*>(w_bytes_usm);
    RSL_H6_BEGIN(LWS)
        const uint8_t* row = w_bytes + m * bytes_per_row;
        for (int b = 0; b < blocks_per_row; ++b) {
            const uint8_t* blk = row + b * 210;
            const uint8_t* ql = blk;
            const uint8_t* qh = blk + 128;
            const int8_t* scales = reinterpret_cast<const int8_t*>(blk + 192);
            const float d = bits_to_f32(static_cast<uint16_t>(blk[208]) | (static_cast<uint16_t>(blk[209]) << 8));
            const int x_base = b * 256;
            for (int n = 0; n < 2; ++n) {
                for (int l = 0; l < 32; ++l) {
                    const int is = l / 16 + n * 8;
                    const int qhb = qh[32 * n + l];
                    const int q1 = (static_cast<int>(ql[64 * n + l] & 0x0F) | ((qhb >> 0) & 0x03) << 4) - 32;
                    const int q2 = (static_cast<int>(ql[64 * n + l + 32] & 0x0F) | ((qhb >> 2) & 0x03) << 4) - 32;
                    const int q3 = (static_cast<int>(ql[64 * n + l] >> 4) | ((qhb >> 4) & 0x03) << 4) - 32;
                    const int q4 = (static_cast<int>(ql[64 * n + l + 32] >> 4) | ((qhb >> 6) & 0x03) << 4) - 32;
                    const int base = n * 128 + l;
                    dot += d * static_cast<float>(scales[is])     * static_cast<float>(q1) * attn_usm[x_base + base];
                    dot += d * static_cast<float>(scales[is + 2]) * static_cast<float>(q2) * attn_usm[x_base + base + 32];
                    dot += d * static_cast<float>(scales[is + 4]) * static_cast<float>(q3) * attn_usm[x_base + base + 64];
                    dot += d * static_cast<float>(scales[is + 6]) * static_cast<float>(q4) * attn_usm[x_base + base + 96];
                }
            }
        }
    RSL_H6_END
}

// ---- IQ4_NL (block 32, 18 bytes) ----
template <std::size_t LWS_T>
inline void matvec_iq4_nl_add_rmsnorm_usm_impl(
    sycl::queue& q, const void* w_bytes_usm, const float* attn_usm,
    float* hidden_usm, const float* w_norm_usm, float* y_norm_usm,
    int M, int K, float eps) {
    constexpr std::size_t LWS = LWS_T;
    const int blocks_per_row = K / 32;
    const int bytes_per_row = blocks_per_row * 18;
    const uint8_t* w_bytes = static_cast<const uint8_t*>(w_bytes_usm);
    RSL_H6_BEGIN(LWS)
        const uint8_t* row = w_bytes + m * bytes_per_row;
        for (int b = 0; b < blocks_per_row; ++b) {
            const uint8_t* blk = row + b * 18;
            const float d = bits_to_f32(static_cast<uint16_t>(blk[0]) | (static_cast<uint16_t>(blk[1]) << 8));
            const uint8_t* qs = blk + 2;
            const int x_base = b * 32;
            for (int j = 0; j < 16; ++j) {
                const uint8_t qb = qs[j];
                dot += d * static_cast<float>(KVALUES_IQ4XS_SYCL[qb & 0x0F]) * attn_usm[x_base + j];
                dot += d * static_cast<float>(KVALUES_IQ4XS_SYCL[qb >> 4])   * attn_usm[x_base + j + 16];
            }
        }
    RSL_H6_END
}

// ---- IQ4_XS (block 256, 136 bytes) ----
template <std::size_t LWS_T>
inline void matvec_iq4_xs_add_rmsnorm_usm_impl(
    sycl::queue& q, const void* w_bytes_usm, const float* attn_usm,
    float* hidden_usm, const float* w_norm_usm, float* y_norm_usm,
    int M, int K, float eps) {
    constexpr std::size_t LWS = LWS_T;
    const int blocks_per_row = K / 256;
    const int bytes_per_row = blocks_per_row * 136;
    const uint8_t* w_bytes = static_cast<const uint8_t*>(w_bytes_usm);
    RSL_H6_BEGIN(LWS)
        const uint8_t* row = w_bytes + m * bytes_per_row;
        for (int b = 0; b < blocks_per_row; ++b) {
            const uint8_t* blk = row + b * 136;
            const float d = bits_to_f32(static_cast<uint16_t>(blk[0]) | (static_cast<uint16_t>(blk[1]) << 8));
            const uint16_t scales_h = static_cast<uint16_t>(blk[2]) | (static_cast<uint16_t>(blk[3]) << 8);
            const uint8_t* scales_l = blk + 4;
            const uint8_t* qs = blk + 8;
            const int x_base = b * 256;
            for (int ib = 0; ib < 8; ++ib) {
                const uint8_t lo_nibble = (ib % 2 == 0) ? (scales_l[ib / 2] & 0x0F) : (scales_l[ib / 2] >> 4);
                const uint8_t hi_bits = static_cast<uint8_t>((scales_h >> (2 * ib)) & 0x03);
                const int8_t ls = static_cast<int8_t>(static_cast<uint8_t>(lo_nibble | (hi_bits << 4))) - 32;
                const float sub_d = d * static_cast<float>(ls);
                const int q_off = ib * 16, x_off = ib * 32;
                for (int j = 0; j < 16; ++j) {
                    const uint8_t qb = qs[q_off + j];
                    dot += sub_d * static_cast<float>(KVALUES_IQ4XS_SYCL[qb & 0x0F]) * attn_usm[x_base + x_off + j];
                    dot += sub_d * static_cast<float>(KVALUES_IQ4XS_SYCL[qb >> 4])   * attn_usm[x_base + x_off + 16 + j];
                }
            }
        }
    RSL_H6_END
}

// ---- IQ1_S (block 256, 50 bytes) ----
template <std::size_t LWS_T>
inline void matvec_iq1_s_add_rmsnorm_usm_impl(
    sycl::queue& q, const void* w_bytes_usm, const float* attn_usm,
    float* hidden_usm, const float* w_norm_usm, float* y_norm_usm,
    int M, int K, float eps) {
    constexpr std::size_t LWS = LWS_T;
    constexpr float IQ1S_DELTA = 0.125f;
    const int blocks_per_row = K / 256;
    const int bytes_per_row = blocks_per_row * 50;
    const uint8_t* w_bytes = static_cast<const uint8_t*>(w_bytes_usm);
    RSL_H6_BEGIN(LWS)
        const uint8_t* row = w_bytes + m * bytes_per_row;
        for (int b = 0; b < blocks_per_row; ++b) {
            const uint8_t* blk = row + b * 50;
            const float d = bits_to_f32(static_cast<uint16_t>(blk[0]) | (static_cast<uint16_t>(blk[1]) << 8));
            const uint8_t* qs = blk + 2;
            const uint8_t* qh_bytes = blk + 34;
            const int x_base = b * 256;
            for (int ib32 = 0; ib32 < 8; ++ib32) {
                const uint16_t qh = static_cast<uint16_t>(qh_bytes[ib32 * 2]) | (static_cast<uint16_t>(qh_bytes[ib32 * 2 + 1]) << 8);
                const float dl = d * (2.0f * static_cast<float>((qh >> 12) & 7) + 1.0f);
                const float delta = (qh & 0x8000) ? (-1.0f - IQ1S_DELTA) : (-1.0f + IQ1S_DELTA);
                const int x_off = x_base + ib32 * 32;
                for (int l = 0; l < 4; ++l) {
                    const std::uint32_t idx = static_cast<std::uint32_t>(qs[ib32 * 4 + l]) | ((static_cast<std::uint32_t>(qh >> (3 * l)) & 7u) << 8);
                    const std::uint64_t grid_bits = rsl::IQ1S_GRID_SYCL[idx];
                    for (int j = 0; j < 8; ++j) {
                        const std::int8_t gi = static_cast<std::int8_t>((grid_bits >> (j * 8)) & 0xFFu);
                        dot += dl * (static_cast<float>(gi) + delta) * attn_usm[x_off + l * 8 + j];
                    }
                }
            }
        }
    RSL_H6_END
}

// ---- IQ1_M (block 256, 56 bytes) ----
template <std::size_t LWS_T>
inline void matvec_iq1_m_add_rmsnorm_usm_impl(
    sycl::queue& q, const void* w_bytes_usm, const float* attn_usm,
    float* hidden_usm, const float* w_norm_usm, float* y_norm_usm,
    int M, int K, float eps) {
    constexpr std::size_t LWS = LWS_T;
    constexpr float IQ1S_DELTA = 0.125f;
    const int blocks_per_row = K / 256;
    const int bytes_per_row = blocks_per_row * 56;
    const uint8_t* w_bytes = static_cast<const uint8_t*>(w_bytes_usm);
    RSL_H6_BEGIN(LWS)
        const uint8_t* row = w_bytes + m * bytes_per_row;
        for (int b = 0; b < blocks_per_row; ++b) {
            const uint8_t* blk = row + b * 56;
            const uint8_t* qs = blk;
            const uint8_t* qh = blk + 32;
            const uint8_t* sc_bytes = blk + 48;
            std::uint16_t sc[4];
            for (int ii = 0; ii < 4; ++ii)
                sc[ii] = static_cast<std::uint16_t>(sc_bytes[ii * 2]) | (static_cast<std::uint16_t>(sc_bytes[ii * 2 + 1]) << 8);
            const std::uint16_t d_bits = (std::uint16_t)((sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0u) | ((sc[2] >> 4) & 0x0F00u) | (sc[3] & 0xF000u));
            const float d = bits_to_f32(d_bits);
            const int x_base = b * 256;
            int x_off = 0;
            for (int ib = 0; ib < 8; ++ib) {
                const std::uint16_t s_word = sc[ib / 2];
                const int shift0 = 6 * (ib % 2), shift1 = 6 * (ib % 2) + 3;
                const float dl1 = d * (2.0f * static_cast<float>((s_word >> shift0) & 0x7u) + 1.0f);
                const float dl2 = d * (2.0f * static_cast<float>((s_word >> shift1) & 0x7u) + 1.0f);
                const std::uint8_t qh0 = qh[ib * 2], qh1 = qh[ib * 2 + 1];
                const float delta_l[4] = {
                    (qh0 & 0x08u) ? (-1.0f - IQ1S_DELTA) : (-1.0f + IQ1S_DELTA),
                    (qh0 & 0x80u) ? (-1.0f - IQ1S_DELTA) : (-1.0f + IQ1S_DELTA),
                    (qh1 & 0x08u) ? (-1.0f - IQ1S_DELTA) : (-1.0f + IQ1S_DELTA),
                    (qh1 & 0x80u) ? (-1.0f - IQ1S_DELTA) : (-1.0f + IQ1S_DELTA),
                };
                const std::size_t idx_l[4] = {
                    static_cast<std::size_t>(qs[ib * 4 + 0]) | (static_cast<std::size_t>(qh0 & 0x07u) << 8),
                    static_cast<std::size_t>(qs[ib * 4 + 1]) | (static_cast<std::size_t>((qh0 >> 4) & 0x07u) << 8),
                    static_cast<std::size_t>(qs[ib * 4 + 2]) | (static_cast<std::size_t>(qh1 & 0x07u) << 8),
                    static_cast<std::size_t>(qs[ib * 4 + 3]) | (static_cast<std::size_t>((qh1 >> 4) & 0x07u) << 8),
                };
                const float dl_l[4] = { dl1, dl1, dl2, dl2 };
                for (int l = 0; l < 4; ++l) {
                    const std::uint64_t grid_bits = rsl::IQ1S_GRID_SYCL[idx_l[l]];
                    const float dl = dl_l[l];
                    const float dl_delta = dl * delta_l[l];
                    for (int j = 0; j < 8; ++j) {
                        const std::int8_t gi = static_cast<std::int8_t>((grid_bits >> (j * 8)) & 0xFFu);
                        const float val_w = dl * static_cast<float>(gi) + dl_delta;
                        dot += val_w * attn_usm[x_base + x_off + 8 * l + j];
                    }
                }
                x_off += 32;
            }
        }
    RSL_H6_END
}

// ---- IQ2_XXS (block 256, 66 bytes) ----
template <std::size_t LWS_T>
inline void matvec_iq2_xxs_add_rmsnorm_usm_impl(
    sycl::queue& q, const void* w_bytes_usm, const float* attn_usm,
    float* hidden_usm, const float* w_norm_usm, float* y_norm_usm,
    int M, int K, float eps) {
    constexpr std::size_t LWS = LWS_T;
    const int blocks_per_row = K / 256;
    const int bytes_per_row = blocks_per_row * 66;
    const uint8_t* w_bytes = static_cast<const uint8_t*>(w_bytes_usm);
    RSL_H6_BEGIN(LWS)
        const uint8_t* row = w_bytes + m * bytes_per_row;
        for (int b = 0; b < blocks_per_row; ++b) {
            const uint8_t* blk = row + b * 66;
            const float d = bits_to_f32(static_cast<uint16_t>(blk[0]) | (static_cast<uint16_t>(blk[1]) << 8));
            const uint8_t* qs = blk + 2;
            const int x_base = b * 256;
            for (int ib32 = 0; ib32 < 8; ++ib32) {
                const std::uint32_t aux0 = static_cast<std::uint32_t>(qs[8 * ib32]) | (static_cast<std::uint32_t>(qs[8 * ib32 + 1]) << 8) | (static_cast<std::uint32_t>(qs[8 * ib32 + 2]) << 16) | (static_cast<std::uint32_t>(qs[8 * ib32 + 3]) << 24);
                const std::uint32_t aux1 = static_cast<std::uint32_t>(qs[8 * ib32 + 4]) | (static_cast<std::uint32_t>(qs[8 * ib32 + 5]) << 8) | (static_cast<std::uint32_t>(qs[8 * ib32 + 6]) << 16) | (static_cast<std::uint32_t>(qs[8 * ib32 + 7]) << 24);
                const float db = d * (0.5f + static_cast<float>(aux1 >> 28)) * 0.25f;
                for (int l = 0; l < 4; ++l) {
                    const std::size_t grid_idx = (aux0 >> (8 * l)) & 0xFFu;
                    const std::uint64_t grid_bits = rsl::IQ2XXS_GRID_SYCL[grid_idx];
                    const std::uint8_t signs = rsl::KSIGNS_IQ2XS_SYCL[(aux1 >> (7 * l)) & 127u];
                    const int x_off = x_base + ib32 * 32 + l * 8;
                    for (int j = 0; j < 8; ++j) {
                        const std::uint8_t gi = static_cast<std::uint8_t>((grid_bits >> (j * 8)) & 0xFFu);
                        const float s = (signs & rsl::KMASK_IQ2XS_SYCL[j]) ? -1.0f : 1.0f;
                        dot += db * static_cast<float>(gi) * s * attn_usm[x_off + j];
                    }
                }
            }
        }
    RSL_H6_END
}

// ---- IQ2_XS (block 256, 74 bytes) ----
template <std::size_t LWS_T>
inline void matvec_iq2_xs_add_rmsnorm_usm_impl(
    sycl::queue& q, const void* w_bytes_usm, const float* attn_usm,
    float* hidden_usm, const float* w_norm_usm, float* y_norm_usm,
    int M, int K, float eps) {
    constexpr std::size_t LWS = LWS_T;
    const int blocks_per_row = K / 256;
    const int bytes_per_row = blocks_per_row * 74;
    const uint8_t* w_bytes = static_cast<const uint8_t*>(w_bytes_usm);
    RSL_H6_BEGIN(LWS)
        const uint8_t* row = w_bytes + m * bytes_per_row;
        for (int b = 0; b < blocks_per_row; ++b) {
            const uint8_t* blk = row + b * 74;
            const float d = bits_to_f32(static_cast<uint16_t>(blk[0]) | (static_cast<uint16_t>(blk[1]) << 8));
            const uint8_t* qs = blk + 2;
            const uint8_t* scales = blk + 2 + 64;
            const int x_base = b * 256;
            for (int ib32 = 0; ib32 < 8; ++ib32) {
                const std::uint8_t scale_byte = scales[ib32];
                const float db_lo = d * (0.5f + static_cast<float>(scale_byte & 0x0Fu)) * 0.25f;
                const float db_hi = d * (0.5f + static_cast<float>(scale_byte >> 4)) * 0.25f;
                const int base = 8 * ib32;
                for (int l = 0; l < 4; ++l) {
                    const std::uint16_t qd = static_cast<std::uint16_t>(qs[base + 2 * l]) | (static_cast<std::uint16_t>(qs[base + 2 * l + 1]) << 8);
                    const std::size_t grid_idx = static_cast<std::size_t>(qd & 0x1FFu);
                    const std::size_t sign_idx = static_cast<std::size_t>(qd >> 9);
                    const std::uint64_t grid_bits = rsl::IQ2XS_GRID_SYCL[grid_idx];
                    const std::uint8_t signs = rsl::KSIGNS_IQ2XS_SYCL[sign_idx];
                    const float db = (l < 2) ? db_lo : db_hi;
                    const int x_off = x_base + ib32 * 32 + l * 8;
                    for (int j = 0; j < 8; ++j) {
                        const std::uint8_t gi = static_cast<std::uint8_t>((grid_bits >> (j * 8)) & 0xFFu);
                        const float s = (signs & rsl::KMASK_IQ2XS_SYCL[j]) ? -1.0f : 1.0f;
                        dot += db * static_cast<float>(gi) * s * attn_usm[x_off + j];
                    }
                }
            }
        }
    RSL_H6_END
}

// ---- IQ2_S (block 256, 82 bytes) ----
template <std::size_t LWS_T>
inline void matvec_iq2_s_add_rmsnorm_usm_impl(
    sycl::queue& q, const void* w_bytes_usm, const float* attn_usm,
    float* hidden_usm, const float* w_norm_usm, float* y_norm_usm,
    int M, int K, float eps) {
    constexpr std::size_t LWS = LWS_T;
    const int blocks_per_row = K / 256;
    const int bytes_per_row = blocks_per_row * 82;
    const uint8_t* w_bytes = static_cast<const uint8_t*>(w_bytes_usm);
    RSL_H6_BEGIN(LWS)
        const uint8_t* row = w_bytes + m * bytes_per_row;
        for (int b = 0; b < blocks_per_row; ++b) {
            const uint8_t* blk = row + b * 82;
            const float d = bits_to_f32(static_cast<uint16_t>(blk[0]) | (static_cast<uint16_t>(blk[1]) << 8));
            const uint8_t* qs_lo = blk + 2;
            const uint8_t* signs = blk + 2 + 32;
            const uint8_t* qh = blk + 2 + 64;
            const uint8_t* scales = blk + 2 + 64 + 8;
            const int x_base = b * 256;
            for (int ib32 = 0; ib32 < 8; ++ib32) {
                const std::uint8_t scale_byte = scales[ib32];
                const float db_lo = d * (0.5f + static_cast<float>(scale_byte & 0x0Fu)) * 0.25f;
                const float db_hi = d * (0.5f + static_cast<float>(scale_byte >> 4)) * 0.25f;
                const int qs_off = ib32 * 4;
                const std::uint8_t qh_byte = qh[ib32];
                for (int l = 0; l < 4; ++l) {
                    const std::size_t high_bits = (static_cast<std::size_t>(qh_byte) << (8 - 2 * l)) & 0x300u;
                    const std::size_t grid_idx = static_cast<std::size_t>(qs_lo[qs_off + l]) | high_bits;
                    const std::uint8_t sign_byte = signs[qs_off + l];
                    const std::uint64_t grid_bits = rsl::IQ2S_GRID_SYCL[grid_idx];
                    const float db = (l < 2) ? db_lo : db_hi;
                    const int x_off = x_base + ib32 * 32 + l * 8;
                    for (int j = 0; j < 8; ++j) {
                        const std::uint8_t gi = static_cast<std::uint8_t>((grid_bits >> (j * 8)) & 0xFFu);
                        const float s = (sign_byte & rsl::KMASK_IQ2XS_SYCL[j]) ? -1.0f : 1.0f;
                        dot += db * static_cast<float>(gi) * s * attn_usm[x_off + j];
                    }
                }
            }
        }
    RSL_H6_END
}

// ---- IQ3_XXS (block 256, 98 bytes) ----
template <std::size_t LWS_T>
inline void matvec_iq3_xxs_add_rmsnorm_usm_impl(
    sycl::queue& q, const void* w_bytes_usm, const float* attn_usm,
    float* hidden_usm, const float* w_norm_usm, float* y_norm_usm,
    int M, int K, float eps) {
    constexpr std::size_t LWS = LWS_T;
    const int blocks_per_row = K / 256;
    const int bytes_per_row = blocks_per_row * 98;
    const uint8_t* w_bytes = static_cast<const uint8_t*>(w_bytes_usm);
    RSL_H6_BEGIN(LWS)
        const uint8_t* row = w_bytes + m * bytes_per_row;
        for (int b = 0; b < blocks_per_row; ++b) {
            const uint8_t* blk = row + b * 98;
            const float d = bits_to_f32(static_cast<uint16_t>(blk[0]) | (static_cast<uint16_t>(blk[1]) << 8));
            const uint8_t* qs_grid = blk + 2;
            const uint8_t* qs_sas = blk + 2 + 64;
            const int x_base = b * 256;
            for (int ib32 = 0; ib32 < 8; ++ib32) {
                const std::uint32_t aux32 = static_cast<std::uint32_t>(qs_sas[4 * ib32]) | (static_cast<std::uint32_t>(qs_sas[4 * ib32 + 1]) << 8) | (static_cast<std::uint32_t>(qs_sas[4 * ib32 + 2]) << 16) | (static_cast<std::uint32_t>(qs_sas[4 * ib32 + 3]) << 24);
                const float db = d * (0.5f + static_cast<float>(aux32 >> 28)) * 0.5f;
                const int qs_off = 8 * ib32;
                for (int l = 0; l < 4; ++l) {
                    const std::size_t g1_idx = static_cast<std::size_t>(qs_grid[qs_off + 2 * l]);
                    const std::size_t g2_idx = static_cast<std::size_t>(qs_grid[qs_off + 2 * l + 1]);
                    const std::uint32_t grid1 = rsl::IQ3XXS_GRID_SYCL[g1_idx];
                    const std::uint32_t grid2 = rsl::IQ3XXS_GRID_SYCL[g2_idx];
                    const std::uint8_t signs = rsl::KSIGNS_IQ2XS_SYCL[(aux32 >> (7 * l)) & 127u];
                    const int x_off = x_base + ib32 * 32 + l * 8;
                    for (int j = 0; j < 4; ++j) {
                        const std::uint8_t g1 = static_cast<std::uint8_t>((grid1 >> (j * 8)) & 0xFFu);
                        const std::uint8_t g2 = static_cast<std::uint8_t>((grid2 >> (j * 8)) & 0xFFu);
                        const float s_lo = (signs & rsl::KMASK_IQ2XS_SYCL[j]) ? -1.0f : 1.0f;
                        const float s_hi = (signs & rsl::KMASK_IQ2XS_SYCL[j + 4]) ? -1.0f : 1.0f;
                        dot += db * static_cast<float>(g1) * s_lo * attn_usm[x_off + j];
                        dot += db * static_cast<float>(g2) * s_hi * attn_usm[x_off + j + 4];
                    }
                }
            }
        }
    RSL_H6_END
}

// ---- IQ3_S (block 256, 110 bytes) ----
template <std::size_t LWS_T>
inline void matvec_iq3_s_add_rmsnorm_usm_impl(
    sycl::queue& q, const void* w_bytes_usm, const float* attn_usm,
    float* hidden_usm, const float* w_norm_usm, float* y_norm_usm,
    int M, int K, float eps) {
    constexpr std::size_t LWS = LWS_T;
    const int blocks_per_row = K / 256;
    const int bytes_per_row = blocks_per_row * 110;
    const uint8_t* w_bytes = static_cast<const uint8_t*>(w_bytes_usm);
    RSL_H6_BEGIN(LWS)
        const uint8_t* row = w_bytes + m * bytes_per_row;
        for (int b = 0; b < blocks_per_row; ++b) {
            const uint8_t* blk = row + b * 110;
            const float d = bits_to_f32(static_cast<uint16_t>(blk[0]) | (static_cast<uint16_t>(blk[1]) << 8));
            const uint8_t* qs = blk + 2;
            const uint8_t* qh = blk + 2 + 64;
            const uint8_t* signs = blk + 2 + 64 + 8;
            const uint8_t* scales = blk + 2 + 64 + 8 + 32;
            const int x_base = b * 256;
            for (int ib32 = 0; ib32 < 8; ++ib32) {
                const int pair = ib32 >> 1;
                const std::uint8_t scale_byte = scales[pair];
                const float db = (ib32 & 1)
                    ? d * (1.0f + 2.0f * static_cast<float>(scale_byte >> 4))
                    : d * (1.0f + 2.0f * static_cast<float>(scale_byte & 0x0Fu));
                const int qs_off = ib32 * 8, signs_off = ib32 * 4;
                const std::uint8_t qh_byte = qh[ib32];
                const int x_off_block = ib32 * 32;
                for (int l = 0; l < 4; ++l) {
                    const std::size_t g1_idx = static_cast<std::size_t>(qs[qs_off + 2 * l]) | ((static_cast<std::size_t>(qh_byte) << (8 - 2 * l)) & 0x100u);
                    const std::size_t g2_idx = static_cast<std::size_t>(qs[qs_off + 2 * l + 1]) | ((static_cast<std::size_t>(qh_byte) << (7 - 2 * l)) & 0x100u);
                    const std::uint32_t grid1 = rsl::IQ3S_GRID_SYCL[g1_idx];
                    const std::uint32_t grid2 = rsl::IQ3S_GRID_SYCL[g2_idx];
                    const std::uint8_t sign_byte = signs[signs_off + l];
                    const int x_off = x_base + x_off_block + l * 8;
                    for (int j = 0; j < 4; ++j) {
                        const std::uint8_t g1 = static_cast<std::uint8_t>((grid1 >> (j * 8)) & 0xFFu);
                        const std::uint8_t g2 = static_cast<std::uint8_t>((grid2 >> (j * 8)) & 0xFFu);
                        const float s_lo = (sign_byte & rsl::KMASK_IQ2XS_SYCL[j]) ? -1.0f : 1.0f;
                        const float s_hi = (sign_byte & rsl::KMASK_IQ2XS_SYCL[j + 4]) ? -1.0f : 1.0f;
                        dot += db * static_cast<float>(g1) * s_lo * attn_usm[x_off + j];
                        dot += db * static_cast<float>(g2) * s_hi * attn_usm[x_off + j + 4];
                    }
                }
            }
        }
    RSL_H6_END
}

// ---- PTQ1_0 (block 128, 28 bytes) ----
// Trit decode identical to `matvec_ptq1_0_packed_f32_usm_impl`; the d scale
// folds per block into `dot`, then RSL_H6_END adds the residual + rmsnorm.
template <std::size_t LWS_T>
inline void matvec_ptq1_0_add_rmsnorm_usm_impl(
    sycl::queue& q, const void* w_bytes_usm, const float* attn_usm,
    float* hidden_usm, const float* w_norm_usm, float* y_norm_usm,
    int M, int K, float eps) {
    constexpr std::size_t LWS = LWS_T;
    const int blocks_per_row = K / 128;
    const int bytes_per_row = blocks_per_row * 28;
    const uint8_t* w_bytes = static_cast<const uint8_t*>(w_bytes_usm);
    RSL_H6_BEGIN(LWS)
        const uint8_t pow3[5] = {1, 3, 9, 27, 81};
        const uint8_t* row = w_bytes + m * bytes_per_row;
        for (int b = 0; b < blocks_per_row; ++b) {
            const uint8_t* blk = row + b * 28;
            const uint8_t* qs = blk;
            const uint8_t* qh = blk + 24;
            const float d = bits_to_f32(static_cast<uint16_t>(blk[26])
                                        | (static_cast<uint16_t>(blk[27]) << 8));
            const int x_base = b * 128;
            float sum = 0.0f;
            for (int n = 0; n < 5; ++n) {
                const uint8_t p3 = pow3[n];
                const int e0 = x_base + n * 16;
                for (int mm = 0; mm < 16; ++mm) {
                    const int trit = ((static_cast<int>(static_cast<uint8_t>(qs[mm] * p3)) * 3) >> 8) - 1;
                    sum += static_cast<float>(trit) * attn_usm[e0 + mm];
                }
            }
            for (int n = 0; n < 5; ++n) {
                const uint8_t p3 = pow3[n];
                const int e0 = x_base + 80 + n * 8;
                for (int mm = 0; mm < 8; ++mm) {
                    const int trit = ((static_cast<int>(static_cast<uint8_t>(qs[16 + mm] * p3)) * 3) >> 8) - 1;
                    sum += static_cast<float>(trit) * attn_usm[e0 + mm];
                }
            }
            for (int n = 0; n < 4; ++n) {
                const uint8_t p3 = pow3[n];
                const int e0 = x_base + 120 + n * 2;
                for (int hh = 0; hh < 2; ++hh) {
                    const int trit = ((static_cast<int>(static_cast<uint8_t>(qh[hh] * p3)) * 3) >> 8) - 1;
                    sum += static_cast<float>(trit) * attn_usm[e0 + hh];
                }
            }
            dot += d * sum;
        }
    RSL_H6_END
}

extern "C" {

#define RSL_H6_WRAPPER_BODY(IMPL) \
    if (s == nullptr || w_bytes_usm == nullptr || attn_usm == nullptr \
        || hidden_usm == nullptr || w_norm_usm == nullptr || y_norm_usm == nullptr) return; \
    if (M <= 0 || K <= 0) return; \
    auto& q = s->q; \
    const int eff_lws = (lws <= 0) ? 128 : lws; \
    switch (eff_lws) { \
        case 32:  IMPL<32>(q, w_bytes_usm, attn_usm, hidden_usm, w_norm_usm, y_norm_usm, M, K, eps); break; \
        case 64:  IMPL<64>(q, w_bytes_usm, attn_usm, hidden_usm, w_norm_usm, y_norm_usm, M, K, eps); break; \
        case 128: IMPL<128>(q, w_bytes_usm, attn_usm, hidden_usm, w_norm_usm, y_norm_usm, M, K, eps); break; \
        case 256: IMPL<256>(q, w_bytes_usm, attn_usm, hidden_usm, w_norm_usm, y_norm_usm, M, K, eps); break; \
        default:  IMPL<128>(q, w_bytes_usm, attn_usm, hidden_usm, w_norm_usm, y_norm_usm, M, K, eps); break; \
    }

void rsl_matvec_q8_0_add_rmsnorm_usm(rsl_stream* s, const void* w_bytes_usm,
    const float* attn_usm, float* hidden_usm, const float* w_norm_usm,
    float* y_norm_usm, int M, int K, float eps, int lws)
    RSL_FFI_BODY_VOID("rsl_matvec_q8_0_add_rmsnorm_usm", {
    if (K % 32 != 0) return;
    RSL_H6_WRAPPER_BODY(matvec_q8_0_add_rmsnorm_usm_impl)
})

void rsl_matvec_q5_k_add_rmsnorm_usm(rsl_stream* s, const void* w_bytes_usm,
    const float* attn_usm, float* hidden_usm, const float* w_norm_usm,
    float* y_norm_usm, int M, int K, float eps, int lws)
    RSL_FFI_BODY_VOID("rsl_matvec_q5_k_add_rmsnorm_usm", {
    if (K % 256 != 0) return;
    RSL_H6_WRAPPER_BODY(matvec_q5_k_add_rmsnorm_usm_impl)
})

void rsl_matvec_q6_k_add_rmsnorm_usm(rsl_stream* s, const void* w_bytes_usm,
    const float* attn_usm, float* hidden_usm, const float* w_norm_usm,
    float* y_norm_usm, int M, int K, float eps, int lws)
    RSL_FFI_BODY_VOID("rsl_matvec_q6_k_add_rmsnorm_usm", {
    if (K % 256 != 0) return;
    RSL_H6_WRAPPER_BODY(matvec_q6_k_add_rmsnorm_usm_impl)
})

void rsl_matvec_iq4_nl_add_rmsnorm_usm(rsl_stream* s, const void* w_bytes_usm,
    const float* attn_usm, float* hidden_usm, const float* w_norm_usm,
    float* y_norm_usm, int M, int K, float eps, int lws)
    RSL_FFI_BODY_VOID("rsl_matvec_iq4_nl_add_rmsnorm_usm", {
    if (K % 32 != 0) return;
    RSL_H6_WRAPPER_BODY(matvec_iq4_nl_add_rmsnorm_usm_impl)
})

void rsl_matvec_iq4_xs_add_rmsnorm_usm(rsl_stream* s, const void* w_bytes_usm,
    const float* attn_usm, float* hidden_usm, const float* w_norm_usm,
    float* y_norm_usm, int M, int K, float eps, int lws)
    RSL_FFI_BODY_VOID("rsl_matvec_iq4_xs_add_rmsnorm_usm", {
    if (K % 256 != 0) return;
    RSL_H6_WRAPPER_BODY(matvec_iq4_xs_add_rmsnorm_usm_impl)
})

void rsl_matvec_iq1_s_add_rmsnorm_usm(rsl_stream* s, const void* w_bytes_usm,
    const float* attn_usm, float* hidden_usm, const float* w_norm_usm,
    float* y_norm_usm, int M, int K, float eps, int lws)
    RSL_FFI_BODY_VOID("rsl_matvec_iq1_s_add_rmsnorm_usm", {
    if (K % 256 != 0) return;
    RSL_H6_WRAPPER_BODY(matvec_iq1_s_add_rmsnorm_usm_impl)
})

void rsl_matvec_iq1_m_add_rmsnorm_usm(rsl_stream* s, const void* w_bytes_usm,
    const float* attn_usm, float* hidden_usm, const float* w_norm_usm,
    float* y_norm_usm, int M, int K, float eps, int lws)
    RSL_FFI_BODY_VOID("rsl_matvec_iq1_m_add_rmsnorm_usm", {
    if (K % 256 != 0) return;
    RSL_H6_WRAPPER_BODY(matvec_iq1_m_add_rmsnorm_usm_impl)
})

void rsl_matvec_iq2_xxs_add_rmsnorm_usm(rsl_stream* s, const void* w_bytes_usm,
    const float* attn_usm, float* hidden_usm, const float* w_norm_usm,
    float* y_norm_usm, int M, int K, float eps, int lws)
    RSL_FFI_BODY_VOID("rsl_matvec_iq2_xxs_add_rmsnorm_usm", {
    if (K % 256 != 0) return;
    RSL_H6_WRAPPER_BODY(matvec_iq2_xxs_add_rmsnorm_usm_impl)
})

void rsl_matvec_iq2_xs_add_rmsnorm_usm(rsl_stream* s, const void* w_bytes_usm,
    const float* attn_usm, float* hidden_usm, const float* w_norm_usm,
    float* y_norm_usm, int M, int K, float eps, int lws)
    RSL_FFI_BODY_VOID("rsl_matvec_iq2_xs_add_rmsnorm_usm", {
    if (K % 256 != 0) return;
    RSL_H6_WRAPPER_BODY(matvec_iq2_xs_add_rmsnorm_usm_impl)
})

void rsl_matvec_iq2_s_add_rmsnorm_usm(rsl_stream* s, const void* w_bytes_usm,
    const float* attn_usm, float* hidden_usm, const float* w_norm_usm,
    float* y_norm_usm, int M, int K, float eps, int lws)
    RSL_FFI_BODY_VOID("rsl_matvec_iq2_s_add_rmsnorm_usm", {
    if (K % 256 != 0) return;
    RSL_H6_WRAPPER_BODY(matvec_iq2_s_add_rmsnorm_usm_impl)
})

void rsl_matvec_iq3_xxs_add_rmsnorm_usm(rsl_stream* s, const void* w_bytes_usm,
    const float* attn_usm, float* hidden_usm, const float* w_norm_usm,
    float* y_norm_usm, int M, int K, float eps, int lws)
    RSL_FFI_BODY_VOID("rsl_matvec_iq3_xxs_add_rmsnorm_usm", {
    if (K % 256 != 0) return;
    RSL_H6_WRAPPER_BODY(matvec_iq3_xxs_add_rmsnorm_usm_impl)
})

void rsl_matvec_iq3_s_add_rmsnorm_usm(rsl_stream* s, const void* w_bytes_usm,
    const float* attn_usm, float* hidden_usm, const float* w_norm_usm,
    float* y_norm_usm, int M, int K, float eps, int lws)
    RSL_FFI_BODY_VOID("rsl_matvec_iq3_s_add_rmsnorm_usm", {
    if (K % 256 != 0) return;
    RSL_H6_WRAPPER_BODY(matvec_iq3_s_add_rmsnorm_usm_impl)
})

void rsl_matvec_ptq1_0_add_rmsnorm_usm(rsl_stream* s, const void* w_bytes_usm,
    const float* attn_usm, float* hidden_usm, const float* w_norm_usm,
    float* y_norm_usm, int M, int K, float eps, int lws)
    RSL_FFI_BODY_VOID("rsl_matvec_ptq1_0_add_rmsnorm_usm", {
    if (K % 128 != 0) return;
    RSL_H6_WRAPPER_BODY(matvec_ptq1_0_add_rmsnorm_usm_impl)
})

}  // extern "C" — close before the H8 mixed-precision templates

// =========================================================================
// H8: F16-input mixed-precision matvec kernels (per packed-quant dtype)
// =========================================================================
//
// Identical per-output-row topology + dequant arithmetic to the
// `matvec_X_packed_f32_usm` kernels, but the activation vector is read
// as F16 (`uint16_t` bits → `bits_to_f32`) instead of F32. Weights are
// unchanged (still the packed-quant bytes); the accumulator + output
// stay F32. Halves activation-vector memory traffic — meaningful on
// Iris Xe's narrow memory bus where each output row re-reads the full
// K-length activation.
//
// Gated behind `RUSTLLAMA_MIXED_PRECISION_MATVEC=1` (default off): F16
// activations lose ~3 mantissa bits, so this stays opt-in until
// validated within tolerance on hardware (Open Risk #7 — F16
// underflow on near-zero activations).
#define RSL_H8_BEGIN(LWS) \
    const std::size_t global = ((static_cast<std::size_t>(M) + (LWS) - 1) / (LWS)) * (LWS); \
    q.submit([&](sycl::handler& h) { \
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(LWS)), \
            [=](sycl::nd_item<1> it) { \
                const int m = static_cast<int>(it.get_global_id(0)); \
                if (m >= M) return; \
                float dot = 0.0f;
#define RSL_H8_END \
                out_usm[m] = dot; \
            }); \
    }).wait();
#define XLD(i) bits_to_f32(x_f16[(i)])

// ---- Q8_0 ----
template <std::size_t LWS_T>
inline void matvec_q8_0_f16in_packed_f32_usm_impl(
    sycl::queue& q, const void* w_bytes_usm, const uint16_t* x_f16, float* out_usm, int M, int K) {
    constexpr std::size_t LWS = LWS_T;
    const int blocks_per_row = K / 32;
    const int bytes_per_row = blocks_per_row * 34;
    const uint8_t* w_bytes = static_cast<const uint8_t*>(w_bytes_usm);
    RSL_H8_BEGIN(LWS)
        const uint8_t* row = w_bytes + m * bytes_per_row;
        for (int blk_i = 0; blk_i < blocks_per_row; ++blk_i) {
            const uint8_t* blk = row + blk_i * 34;
            const float scale = bits_to_f32(static_cast<uint16_t>(blk[0]) | (static_cast<uint16_t>(blk[1]) << 8));
            const int x_off = blk_i * 32;
            for (int dd = 0; dd < 32; ++dd)
                dot += scale * static_cast<float>(static_cast<int8_t>(blk[2 + dd])) * XLD(x_off + dd);
        }
    RSL_H8_END
}

// ---- Q4_K ----
template <std::size_t LWS_T>
inline void matvec_q4_k_f16in_packed_f32_usm_impl(
    sycl::queue& q, const void* w_bytes_usm, const uint16_t* x_f16, float* out_usm, int M, int K) {
    constexpr std::size_t LWS = LWS_T;
    const int blocks_per_row = K / 256;
    const int bytes_per_row = blocks_per_row * 144;
    const uint8_t* w_bytes = static_cast<const uint8_t*>(w_bytes_usm);
    RSL_H8_BEGIN(LWS)
        const uint8_t* row = w_bytes + m * bytes_per_row;
        for (int b = 0; b < blocks_per_row; ++b) {
            const uint8_t* blk = row + b * 144;
            const float d = bits_to_f32(static_cast<uint16_t>(blk[0]) | (static_cast<uint16_t>(blk[1]) << 8));
            const float dmin = bits_to_f32(static_cast<uint16_t>(blk[2]) | (static_cast<uint16_t>(blk[3]) << 8));
            uint8_t sc[8], mn[8];
            { const uint8_t* sb = blk + 4;
              for (int j = 0; j < 8; ++j) {
                  if (j < 4) { sc[j] = sb[j] & 0x3F; mn[j] = sb[j + 4] & 0x3F; }
                  else { sc[j] = (sb[j + 4] & 0x0F) | ((sb[j - 4] >> 6) << 4); mn[j] = (sb[j + 4] >> 4) | ((sb[j] >> 6) << 4); } } }
            const uint8_t* qs = blk + 16;
            const int x_base = b * 256;
            for (int group = 0; group < 4; ++group) {
                const float d_lo = d * static_cast<float>(sc[group * 2]);
                const float m_lo = dmin * static_cast<float>(mn[group * 2]);
                const float d_hi = d * static_cast<float>(sc[group * 2 + 1]);
                const float m_hi = dmin * static_cast<float>(mn[group * 2 + 1]);
                const uint8_t* qc = qs + group * 32;
                const int x_lo_off = x_base + group * 64, x_hi_off = x_lo_off + 32;
                for (int l = 0; l < 32; ++l) {
                    const uint8_t qb = qc[l];
                    dot += (d_lo * static_cast<float>(qb & 0x0F) - m_lo) * XLD(x_lo_off + l);
                    dot += (d_hi * static_cast<float>(qb >> 4)   - m_hi) * XLD(x_hi_off + l);
                }
            }
        }
    RSL_H8_END
}

// ---- Q5_K ----
template <std::size_t LWS_T>
inline void matvec_q5_k_f16in_packed_f32_usm_impl(
    sycl::queue& q, const void* w_bytes_usm, const uint16_t* x_f16, float* out_usm, int M, int K) {
    constexpr std::size_t LWS = LWS_T;
    const int blocks_per_row = K / 256;
    const int bytes_per_row = blocks_per_row * 176;
    const uint8_t* w_bytes = static_cast<const uint8_t*>(w_bytes_usm);
    RSL_H8_BEGIN(LWS)
        const uint8_t* row = w_bytes + m * bytes_per_row;
        for (int b = 0; b < blocks_per_row; ++b) {
            const uint8_t* blk = row + b * 176;
            const float d = bits_to_f32(static_cast<uint16_t>(blk[0]) | (static_cast<uint16_t>(blk[1]) << 8));
            const float dmin = bits_to_f32(static_cast<uint16_t>(blk[2]) | (static_cast<uint16_t>(blk[3]) << 8));
            const uint8_t* sb = blk + 4;
            uint8_t sc[8], mn[8];
            for (int j = 0; j < 8; ++j) {
                if (j < 4) { sc[j] = sb[j] & 0x3F; mn[j] = sb[j + 4] & 0x3F; }
                else { sc[j] = (sb[j + 4] & 0x0F) | ((sb[j - 4] >> 6) << 4); mn[j] = (sb[j + 4] >> 4) | ((sb[j] >> 6) << 4); }
            }
            const uint8_t* qh = blk + 16;
            const uint8_t* qs = blk + 48;
            const int x_base = b * 256;
            for (int group = 0; group < 4; ++group) {
                const uint8_t* qc = qs + group * 32;
                const float d_lo = d * static_cast<float>(sc[group * 2]);
                const float m_lo = dmin * static_cast<float>(mn[group * 2]);
                const float d_hi = d * static_cast<float>(sc[group * 2 + 1]);
                const float m_hi = dmin * static_cast<float>(mn[group * 2 + 1]);
                const int bit_lo = group * 2, bit_hi = group * 2 + 1;
                const int x_lo_off = x_base + group * 64, x_hi_off = x_lo_off + 32;
                for (int l = 0; l < 32; ++l) {
                    const uint8_t qb = qc[l], qhb = qh[l];
                    const uint32_t lo = static_cast<uint32_t>(qb & 0x0F) | (static_cast<uint32_t>((qhb >> bit_lo) & 1) << 4);
                    const uint32_t hi = static_cast<uint32_t>(qb >> 4) | (static_cast<uint32_t>((qhb >> bit_hi) & 1) << 4);
                    dot += (d_lo * static_cast<float>(lo) - m_lo) * XLD(x_lo_off + l);
                    dot += (d_hi * static_cast<float>(hi) - m_hi) * XLD(x_hi_off + l);
                }
            }
        }
    RSL_H8_END
}

// ---- Q6_K ----
template <std::size_t LWS_T>
inline void matvec_q6_k_f16in_packed_f32_usm_impl(
    sycl::queue& q, const void* w_bytes_usm, const uint16_t* x_f16, float* out_usm, int M, int K) {
    constexpr std::size_t LWS = LWS_T;
    const int blocks_per_row = K / 256;
    const int bytes_per_row = blocks_per_row * 210;
    const uint8_t* w_bytes = static_cast<const uint8_t*>(w_bytes_usm);
    RSL_H8_BEGIN(LWS)
        const uint8_t* row = w_bytes + m * bytes_per_row;
        for (int b = 0; b < blocks_per_row; ++b) {
            const uint8_t* blk = row + b * 210;
            const uint8_t* ql = blk;
            const uint8_t* qh = blk + 128;
            const int8_t* scales = reinterpret_cast<const int8_t*>(blk + 192);
            const float d = bits_to_f32(static_cast<uint16_t>(blk[208]) | (static_cast<uint16_t>(blk[209]) << 8));
            const int x_base = b * 256;
            for (int n = 0; n < 2; ++n) {
                for (int l = 0; l < 32; ++l) {
                    const int is = l / 16 + n * 8;
                    const int qhb = qh[32 * n + l];
                    const int q1 = (static_cast<int>(ql[64 * n + l] & 0x0F) | ((qhb >> 0) & 0x03) << 4) - 32;
                    const int q2 = (static_cast<int>(ql[64 * n + l + 32] & 0x0F) | ((qhb >> 2) & 0x03) << 4) - 32;
                    const int q3 = (static_cast<int>(ql[64 * n + l] >> 4) | ((qhb >> 4) & 0x03) << 4) - 32;
                    const int q4 = (static_cast<int>(ql[64 * n + l + 32] >> 4) | ((qhb >> 6) & 0x03) << 4) - 32;
                    const int base = n * 128 + l;
                    dot += d * static_cast<float>(scales[is])     * static_cast<float>(q1) * XLD(x_base + base);
                    dot += d * static_cast<float>(scales[is + 2]) * static_cast<float>(q2) * XLD(x_base + base + 32);
                    dot += d * static_cast<float>(scales[is + 4]) * static_cast<float>(q3) * XLD(x_base + base + 64);
                    dot += d * static_cast<float>(scales[is + 6]) * static_cast<float>(q4) * XLD(x_base + base + 96);
                }
            }
        }
    RSL_H8_END
}

// ---- IQ4_NL ----
template <std::size_t LWS_T>
inline void matvec_iq4_nl_f16in_packed_f32_usm_impl(
    sycl::queue& q, const void* w_bytes_usm, const uint16_t* x_f16, float* out_usm, int M, int K) {
    constexpr std::size_t LWS = LWS_T;
    const int blocks_per_row = K / 32;
    const int bytes_per_row = blocks_per_row * 18;
    const uint8_t* w_bytes = static_cast<const uint8_t*>(w_bytes_usm);
    RSL_H8_BEGIN(LWS)
        const uint8_t* row = w_bytes + m * bytes_per_row;
        for (int b = 0; b < blocks_per_row; ++b) {
            const uint8_t* blk = row + b * 18;
            const float d = bits_to_f32(static_cast<uint16_t>(blk[0]) | (static_cast<uint16_t>(blk[1]) << 8));
            const uint8_t* qs = blk + 2;
            const int x_base = b * 32;
            for (int j = 0; j < 16; ++j) {
                const uint8_t qb = qs[j];
                dot += d * static_cast<float>(KVALUES_IQ4XS_SYCL[qb & 0x0F]) * XLD(x_base + j);
                dot += d * static_cast<float>(KVALUES_IQ4XS_SYCL[qb >> 4])   * XLD(x_base + j + 16);
            }
        }
    RSL_H8_END
}

// ---- IQ4_XS ----
template <std::size_t LWS_T>
inline void matvec_iq4_xs_f16in_packed_f32_usm_impl(
    sycl::queue& q, const void* w_bytes_usm, const uint16_t* x_f16, float* out_usm, int M, int K) {
    constexpr std::size_t LWS = LWS_T;
    const int blocks_per_row = K / 256;
    const int bytes_per_row = blocks_per_row * 136;
    const uint8_t* w_bytes = static_cast<const uint8_t*>(w_bytes_usm);
    RSL_H8_BEGIN(LWS)
        const uint8_t* row = w_bytes + m * bytes_per_row;
        for (int b = 0; b < blocks_per_row; ++b) {
            const uint8_t* blk = row + b * 136;
            const float d = bits_to_f32(static_cast<uint16_t>(blk[0]) | (static_cast<uint16_t>(blk[1]) << 8));
            const uint16_t scales_h = static_cast<uint16_t>(blk[2]) | (static_cast<uint16_t>(blk[3]) << 8);
            const uint8_t* scales_l = blk + 4;
            const uint8_t* qs = blk + 8;
            const int x_base = b * 256;
            for (int ib = 0; ib < 8; ++ib) {
                const uint8_t lo_nibble = (ib % 2 == 0) ? (scales_l[ib / 2] & 0x0F) : (scales_l[ib / 2] >> 4);
                const uint8_t hi_bits = static_cast<uint8_t>((scales_h >> (2 * ib)) & 0x03);
                const int8_t ls = static_cast<int8_t>(static_cast<uint8_t>(lo_nibble | (hi_bits << 4))) - 32;
                const float sub_d = d * static_cast<float>(ls);
                const int q_off = ib * 16, x_off = ib * 32;
                for (int j = 0; j < 16; ++j) {
                    const uint8_t qb = qs[q_off + j];
                    dot += sub_d * static_cast<float>(KVALUES_IQ4XS_SYCL[qb & 0x0F]) * XLD(x_base + x_off + j);
                    dot += sub_d * static_cast<float>(KVALUES_IQ4XS_SYCL[qb >> 4])   * XLD(x_base + x_off + 16 + j);
                }
            }
        }
    RSL_H8_END
}

// ---- IQ1_S ----
template <std::size_t LWS_T>
inline void matvec_iq1_s_f16in_packed_f32_usm_impl(
    sycl::queue& q, const void* w_bytes_usm, const uint16_t* x_f16, float* out_usm, int M, int K) {
    constexpr std::size_t LWS = LWS_T;
    constexpr float IQ1S_DELTA = 0.125f;
    const int blocks_per_row = K / 256;
    const int bytes_per_row = blocks_per_row * 50;
    const uint8_t* w_bytes = static_cast<const uint8_t*>(w_bytes_usm);
    RSL_H8_BEGIN(LWS)
        const uint8_t* row = w_bytes + m * bytes_per_row;
        for (int b = 0; b < blocks_per_row; ++b) {
            const uint8_t* blk = row + b * 50;
            const float d = bits_to_f32(static_cast<uint16_t>(blk[0]) | (static_cast<uint16_t>(blk[1]) << 8));
            const uint8_t* qs = blk + 2;
            const uint8_t* qh_bytes = blk + 34;
            const int x_base = b * 256;
            for (int ib32 = 0; ib32 < 8; ++ib32) {
                const uint16_t qh = static_cast<uint16_t>(qh_bytes[ib32 * 2]) | (static_cast<uint16_t>(qh_bytes[ib32 * 2 + 1]) << 8);
                const float dl = d * (2.0f * static_cast<float>((qh >> 12) & 7) + 1.0f);
                const float delta = (qh & 0x8000) ? (-1.0f - IQ1S_DELTA) : (-1.0f + IQ1S_DELTA);
                const int x_off = x_base + ib32 * 32;
                for (int l = 0; l < 4; ++l) {
                    const std::uint32_t idx = static_cast<std::uint32_t>(qs[ib32 * 4 + l]) | ((static_cast<std::uint32_t>(qh >> (3 * l)) & 7u) << 8);
                    const std::uint64_t grid_bits = rsl::IQ1S_GRID_SYCL[idx];
                    for (int j = 0; j < 8; ++j) {
                        const std::int8_t gi = static_cast<std::int8_t>((grid_bits >> (j * 8)) & 0xFFu);
                        dot += dl * (static_cast<float>(gi) + delta) * XLD(x_off + l * 8 + j);
                    }
                }
            }
        }
    RSL_H8_END
}

// ---- IQ1_M ----
template <std::size_t LWS_T>
inline void matvec_iq1_m_f16in_packed_f32_usm_impl(
    sycl::queue& q, const void* w_bytes_usm, const uint16_t* x_f16, float* out_usm, int M, int K) {
    constexpr std::size_t LWS = LWS_T;
    constexpr float IQ1S_DELTA = 0.125f;
    const int blocks_per_row = K / 256;
    const int bytes_per_row = blocks_per_row * 56;
    const uint8_t* w_bytes = static_cast<const uint8_t*>(w_bytes_usm);
    RSL_H8_BEGIN(LWS)
        const uint8_t* row = w_bytes + m * bytes_per_row;
        for (int b = 0; b < blocks_per_row; ++b) {
            const uint8_t* blk = row + b * 56;
            const uint8_t* qs = blk;
            const uint8_t* qh = blk + 32;
            const uint8_t* sc_bytes = blk + 48;
            std::uint16_t sc[4];
            for (int ii = 0; ii < 4; ++ii)
                sc[ii] = static_cast<std::uint16_t>(sc_bytes[ii * 2]) | (static_cast<std::uint16_t>(sc_bytes[ii * 2 + 1]) << 8);
            const std::uint16_t d_bits = (std::uint16_t)((sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0u) | ((sc[2] >> 4) & 0x0F00u) | (sc[3] & 0xF000u));
            const float d = bits_to_f32(d_bits);
            const int x_base = b * 256;
            int x_off = 0;
            for (int ib = 0; ib < 8; ++ib) {
                const std::uint16_t s_word = sc[ib / 2];
                const int shift0 = 6 * (ib % 2), shift1 = 6 * (ib % 2) + 3;
                const float dl1 = d * (2.0f * static_cast<float>((s_word >> shift0) & 0x7u) + 1.0f);
                const float dl2 = d * (2.0f * static_cast<float>((s_word >> shift1) & 0x7u) + 1.0f);
                const std::uint8_t qh0 = qh[ib * 2], qh1 = qh[ib * 2 + 1];
                const float delta_l[4] = {
                    (qh0 & 0x08u) ? (-1.0f - IQ1S_DELTA) : (-1.0f + IQ1S_DELTA),
                    (qh0 & 0x80u) ? (-1.0f - IQ1S_DELTA) : (-1.0f + IQ1S_DELTA),
                    (qh1 & 0x08u) ? (-1.0f - IQ1S_DELTA) : (-1.0f + IQ1S_DELTA),
                    (qh1 & 0x80u) ? (-1.0f - IQ1S_DELTA) : (-1.0f + IQ1S_DELTA), };
                const std::size_t idx_l[4] = {
                    static_cast<std::size_t>(qs[ib * 4 + 0]) | (static_cast<std::size_t>(qh0 & 0x07u) << 8),
                    static_cast<std::size_t>(qs[ib * 4 + 1]) | (static_cast<std::size_t>((qh0 >> 4) & 0x07u) << 8),
                    static_cast<std::size_t>(qs[ib * 4 + 2]) | (static_cast<std::size_t>(qh1 & 0x07u) << 8),
                    static_cast<std::size_t>(qs[ib * 4 + 3]) | (static_cast<std::size_t>((qh1 >> 4) & 0x07u) << 8), };
                const float dl_l[4] = { dl1, dl1, dl2, dl2 };
                for (int l = 0; l < 4; ++l) {
                    const std::uint64_t grid_bits = rsl::IQ1S_GRID_SYCL[idx_l[l]];
                    const float dl = dl_l[l];
                    const float dl_delta = dl * delta_l[l];
                    for (int j = 0; j < 8; ++j) {
                        const std::int8_t gi = static_cast<std::int8_t>((grid_bits >> (j * 8)) & 0xFFu);
                        const float val_w = dl * static_cast<float>(gi) + dl_delta;
                        dot += val_w * XLD(x_base + x_off + 8 * l + j);
                    }
                }
                x_off += 32;
            }
        }
    RSL_H8_END
}

// ---- IQ2_XXS ----
template <std::size_t LWS_T>
inline void matvec_iq2_xxs_f16in_packed_f32_usm_impl(
    sycl::queue& q, const void* w_bytes_usm, const uint16_t* x_f16, float* out_usm, int M, int K) {
    constexpr std::size_t LWS = LWS_T;
    const int blocks_per_row = K / 256;
    const int bytes_per_row = blocks_per_row * 66;
    const uint8_t* w_bytes = static_cast<const uint8_t*>(w_bytes_usm);
    RSL_H8_BEGIN(LWS)
        const uint8_t* row = w_bytes + m * bytes_per_row;
        for (int b = 0; b < blocks_per_row; ++b) {
            const uint8_t* blk = row + b * 66;
            const float d = bits_to_f32(static_cast<uint16_t>(blk[0]) | (static_cast<uint16_t>(blk[1]) << 8));
            const uint8_t* qs = blk + 2;
            const int x_base = b * 256;
            for (int ib32 = 0; ib32 < 8; ++ib32) {
                const std::uint32_t aux0 = static_cast<std::uint32_t>(qs[8 * ib32]) | (static_cast<std::uint32_t>(qs[8 * ib32 + 1]) << 8) | (static_cast<std::uint32_t>(qs[8 * ib32 + 2]) << 16) | (static_cast<std::uint32_t>(qs[8 * ib32 + 3]) << 24);
                const std::uint32_t aux1 = static_cast<std::uint32_t>(qs[8 * ib32 + 4]) | (static_cast<std::uint32_t>(qs[8 * ib32 + 5]) << 8) | (static_cast<std::uint32_t>(qs[8 * ib32 + 6]) << 16) | (static_cast<std::uint32_t>(qs[8 * ib32 + 7]) << 24);
                const float db = d * (0.5f + static_cast<float>(aux1 >> 28)) * 0.25f;
                for (int l = 0; l < 4; ++l) {
                    const std::size_t grid_idx = (aux0 >> (8 * l)) & 0xFFu;
                    const std::uint64_t grid_bits = rsl::IQ2XXS_GRID_SYCL[grid_idx];
                    const std::uint8_t signs = rsl::KSIGNS_IQ2XS_SYCL[(aux1 >> (7 * l)) & 127u];
                    const int x_off = x_base + ib32 * 32 + l * 8;
                    for (int j = 0; j < 8; ++j) {
                        const std::uint8_t gi = static_cast<std::uint8_t>((grid_bits >> (j * 8)) & 0xFFu);
                        const float s = (signs & rsl::KMASK_IQ2XS_SYCL[j]) ? -1.0f : 1.0f;
                        dot += db * static_cast<float>(gi) * s * XLD(x_off + j);
                    }
                }
            }
        }
    RSL_H8_END
}

// ---- IQ2_XS ----
template <std::size_t LWS_T>
inline void matvec_iq2_xs_f16in_packed_f32_usm_impl(
    sycl::queue& q, const void* w_bytes_usm, const uint16_t* x_f16, float* out_usm, int M, int K) {
    constexpr std::size_t LWS = LWS_T;
    const int blocks_per_row = K / 256;
    const int bytes_per_row = blocks_per_row * 74;
    const uint8_t* w_bytes = static_cast<const uint8_t*>(w_bytes_usm);
    RSL_H8_BEGIN(LWS)
        const uint8_t* row = w_bytes + m * bytes_per_row;
        for (int b = 0; b < blocks_per_row; ++b) {
            const uint8_t* blk = row + b * 74;
            const float d = bits_to_f32(static_cast<uint16_t>(blk[0]) | (static_cast<uint16_t>(blk[1]) << 8));
            const uint8_t* qs = blk + 2;
            const uint8_t* scales = blk + 2 + 64;
            const int x_base = b * 256;
            for (int ib32 = 0; ib32 < 8; ++ib32) {
                const std::uint8_t scale_byte = scales[ib32];
                const float db_lo = d * (0.5f + static_cast<float>(scale_byte & 0x0Fu)) * 0.25f;
                const float db_hi = d * (0.5f + static_cast<float>(scale_byte >> 4)) * 0.25f;
                const int base = 8 * ib32;
                for (int l = 0; l < 4; ++l) {
                    const std::uint16_t qd = static_cast<std::uint16_t>(qs[base + 2 * l]) | (static_cast<std::uint16_t>(qs[base + 2 * l + 1]) << 8);
                    const std::size_t grid_idx = static_cast<std::size_t>(qd & 0x1FFu);
                    const std::size_t sign_idx = static_cast<std::size_t>(qd >> 9);
                    const std::uint64_t grid_bits = rsl::IQ2XS_GRID_SYCL[grid_idx];
                    const std::uint8_t signs = rsl::KSIGNS_IQ2XS_SYCL[sign_idx];
                    const float db = (l < 2) ? db_lo : db_hi;
                    const int x_off = x_base + ib32 * 32 + l * 8;
                    for (int j = 0; j < 8; ++j) {
                        const std::uint8_t gi = static_cast<std::uint8_t>((grid_bits >> (j * 8)) & 0xFFu);
                        const float s = (signs & rsl::KMASK_IQ2XS_SYCL[j]) ? -1.0f : 1.0f;
                        dot += db * static_cast<float>(gi) * s * XLD(x_off + j);
                    }
                }
            }
        }
    RSL_H8_END
}

// ---- IQ2_S ----
template <std::size_t LWS_T>
inline void matvec_iq2_s_f16in_packed_f32_usm_impl(
    sycl::queue& q, const void* w_bytes_usm, const uint16_t* x_f16, float* out_usm, int M, int K) {
    constexpr std::size_t LWS = LWS_T;
    const int blocks_per_row = K / 256;
    const int bytes_per_row = blocks_per_row * 82;
    const uint8_t* w_bytes = static_cast<const uint8_t*>(w_bytes_usm);
    RSL_H8_BEGIN(LWS)
        const uint8_t* row = w_bytes + m * bytes_per_row;
        for (int b = 0; b < blocks_per_row; ++b) {
            const uint8_t* blk = row + b * 82;
            const float d = bits_to_f32(static_cast<uint16_t>(blk[0]) | (static_cast<uint16_t>(blk[1]) << 8));
            const uint8_t* qs_lo = blk + 2;
            const uint8_t* signs = blk + 2 + 32;
            const uint8_t* qh = blk + 2 + 64;
            const uint8_t* scales = blk + 2 + 64 + 8;
            const int x_base = b * 256;
            for (int ib32 = 0; ib32 < 8; ++ib32) {
                const std::uint8_t scale_byte = scales[ib32];
                const float db_lo = d * (0.5f + static_cast<float>(scale_byte & 0x0Fu)) * 0.25f;
                const float db_hi = d * (0.5f + static_cast<float>(scale_byte >> 4)) * 0.25f;
                const int qs_off = ib32 * 4;
                const std::uint8_t qh_byte = qh[ib32];
                for (int l = 0; l < 4; ++l) {
                    const std::size_t high_bits = (static_cast<std::size_t>(qh_byte) << (8 - 2 * l)) & 0x300u;
                    const std::size_t grid_idx = static_cast<std::size_t>(qs_lo[qs_off + l]) | high_bits;
                    const std::uint8_t sign_byte = signs[qs_off + l];
                    const std::uint64_t grid_bits = rsl::IQ2S_GRID_SYCL[grid_idx];
                    const float db = (l < 2) ? db_lo : db_hi;
                    const int x_off = x_base + ib32 * 32 + l * 8;
                    for (int j = 0; j < 8; ++j) {
                        const std::uint8_t gi = static_cast<std::uint8_t>((grid_bits >> (j * 8)) & 0xFFu);
                        const float s = (sign_byte & rsl::KMASK_IQ2XS_SYCL[j]) ? -1.0f : 1.0f;
                        dot += db * static_cast<float>(gi) * s * XLD(x_off + j);
                    }
                }
            }
        }
    RSL_H8_END
}

// ---- IQ3_XXS ----
template <std::size_t LWS_T>
inline void matvec_iq3_xxs_f16in_packed_f32_usm_impl(
    sycl::queue& q, const void* w_bytes_usm, const uint16_t* x_f16, float* out_usm, int M, int K) {
    constexpr std::size_t LWS = LWS_T;
    const int blocks_per_row = K / 256;
    const int bytes_per_row = blocks_per_row * 98;
    const uint8_t* w_bytes = static_cast<const uint8_t*>(w_bytes_usm);
    RSL_H8_BEGIN(LWS)
        const uint8_t* row = w_bytes + m * bytes_per_row;
        for (int b = 0; b < blocks_per_row; ++b) {
            const uint8_t* blk = row + b * 98;
            const float d = bits_to_f32(static_cast<uint16_t>(blk[0]) | (static_cast<uint16_t>(blk[1]) << 8));
            const uint8_t* qs_grid = blk + 2;
            const uint8_t* qs_sas = blk + 2 + 64;
            const int x_base = b * 256;
            for (int ib32 = 0; ib32 < 8; ++ib32) {
                const std::uint32_t aux32 = static_cast<std::uint32_t>(qs_sas[4 * ib32]) | (static_cast<std::uint32_t>(qs_sas[4 * ib32 + 1]) << 8) | (static_cast<std::uint32_t>(qs_sas[4 * ib32 + 2]) << 16) | (static_cast<std::uint32_t>(qs_sas[4 * ib32 + 3]) << 24);
                const float db = d * (0.5f + static_cast<float>(aux32 >> 28)) * 0.5f;
                const int qs_off = 8 * ib32;
                for (int l = 0; l < 4; ++l) {
                    const std::size_t g1_idx = static_cast<std::size_t>(qs_grid[qs_off + 2 * l]);
                    const std::size_t g2_idx = static_cast<std::size_t>(qs_grid[qs_off + 2 * l + 1]);
                    const std::uint32_t grid1 = rsl::IQ3XXS_GRID_SYCL[g1_idx];
                    const std::uint32_t grid2 = rsl::IQ3XXS_GRID_SYCL[g2_idx];
                    const std::uint8_t signs = rsl::KSIGNS_IQ2XS_SYCL[(aux32 >> (7 * l)) & 127u];
                    const int x_off = x_base + ib32 * 32 + l * 8;
                    for (int j = 0; j < 4; ++j) {
                        const std::uint8_t g1 = static_cast<std::uint8_t>((grid1 >> (j * 8)) & 0xFFu);
                        const std::uint8_t g2 = static_cast<std::uint8_t>((grid2 >> (j * 8)) & 0xFFu);
                        const float s_lo = (signs & rsl::KMASK_IQ2XS_SYCL[j]) ? -1.0f : 1.0f;
                        const float s_hi = (signs & rsl::KMASK_IQ2XS_SYCL[j + 4]) ? -1.0f : 1.0f;
                        dot += db * static_cast<float>(g1) * s_lo * XLD(x_off + j);
                        dot += db * static_cast<float>(g2) * s_hi * XLD(x_off + j + 4);
                    }
                }
            }
        }
    RSL_H8_END
}

// ---- IQ3_S ----
template <std::size_t LWS_T>
inline void matvec_iq3_s_f16in_packed_f32_usm_impl(
    sycl::queue& q, const void* w_bytes_usm, const uint16_t* x_f16, float* out_usm, int M, int K) {
    constexpr std::size_t LWS = LWS_T;
    const int blocks_per_row = K / 256;
    const int bytes_per_row = blocks_per_row * 110;
    const uint8_t* w_bytes = static_cast<const uint8_t*>(w_bytes_usm);
    RSL_H8_BEGIN(LWS)
        const uint8_t* row = w_bytes + m * bytes_per_row;
        for (int b = 0; b < blocks_per_row; ++b) {
            const uint8_t* blk = row + b * 110;
            const float d = bits_to_f32(static_cast<uint16_t>(blk[0]) | (static_cast<uint16_t>(blk[1]) << 8));
            const uint8_t* qs = blk + 2;
            const uint8_t* qh = blk + 2 + 64;
            const uint8_t* signs = blk + 2 + 64 + 8;
            const uint8_t* scales = blk + 2 + 64 + 8 + 32;
            const int x_base = b * 256;
            for (int ib32 = 0; ib32 < 8; ++ib32) {
                const int pair = ib32 >> 1;
                const std::uint8_t scale_byte = scales[pair];
                const float db = (ib32 & 1)
                    ? d * (1.0f + 2.0f * static_cast<float>(scale_byte >> 4))
                    : d * (1.0f + 2.0f * static_cast<float>(scale_byte & 0x0Fu));
                const int qs_off = ib32 * 8, signs_off = ib32 * 4;
                const std::uint8_t qh_byte = qh[ib32];
                const int x_off_block = ib32 * 32;
                for (int l = 0; l < 4; ++l) {
                    const std::size_t g1_idx = static_cast<std::size_t>(qs[qs_off + 2 * l]) | ((static_cast<std::size_t>(qh_byte) << (8 - 2 * l)) & 0x100u);
                    const std::size_t g2_idx = static_cast<std::size_t>(qs[qs_off + 2 * l + 1]) | ((static_cast<std::size_t>(qh_byte) << (7 - 2 * l)) & 0x100u);
                    const std::uint32_t grid1 = rsl::IQ3S_GRID_SYCL[g1_idx];
                    const std::uint32_t grid2 = rsl::IQ3S_GRID_SYCL[g2_idx];
                    const std::uint8_t sign_byte = signs[signs_off + l];
                    const int x_off = x_base + x_off_block + l * 8;
                    for (int j = 0; j < 4; ++j) {
                        const std::uint8_t g1 = static_cast<std::uint8_t>((grid1 >> (j * 8)) & 0xFFu);
                        const std::uint8_t g2 = static_cast<std::uint8_t>((grid2 >> (j * 8)) & 0xFFu);
                        const float s_lo = (sign_byte & rsl::KMASK_IQ2XS_SYCL[j]) ? -1.0f : 1.0f;
                        const float s_hi = (sign_byte & rsl::KMASK_IQ2XS_SYCL[j + 4]) ? -1.0f : 1.0f;
                        dot += db * static_cast<float>(g1) * s_lo * XLD(x_off + j);
                        dot += db * static_cast<float>(g2) * s_hi * XLD(x_off + j + 4);
                    }
                }
            }
        }
    RSL_H8_END
}

// ---- PTQ1_0 ----
// Same trit decode as the packed matvec; activation read as F16 via XLD.
// The d scale folds per block into `dot`.
template <std::size_t LWS_T>
inline void matvec_ptq1_0_f16in_packed_f32_usm_impl(
    sycl::queue& q, const void* w_bytes_usm, const uint16_t* x_f16, float* out_usm, int M, int K) {
    constexpr std::size_t LWS = LWS_T;
    const int blocks_per_row = K / 128;
    const int bytes_per_row = blocks_per_row * 28;
    const uint8_t* w_bytes = static_cast<const uint8_t*>(w_bytes_usm);
    RSL_H8_BEGIN(LWS)
        const uint8_t pow3[5] = {1, 3, 9, 27, 81};
        const uint8_t* row = w_bytes + m * bytes_per_row;
        for (int b = 0; b < blocks_per_row; ++b) {
            const uint8_t* blk = row + b * 28;
            const uint8_t* qs = blk;
            const uint8_t* qh = blk + 24;
            const float d = bits_to_f32(static_cast<uint16_t>(blk[26])
                                        | (static_cast<uint16_t>(blk[27]) << 8));
            const int x_base = b * 128;
            float sum = 0.0f;
            for (int n = 0; n < 5; ++n) {
                const uint8_t p3 = pow3[n];
                const int e0 = x_base + n * 16;
                for (int mm = 0; mm < 16; ++mm) {
                    const int trit = ((static_cast<int>(static_cast<uint8_t>(qs[mm] * p3)) * 3) >> 8) - 1;
                    sum += static_cast<float>(trit) * XLD(e0 + mm);
                }
            }
            for (int n = 0; n < 5; ++n) {
                const uint8_t p3 = pow3[n];
                const int e0 = x_base + 80 + n * 8;
                for (int mm = 0; mm < 8; ++mm) {
                    const int trit = ((static_cast<int>(static_cast<uint8_t>(qs[16 + mm] * p3)) * 3) >> 8) - 1;
                    sum += static_cast<float>(trit) * XLD(e0 + mm);
                }
            }
            for (int n = 0; n < 4; ++n) {
                const uint8_t p3 = pow3[n];
                const int e0 = x_base + 120 + n * 2;
                for (int hh = 0; hh < 2; ++hh) {
                    const int trit = ((static_cast<int>(static_cast<uint8_t>(qh[hh] * p3)) * 3) >> 8) - 1;
                    sum += static_cast<float>(trit) * XLD(e0 + hh);
                }
            }
            dot += d * sum;
        }
    RSL_H8_END
}

extern "C" {

#define RSL_H8_WRAPPER(NAME, IMPL, ALIGN) \
void NAME(rsl_stream* s, const void* w_bytes_usm, const uint16_t* x_f16, \
          float* out_usm, int M, int K, int lws) RSL_FFI_BODY_VOID(#NAME, { \
    if (s == nullptr || w_bytes_usm == nullptr || x_f16 == nullptr || out_usm == nullptr) return; \
    if (M <= 0 || K <= 0 || (K % (ALIGN)) != 0) return; \
    auto& q = s->q; \
    const int eff_lws = (lws <= 0) ? static_cast<int>(RSL_LWS) : lws; \
    switch (eff_lws) { \
        case 16:  IMPL<16>(q, w_bytes_usm, x_f16, out_usm, M, K); break; \
        case 32:  IMPL<32>(q, w_bytes_usm, x_f16, out_usm, M, K); break; \
        case 64:  IMPL<64>(q, w_bytes_usm, x_f16, out_usm, M, K); break; \
        case 128: IMPL<128>(q, w_bytes_usm, x_f16, out_usm, M, K); break; \
        case 256: IMPL<256>(q, w_bytes_usm, x_f16, out_usm, M, K); break; \
        default:  IMPL<64>(q, w_bytes_usm, x_f16, out_usm, M, K); break; \
    } \
})

RSL_H8_WRAPPER(rsl_matvec_q8_0_f16in_packed_f32_usm,  matvec_q8_0_f16in_packed_f32_usm_impl,  32)
RSL_H8_WRAPPER(rsl_matvec_q4_k_f16in_packed_f32_usm,  matvec_q4_k_f16in_packed_f32_usm_impl,  256)
RSL_H8_WRAPPER(rsl_matvec_q5_k_f16in_packed_f32_usm,  matvec_q5_k_f16in_packed_f32_usm_impl,  256)
RSL_H8_WRAPPER(rsl_matvec_q6_k_f16in_packed_f32_usm,  matvec_q6_k_f16in_packed_f32_usm_impl,  256)
RSL_H8_WRAPPER(rsl_matvec_iq4_nl_f16in_packed_f32_usm, matvec_iq4_nl_f16in_packed_f32_usm_impl, 32)
RSL_H8_WRAPPER(rsl_matvec_iq4_xs_f16in_packed_f32_usm, matvec_iq4_xs_f16in_packed_f32_usm_impl, 256)
RSL_H8_WRAPPER(rsl_matvec_iq1_s_f16in_packed_f32_usm,  matvec_iq1_s_f16in_packed_f32_usm_impl,  256)
RSL_H8_WRAPPER(rsl_matvec_iq1_m_f16in_packed_f32_usm,  matvec_iq1_m_f16in_packed_f32_usm_impl,  256)
RSL_H8_WRAPPER(rsl_matvec_iq2_xxs_f16in_packed_f32_usm, matvec_iq2_xxs_f16in_packed_f32_usm_impl, 256)
RSL_H8_WRAPPER(rsl_matvec_iq2_xs_f16in_packed_f32_usm, matvec_iq2_xs_f16in_packed_f32_usm_impl, 256)
RSL_H8_WRAPPER(rsl_matvec_iq2_s_f16in_packed_f32_usm,  matvec_iq2_s_f16in_packed_f32_usm_impl,  256)
RSL_H8_WRAPPER(rsl_matvec_iq3_xxs_f16in_packed_f32_usm, matvec_iq3_xxs_f16in_packed_f32_usm_impl, 256)
RSL_H8_WRAPPER(rsl_matvec_iq3_s_f16in_packed_f32_usm,  matvec_iq3_s_f16in_packed_f32_usm_impl,  256)
RSL_H8_WRAPPER(rsl_matvec_ptq1_0_f16in_packed_f32_usm, matvec_ptq1_0_f16in_packed_f32_usm_impl, 128)

// Rotary positional embedding, half-split (neox / HF-converted-GGUF
// convention): for each head, pair (qk[j], qk[j + head_dim/2]) gets
// rotated by angle = pos * inv_freq[j]. In-place on qk. Work-item per
// (head, j) pair; head_dim/2 pairs per head.
void rsl_rope(rsl_stream* s,
              uint16_t* qk, int n_heads, int head_dim, int pos,
              const uint16_t* inv_freq) RSL_FFI_BODY_VOID("rsl_rope", {
    if (s == nullptr || qk == nullptr || inv_freq == nullptr) return;
    if (n_heads <= 0 || head_dim <= 0 || (head_dim % 2) != 0) return;
    auto& q = s->q;
    const int half = head_dim / 2;
    const std::size_t total = static_cast<std::size_t>(n_heads) * head_dim;
    auto* qk_d = sycl::malloc_device<uint16_t>(total, q);
    auto* freq_d = sycl::malloc_device<uint16_t>(half, q);
    if (!qk_d || !freq_d) {
        if (qk_d) sycl::free(qk_d, q);
        if (freq_d) sycl::free(freq_d, q);
        return;
    }
    q.memcpy(qk_d, qk, total * sizeof(uint16_t));
    q.memcpy(freq_d, inv_freq, half * sizeof(uint16_t));
    q.wait();
    const int rope_total = n_heads * half;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(round_up_to_lws(rope_total)),
                              sycl::range<1>(RSL_LWS)),
            [=](sycl::nd_item<1> it) {
            const int flat = static_cast<int>(it.get_global_id(0));
            if (flat >= rope_total) return;
            const int head = flat / half;
            const int j = flat % half;
            const float freq = bits_to_f32(freq_d[j]);
            const float angle = static_cast<float>(pos) * freq;
            const float c = sycl::cos(angle);
            const float si = sycl::sin(angle);
            const int base = head * head_dim;
            const float x0 = bits_to_f32(qk_d[base + j]);
            const float x1 = bits_to_f32(qk_d[base + j + half]);
            qk_d[base + j] = f32_to_bits(x0 * c - x1 * si);
            qk_d[base + j + half] = f32_to_bits(x0 * si + x1 * c);
        });
    }).wait();
    q.memcpy(qk, qk_d, total * sizeof(uint16_t)).wait();
    sycl::free(qk_d, q);
    sycl::free(freq_d, q);
})

// In-place softmax across the kv_len dimension, fused with the
// `scale` multiply and (optional) additive mask. Layout:
//   qk_scores[n_heads, seq, kv_len], row-major.
// Each work-item handles one (head, query) pair — a `kv_len`-wide
// pass that does max + exp/sum + normalize in three loops. We use
// the standard "subtract max before exp" trick for numerical
// stability.
//
// `mask` is optional: when null we treat it as all-zeros (no
// position is masked).
void rsl_softmax_attn(rsl_stream* s,
                      uint16_t* qk_scores, const uint16_t* mask,
                      int n_heads, int seq, int kv_len, float scale) RSL_FFI_BODY_VOID("rsl_softmax_attn", {
    if (s == nullptr || qk_scores == nullptr) return;
    if (n_heads <= 0 || seq <= 0 || kv_len <= 0) return;
    auto& q = s->q;
    const std::size_t total =
        static_cast<std::size_t>(n_heads) * seq * kv_len;
    auto* scores_d = sycl::malloc_device<uint16_t>(total, q);
    uint16_t* mask_d = nullptr;
    if (mask) {
        mask_d = sycl::malloc_device<uint16_t>(kv_len, q);
        if (!mask_d) {
            sycl::free(scores_d, q);
            return;
        }
        q.memcpy(mask_d, mask, kv_len * sizeof(uint16_t));
    }
    q.memcpy(scores_d, qk_scores, total * sizeof(uint16_t));
    q.wait();
    const bool has_mask = (mask_d != nullptr);
    const int softmax_total = n_heads * seq;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(round_up_to_lws(softmax_total)),
                              sycl::range<1>(RSL_LWS)),
            [=](sycl::nd_item<1> it) {
            const int flat = static_cast<int>(it.get_global_id(0));
            if (flat >= softmax_total) return;
            const int head = flat / seq;
            const int q_pos = flat % seq;
            const int base =
                head * seq * kv_len + q_pos * kv_len;

            // Pass 1: find max for numerical stability.
            float m = -INFINITY;
            for (int k = 0; k < kv_len; ++k) {
                float v = bits_to_f32(scores_d[base + k]) * scale;
                if (has_mask) v += bits_to_f32(mask_d[k]);
                if (v > m) m = v;
            }
            // Pass 2: exp(v - m), accumulate sum, write the exp back.
            float sum = 0.0f;
            for (int k = 0; k < kv_len; ++k) {
                float v = bits_to_f32(scores_d[base + k]) * scale;
                if (has_mask) v += bits_to_f32(mask_d[k]);
                float e = sycl::exp(v - m);
                scores_d[base + k] = f32_to_bits(e);
                sum += e;
            }
            // Pass 3: normalize.
            const float inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
            for (int k = 0; k < kv_len; ++k) {
                float e = bits_to_f32(scores_d[base + k]) * inv_sum;
                scores_d[base + k] = f32_to_bits(e);
            }
        });
    }).wait();
    q.memcpy(qk_scores, scores_d, total * sizeof(uint16_t)).wait();
    sycl::free(scores_d, q);
    if (mask_d) sycl::free(mask_d, q);
})

// SwiGLU activation: out[i] = silu(x[i]) * y[i] where
// silu(v) = v * sigmoid(v) = v / (1 + exp(-v)).
// One work-item per element.
void rsl_silu_mul(rsl_stream* s,
                  const uint16_t* x, const uint16_t* y, uint16_t* out, int n) RSL_FFI_BODY_VOID("rsl_silu_mul", {
    if (s == nullptr || x == nullptr || y == nullptr || out == nullptr) return;
    if (n <= 0) return;
    auto& q = s->q;
    const std::size_t total = static_cast<std::size_t>(n);
    auto* xd = sycl::malloc_device<uint16_t>(total, q);
    auto* yd = sycl::malloc_device<uint16_t>(total, q);
    auto* od = sycl::malloc_device<uint16_t>(total, q);
    if (!xd || !yd || !od) {
        if (xd) sycl::free(xd, q);
        if (yd) sycl::free(yd, q);
        if (od) sycl::free(od, q);
        return;
    }
    q.memcpy(xd, x, total * sizeof(uint16_t));
    q.memcpy(yd, y, total * sizeof(uint16_t));
    q.wait();
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(round_up_to_lws(total)),
                              sycl::range<1>(RSL_LWS)),
            [=](sycl::nd_item<1> it) {
            const std::size_t i = it.get_global_id(0);
            if (i >= total) return;
            const float xv = bits_to_f32(xd[i]);
            const float yv = bits_to_f32(yd[i]);
            // silu(x) = x * sigmoid(x); 1 / (1 + exp(-x)) is the
            // numerically stable form for both signs.
            const float silu = xv / (1.0f + sycl::exp(-xv));
            od[i] = f32_to_bits(silu * yv);
        });
    }).wait();
    q.memcpy(out, od, total * sizeof(uint16_t)).wait();
    sycl::free(xd, q);
    sycl::free(yd, q);
    sycl::free(od, q);
})

// Gather rows from `table[V, d]` indexed by `ids[N]` into `out[N, d]`.
// Negative ids (or ids beyond table bounds) produce a zero row — same
// safety net the CPU side uses for tokens that decode-then-detokenize
// outside the vocab.
void rsl_embedding_lookup(rsl_stream* s,
                          const uint16_t* table, const int32_t* ids,
                          uint16_t* out, int n_ids, int d) RSL_FFI_BODY_VOID("rsl_embedding_lookup", {
    if (s == nullptr || table == nullptr || ids == nullptr || out == nullptr) return;
    if (n_ids <= 0 || d <= 0) return;
    auto& q = s->q;
    // The table can be large; copy into device memory once per call.
    // A persistent-table caching layer is a v1.x optimization.
    // For now we just need correctness on the GPU side.
    // NOTE: callers that already have `table` resident in USM should
    // bypass this function — that's the path the engine will take in
    // Phase 4 once load-time placement is in place.
    const std::size_t out_sz = static_cast<std::size_t>(n_ids) * d;
    auto* td = sycl::malloc_device<uint16_t>(0, q);  // sized below.
    (void)td;
    // We don't know V here — caller passes `table` sized [V, d] but we
    // only need the rows we look up. Two options:
    //   1. Copy the entire table (caller knows V).
    //   2. Gather row-by-row from the host pointer (slow if N is large).
    // For Phase 3.1 we expect this to be called only at the very start
    // of a forward pass with N tiny (decode: N=1; prefill: N up to
    // ctx_size). The CPU-side host pointer + small-N case → row-by-row
    // is fine.
    auto* od = sycl::malloc_device<uint16_t>(out_sz, q);
    auto* ids_d = sycl::malloc_device<int32_t>(n_ids, q);
    if (!od || !ids_d) {
        if (od) sycl::free(od, q);
        if (ids_d) sycl::free(ids_d, q);
        return;
    }
    q.memcpy(ids_d, ids, n_ids * sizeof(int32_t)).wait();

    // For each id, host-side memcpy the row into `od`. This avoids
    // staging the full table just to fetch a handful of rows.
    // (Negative ids zero the row.)
    for (int i = 0; i < n_ids; ++i) {
        const int32_t row = ids[i];
        if (row < 0) {
            q.memset(od + static_cast<std::size_t>(i) * d, 0,
                     d * sizeof(uint16_t));
        } else {
            q.memcpy(od + static_cast<std::size_t>(i) * d,
                     table + static_cast<std::size_t>(row) * d,
                     d * sizeof(uint16_t));
        }
    }
    q.wait();
    q.memcpy(out, od, out_sz * sizeof(uint16_t)).wait();
    sycl::free(od, q);
    sycl::free(ids_d, q);
})


// Phase 4+ kernels — stubs in both modes. Filled in alongside the
// hybrid CPU/GPU model forward pass.
void rsl_gemm_q4k_f16(rsl_stream*, const void*, const uint16_t*, uint16_t*,
                      int, int, int) {
    // phase 4
}

void rsl_gemm_q8_0_f16(rsl_stream*, const void*, const uint16_t*, uint16_t*,
                       int, int, int) {
    // phase 4
}

void rsl_sample_argmax(rsl_stream*, const uint16_t*, int, int32_t*) {
    // phase 3.2 — full sampler stays on the CPU for v1; argmax on
    // GPU only matters once we add speculative decoding's
    // verification step.
}

// ----- NVFP4 kernels -----
//
// Hardware FP4 tensor cores are Blackwell-only; on Intel / AMD GPUs
// we software-decode FP4 → f32 inside the kernel and accumulate in
// f32, then round to f16 on store. This is correctness-preserving
// (FP4's non-uniform spacing carries through unchanged) but doesn't
// hit Blackwell's 2× BF16 throughput rate. The memory savings (~8×
// over F32 weights, ~4× over F16) are real on every backend.
//
// Both kernels live behind the same SYCL device path as the rest
// of the GEMM family. The non-SYCL stubs return zero-output.


namespace {

// E2M1 codebook (mirrors `rustllama_kernels_cpu::nvfp4::NVFP4_CODEBOOK`).
// Sixteen f32 values indexed by the 4-bit code. Kept inside the
// kernel TU so the GPU side has the table without an extra device
// pointer crossing the ABI.
constexpr float kNvfp4Codebook[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

// FP8 E4M3 → f32. Mirrors the CPU implementation exactly so the
// per-block scales decode bit-identically.
inline float e4m3_to_f32_dev(uint8_t b) {
    const bool sign = (b & 0x80) != 0;
    const uint8_t exp = (b >> 3) & 0x0F;
    const uint8_t mant = b & 0x07;
    if (exp == 0x0F && mant == 0x07) {
        // NaN — propagate. SYCL math handles NaN propagation natively.
        return sycl::nan(0u);
    }
    float val;
    if (exp == 0) {
        val = static_cast<float>(mant) * (1.0f / 512.0f);
    } else {
        const float m = 1.0f + static_cast<float>(mant) / 8.0f;
        const int e = static_cast<int>(exp) - 7;
        val = m * sycl::native::powr(2.0f, static_cast<float>(e));
    }
    return sign ? -val : val;
}

}  // namespace

void rsl_dequant_nvfp4(rsl_stream* s,
                       const void* w_nvfp4,
                       uint16_t* out,
                       int n_blocks) {
    if (s == nullptr || w_nvfp4 == nullptr || out == nullptr) return;
    if (n_blocks <= 0) return;
    auto& q = s->q;
    constexpr int kBlockBytes = 9;
    constexpr int kBlockElems = 16;
    const std::size_t in_sz = static_cast<std::size_t>(n_blocks) * kBlockBytes;
    const std::size_t out_sz = static_cast<std::size_t>(n_blocks) * kBlockElems;
    auto* wd = sycl::malloc_device<uint8_t>(in_sz, q);
    auto* od = sycl::malloc_device<uint16_t>(out_sz, q);
    if (!wd || !od) {
        if (wd) sycl::free(wd, q);
        if (od) sycl::free(od, q);
        return;
    }
    q.memcpy(wd, w_nvfp4, in_sz);
    q.wait();
    // One work-item per block; each unpacks 8 bytes → 16 f16s.
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(round_up_to_lws(n_blocks)),
                              sycl::range<1>(RSL_LWS)),
            [=](sycl::nd_item<1> it) {
            const int b = static_cast<int>(it.get_global_id(0));
            if (b >= n_blocks) return;
            const int in_off = b * kBlockBytes;
            const int out_off = b * kBlockElems;
            const float scale = e4m3_to_f32_dev(wd[in_off + 8]);
            for (int j = 0; j < 8; ++j) {
                const uint8_t byte = wd[in_off + j];
                const int lo = byte & 0x0F;
                const int hi = (byte >> 4) & 0x0F;
                od[out_off + j * 2] = f32_to_bits(kNvfp4Codebook[lo] * scale);
                od[out_off + j * 2 + 1] = f32_to_bits(kNvfp4Codebook[hi] * scale);
            }
        });
    }).wait();
    q.memcpy(out, od, out_sz * sizeof(uint16_t)).wait();
    sycl::free(wd, q);
    sycl::free(od, q);
}

void rsl_matvec_nvfp4_f16(rsl_stream* s,
                          const void* w_nvfp4,
                          const uint16_t* x,
                          uint16_t* out,
                          int M, int K) {
    if (s == nullptr || w_nvfp4 == nullptr || x == nullptr || out == nullptr) return;
    if (M <= 0 || K <= 0) return;
    constexpr int kBlockBytes = 9;
    constexpr int kBlockElems = 16;
    if ((K % kBlockElems) != 0) return;  // shape constraint
    const int blocks_per_row = K / kBlockElems;
    auto& q = s->q;
    const std::size_t w_sz =
        static_cast<std::size_t>(M) * blocks_per_row * kBlockBytes;
    const std::size_t x_sz = static_cast<std::size_t>(K) * sizeof(uint16_t);
    const std::size_t out_sz = static_cast<std::size_t>(M) * sizeof(uint16_t);
    auto* wd = sycl::malloc_device<uint8_t>(w_sz, q);
    auto* xd = sycl::malloc_device<uint16_t>(K, q);
    auto* od = sycl::malloc_device<uint16_t>(M, q);
    if (!wd || !xd || !od) {
        if (wd) sycl::free(wd, q);
        if (xd) sycl::free(xd, q);
        if (od) sycl::free(od, q);
        return;
    }
    q.memcpy(wd, w_nvfp4, w_sz);
    q.memcpy(xd, x, x_sz);
    q.wait();
    // One work-item per output row. Per row: walk blocks_per_row
    // blocks, dequant inline, accumulate dot-product in f32.
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(round_up_to_lws(M)),
                              sycl::range<1>(RSL_LWS)),
            [=](sycl::nd_item<1> it) {
            const int row = static_cast<int>(it.get_global_id(0));
            if (row >= M) return;
            const int row_start = row * blocks_per_row * kBlockBytes;
            float acc = 0.0f;
            for (int b = 0; b < blocks_per_row; ++b) {
                const int off = row_start + b * kBlockBytes;
                const float scale = e4m3_to_f32_dev(wd[off + 8]);
                for (int j = 0; j < 8; ++j) {
                    const uint8_t byte = wd[off + j];
                    const int lo = byte & 0x0F;
                    const int hi = (byte >> 4) & 0x0F;
                    const int x_idx = b * kBlockElems + j * 2;
                    acc += scale * kNvfp4Codebook[lo] * bits_to_f32(xd[x_idx]);
                    acc += scale * kNvfp4Codebook[hi] * bits_to_f32(xd[x_idx + 1]);
                }
            }
            od[row] = f32_to_bits(acc);
        });
    }).wait();
    q.memcpy(out, od, out_sz).wait();
    sycl::free(wd, q);
    sycl::free(xd, q);
    sycl::free(od, q);
}

// G5: K-quant → F32 dequant kernels (USM-in/USM-out). Used by the
// quantize pipeline to skip CPU dequant for IQ tensors whose source
// is Q3_K / Q4_K / Q5_K / Q6_K. Caller manages USM buffers; this fn
// just dispatches the kernel and waits.
//
// Per-block work-item topology (one work-item per 256-element
// super-block): trivially parallel, no cross-block dependence. For
// an 80 MiB Q4_K tensor that's ~80000 work-items — saturates the
// iGPU's EUs easily. Arithmetic mirrors the CPU references in
// rustllama_gguf::dequant exactly so parity tests can pin them.

void rsl_dequant_q4_k_to_f32_usm(rsl_stream* s,
                                  const void* bytes_usm,
                                  float* out_usm,
                                  int n_blocks) {
    if (s == nullptr || bytes_usm == nullptr || out_usm == nullptr) return;
    if (n_blocks <= 0) return;
    auto& q = s->q;
    const uint8_t* bytes = static_cast<const uint8_t*>(bytes_usm);
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(n_blocks), [=](sycl::id<1> id) {
            const int b = static_cast<int>(id[0]);
            const uint8_t* blk = bytes + b * 144;
            uint16_t d_bits = static_cast<uint16_t>(blk[0])
                              | (static_cast<uint16_t>(blk[1]) << 8);
            uint16_t dmin_bits = static_cast<uint16_t>(blk[2])
                                 | (static_cast<uint16_t>(blk[3]) << 8);
            const float d = bits_to_f32(d_bits);
            const float dmin = bits_to_f32(dmin_bits);
            const uint8_t* sb = blk + 4;
            uint8_t sc[8], mn[8];
            for (int j = 0; j < 8; ++j) {
                if (j < 4) {
                    sc[j] = sb[j] & 0x3F;
                    mn[j] = sb[j + 4] & 0x3F;
                } else {
                    sc[j] = (sb[j + 4] & 0x0F) | ((sb[j - 4] >> 6) << 4);
                    mn[j] = (sb[j + 4] >> 4) | ((sb[j] >> 6) << 4);
                }
            }
            const uint8_t* qs = blk + 16;
            float* out_blk = out_usm + b * 256;
            for (int group = 0; group < 4; ++group) {
                const uint8_t* qc = qs + group * 32;
                const float d_lo = d * static_cast<float>(sc[group * 2]);
                const float m_lo = dmin * static_cast<float>(mn[group * 2]);
                const float d_hi = d * static_cast<float>(sc[group * 2 + 1]);
                const float m_hi = dmin * static_cast<float>(mn[group * 2 + 1]);
                const int dst_lo = group * 64;
                for (int l = 0; l < 32; ++l) {
                    const uint8_t qb = qc[l];
                    out_blk[dst_lo + l] =
                        d_lo * static_cast<float>(qb & 0x0F) - m_lo;
                    out_blk[dst_lo + 32 + l] =
                        d_hi * static_cast<float>(qb >> 4) - m_hi;
                }
            }
        });
    }).wait();
}

void rsl_dequant_q3_k_to_f32_usm(rsl_stream* s,
                                  const void* bytes_usm,
                                  float* out_usm,
                                  int n_blocks) {
    if (s == nullptr || bytes_usm == nullptr || out_usm == nullptr) return;
    if (n_blocks <= 0) return;
    auto& q = s->q;
    const uint8_t* bytes = static_cast<const uint8_t*>(bytes_usm);
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(n_blocks), [=](sycl::id<1> id) {
            const int b = static_cast<int>(id[0]);
            const uint8_t* blk = bytes + b * 110;
            const uint8_t* hmask = blk;
            const uint8_t* qs = blk + 32;
            const uint8_t* sc_raw = blk + 32 + 64;
            uint16_t d_bits = static_cast<uint16_t>(blk[108])
                              | (static_cast<uint16_t>(blk[109]) << 8);
            const float d_all = bits_to_f32(d_bits);
            const uint32_t KMASK1 = 0x03030303u;
            const uint32_t KMASK2 = 0x0f0f0f0fu;
            uint32_t aux[4];
            aux[0] = static_cast<uint32_t>(sc_raw[0])
                     | (static_cast<uint32_t>(sc_raw[1]) << 8)
                     | (static_cast<uint32_t>(sc_raw[2]) << 16)
                     | (static_cast<uint32_t>(sc_raw[3]) << 24);
            aux[1] = static_cast<uint32_t>(sc_raw[4])
                     | (static_cast<uint32_t>(sc_raw[5]) << 8)
                     | (static_cast<uint32_t>(sc_raw[6]) << 16)
                     | (static_cast<uint32_t>(sc_raw[7]) << 24);
            aux[2] = static_cast<uint32_t>(sc_raw[8])
                     | (static_cast<uint32_t>(sc_raw[9]) << 8)
                     | (static_cast<uint32_t>(sc_raw[10]) << 16)
                     | (static_cast<uint32_t>(sc_raw[11]) << 24);
            uint32_t tmp = aux[2];
            aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
            aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
            aux[0] = (aux[0] & KMASK2) | ((tmp & KMASK1) << 4);
            aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
            int8_t scales[16];
            for (int j = 0; j < 16; ++j) {
                scales[j] = static_cast<int8_t>(
                    static_cast<uint8_t>((aux[j >> 2] >> ((j & 3) * 8)) & 0xFF));
            }
            float* out_blk = out_usm + b * 256;
            int q_cursor = 0;
            int y_cursor = 0;
            uint8_t m = 1;
            int is = 0;
            for (int chunk = 0; chunk < 2; ++chunk) {
                uint32_t shift = 0;
                for (int j = 0; j < 4; ++j) {
                    float dl = d_all * (static_cast<float>(scales[is]) - 32.0f);
                    ++is;
                    for (int l = 0; l < 16; ++l) {
                        const int lo = static_cast<int>(
                            (qs[q_cursor + l] >> shift) & 3);
                        const int hi_sub = (hmask[l] & m) != 0 ? 0 : 4;
                        out_blk[y_cursor++] =
                            dl * static_cast<float>(lo - hi_sub);
                    }
                    dl = d_all * (static_cast<float>(scales[is]) - 32.0f);
                    ++is;
                    for (int l = 0; l < 16; ++l) {
                        const int lo = static_cast<int>(
                            (qs[q_cursor + l + 16] >> shift) & 3);
                        const int hi_sub = (hmask[l + 16] & m) != 0 ? 0 : 4;
                        out_blk[y_cursor++] =
                            dl * static_cast<float>(lo - hi_sub);
                    }
                    shift += 2;
                    m = static_cast<uint8_t>(m << 1);
                }
                q_cursor += 32;
            }
        });
    }).wait();
}

void rsl_dequant_q5_k_to_f32_usm(rsl_stream* s,
                                  const void* bytes_usm,
                                  float* out_usm,
                                  int n_blocks) {
    if (s == nullptr || bytes_usm == nullptr || out_usm == nullptr) return;
    if (n_blocks <= 0) return;
    auto& q = s->q;
    const uint8_t* bytes = static_cast<const uint8_t*>(bytes_usm);
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(n_blocks), [=](sycl::id<1> id) {
            const int b = static_cast<int>(id[0]);
            const uint8_t* blk = bytes + b * 176;
            uint16_t d_bits = static_cast<uint16_t>(blk[0])
                              | (static_cast<uint16_t>(blk[1]) << 8);
            uint16_t dmin_bits = static_cast<uint16_t>(blk[2])
                                 | (static_cast<uint16_t>(blk[3]) << 8);
            const float d = bits_to_f32(d_bits);
            const float dmin = bits_to_f32(dmin_bits);
            const uint8_t* sb = blk + 4;
            const uint8_t* qh = blk + 16;
            const uint8_t* qs = blk + 48;
            uint8_t sc[8], mn[8];
            for (int j = 0; j < 8; ++j) {
                if (j < 4) {
                    sc[j] = sb[j] & 0x3F;
                    mn[j] = sb[j + 4] & 0x3F;
                } else {
                    sc[j] = (sb[j + 4] & 0x0F) | ((sb[j - 4] >> 6) << 4);
                    mn[j] = (sb[j + 4] >> 4) | ((sb[j] >> 6) << 4);
                }
            }
            float* out_blk = out_usm + b * 256;
            for (int group = 0; group < 4; ++group) {
                const uint8_t* qc = qs + group * 32;
                const float d_lo = d * static_cast<float>(sc[group * 2]);
                const float m_lo = dmin * static_cast<float>(mn[group * 2]);
                const float d_hi = d * static_cast<float>(sc[group * 2 + 1]);
                const float m_hi = dmin * static_cast<float>(mn[group * 2 + 1]);
                const int bit_lo = group * 2;
                const int bit_hi = group * 2 + 1;
                const int dst = group * 64;
                for (int l = 0; l < 32; ++l) {
                    const uint8_t qb = qc[l];
                    const uint8_t qhb = qh[l];
                    const int lo = static_cast<int>(qb & 0x0F);
                    const int hi = static_cast<int>(qb >> 4);
                    const int lo_full = lo + (((qhb >> bit_lo) & 1) ? 16 : 0);
                    const int hi_full = hi + (((qhb >> bit_hi) & 1) ? 16 : 0);
                    out_blk[dst + l] = d_lo * static_cast<float>(lo_full) - m_lo;
                    out_blk[dst + 32 + l] = d_hi * static_cast<float>(hi_full) - m_hi;
                }
            }
        });
    }).wait();
}

void rsl_dequant_q6_k_to_f32_usm(rsl_stream* s,
                                  const void* bytes_usm,
                                  float* out_usm,
                                  int n_blocks) {
    if (s == nullptr || bytes_usm == nullptr || out_usm == nullptr) return;
    if (n_blocks <= 0) return;
    auto& q = s->q;
    const uint8_t* bytes = static_cast<const uint8_t*>(bytes_usm);
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(n_blocks), [=](sycl::id<1> id) {
            const int b = static_cast<int>(id[0]);
            const uint8_t* blk = bytes + b * 210;
            const uint8_t* ql = blk;
            const uint8_t* qh = blk + 128;
            const uint8_t* scales_raw = blk + 128 + 64;
            uint16_t d_bits = static_cast<uint16_t>(blk[208])
                              | (static_cast<uint16_t>(blk[209]) << 8);
            const float d = bits_to_f32(d_bits);
            float* out_blk = out_usm + b * 256;
            for (int n = 0; n < 2; ++n) {
                for (int l = 0; l < 32; ++l) {
                    const int is = l / 16 + n * 8;
                    const int q1 = static_cast<int>(
                        (ql[64 * n + l] & 0x0F)
                        | ((static_cast<int>(qh[32 * n + l] >> 0) & 0x03) << 4)
                    ) - 32;
                    const int q2 = static_cast<int>(
                        (ql[64 * n + l + 32] & 0x0F)
                        | ((static_cast<int>(qh[32 * n + l] >> 2) & 0x03) << 4)
                    ) - 32;
                    const int q3 = static_cast<int>(
                        (ql[64 * n + l] >> 4)
                        | ((static_cast<int>(qh[32 * n + l] >> 4) & 0x03) << 4)
                    ) - 32;
                    const int q4 = static_cast<int>(
                        (ql[64 * n + l + 32] >> 4)
                        | ((static_cast<int>(qh[32 * n + l] >> 6) & 0x03) << 4)
                    ) - 32;
                    const int base = n * 128 + l;
                    const float s0 = static_cast<float>(
                        static_cast<int8_t>(scales_raw[is]));
                    const float s1 = static_cast<float>(
                        static_cast<int8_t>(scales_raw[is + 2]));
                    const float s2 = static_cast<float>(
                        static_cast<int8_t>(scales_raw[is + 4]));
                    const float s3 = static_cast<float>(
                        static_cast<int8_t>(scales_raw[is + 6]));
                    out_blk[base] = d * s0 * static_cast<float>(q1);
                    out_blk[base + 32] = d * s1 * static_cast<float>(q2);
                    out_blk[base + 64] = d * s2 * static_cast<float>(q3);
                    out_blk[base + 96] = d * s3 * static_cast<float>(q4);
                }
            }
        });
    }).wait();
}


}  // extern "C"
