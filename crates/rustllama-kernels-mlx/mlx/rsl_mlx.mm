// Apple-Metal / MLX kernels for rustllama's macOS backend.
//
// Objective-C++ host shim: it owns the Metal device/queue + MLX stream and
// exposes the `rsl_mlx_*` C ABI that `src/lib.rs` binds to. It is the peer
// of the CUDA crate's `cuda/rsl_cuda.cu` and the SYCL crate's
// `cpp/rsl_kernels.cpp`.
//
// ===================== STATUS: PHASE 4 (compute surface) ================
// The real Metal path (macOS/Apple-Silicon, guarded by RSL_MLX_HAVE_METAL=1)
// now covers the core forward-pass compute surface. Live:
//   * rsl_mlx_device_count / _device_info — MTLCopyAllDevices enumeration,
//     name / recommendedMaxWorkingSetSize / registryID (uuid left zeroed;
//     the tuner keys on registryID on Apple).
//   * stream_create/_destroy — a stream owns an MTLDevice + MTLCommandQueue.
//   * The pointer<->MTLBuffer bridge — malloc_device / malloc_from_host / free
//     back a StorageModeShared MTLBuffer whose `.contents` is the raw "device
//     pointer" handed to Rust; a global registry maps [base,base+len) spans
//     back to their MTLBuffer (see mlx_registry_*/mlx_resolve above).
//     memcpy_h2d/d2h are plain memcpys (unified memory; offsets auto-honored).
//   * A shared MTLLibrary + pipeline-by-name cache (mlx_library / mlx_pipeline)
//     and the mlx_run device-resident dispatch helper.
//   * rsl_mlx_matvec_f32 + rsl_mlx_rmsnorm_f32 — host-pointer reference paths.
//   * Forward-pass primitives: add_rmsnorm, rope, silu_mul, embedding_lookup.
//   * FlashAttention F32: GQA online-softmax decode + causal prefill.
//   * argmax sampling; packed-quant matvec Q8_0 + Q4_0 + the full K-quant
//     family (Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_K), single + batched — the common
//     dense + K-quant GGUF formats.
// Each kernel body lives in rsl_mlx.metal (compiled + EMBEDDED by build.rs,
// loaded via newLibraryWithData:).
//
// !!! WRITE-BLIND — AUTHORED ON A NON-APPLE HOST (Phase 4). The Metal shaders
// and this real path have NEVER been compiled (no `xcrun metal` off-Mac) or
// run on a GPU. They are byte-exact ports of the SYCL/CPU references; the first
// Mac build is expected to need MSL/ObjC++ fixes, and every kernel must pass
// the Metal parity harness before it is trusted. On a non-Apple host NONE of
// this compiles — the crate links the generated no-op stub (build.rs
// build_mlx_stub) and device_count()==0 → CPU/SYCL/CUDA.
//
// ===================== STILL INERT (-1, Mac-pending) ====================
// The Metal compute surface is now COMPLETE — every entry point declared in
// rsl_mlx.h is implemented: device enumeration + the pointer<->MTLBuffer bridge,
// the forward-pass primitives (rmsnorm/add_rmsnorm/rope/silu_mul/embedding),
// FlashAttention F32 (decode+prefill) + ALL quantized-KV variants
// (q4_0/nvfp4/mxfp4/6/8/q8_0/tq), argmax, the Prism Hadamard, and the full
// packed-quant matvec set (every GGUF/MLX quant the loader produces: base +
// K-quants + IQ4 + IQ1/2/3 grids + MXFP/NVFP + PTQ1_0/PQ2_0). Nothing returns
// the inert -1 anymore. WRITE-BLIND: none of it has been through `xcrun metal`
// or an Apple GPU — the first Mac build needs a compile pass + the Metal parity
// harness before any of it is trusted.
//
// Keep this file, `rsl_mlx.h`, `rsl_mlx.def`, and the `extern "C"` block in
// `src/lib.rs` in lock-step.

#include "rsl_mlx.h"

#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <new>

// Flip on in Phase 1 (build.rs `-DRSL_MLX_HAVE_METAL=1`) once the Metal
// shader library (`rsl_mlx.metal` → `.metallib`) + mlx-c link are wired.
#ifndef RSL_MLX_HAVE_METAL
#define RSL_MLX_HAVE_METAL 0
#endif

#if RSL_MLX_HAVE_METAL
// Phase 1a REAL Metal path. Pure Metal — NO mlx-c (libmlx) yet: the device +
// buffer bridge + f32 matvec need only the system Metal/Foundation
// frameworks, so MLX_LIB_DIR stays optional (build.rs).
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <dispatch/dispatch.h>  // dispatch_data_create for newLibraryWithData:
#include <mutex>
#include <vector>
// The Metal shader library: build.rs compiles rsl_mlx.metal → a .metallib and
// bin2c's it into this generated header as `static const unsigned char
// rsl_mlx_metallib[]` + `rsl_mlx_metallib_len`. We EMBED the metallib (rather
// than ship a sidecar .metallib and newLibraryWithURL:) so librsl_mlx.dylib is
// self-contained and relocatable — it survives being copied beside the binary
// in the @rpath bundle layout with no runtime path lookup. The `-I$OUT_DIR`
// that puts this header on the include path is added by build.rs alongside
// `-DRSL_MLX_HAVE_METAL=1`, so this #include is only reached on the real path.
#include "rsl_mlx_metallib.h"
#endif

// Per-thread kernel-error latch, mirroring the CUDA/SYCL crates. A
// launch/encode failure bumps it; the Rust wrapper drains it via
// rsl_mlx_consume_error_count() after each call and turns a non-zero count
// into a CPU fallback.
static thread_local int g_rsl_mlx_errors = 0;

extern "C" int rsl_mlx_consume_error_count(void) {
    int n = g_rsl_mlx_errors;
    g_rsl_mlx_errors = 0;
    return n;
}

#if RSL_MLX_HAVE_METAL
// ======================================================================
// Phase 1a Metal runtime: device list + the pointer<->MTLBuffer bridge +
// the command queue / compute pipeline. All of this is the macOS-real path,
// compiled only when build.rs defines RSL_MLX_HAVE_METAL=1; the inert
// non-Apple stub comes from the generated rsl_mlx_stub.c instead and never
// sees any of it.
// ======================================================================

// ---- Metal device list (retained once) --------------------------------
//
// MTLCopyAllDevices() enumerates every GPU; on Apple Silicon that is the one
// unified SoC GPU. We retain the array in a file-static so index->device is
// stable across calls (the tuner keys on MTLDevice.registryID, resolved in
// rsl_mlx_device_info). Fallback to MTLCreateSystemDefaultDevice() covers the
// rare case where MTLCopyAllDevices reports empty.
static NSArray<id<MTLDevice>> *g_mlx_devices = nil;  // strong (ARC)
static std::once_flag g_mlx_devices_once;

static NSArray<id<MTLDevice>> *mlx_devices(void) {
    std::call_once(g_mlx_devices_once, []() {
        @autoreleasepool {
            NSArray<id<MTLDevice>> *devs = MTLCopyAllDevices();
            if (devs == nil || devs.count == 0) {
                id<MTLDevice> d = MTLCreateSystemDefaultDevice();
                devs = d ? @[ d ] : @[];
            }
            g_mlx_devices = devs;  // ARC retains into the strong global
        }
    });
    return g_mlx_devices;
}

static id<MTLDevice> mlx_device_at(int idx) {
    NSArray<id<MTLDevice>> *devs = mlx_devices();
    if (idx < 0 || (NSUInteger)idx >= devs.count) return nil;
    return devs[(NSUInteger)idx];
}

// ---- The pointer<->MTLBuffer registry (the crux) -----------------------
//
// rustllama drives the device-resident path with CUDA-style raw device
// pointers and pointer arithmetic (see MlxDeviceBuffer::copy_from_host_at in
// src/lib.rs, which does `ptr + byte_offset`). Metal has no such pointers —
// it has MTLBuffer OBJECTS. Apple Silicon is UNIFIED MEMORY, so a
// StorageModeShared buffer's `.contents` IS a real host-addressable address:
// we hand that pointer to Rust as the "device pointer" and keep a registry
// mapping each [base, base+len) span back to its owning MTLBuffer. That lets
//   * free() find + release the right buffer from just the base pointer, and
//   * a kernel dispatch bind a buffer at a byte offset via
//     setBuffer:offset: (mlx_resolve, below) for an interior pointer.
// Because contents is host-addressable, memcpy_h2d/d2h are plain memcpys that
// already honor offsets (the pointer is the true address) — no blit needed.
//
// We store the MTLBuffer as a manually-retained opaque pointer
// (__bridge_retained) rather than a __strong member so the registry is a POD
// vector, sidestepping any ARC/std::vector move-retain subtlety under
// reallocation.
struct MlxAlloc {
    void *base;          // MTLBuffer.contents — the pointer handed to Rust
    void *buf_retained;  // (__bridge_retained id<MTLBuffer>) — a +1 we own
    size_t len;
};
static std::vector<MlxAlloc> g_mlx_allocs;
static std::mutex g_mlx_alloc_mutex;

static void mlx_registry_insert(void *base, id<MTLBuffer> buf, size_t len) {
    MlxAlloc a;
    a.base = base;
    a.buf_retained = (__bridge_retained void *)buf;  // transfer +1 into the POD slot
    a.len = len;
    std::lock_guard<std::mutex> lk(g_mlx_alloc_mutex);
    g_mlx_allocs.push_back(a);
}

// Index of the allocation whose [base, base+len) contains `ptr` (base
// inclusive); -1 if none. Caller holds g_mlx_alloc_mutex.
static long mlx_registry_find_locked(const void *ptr) {
    const uint8_t *p = (const uint8_t *)ptr;
    for (size_t i = 0; i < g_mlx_allocs.size(); ++i) {
        const uint8_t *b = (const uint8_t *)g_mlx_allocs[i].base;
        if (p >= b && p < b + g_mlx_allocs[i].len) return (long)i;
    }
    return -1;
}

static void mlx_registry_remove(void *ptr) {
    std::lock_guard<std::mutex> lk(g_mlx_alloc_mutex);
    long i = mlx_registry_find_locked(ptr);
    if (i < 0) return;  // foreign / double free — ignore defensively
    // Reclaim the +1 taken at insert; ARC releases the buffer when `buf` (a
    // local strong) leaves this scope.
    id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)g_mlx_allocs[(size_t)i].buf_retained;
    (void)buf;
    g_mlx_allocs.erase(g_mlx_allocs.begin() + i);
}

// Resolve a (possibly interior) device pointer to its owning MTLBuffer + byte
// offset, for binding at a non-zero offset via setBuffer:offset:.
//   [[maybe_unused]]: exercised by the DEVICE-RESIDENT packed/forward kernels,
//   which are Phase 1b. Phase 1a's only dispatched kernel is the host-pointer
//   matvec_f32, which stages into fresh buffers bound at offset 0.
[[maybe_unused]] static id<MTLBuffer> mlx_resolve(const void *ptr, size_t *out_offset) {
    std::lock_guard<std::mutex> lk(g_mlx_alloc_mutex);
    long i = mlx_registry_find_locked(ptr);
    if (i < 0) {
        if (out_offset) *out_offset = 0;
        return nil;
    }
    const MlxAlloc &a = g_mlx_allocs[(size_t)i];
    if (out_offset)
        *out_offset = (size_t)((const uint8_t *)ptr - (const uint8_t *)a.base);
    return (__bridge id<MTLBuffer>)a.buf_retained;  // +0; kept alive by the registry
}

// ---- Command queue + compute pipeline cache ---------------------------
//
// The host-pointer reference kernels (rsl_mlx_matvec_f32) carry no stream, so
// they run on GPU 0 through a lazily-built, cached command queue + pipeline.
// Both are keyed by device to future-proof multi-GPU, though Apple Silicon has
// exactly one GPU so each builds once.
static id<MTLCommandQueue> g_mlx_default_queue = nil;  // strong
static id<MTLDevice> g_mlx_default_queue_dev = nil;    // strong (which dev it's for)
static std::mutex g_mlx_queue_mutex;

static id<MTLCommandQueue> mlx_default_queue(id<MTLDevice> dev) {
    std::lock_guard<std::mutex> lk(g_mlx_queue_mutex);
    if (g_mlx_default_queue != nil && g_mlx_default_queue_dev == dev)
        return g_mlx_default_queue;
    id<MTLCommandQueue> q = [dev newCommandQueue];
    if (q == nil) return nil;
    g_mlx_default_queue = q;
    g_mlx_default_queue_dev = dev;
    return g_mlx_default_queue;
}

// ---- Shared MTLLibrary (embedded metallib) + pipeline-by-name cache ----
//
// The Phase-1a f32 matvec used a single hardcoded pipeline; Phase 4 adds a
// whole kernel surface, so we cache the loaded MTLLibrary once and build a
// MTLComputePipelineState per kernel name on first use (keyed by device to
// future-proof multi-GPU, though Apple Silicon has one GPU so each builds
// once). nil on failure latches an error so the caller falls back to CPU.
static id<MTLLibrary> g_mlx_library = nil;      // strong — keep the lib alive
static id<MTLDevice>  g_mlx_library_dev = nil;  // strong (which dev it's for)
static std::mutex g_mlx_library_mutex;

static id<MTLLibrary> mlx_library(id<MTLDevice> dev) {
    std::lock_guard<std::mutex> lk(g_mlx_library_mutex);
    if (g_mlx_library != nil && g_mlx_library_dev == dev) return g_mlx_library;
    NSError *err = nil;
    // DISPATCH_DATA_DESTRUCTOR_DEFAULT makes dispatch_data copy the bytes, so
    // the static array's lifetime does not matter after this call.
    dispatch_data_t dd = dispatch_data_create(
        rsl_mlx_metallib, (size_t)rsl_mlx_metallib_len, nullptr,
        DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    id<MTLLibrary> lib = [dev newLibraryWithData:dd error:&err];
    if (lib == nil) {
        std::fprintf(stderr, "rsl_mlx: newLibraryWithData failed: %s\n",
                     err ? err.localizedDescription.UTF8String : "(nil)");
        g_rsl_mlx_errors++;
        return nil;
    }
    g_mlx_library = lib;
    g_mlx_library_dev = dev;
    return g_mlx_library;
}

static NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *g_mlx_psos = nil;  // strong
static id<MTLDevice> g_mlx_psos_dev = nil;  // strong (which dev the cache is for)
static std::mutex g_mlx_psos_mutex;

// Build (once, cached) the compute pipeline named `name` for `dev` from the
// embedded metallib. nil on failure (error latched).
static id<MTLComputePipelineState> mlx_pipeline(id<MTLDevice> dev, NSString *name) {
    std::lock_guard<std::mutex> lk(g_mlx_psos_mutex);
    if (g_mlx_psos == nil || g_mlx_psos_dev != dev) {
        g_mlx_psos = [NSMutableDictionary dictionary];  // ARC retains into the strong global
        g_mlx_psos_dev = dev;
    }
    id<MTLComputePipelineState> cached = [g_mlx_psos objectForKey:name];
    if (cached != nil) return cached;
    id<MTLLibrary> lib = mlx_library(dev);
    if (lib == nil) return nil;
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (fn == nil) {
        std::fprintf(stderr, "rsl_mlx: kernel %s not found\n", name.UTF8String);
        g_rsl_mlx_errors++;
        return nil;
    }
    NSError *err = nil;
    id<MTLComputePipelineState> pso =
        [dev newComputePipelineStateWithFunction:fn error:&err];
    if (pso == nil) {
        std::fprintf(stderr, "rsl_mlx: pipeline %s build failed: %s\n", name.UTF8String,
                     err ? err.localizedDescription.UTF8String : "(nil)");
        g_rsl_mlx_errors++;
        return nil;
    }
    [g_mlx_psos setObject:pso forKey:name];
    return pso;
}

// Back-compat alias for the Phase-1a host-pointer matvec path.
static id<MTLComputePipelineState> mlx_matvec_pso(id<MTLDevice> dev) {
    return mlx_pipeline(dev, @"rsl_mlx_matvec_f32_kernel");
}

// ---- Device-resident dispatch helper -----------------------------------
//
// Build a command buffer + compute encoder on the stream, let `bind` set the
// buffers/bytes (buffers resolved by the caller from the global registry),
// dispatch `grid` threadgroups of `tpg`, commit + wait (Phase-4 runs
// synchronously, like the matvec_f32 path). Returns 0 on success, -1 on a
// pipeline/encode/GPU error (latched → CPU fallback). Must be called inside
// an @autoreleasepool-able scope.
typedef void (^MlxBindBlock)(id<MTLComputeCommandEncoder>);
static int mlx_run(rsl_mlx_stream *s, NSString *name, MTLSize grid, MTLSize tpg,
                   MlxBindBlock bind) {
    if (s == nullptr) return -1;
    @autoreleasepool {
        id<MTLComputePipelineState> pso = mlx_pipeline(s->device, name);
        if (pso == nil) return -1;  // error already latched
        id<MTLCommandBuffer> cb = [s->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        bind(enc);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tpg];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) { g_rsl_mlx_errors++; return -1; }
        return 0;
    }
}

// Shared body for the packed-quant matvecs: resolve W/x/out from the registry,
// dispatch `kernel_name` over (M rows, N cols) threadgroups of 32 lanes. K must
// be a multiple of the format's 32-wide block. N == 1 is the single case.
static int mlx_packed_matvec_align(rsl_mlx_stream *s, NSString *kernel_name,
                             const void *w, const float *x, float *out,
                             int M, int K, int N, int k_align) {
    if (s == nullptr || w == nullptr || x == nullptr || out == nullptr) return -1;
    if (M <= 0 || K <= 0 || N <= 0 || k_align <= 0 || (K % k_align) != 0) return -1;
    size_t ow = 0, ox = 0, oo = 0;
    id<MTLBuffer> bw = mlx_resolve(w, &ow);
    id<MTLBuffer> bx = mlx_resolve(x, &ox);
    id<MTLBuffer> bo = mlx_resolve(out, &oo);
    if (bw == nil || bx == nil || bo == nil) { g_rsl_mlx_errors++; return -1; }
    int kk = K, mm = M, nn = N;
    MTLSize grid = MTLSizeMake((NSUInteger)M, (NSUInteger)N, 1);
    MTLSize tpg = MTLSizeMake(32, 1, 1);
    return mlx_run(s, kernel_name, grid, tpg, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bw offset:ow atIndex:0];
        [enc setBuffer:bx offset:ox atIndex:1];
        [enc setBuffer:bo offset:oo atIndex:2];
        [enc setBytes:&kk length:sizeof(int) atIndex:3];
        [enc setBytes:&mm length:sizeof(int) atIndex:4];
        [enc setBytes:&nn length:sizeof(int) atIndex:5];
    });
}

// Default 32-wide-block packed matvec (Q8_0/Q4_0/Q5_0/Q4_1/Q5_1/IQ4_NL/MXFP*).
// K-quants pass 256 and NVFP4 passes 16 via mlx_packed_matvec_align directly.
static int mlx_packed_matvec(rsl_mlx_stream *s, NSString *kernel_name,
                             const void *w, const float *x, float *out,
                             int M, int K, int N) {
    return mlx_packed_matvec_align(s, kernel_name, w, x, out, M, K, N, 32);
}

// Shared gates + dispatch for the scale-free quantized-KV flash kernels
// (q4_0/nvfp4/mxfp4/6/8): F32 q/out, packed byte K/V. `k_align` = the KV block
// width head_dim must divide (16 for nvfp4, else 32). head_dim <= 256.
static int mlx_flash_kv_decode(rsl_mlx_stream *s, NSString *name, const float *q,
                               const void *k, const void *v, float *out,
                               int n_heads, int n_kv_heads, int head_dim,
                               int max_ctx, int kv_len, int k_align) {
    if (!s || !q || !k || !v || !out) return -1;
    if (n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0 || max_ctx <= 0 || kv_len <= 0) return -1;
    if ((n_heads % n_kv_heads) != 0 || head_dim > 256 || (head_dim % k_align) != 0) return -1;
    size_t oq = 0, ok = 0, ov = 0, oo = 0;
    id<MTLBuffer> bq = mlx_resolve(q, &oq), bk = mlx_resolve(k, &ok), bv = mlx_resolve(v, &ov), bo = mlx_resolve(out, &oo);
    if (bq == nil || bk == nil || bv == nil || bo == nil) { g_rsl_mlx_errors++; return -1; }
    int nh = n_heads, nkv = n_kv_heads, hd = head_dim, mc = max_ctx, kl = kv_len;
    MTLSize grid = MTLSizeMake((NSUInteger)n_heads, 1, 1), tpg = MTLSizeMake(32, 1, 1);
    return mlx_run(s, name, grid, tpg, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bq offset:oq atIndex:0];
        [enc setBuffer:bk offset:ok atIndex:1];
        [enc setBuffer:bv offset:ov atIndex:2];
        [enc setBuffer:bo offset:oo atIndex:3];
        [enc setBytes:&nh length:sizeof(int) atIndex:4];
        [enc setBytes:&nkv length:sizeof(int) atIndex:5];
        [enc setBytes:&hd length:sizeof(int) atIndex:6];
        [enc setBytes:&mc length:sizeof(int) atIndex:7];
        [enc setBytes:&kl length:sizeof(int) atIndex:8];
    });
}
static int mlx_flash_kv_prefill(rsl_mlx_stream *s, NSString *name, const float *q,
                                const void *k, const void *v, float *out,
                                int n_heads, int n_kv_heads, int head_dim,
                                int max_ctx, int kv_len_base, int n_new, int k_align) {
    if (!s || !q || !k || !v || !out) return -1;
    if (n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0 || max_ctx <= 0 || n_new <= 0) return -1;
    if ((n_heads % n_kv_heads) != 0 || head_dim > 256 || (head_dim % k_align) != 0) return -1;
    if (kv_len_base < 0 || kv_len_base + n_new > max_ctx) return -1;
    size_t oq = 0, ok = 0, ov = 0, oo = 0;
    id<MTLBuffer> bq = mlx_resolve(q, &oq), bk = mlx_resolve(k, &ok), bv = mlx_resolve(v, &ov), bo = mlx_resolve(out, &oo);
    if (bq == nil || bk == nil || bv == nil || bo == nil) { g_rsl_mlx_errors++; return -1; }
    int nh = n_heads, nkv = n_kv_heads, hd = head_dim, mc = max_ctx, kb = kv_len_base, nn = n_new;
    MTLSize grid = MTLSizeMake((NSUInteger)n_heads, (NSUInteger)n_new, 1), tpg = MTLSizeMake(32, 1, 1);
    return mlx_run(s, name, grid, tpg, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bq offset:oq atIndex:0];
        [enc setBuffer:bk offset:ok atIndex:1];
        [enc setBuffer:bv offset:ov atIndex:2];
        [enc setBuffer:bo offset:oo atIndex:3];
        [enc setBytes:&nh length:sizeof(int) atIndex:4];
        [enc setBytes:&nkv length:sizeof(int) atIndex:5];
        [enc setBytes:&hd length:sizeof(int) atIndex:6];
        [enc setBytes:&mc length:sizeof(int) atIndex:7];
        [enc setBytes:&kb length:sizeof(int) atIndex:8];
        [enc setBytes:&nn length:sizeof(int) atIndex:9];
    });
}
#endif  // RSL_MLX_HAVE_METAL

// ---- Device query -----------------------------------------------------

extern "C" int rsl_mlx_device_count(void) {
#if RSL_MLX_HAVE_METAL
    // On Apple Silicon this is normally 1 (the unified SoC GPU).
    return (int)mlx_devices().count;
#else
    return 0; // inert stub: no Metal device
#endif
}

extern "C" int rsl_mlx_device_info(int idx, char *name, int name_cap,
                                   unsigned long long *total_mem,
                                   unsigned long long *registry_id,
                                   unsigned char *uuid /* 16 bytes or null */) {
    (void)idx;
    if (name && name_cap > 0) name[0] = '\0';
    if (total_mem) *total_mem = 0;
    if (registry_id) *registry_id = 0;
    if (uuid) std::memset(uuid, 0, 16);
#if RSL_MLX_HAVE_METAL
    @autoreleasepool {
        id<MTLDevice> dev = mlx_device_at(idx);
        if (dev == nil) return -1;
        // Copy the device name into the caller's fixed C buffer (NUL-terminated
        // within name_cap). dev.name.UTF8String is valid for this autorelease
        // pool's lifetime, so we copy immediately.
        if (name && name_cap > 0) {
            const char *n = dev.name.UTF8String;
            if (n) {
                std::strncpy(name, n, (size_t)name_cap - 1);
                name[name_cap - 1] = '\0';
            } else {
                name[0] = '\0';
            }
        }
        // recommendedMaxWorkingSetSize is the driver's suggested usable working
        // set — the right VRAM/unified-pool budget proxy for the placement
        // layer (there is no separate "dedicated VRAM" on unified memory).
        if (total_mem) *total_mem = (unsigned long long)dev.recommendedMaxWorkingSetSize;
        // registryID is Metal's stable, driver-invariant device id (uint64 from
        // the IO registry) — the tuner's fingerprint key on Apple.
        if (registry_id) *registry_id = (unsigned long long)dev.registryID;
        // uuid: Metal has no 16-byte device UUID, and Phase 1a leaves it zeroed
        // (already memset above) — the tuner keys on registry_id here. A
        // synthesized uuid can come later if cross-backend dedup needs it.
        return 0;
    }
#else
    return -1; // inert: no device to describe
#endif
}

// ---- Host-pointer reference kernels (parity-pattern establishers) -----

extern "C" int rsl_mlx_rmsnorm_f32(const float *x, const float *w, float *y,
                                   int n_rows, int d, float eps) {
#if RSL_MLX_HAVE_METAL
    if (!x || !w || !y || n_rows <= 0 || d <= 0) return -1;
    @autoreleasepool {
        // Stream-less host-pointer reference path → GPU 0 + cached queue/pso,
        // staging x/w into fresh shared buffers and reading y straight back
        // (same shape as rsl_mlx_matvec_f32).
        id<MTLDevice> dev = mlx_device_at(0);
        if (dev == nil) { g_rsl_mlx_errors++; return -1; }
        id<MTLComputePipelineState> pso = mlx_pipeline(dev, @"rsl_mlx_rmsnorm_f32_kernel");
        if (pso == nil) return -1;  // error already latched
        id<MTLCommandQueue> q = mlx_default_queue(dev);
        if (q == nil) { g_rsl_mlx_errors++; return -1; }

        const NSUInteger xbytes = (NSUInteger)n_rows * (NSUInteger)d * sizeof(float);
        const NSUInteger wbytes = (NSUInteger)d * sizeof(float);
        id<MTLBuffer> bX = [dev newBufferWithBytes:x length:xbytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bW = [dev newBufferWithBytes:w length:wbytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bY = [dev newBufferWithLength:xbytes options:MTLResourceStorageModeShared];
        if (bX == nil || bW == nil || bY == nil) { g_rsl_mlx_errors++; return -1; }

        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bX offset:0 atIndex:0];
        [enc setBuffer:bW offset:0 atIndex:1];
        [enc setBuffer:bY offset:0 atIndex:2];
        int dd = d;
        float ee = eps;
        [enc setBytes:&dd length:sizeof(int) atIndex:3];
        [enc setBytes:&ee length:sizeof(float) atIndex:4];
        // One 32-lane threadgroup (== one SIMD-group) per row; lanes stride d
        // and simd_sum-reduce the sum of squares.
        MTLSize grid = MTLSizeMake((NSUInteger)n_rows, 1, 1);
        MTLSize tpg = MTLSizeMake(32, 1, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tpg];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) { g_rsl_mlx_errors++; return -1; }
        std::memcpy(y, [bY contents], (size_t)xbytes);
        return 0;
    }
#else
    (void)x; (void)w; (void)y; (void)n_rows; (void)d; (void)eps;
    return -1;
#endif
}

extern "C" int rsl_mlx_matvec_f32(const float *W, const float *x, float *out,
                                  int m_rows, int k_dim) {
#if RSL_MLX_HAVE_METAL
    if (!W || !x || !out || m_rows <= 0 || k_dim <= 0) return -1;
    @autoreleasepool {
        // Stream-less host-pointer reference path → GPU 0 + the cached queue
        // and pipeline. (The hot path is the device-resident packed kernels,
        // Phase 1b; this entry is the parity/smoke establisher.)
        id<MTLDevice> dev = mlx_device_at(0);
        if (dev == nil) { g_rsl_mlx_errors++; return -1; }
        id<MTLComputePipelineState> pso = mlx_matvec_pso(dev);
        if (pso == nil) return -1;  // error already latched by mlx_matvec_pso
        id<MTLCommandQueue> q = mlx_default_queue(dev);
        if (q == nil) { g_rsl_mlx_errors++; return -1; }

        const NSUInteger wbytes = (NSUInteger)m_rows * (NSUInteger)k_dim * sizeof(float);
        const NSUInteger xbytes = (NSUInteger)k_dim * sizeof(float);
        const NSUInteger obytes = (NSUInteger)m_rows * sizeof(float);
        // Arbitrary Rust-slice host pointers are not page-aligned MTLBuffers, so
        // we stage W/x into fresh StorageModeShared buffers (on unified memory
        // that memcpy is the only copy — no separate H2D blit) and read the
        // result straight back out of the output buffer's contents.
        id<MTLBuffer> bW = [dev newBufferWithBytes:W length:wbytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bX = [dev newBufferWithBytes:x length:xbytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bO = [dev newBufferWithLength:obytes options:MTLResourceStorageModeShared];
        if (bW == nil || bX == nil || bO == nil) { g_rsl_mlx_errors++; return -1; }

        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bW offset:0 atIndex:0];
        [enc setBuffer:bX offset:0 atIndex:1];
        [enc setBuffer:bO offset:0 atIndex:2];
        int K = k_dim;
        [enc setBytes:&K length:sizeof(int) atIndex:3];
        // One 32-lane threadgroup (== one SIMD-group on Apple GPUs) per output
        // row; the kernel strides K across lanes and simd_sum-reduces. The grid
        // is exactly M threadgroups, so threadgroup_position_in_grid IS the row
        // and no bounds check is needed in the shader.
        MTLSize grid = MTLSizeMake((NSUInteger)m_rows, 1, 1);
        MTLSize tpg = MTLSizeMake(32, 1, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tpg];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];  // Phase 1a runs synchronously
        if (cb.status == MTLCommandBufferStatusError) { g_rsl_mlx_errors++; return -1; }
        std::memcpy(out, [bO contents], (size_t)obytes);
        return 0;
    }
#else
    (void)W; (void)x; (void)out; (void)m_rows; (void)k_dim;
    return -1;
#endif
}

// ---- Stream / device-buffer lifecycle ---------------------------------
//
// A stream owns the device it is bound to + a fresh MTLCommandQueue (the MLX
// analogue of the CUDA/SYCL stream). It is driven from one owning worker
// thread (src/lib.rs marks MlxStream Send-but-not-Sync), so no per-stream
// locking is needed here. A "device buffer" is a StorageModeShared MTLBuffer
// whose `.contents` pointer is handed to Rust and tracked in the global
// registry (above) so free()/resolve() can recover the owning MTLBuffer.
#if RSL_MLX_HAVE_METAL
// The concrete stream. ObjC object members are __strong under ARC, so this
// MUST be created/destroyed with C++ new/delete (which run ARC retain on
// construction and release on destruction) — never malloc/free.
struct rsl_mlx_stream {
    id<MTLDevice> device;       // strong
    id<MTLCommandQueue> queue;  // strong
    int device_index;
};
#endif

extern "C" rsl_mlx_stream *rsl_mlx_stream_create(int device_index) {
#if RSL_MLX_HAVE_METAL
    @autoreleasepool {
        id<MTLDevice> dev = mlx_device_at(device_index);
        if (dev == nil) return nullptr;
        id<MTLCommandQueue> q = [dev newCommandQueue];
        if (q == nil) return nullptr;
        rsl_mlx_stream *s = new (std::nothrow) rsl_mlx_stream();
        if (s == nullptr) return nullptr;
        s->device = dev;  // ARC retains into the strong members
        s->queue = q;
        s->device_index = device_index;
        return s;
    }
#else
    (void)device_index;
    return nullptr;
#endif
}

extern "C" void rsl_mlx_stream_destroy(rsl_mlx_stream *s) {
#if RSL_MLX_HAVE_METAL
    if (s) delete s;  // dtor ARC-releases device + queue
#else
    (void)s;
#endif
}

extern "C" void *rsl_mlx_malloc_device(rsl_mlx_stream *s,
                                       unsigned long long n_bytes) {
#if RSL_MLX_HAVE_METAL
    if (s == nullptr || n_bytes == 0) return nullptr;
    @autoreleasepool {
        // StorageModeShared: CPU+GPU coherent, zero-copy on Apple's unified
        // memory. `.contents` is the host-addressable base we return as the
        // "device pointer".
        id<MTLBuffer> buf = [s->device newBufferWithLength:(NSUInteger)n_bytes
                                                   options:MTLResourceStorageModeShared];
        if (buf == nil) { g_rsl_mlx_errors++; return nullptr; }
        void *base = [buf contents];
        mlx_registry_insert(base, buf, (size_t)n_bytes);
        return base;
    }
#else
    (void)s; (void)n_bytes;
    return nullptr;
#endif
}

extern "C" void *rsl_mlx_malloc_from_host(rsl_mlx_stream *s, const void *src,
                                          unsigned long long n_bytes) {
#if RSL_MLX_HAVE_METAL
    void *base = rsl_mlx_malloc_device(s, n_bytes);
    if (base == nullptr) return nullptr;
    // Shared storage → the upload is a plain memcpy into the buffer's own
    // bytes. (`src` may be NULL to leave the buffer uninitialized.)
    if (src && n_bytes) std::memcpy(base, src, (size_t)n_bytes);
    return base;
#else
    (void)s; (void)src; (void)n_bytes;
    return nullptr;
#endif
}

extern "C" void rsl_mlx_free(rsl_mlx_stream *s, void *dev_ptr) {
#if RSL_MLX_HAVE_METAL
    (void)s;  // buffers are owned by the global registry, not the stream
    if (dev_ptr == nullptr) return;
    mlx_registry_remove(dev_ptr);  // finds the owning MTLBuffer + ARC-releases it
#else
    (void)s; (void)dev_ptr;
#endif
}

extern "C" int rsl_mlx_memcpy_h2d(rsl_mlx_stream *s, void *dst_dev,
                                  const void *src_host,
                                  unsigned long long n_bytes) {
    (void)s;
    // On unified memory this is a plain host memcpy: dst_dev is a shared
    // MTLBuffer's contents pointer (possibly base+offset — the Rust wrapper
    // does the offset arithmetic, and because the pointer is the true host
    // address the offset is already honored, no blit or registry lookup).
    if (dst_dev && src_host && n_bytes) std::memcpy(dst_dev, src_host, (size_t)n_bytes);
    return 0;
}

extern "C" int rsl_mlx_memcpy_d2h(rsl_mlx_stream *s, void *dst_host,
                                  const void *src_dev,
                                  unsigned long long n_bytes) {
    (void)s;
    if (dst_host && src_dev && n_bytes) std::memcpy(dst_host, src_dev, (size_t)n_bytes);
    return 0;
}

// ---- Kernels ----------------------------------------------------------
//
// All kernel entry points below are Phase-1 TODO. They return -1 in Phase 0
// (never reached while device_count()==0). Phase 1 replaces each body with a
// Metal compute dispatch (or an mlx-c op) that is a byte-exact port of the
// CPU reference, matched by the parity harness. The `RSL_MLX_STUB_KERNEL`
// macro collapses the boilerplate no-op body for the many signatures.

#define RSL_MLX_STUB_KERNEL { return -1; }

// PTQ1_0 (Bonsai ternary, 128-wide) + Prism Hadamard — LIVE.
extern "C" int rsl_mlx_matvec_ptq1_0_packed_f32(rsl_mlx_stream *s, const void *w,
    const float *x, float *out, int M, int K) {
#if RSL_MLX_HAVE_METAL
    return mlx_packed_matvec_align(s, @"rsl_mlx_matvec_ptq1_0_packed_f32_kernel", w, x, out, M, K, 1, 128);
#else
    (void)s;(void)w;(void)x;(void)out;(void)M;(void)K; return -1;
#endif
}
extern "C" int rsl_mlx_matvec_ptq1_0_packed_f32_batched(rsl_mlx_stream *s, const void *w,
    const float *x, float *out, int M, int K, int N) {
#if RSL_MLX_HAVE_METAL
    return mlx_packed_matvec_align(s, @"rsl_mlx_matvec_ptq1_0_packed_f32_kernel", w, x, out, M, K, N, 128);
#else
    (void)s;(void)w;(void)x;(void)out;(void)M;(void)K;(void)N; return -1;
#endif
}
extern "C" int rsl_mlx_hadamard_forward(rsl_mlx_stream *s, const float *x,
    const float *signs, float *out, int n_elems, int block) {
#if RSL_MLX_HAVE_METAL
    if (!s || !x || !signs || !out || n_elems <= 0 || block <= 0) return -1;
    // block: power of two <= 4096 (the kernel's threadgroup slm) dividing n_elems.
    if (block > 4096 || (block & (block - 1)) != 0 || (n_elems % block) != 0) return -1;
    size_t ox = 0, osg = 0, oo = 0;
    id<MTLBuffer> bx = mlx_resolve(x, &ox);
    id<MTLBuffer> bsg = mlx_resolve(signs, &osg);
    id<MTLBuffer> bo = mlx_resolve(out, &oo);
    if (bx == nil || bsg == nil || bo == nil) { g_rsl_mlx_errors++; return -1; }
    int ne = n_elems, blk = block;
    const NSUInteger lws = block < 256 ? (NSUInteger)block : 256;
    MTLSize grid = MTLSizeMake((NSUInteger)(n_elems / block), 1, 1);
    MTLSize tpg = MTLSizeMake(lws, 1, 1);
    return mlx_run(s, @"rsl_mlx_hadamard_forward_kernel", grid, tpg, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bx offset:ox atIndex:0];
        [enc setBuffer:bsg offset:osg atIndex:1];
        [enc setBuffer:bo offset:oo atIndex:2];
        [enc setBytes:&ne length:sizeof(int) atIndex:3];
        [enc setBytes:&blk length:sizeof(int) atIndex:4];
    });
#else
    (void)s;(void)x;(void)signs;(void)out;(void)n_elems;(void)block; return -1;
#endif
}

// Packed-quant matvecs (single + batched). TODO(phase1): one Metal kernel
// per quant; the on-device dequant math ports byte-exact from the CPU
// reference (rustllama-kernels-cpu) and the SYCL/CUDA kernels. The
// codebook/grid tables the IQ quants need (IQ1S/IQ2*/IQ3* grids +
// KSIGNS/KMASK) should be staged into `.metal` constants by build.rs, the
// same way the SYCL/CUDA builds emit `iq_grids*.inl`.
#define RSL_MLX_DEFINE_PACKED(NAME)                                           \
    extern "C" int rsl_mlx_##NAME(rsl_mlx_stream *s, const void *w,           \
        const float *x, float *out, int M, int K)                             \
        { (void)s;(void)w;(void)x;(void)out;(void)M;(void)K; RSL_MLX_STUB_KERNEL } \
    extern "C" int rsl_mlx_##NAME##_batched(rsl_mlx_stream *s, const void *w, \
        const float *x, float *out, int M, int K, int N)                      \
        { (void)s;(void)w;(void)x;(void)out;(void)M;(void)K;(void)N; RSL_MLX_STUB_KERNEL }
// NOTE: the full K-quant family (Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_K) + Q8_0/Q4_0
// (and their batched forms) are IMPLEMENTED below (real Metal dispatch), so
// they are omitted from this inert-stub list.
// (All packed quants are now implemented below — the IQ-grid family via the
// RSL_MLX_K256 macro. This inert-stub list is intentionally empty.)
#undef RSL_MLX_DEFINE_PACKED

// --- Q8_0 / Q4_0 packed matvec: LIVE (Metal dispatch via mlx_packed_matvec) ---
// The two simplest GGUF quants (32-wide blocks, no 6-bit scale unpack / grid
// tables), ported byte-exact from rsl_matvec_{q8_0,q4_0}_packed_f32_usm. They
// establish the "dequant W's block layout in the matvec inner loop" pattern the
// remaining packed quants above specialize.
extern "C" int rsl_mlx_matvec_q8_0_packed_f32(rsl_mlx_stream *s, const void *w,
    const float *x, float *out, int M, int K) {
#if RSL_MLX_HAVE_METAL
    return mlx_packed_matvec(s, @"rsl_mlx_matvec_q8_0_packed_f32_kernel", w, x, out, M, K, 1);
#else
    (void)s;(void)w;(void)x;(void)out;(void)M;(void)K; return -1;
#endif
}
extern "C" int rsl_mlx_matvec_q8_0_packed_f32_batched(rsl_mlx_stream *s, const void *w,
    const float *x, float *out, int M, int K, int N) {
#if RSL_MLX_HAVE_METAL
    return mlx_packed_matvec(s, @"rsl_mlx_matvec_q8_0_packed_f32_kernel", w, x, out, M, K, N);
#else
    (void)s;(void)w;(void)x;(void)out;(void)M;(void)K;(void)N; return -1;
#endif
}
extern "C" int rsl_mlx_matvec_q4_0_packed_f32(rsl_mlx_stream *s, const void *w,
    const float *x, float *out, int M, int K) {
#if RSL_MLX_HAVE_METAL
    return mlx_packed_matvec(s, @"rsl_mlx_matvec_q4_0_packed_f32_kernel", w, x, out, M, K, 1);
#else
    (void)s;(void)w;(void)x;(void)out;(void)M;(void)K; return -1;
#endif
}
extern "C" int rsl_mlx_matvec_q4_0_packed_f32_batched(rsl_mlx_stream *s, const void *w,
    const float *x, float *out, int M, int K, int N) {
#if RSL_MLX_HAVE_METAL
    return mlx_packed_matvec(s, @"rsl_mlx_matvec_q4_0_packed_f32_kernel", w, x, out, M, K, N);
#else
    (void)s;(void)w;(void)x;(void)out;(void)M;(void)K;(void)N; return -1;
#endif
}

// --- Q4_K / Q6_K packed matvec: LIVE. K-quant super-blocks (256-wide), so they
// add a K % 256 guard on top of the shared mlx_packed_matvec dispatch. Ported
// byte-exact from rsl_matvec_{q4_k,q6_k}_packed_f32_usm (6-bit scale unpack /
// ql+qh 6-bit layout). ---
extern "C" int rsl_mlx_matvec_q4_k_packed_f32(rsl_mlx_stream *s, const void *w,
    const float *x, float *out, int M, int K) {
#if RSL_MLX_HAVE_METAL
    if ((K % 256) != 0) return -1;
    return mlx_packed_matvec(s, @"rsl_mlx_matvec_q4_k_packed_f32_kernel", w, x, out, M, K, 1);
#else
    (void)s;(void)w;(void)x;(void)out;(void)M;(void)K; return -1;
#endif
}
extern "C" int rsl_mlx_matvec_q4_k_packed_f32_batched(rsl_mlx_stream *s, const void *w,
    const float *x, float *out, int M, int K, int N) {
#if RSL_MLX_HAVE_METAL
    if ((K % 256) != 0) return -1;
    return mlx_packed_matvec(s, @"rsl_mlx_matvec_q4_k_packed_f32_kernel", w, x, out, M, K, N);
#else
    (void)s;(void)w;(void)x;(void)out;(void)M;(void)K;(void)N; return -1;
#endif
}
extern "C" int rsl_mlx_matvec_q6_k_packed_f32(rsl_mlx_stream *s, const void *w,
    const float *x, float *out, int M, int K) {
#if RSL_MLX_HAVE_METAL
    if ((K % 256) != 0) return -1;
    return mlx_packed_matvec(s, @"rsl_mlx_matvec_q6_k_packed_f32_kernel", w, x, out, M, K, 1);
#else
    (void)s;(void)w;(void)x;(void)out;(void)M;(void)K; return -1;
#endif
}
extern "C" int rsl_mlx_matvec_q6_k_packed_f32_batched(rsl_mlx_stream *s, const void *w,
    const float *x, float *out, int M, int K, int N) {
#if RSL_MLX_HAVE_METAL
    if ((K % 256) != 0) return -1;
    return mlx_packed_matvec(s, @"rsl_mlx_matvec_q6_k_packed_f32_kernel", w, x, out, M, K, N);
#else
    (void)s;(void)w;(void)x;(void)out;(void)M;(void)K;(void)N; return -1;
#endif
}

// --- Q5_K / Q2_K / Q3_K / Q8_K packed matvec: LIVE. Completes the K-quant
// family on Metal; same 256-wide super-block K%256 guard + shared
// mlx_packed_matvec dispatch as Q4_K/Q6_K, ported byte-exact from the
// rsl_matvec_{q5_k,q2_k,q3_k,q8_k}_packed_f32_usm SYCL refs. ---
#if RSL_MLX_HAVE_METAL
#define RSL_MLX_KQUANT(NAME, KERNEL)                                               \
    extern "C" int rsl_mlx_##NAME(rsl_mlx_stream *s, const void *w,                \
        const float *x, float *out, int M, int K) {                               \
        if ((K % 256) != 0) return -1;                                            \
        return mlx_packed_matvec(s, @KERNEL, w, x, out, M, K, 1);                  \
    }                                                                              \
    extern "C" int rsl_mlx_##NAME##_batched(rsl_mlx_stream *s, const void *w,      \
        const float *x, float *out, int M, int K, int N) {                        \
        if ((K % 256) != 0) return -1;                                            \
        return mlx_packed_matvec(s, @KERNEL, w, x, out, M, K, N);                  \
    }
#else
#define RSL_MLX_KQUANT(NAME, KERNEL)                                               \
    extern "C" int rsl_mlx_##NAME(rsl_mlx_stream *s, const void *w,                \
        const float *x, float *out, int M, int K)                                 \
        { (void)s;(void)w;(void)x;(void)out;(void)M;(void)K; return -1; }          \
    extern "C" int rsl_mlx_##NAME##_batched(rsl_mlx_stream *s, const void *w,      \
        const float *x, float *out, int M, int K, int N)                          \
        { (void)s;(void)w;(void)x;(void)out;(void)M;(void)K;(void)N; return -1; }
#endif
RSL_MLX_KQUANT(matvec_q5_k_packed_f32, "rsl_mlx_matvec_q5_k_packed_f32_kernel")
RSL_MLX_KQUANT(matvec_q2_k_packed_f32, "rsl_mlx_matvec_q2_k_packed_f32_kernel")
RSL_MLX_KQUANT(matvec_q3_k_packed_f32, "rsl_mlx_matvec_q3_k_packed_f32_kernel")
RSL_MLX_KQUANT(matvec_q8_k_packed_f32, "rsl_mlx_matvec_q8_k_packed_f32_kernel")
#undef RSL_MLX_KQUANT

// --- Q5_0 / Q4_1 / Q5_1 packed matvec: LIVE. Simple 32-wide blocks (K%32 via
// mlx_packed_matvec), ported byte-exact from the SYCL refs. The RSL_MLX_SIMPLE32
// macro is reused by other 32-wide quants (IQ4_NL, MXFP4/6/8) as they land. ---
#if RSL_MLX_HAVE_METAL
#define RSL_MLX_SIMPLE32(NAME, KERNEL)                                             \
    extern "C" int rsl_mlx_##NAME(rsl_mlx_stream *s, const void *w,                \
        const float *x, float *out, int M, int K)                                 \
        { return mlx_packed_matvec(s, @KERNEL, w, x, out, M, K, 1); }              \
    extern "C" int rsl_mlx_##NAME##_batched(rsl_mlx_stream *s, const void *w,      \
        const float *x, float *out, int M, int K, int N)                          \
        { return mlx_packed_matvec(s, @KERNEL, w, x, out, M, K, N); }
#else
#define RSL_MLX_SIMPLE32(NAME, KERNEL)                                             \
    extern "C" int rsl_mlx_##NAME(rsl_mlx_stream *s, const void *w,                \
        const float *x, float *out, int M, int K)                                 \
        { (void)s;(void)w;(void)x;(void)out;(void)M;(void)K; return -1; }          \
    extern "C" int rsl_mlx_##NAME##_batched(rsl_mlx_stream *s, const void *w,      \
        const float *x, float *out, int M, int K, int N)                          \
        { (void)s;(void)w;(void)x;(void)out;(void)M;(void)K;(void)N; return -1; }
#endif
RSL_MLX_SIMPLE32(matvec_q5_0_packed_f32, "rsl_mlx_matvec_q5_0_packed_f32_kernel")
RSL_MLX_SIMPLE32(matvec_q4_1_packed_f32, "rsl_mlx_matvec_q4_1_packed_f32_kernel")
RSL_MLX_SIMPLE32(matvec_q5_1_packed_f32, "rsl_mlx_matvec_q5_1_packed_f32_kernel")
// IQ4_NL is a 32-wide codebook quant → the same simple path; IQ4_XS is 256-wide
// (below).
RSL_MLX_SIMPLE32(matvec_iq4_nl_packed_f32, "rsl_mlx_matvec_iq4_nl_packed_f32_kernel")

// IQ4_XS: 256-wide codebook quant — K % 256 guard + shared dispatch.
extern "C" int rsl_mlx_matvec_iq4_xs_packed_f32(rsl_mlx_stream *s, const void *w,
    const float *x, float *out, int M, int K) {
#if RSL_MLX_HAVE_METAL
    if ((K % 256) != 0) return -1;
    return mlx_packed_matvec(s, @"rsl_mlx_matvec_iq4_xs_packed_f32_kernel", w, x, out, M, K, 1);
#else
    (void)s;(void)w;(void)x;(void)out;(void)M;(void)K; return -1;
#endif
}
extern "C" int rsl_mlx_matvec_iq4_xs_packed_f32_batched(rsl_mlx_stream *s, const void *w,
    const float *x, float *out, int M, int K, int N) {
#if RSL_MLX_HAVE_METAL
    if ((K % 256) != 0) return -1;
    return mlx_packed_matvec(s, @"rsl_mlx_matvec_iq4_xs_packed_f32_kernel", w, x, out, M, K, N);
#else
    (void)s;(void)w;(void)x;(void)out;(void)M;(void)K;(void)N; return -1;
#endif
}

// --- MXFP4/6/8 (32-wide, E8M0 scale) via SIMPLE32; NVFP4 (16-wide, E4M3 scale)
// needs K%16 so it uses mlx_packed_matvec_align(…, 16). Ported byte-exact from
// the rsl_matvec_{mxfp4,mxfp6,mxfp8,nvfp4}_packed_f32_usm SYCL refs. ---
RSL_MLX_SIMPLE32(matvec_mxfp4_packed_f32, "rsl_mlx_matvec_mxfp4_packed_f32_kernel")
RSL_MLX_SIMPLE32(matvec_mxfp6_packed_f32, "rsl_mlx_matvec_mxfp6_packed_f32_kernel")
RSL_MLX_SIMPLE32(matvec_mxfp8_packed_f32, "rsl_mlx_matvec_mxfp8_packed_f32_kernel")

extern "C" int rsl_mlx_matvec_nvfp4_packed_f32(rsl_mlx_stream *s, const void *w,
    const float *x, float *out, int M, int K) {
#if RSL_MLX_HAVE_METAL
    return mlx_packed_matvec_align(s, @"rsl_mlx_matvec_nvfp4_packed_f32_kernel", w, x, out, M, K, 1, 16);
#else
    (void)s;(void)w;(void)x;(void)out;(void)M;(void)K; return -1;
#endif
}
extern "C" int rsl_mlx_matvec_nvfp4_packed_f32_batched(rsl_mlx_stream *s, const void *w,
    const float *x, float *out, int M, int K, int N) {
#if RSL_MLX_HAVE_METAL
    return mlx_packed_matvec_align(s, @"rsl_mlx_matvec_nvfp4_packed_f32_kernel", w, x, out, M, K, N, 16);
#else
    (void)s;(void)w;(void)x;(void)out;(void)M;(void)K;(void)N; return -1;
#endif
}

// PQ2_0 (PrismML Bonsai 2-bit): 128-wide super-block, K % 128.
extern "C" int rsl_mlx_matvec_pq2_0_packed_f32(rsl_mlx_stream *s, const void *w,
    const float *x, float *out, int M, int K) {
#if RSL_MLX_HAVE_METAL
    return mlx_packed_matvec_align(s, @"rsl_mlx_matvec_pq2_0_packed_f32_kernel", w, x, out, M, K, 1, 128);
#else
    (void)s;(void)w;(void)x;(void)out;(void)M;(void)K; return -1;
#endif
}
extern "C" int rsl_mlx_matvec_pq2_0_packed_f32_batched(rsl_mlx_stream *s, const void *w,
    const float *x, float *out, int M, int K, int N) {
#if RSL_MLX_HAVE_METAL
    return mlx_packed_matvec_align(s, @"rsl_mlx_matvec_pq2_0_packed_f32_kernel", w, x, out, M, K, N, 128);
#else
    (void)s;(void)w;(void)x;(void)out;(void)M;(void)K;(void)N; return -1;
#endif
}

// --- IQ-grid quants (IQ1_S/IQ1_M/IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S): LIVE.
// All 256-wide codebook quants using the build.rs-staged IQ grids + KSIGNS/
// KMASK Metal constants; ported byte-exact from the SYCL refs. K%256. ---
#if RSL_MLX_HAVE_METAL
#define RSL_MLX_K256(NAME, KERNEL)                                                \
    extern "C" int rsl_mlx_##NAME(rsl_mlx_stream *s, const void *w,                \
        const float *x, float *out, int M, int K)                                 \
        { return mlx_packed_matvec_align(s, @KERNEL, w, x, out, M, K, 1, 256); }   \
    extern "C" int rsl_mlx_##NAME##_batched(rsl_mlx_stream *s, const void *w,      \
        const float *x, float *out, int M, int K, int N)                          \
        { return mlx_packed_matvec_align(s, @KERNEL, w, x, out, M, K, N, 256); }
#else
#define RSL_MLX_K256(NAME, KERNEL)                                                \
    extern "C" int rsl_mlx_##NAME(rsl_mlx_stream *s, const void *w,                \
        const float *x, float *out, int M, int K)                                 \
        { (void)s;(void)w;(void)x;(void)out;(void)M;(void)K; return -1; }          \
    extern "C" int rsl_mlx_##NAME##_batched(rsl_mlx_stream *s, const void *w,      \
        const float *x, float *out, int M, int K, int N)                          \
        { (void)s;(void)w;(void)x;(void)out;(void)M;(void)K;(void)N; return -1; }
#endif
RSL_MLX_K256(matvec_iq2_xxs_packed_f32, "rsl_mlx_matvec_iq2_xxs_packed_f32_kernel")
RSL_MLX_K256(matvec_iq2_xs_packed_f32, "rsl_mlx_matvec_iq2_xs_packed_f32_kernel")
RSL_MLX_K256(matvec_iq2_s_packed_f32, "rsl_mlx_matvec_iq2_s_packed_f32_kernel")
RSL_MLX_K256(matvec_iq3_xxs_packed_f32, "rsl_mlx_matvec_iq3_xxs_packed_f32_kernel")
RSL_MLX_K256(matvec_iq3_s_packed_f32, "rsl_mlx_matvec_iq3_s_packed_f32_kernel")
RSL_MLX_K256(matvec_iq1_s_packed_f32, "rsl_mlx_matvec_iq1_s_packed_f32_kernel")
RSL_MLX_K256(matvec_iq1_m_packed_f32, "rsl_mlx_matvec_iq1_m_packed_f32_kernel")
#undef RSL_MLX_K256

// Forward-pass primitives — LIVE (device-resident Metal dispatch).
extern "C" int rsl_mlx_add_rmsnorm_f32(rsl_mlx_stream *s, float *hidden,
    const float *branch, const float *w, float *y_norm, int n_rows, int d, float eps) {
#if RSL_MLX_HAVE_METAL
    if (!s || !hidden || !branch || !w || !y_norm || n_rows <= 0 || d <= 0) return -1;
    size_t oh = 0, ob = 0, ow = 0, oy = 0;
    id<MTLBuffer> bh = mlx_resolve(hidden, &oh);
    id<MTLBuffer> bb = mlx_resolve(branch, &ob);
    id<MTLBuffer> bw = mlx_resolve(w, &ow);
    id<MTLBuffer> by = mlx_resolve(y_norm, &oy);
    if (bh == nil || bb == nil || bw == nil || by == nil) { g_rsl_mlx_errors++; return -1; }
    int dd = d;
    float ee = eps;
    MTLSize grid = MTLSizeMake((NSUInteger)n_rows, 1, 1);
    MTLSize tpg = MTLSizeMake(32, 1, 1);
    return mlx_run(s, @"rsl_mlx_add_rmsnorm_f32_kernel", grid, tpg, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bh offset:oh atIndex:0];
        [enc setBuffer:bb offset:ob atIndex:1];
        [enc setBuffer:bw offset:ow atIndex:2];
        [enc setBuffer:by offset:oy atIndex:3];
        [enc setBytes:&dd length:sizeof(int) atIndex:4];
        [enc setBytes:&ee length:sizeof(float) atIndex:5];
    });
#else
    (void)s;(void)hidden;(void)branch;(void)w;(void)y_norm;(void)n_rows;(void)d;(void)eps; RSL_MLX_STUB_KERNEL
#endif
}
extern "C" int rsl_mlx_rope_f32(rsl_mlx_stream *s, float *qk, int n_heads,
    int head_dim, int pos, const float *inv_freq) {
#if RSL_MLX_HAVE_METAL
    if (!s || !qk || !inv_freq || n_heads <= 0 || head_dim <= 0 || (head_dim % 2) != 0) return -1;
    size_t oq = 0, of = 0;
    id<MTLBuffer> bq = mlx_resolve(qk, &oq);
    id<MTLBuffer> bf = mlx_resolve(inv_freq, &of);
    if (bq == nil || bf == nil) { g_rsl_mlx_errors++; return -1; }
    int nh = n_heads, hd = head_dim, pp = pos;
    const NSUInteger total = (NSUInteger)n_heads * (NSUInteger)(head_dim / 2);
    const NSUInteger tg = 256;
    MTLSize grid = MTLSizeMake((total + tg - 1) / tg, 1, 1);
    MTLSize tpg = MTLSizeMake(tg, 1, 1);
    return mlx_run(s, @"rsl_mlx_rope_f32_kernel", grid, tpg, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bq offset:oq atIndex:0];
        [enc setBuffer:bf offset:of atIndex:1];
        [enc setBytes:&nh length:sizeof(int) atIndex:2];
        [enc setBytes:&hd length:sizeof(int) atIndex:3];
        [enc setBytes:&pp length:sizeof(int) atIndex:4];
    });
#else
    (void)s;(void)qk;(void)n_heads;(void)head_dim;(void)pos;(void)inv_freq; RSL_MLX_STUB_KERNEL
#endif
}
extern "C" int rsl_mlx_silu_mul_f32(rsl_mlx_stream *s, const float *x,
    const float *y, float *out, int n) {
#if RSL_MLX_HAVE_METAL
    if (!s || !x || !y || !out || n <= 0) return -1;
    size_t ox = 0, oy = 0, oo = 0;
    id<MTLBuffer> bx = mlx_resolve(x, &ox);
    id<MTLBuffer> by = mlx_resolve(y, &oy);
    id<MTLBuffer> bo = mlx_resolve(out, &oo);
    if (bx == nil || by == nil || bo == nil) { g_rsl_mlx_errors++; return -1; }
    int nn = n;
    const NSUInteger tg = 256;
    MTLSize grid = MTLSizeMake(((NSUInteger)n + tg - 1) / tg, 1, 1);
    MTLSize tpg = MTLSizeMake(tg, 1, 1);
    return mlx_run(s, @"rsl_mlx_silu_mul_f32_kernel", grid, tpg, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bx offset:ox atIndex:0];
        [enc setBuffer:by offset:oy atIndex:1];
        [enc setBuffer:bo offset:oo atIndex:2];
        [enc setBytes:&nn length:sizeof(int) atIndex:3];
    });
#else
    (void)s;(void)x;(void)y;(void)out;(void)n; RSL_MLX_STUB_KERNEL
#endif
}
extern "C" int rsl_mlx_embedding_lookup_f32(rsl_mlx_stream *s, const float *table,
    const int *ids, float *out, int n_ids, int d) {
#if RSL_MLX_HAVE_METAL
    if (!s || !table || !ids || !out || n_ids <= 0 || d <= 0) return -1;
    size_t ot = 0, oi = 0, oo = 0;
    id<MTLBuffer> bt = mlx_resolve(table, &ot);
    id<MTLBuffer> bi = mlx_resolve(ids, &oi);   // ids is a DEVICE pointer here
    id<MTLBuffer> bo = mlx_resolve(out, &oo);
    if (bt == nil || bi == nil || bo == nil) { g_rsl_mlx_errors++; return -1; }
    int ni = n_ids, dd = d;
    const unsigned long long total = (unsigned long long)n_ids * (unsigned long long)d;
    const NSUInteger tg = 256;
    MTLSize grid = MTLSizeMake((NSUInteger)((total + tg - 1) / tg), 1, 1);
    MTLSize tpg = MTLSizeMake(tg, 1, 1);
    return mlx_run(s, @"rsl_mlx_embedding_lookup_f32_kernel", grid, tpg, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bt offset:ot atIndex:0];
        [enc setBuffer:bi offset:oi atIndex:1];
        [enc setBuffer:bo offset:oo atIndex:2];
        [enc setBytes:&ni length:sizeof(int) atIndex:3];
        [enc setBytes:&dd length:sizeof(int) atIndex:4];
    });
#else
    (void)s;(void)table;(void)ids;(void)out;(void)n_ids;(void)d; RSL_MLX_STUB_KERNEL
#endif
}

// FlashAttention (F32 K/V) — LIVE. GQA online-softmax: one 32-lane group per
// query (decode) / per (head, q_pos) (prefill); head_dim capped at 256 (the
// shader's threadgroup accumulator). Ports rsl_flash_attn_{decode,prefill}_usm.
extern "C" int rsl_mlx_flash_attn_decode_f32(rsl_mlx_stream *s, const float *q,
    const float *k, const float *v, float *out, int n_heads, int n_kv_heads,
    int head_dim, int max_ctx, int kv_len) {
#if RSL_MLX_HAVE_METAL
    if (!s || !q || !k || !v || !out) return -1;
    if (n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0 || max_ctx <= 0 || kv_len <= 0) return -1;
    if ((n_heads % n_kv_heads) != 0 || head_dim > 256) return -1;
    size_t oq = 0, ok = 0, ov = 0, oo = 0;
    id<MTLBuffer> bq = mlx_resolve(q, &oq);
    id<MTLBuffer> bk = mlx_resolve(k, &ok);
    id<MTLBuffer> bv = mlx_resolve(v, &ov);
    id<MTLBuffer> bo = mlx_resolve(out, &oo);
    if (bq == nil || bk == nil || bv == nil || bo == nil) { g_rsl_mlx_errors++; return -1; }
    int nh = n_heads, nkv = n_kv_heads, hd = head_dim, mc = max_ctx, kl = kv_len;
    MTLSize grid = MTLSizeMake((NSUInteger)n_heads, 1, 1);
    MTLSize tpg = MTLSizeMake(32, 1, 1);
    return mlx_run(s, @"rsl_mlx_flash_attn_decode_f32_kernel", grid, tpg, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bq offset:oq atIndex:0];
        [enc setBuffer:bk offset:ok atIndex:1];
        [enc setBuffer:bv offset:ov atIndex:2];
        [enc setBuffer:bo offset:oo atIndex:3];
        [enc setBytes:&nh length:sizeof(int) atIndex:4];
        [enc setBytes:&nkv length:sizeof(int) atIndex:5];
        [enc setBytes:&hd length:sizeof(int) atIndex:6];
        [enc setBytes:&mc length:sizeof(int) atIndex:7];
        [enc setBytes:&kl length:sizeof(int) atIndex:8];
    });
#else
    (void)s;(void)q;(void)k;(void)v;(void)out;(void)n_heads;(void)n_kv_heads;(void)head_dim;(void)max_ctx;(void)kv_len; RSL_MLX_STUB_KERNEL
#endif
}
extern "C" int rsl_mlx_flash_attn_prefill_f32(rsl_mlx_stream *s, const float *q,
    const float *k, const float *v, float *out, int n_heads, int n_kv_heads,
    int head_dim, int max_ctx, int kv_len_base, int n_new) {
#if RSL_MLX_HAVE_METAL
    if (!s || !q || !k || !v || !out) return -1;
    if (n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0 || max_ctx <= 0 || n_new <= 0) return -1;
    if ((n_heads % n_kv_heads) != 0 || head_dim > 256) return -1;
    if (kv_len_base < 0 || kv_len_base + n_new > max_ctx) return -1;
    size_t oq = 0, ok = 0, ov = 0, oo = 0;
    id<MTLBuffer> bq = mlx_resolve(q, &oq);
    id<MTLBuffer> bk = mlx_resolve(k, &ok);
    id<MTLBuffer> bv = mlx_resolve(v, &ov);
    id<MTLBuffer> bo = mlx_resolve(out, &oo);
    if (bq == nil || bk == nil || bv == nil || bo == nil) { g_rsl_mlx_errors++; return -1; }
    int nh = n_heads, nkv = n_kv_heads, hd = head_dim, mc = max_ctx, kb = kv_len_base, nn = n_new;
    MTLSize grid = MTLSizeMake((NSUInteger)n_heads, (NSUInteger)n_new, 1);
    MTLSize tpg = MTLSizeMake(32, 1, 1);
    return mlx_run(s, @"rsl_mlx_flash_attn_prefill_f32_kernel", grid, tpg, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bq offset:oq atIndex:0];
        [enc setBuffer:bk offset:ok atIndex:1];
        [enc setBuffer:bv offset:ov atIndex:2];
        [enc setBuffer:bo offset:oo atIndex:3];
        [enc setBytes:&nh length:sizeof(int) atIndex:4];
        [enc setBytes:&nkv length:sizeof(int) atIndex:5];
        [enc setBytes:&hd length:sizeof(int) atIndex:6];
        [enc setBytes:&mc length:sizeof(int) atIndex:7];
        [enc setBytes:&kb length:sizeof(int) atIndex:8];
        [enc setBytes:&nn length:sizeof(int) atIndex:9];
    });
#else
    (void)s;(void)q;(void)k;(void)v;(void)out;(void)n_heads;(void)n_kv_heads;(void)head_dim;(void)max_ctx;(void)kv_len_base;(void)n_new; RSL_MLX_STUB_KERNEL
#endif
}

// Quantized-KV FlashAttention. TODO(phase1): dequant K/V on the fly inside
// the flash kernel; byte-exact ports of the CPU reference q4_0/nvfp4/mxfp*/
// turboquant/q8_0 KV kernels.
#define RSL_MLX_DEFINE_FLASH_KV(SUFFIX)                                       \
    extern "C" int rsl_mlx_flash_attn_decode_##SUFFIX(rsl_mlx_stream *s,      \
        const float *q, const void *k_packed, const void *v_packed,          \
        float *out, int n_heads, int n_kv_heads, int head_dim, int max_ctx,  \
        int kv_len)                                                          \
        { (void)s;(void)q;(void)k_packed;(void)v_packed;(void)out;(void)n_heads;(void)n_kv_heads;(void)head_dim;(void)max_ctx;(void)kv_len; RSL_MLX_STUB_KERNEL } \
    extern "C" int rsl_mlx_flash_attn_prefill_##SUFFIX(rsl_mlx_stream *s,     \
        const float *q, const void *k_packed, const void *v_packed,          \
        float *out, int n_heads, int n_kv_heads, int head_dim, int max_ctx,  \
        int kv_len_base, int n_new)                                          \
        { (void)s;(void)q;(void)k_packed;(void)v_packed;(void)out;(void)n_heads;(void)n_kv_heads;(void)head_dim;(void)max_ctx;(void)kv_len_base;(void)n_new; RSL_MLX_STUB_KERNEL }
// Quantized-KV flash (scale-free formats) — LIVE. Real Metal dispatch via the
// mlx_flash_kv_{decode,prefill} helpers; KALIGN = the KV block width.
#if RSL_MLX_HAVE_METAL
#define RSL_MLX_FLASH_KV(SUFFIX, KALIGN)                                           \
    extern "C" int rsl_mlx_flash_attn_decode_##SUFFIX(rsl_mlx_stream *s,           \
        const float *q, const void *k_packed, const void *v_packed, float *out,   \
        int n_heads, int n_kv_heads, int head_dim, int max_ctx, int kv_len)        \
        { return mlx_flash_kv_decode(s, @"rsl_mlx_flash_attn_decode_" #SUFFIX "_kernel", \
            q, k_packed, v_packed, out, n_heads, n_kv_heads, head_dim, max_ctx, kv_len, KALIGN); } \
    extern "C" int rsl_mlx_flash_attn_prefill_##SUFFIX(rsl_mlx_stream *s,          \
        const float *q, const void *k_packed, const void *v_packed, float *out,   \
        int n_heads, int n_kv_heads, int head_dim, int max_ctx, int kv_len_base, int n_new) \
        { return mlx_flash_kv_prefill(s, @"rsl_mlx_flash_attn_prefill_" #SUFFIX "_kernel", \
            q, k_packed, v_packed, out, n_heads, n_kv_heads, head_dim, max_ctx, kv_len_base, n_new, KALIGN); }
#else
#define RSL_MLX_FLASH_KV(SUFFIX, KALIGN)                                           \
    extern "C" int rsl_mlx_flash_attn_decode_##SUFFIX(rsl_mlx_stream *s,           \
        const float *q, const void *k_packed, const void *v_packed, float *out,   \
        int n_heads, int n_kv_heads, int head_dim, int max_ctx, int kv_len)        \
        { (void)s;(void)q;(void)k_packed;(void)v_packed;(void)out;(void)n_heads;(void)n_kv_heads;(void)head_dim;(void)max_ctx;(void)kv_len; return -1; } \
    extern "C" int rsl_mlx_flash_attn_prefill_##SUFFIX(rsl_mlx_stream *s,          \
        const float *q, const void *k_packed, const void *v_packed, float *out,   \
        int n_heads, int n_kv_heads, int head_dim, int max_ctx, int kv_len_base, int n_new) \
        { (void)s;(void)q;(void)k_packed;(void)v_packed;(void)out;(void)n_heads;(void)n_kv_heads;(void)head_dim;(void)max_ctx;(void)kv_len_base;(void)n_new; return -1; }
#endif
RSL_MLX_FLASH_KV(q4_0, 32)
RSL_MLX_FLASH_KV(nvfp4, 16)
RSL_MLX_FLASH_KV(mxfp4, 32)
RSL_MLX_FLASH_KV(mxfp6, 32)
RSL_MLX_FLASH_KV(mxfp8, 32)
#undef RSL_MLX_FLASH_KV

// TurboQuant KV flash carries per-row f32 scales + a `bits` selector, so it
// has its own signature (not the macro above).
// TurboQuant-KV flash — LIVE. 6 device buffers (q/k/v/k_scales/v_scales/out) +
// `bits` + dims; head_dim a power of two <= 256 (the in-kernel inverse-WHT).
extern "C" int rsl_mlx_flash_attn_decode_tq(rsl_mlx_stream *s, const float *q,
    const void *k_packed, const void *v_packed, const float *k_scales,
    const float *v_scales, int bits, float *out, int n_heads, int n_kv_heads,
    int head_dim, int max_ctx, int kv_len) {
#if RSL_MLX_HAVE_METAL
    if (!s || !q || !k_packed || !v_packed || !k_scales || !v_scales || !out) return -1;
    if (bits != 1 && bits != 2 && bits != 4 && bits != 8) return -1;
    if (n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0 || max_ctx <= 0 || kv_len <= 0) return -1;
    if ((n_heads % n_kv_heads) != 0 || head_dim > 256 || (head_dim & (head_dim - 1)) != 0) return -1;
    size_t oq=0,ok=0,ov=0,oks=0,ovs=0,oo=0;
    id<MTLBuffer> bq=mlx_resolve(q,&oq), bk=mlx_resolve(k_packed,&ok), bv=mlx_resolve(v_packed,&ov),
        bks=mlx_resolve(k_scales,&oks), bvs=mlx_resolve(v_scales,&ovs), bo=mlx_resolve(out,&oo);
    if(bq==nil||bk==nil||bv==nil||bks==nil||bvs==nil||bo==nil){g_rsl_mlx_errors++;return -1;}
    int bb=bits,nh=n_heads,nkv=n_kv_heads,hd=head_dim,mc=max_ctx,kl=kv_len;
    MTLSize grid=MTLSizeMake((NSUInteger)n_heads,1,1), tpg=MTLSizeMake(32,1,1);
    return mlx_run(s,@"rsl_mlx_flash_attn_decode_tq_kernel",grid,tpg,^(id<MTLComputeCommandEncoder> enc){
        [enc setBuffer:bq offset:oq atIndex:0];
        [enc setBuffer:bk offset:ok atIndex:1];
        [enc setBuffer:bv offset:ov atIndex:2];
        [enc setBuffer:bks offset:oks atIndex:3];
        [enc setBuffer:bvs offset:ovs atIndex:4];
        [enc setBytes:&bb length:sizeof(int) atIndex:5];
        [enc setBuffer:bo offset:oo atIndex:6];
        [enc setBytes:&nh length:sizeof(int) atIndex:7];
        [enc setBytes:&nkv length:sizeof(int) atIndex:8];
        [enc setBytes:&hd length:sizeof(int) atIndex:9];
        [enc setBytes:&mc length:sizeof(int) atIndex:10];
        [enc setBytes:&kl length:sizeof(int) atIndex:11];
    });
#else
    (void)s;(void)q;(void)k_packed;(void)v_packed;(void)k_scales;(void)v_scales;(void)bits;(void)out;(void)n_heads;(void)n_kv_heads;(void)head_dim;(void)max_ctx;(void)kv_len; return -1;
#endif
}
extern "C" int rsl_mlx_flash_attn_prefill_tq(rsl_mlx_stream *s, const float *q,
    const void *k_packed, const void *v_packed, const float *k_scales,
    const float *v_scales, int bits, float *out, int n_heads, int n_kv_heads,
    int head_dim, int max_ctx, int kv_len_base, int n_new) {
#if RSL_MLX_HAVE_METAL
    if (!s || !q || !k_packed || !v_packed || !k_scales || !v_scales || !out) return -1;
    if (bits != 1 && bits != 2 && bits != 4 && bits != 8) return -1;
    if (n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0 || max_ctx <= 0 || n_new <= 0) return -1;
    if ((n_heads % n_kv_heads) != 0 || head_dim > 256 || (head_dim & (head_dim - 1)) != 0) return -1;
    if (kv_len_base < 0 || kv_len_base + n_new > max_ctx) return -1;
    size_t oq=0,ok=0,ov=0,oks=0,ovs=0,oo=0;
    id<MTLBuffer> bq=mlx_resolve(q,&oq), bk=mlx_resolve(k_packed,&ok), bv=mlx_resolve(v_packed,&ov),
        bks=mlx_resolve(k_scales,&oks), bvs=mlx_resolve(v_scales,&ovs), bo=mlx_resolve(out,&oo);
    if(bq==nil||bk==nil||bv==nil||bks==nil||bvs==nil||bo==nil){g_rsl_mlx_errors++;return -1;}
    int bb=bits,nh=n_heads,nkv=n_kv_heads,hd=head_dim,mc=max_ctx,kb=kv_len_base,nn=n_new;
    MTLSize grid=MTLSizeMake((NSUInteger)n_heads,(NSUInteger)n_new,1), tpg=MTLSizeMake(32,1,1);
    return mlx_run(s,@"rsl_mlx_flash_attn_prefill_tq_kernel",grid,tpg,^(id<MTLComputeCommandEncoder> enc){
        [enc setBuffer:bq offset:oq atIndex:0];
        [enc setBuffer:bk offset:ok atIndex:1];
        [enc setBuffer:bv offset:ov atIndex:2];
        [enc setBuffer:bks offset:oks atIndex:3];
        [enc setBuffer:bvs offset:ovs atIndex:4];
        [enc setBytes:&bb length:sizeof(int) atIndex:5];
        [enc setBuffer:bo offset:oo atIndex:6];
        [enc setBytes:&nh length:sizeof(int) atIndex:7];
        [enc setBytes:&nkv length:sizeof(int) atIndex:8];
        [enc setBytes:&hd length:sizeof(int) atIndex:9];
        [enc setBytes:&mc length:sizeof(int) atIndex:10];
        [enc setBytes:&kb length:sizeof(int) atIndex:11];
        [enc setBytes:&nn length:sizeof(int) atIndex:12];
    });
#else
    (void)s;(void)q;(void)k_packed;(void)v_packed;(void)k_scales;(void)v_scales;(void)bits;(void)out;(void)n_heads;(void)n_kv_heads;(void)head_dim;(void)max_ctx;(void)kv_len_base;(void)n_new; return -1;
#endif
}

// Q8_0 KV flash: i8 slab + per-row f32 scale (its own signature too).
// Q8_0-KV flash — LIVE. i8 slab K/V + per-row f32 scales; 6 device buffers.
extern "C" int rsl_mlx_flash_attn_decode_q8_0(rsl_mlx_stream *s, const float *q,
    const void *k_packed, const void *v_packed, const float *k_scales,
    const float *v_scales, float *out, int n_heads, int n_kv_heads,
    int head_dim, int max_ctx, int kv_len) {
#if RSL_MLX_HAVE_METAL
    if (!s || !q || !k_packed || !v_packed || !k_scales || !v_scales || !out) return -1;
    if (n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0 || max_ctx <= 0 || kv_len <= 0) return -1;
    if ((n_heads % n_kv_heads) != 0 || head_dim > 256) return -1;
    size_t oq=0,ok=0,ov=0,oks=0,ovs=0,oo=0;
    id<MTLBuffer> bq=mlx_resolve(q,&oq), bk=mlx_resolve(k_packed,&ok), bv=mlx_resolve(v_packed,&ov),
        bks=mlx_resolve(k_scales,&oks), bvs=mlx_resolve(v_scales,&ovs), bo=mlx_resolve(out,&oo);
    if(bq==nil||bk==nil||bv==nil||bks==nil||bvs==nil||bo==nil){g_rsl_mlx_errors++;return -1;}
    int nh=n_heads,nkv=n_kv_heads,hd=head_dim,mc=max_ctx,kl=kv_len;
    MTLSize grid=MTLSizeMake((NSUInteger)n_heads,1,1), tpg=MTLSizeMake(32,1,1);
    return mlx_run(s,@"rsl_mlx_flash_attn_decode_q8_0_kernel",grid,tpg,^(id<MTLComputeCommandEncoder> enc){
        [enc setBuffer:bq offset:oq atIndex:0];
        [enc setBuffer:bk offset:ok atIndex:1];
        [enc setBuffer:bv offset:ov atIndex:2];
        [enc setBuffer:bks offset:oks atIndex:3];
        [enc setBuffer:bvs offset:ovs atIndex:4];
        [enc setBuffer:bo offset:oo atIndex:5];
        [enc setBytes:&nh length:sizeof(int) atIndex:6];
        [enc setBytes:&nkv length:sizeof(int) atIndex:7];
        [enc setBytes:&hd length:sizeof(int) atIndex:8];
        [enc setBytes:&mc length:sizeof(int) atIndex:9];
        [enc setBytes:&kl length:sizeof(int) atIndex:10];
    });
#else
    (void)s;(void)q;(void)k_packed;(void)v_packed;(void)k_scales;(void)v_scales;(void)out;(void)n_heads;(void)n_kv_heads;(void)head_dim;(void)max_ctx;(void)kv_len; return -1;
#endif
}
extern "C" int rsl_mlx_flash_attn_prefill_q8_0(rsl_mlx_stream *s, const float *q,
    const void *k_packed, const void *v_packed, const float *k_scales,
    const float *v_scales, float *out, int n_heads, int n_kv_heads,
    int head_dim, int max_ctx, int kv_len_base, int n_new) {
#if RSL_MLX_HAVE_METAL
    if (!s || !q || !k_packed || !v_packed || !k_scales || !v_scales || !out) return -1;
    if (n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0 || max_ctx <= 0 || n_new <= 0) return -1;
    if ((n_heads % n_kv_heads) != 0 || head_dim > 256) return -1;
    if (kv_len_base < 0 || kv_len_base + n_new > max_ctx) return -1;
    size_t oq=0,ok=0,ov=0,oks=0,ovs=0,oo=0;
    id<MTLBuffer> bq=mlx_resolve(q,&oq), bk=mlx_resolve(k_packed,&ok), bv=mlx_resolve(v_packed,&ov),
        bks=mlx_resolve(k_scales,&oks), bvs=mlx_resolve(v_scales,&ovs), bo=mlx_resolve(out,&oo);
    if(bq==nil||bk==nil||bv==nil||bks==nil||bvs==nil||bo==nil){g_rsl_mlx_errors++;return -1;}
    int nh=n_heads,nkv=n_kv_heads,hd=head_dim,mc=max_ctx,kb=kv_len_base,nn=n_new;
    MTLSize grid=MTLSizeMake((NSUInteger)n_heads,(NSUInteger)n_new,1), tpg=MTLSizeMake(32,1,1);
    return mlx_run(s,@"rsl_mlx_flash_attn_prefill_q8_0_kernel",grid,tpg,^(id<MTLComputeCommandEncoder> enc){
        [enc setBuffer:bq offset:oq atIndex:0];
        [enc setBuffer:bk offset:ok atIndex:1];
        [enc setBuffer:bv offset:ov atIndex:2];
        [enc setBuffer:bks offset:oks atIndex:3];
        [enc setBuffer:bvs offset:ovs atIndex:4];
        [enc setBuffer:bo offset:oo atIndex:5];
        [enc setBytes:&nh length:sizeof(int) atIndex:6];
        [enc setBytes:&nkv length:sizeof(int) atIndex:7];
        [enc setBytes:&hd length:sizeof(int) atIndex:8];
        [enc setBytes:&mc length:sizeof(int) atIndex:9];
        [enc setBytes:&kb length:sizeof(int) atIndex:10];
        [enc setBytes:&nn length:sizeof(int) atIndex:11];
    });
#else
    (void)s;(void)q;(void)k_packed;(void)v_packed;(void)k_scales;(void)v_scales;(void)out;(void)n_heads;(void)n_kv_heads;(void)head_dim;(void)max_ctx;(void)kv_len_base;(void)n_new; return -1;
#endif
}

// Sampling — LIVE. Single threadgroup argmax (lowest index on ties).
extern "C" int rsl_mlx_argmax_f32(rsl_mlx_stream *s, const float *logits,
    int vocab, int *out_idx) {
#if RSL_MLX_HAVE_METAL
    if (!s || !logits || !out_idx || vocab <= 0) return -1;
    size_t ol = 0, oo = 0;
    id<MTLBuffer> bl = mlx_resolve(logits, &ol);
    id<MTLBuffer> bo = mlx_resolve(out_idx, &oo);
    if (bl == nil || bo == nil) { g_rsl_mlx_errors++; return -1; }
    int vv = vocab;
    // One threadgroup of 256 (power of two for the tree reduction); the kernel
    // threadgroup arrays are sized 256 to match.
    MTLSize grid = MTLSizeMake(1, 1, 1);
    MTLSize tpg = MTLSizeMake(256, 1, 1);
    return mlx_run(s, @"rsl_mlx_argmax_f32_kernel", grid, tpg, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bl offset:ol atIndex:0];
        [enc setBuffer:bo offset:oo atIndex:1];
        [enc setBytes:&vv length:sizeof(int) atIndex:2];
    });
#else
    (void)s;(void)logits;(void)vocab;(void)out_idx; RSL_MLX_STUB_KERNEL
#endif
}

#undef RSL_MLX_STUB_KERNEL
