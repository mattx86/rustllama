// Apple-Metal / MLX kernels for rustllama's macOS backend.
//
// Objective-C++ host shim: it owns the Metal device/queue + MLX stream and
// exposes the `rsl_mlx_*` C ABI that `src/lib.rs` binds to. It is the peer
// of the CUDA crate's `cuda/rsl_cuda.cu` and the SYCL crate's
// `cpp/rsl_kernels.cpp`.
//
// ============================ STATUS: PHASE 1a ==========================
// The MINIMAL real f32 Metal path is LIVE (macOS/Apple-Silicon, guarded by
// RSL_MLX_HAVE_METAL=1, which build.rs now defines on the real path). Real:
//   * rsl_mlx_device_count / _device_info — MTLCopyAllDevices enumeration,
//     name / recommendedMaxWorkingSetSize / registryID (uuid left zeroed;
//     the tuner keys on registryID on Apple).
//   * stream_create/_destroy — a stream owns an MTLDevice + MTLCommandQueue.
//   * The pointer<->MTLBuffer bridge — malloc_device / malloc_from_host / free
//     back a StorageModeShared MTLBuffer whose `.contents` is the raw "device
//     pointer" handed to Rust; a global registry maps [base,base+len) spans
//     back to their MTLBuffer (see mlx_registry_*/mlx_resolve above).
//     memcpy_h2d/d2h are plain memcpys (unified memory; offsets auto-honored).
//   * rsl_mlx_matvec_f32 — a real Metal compute dispatch of the
//     rsl_mlx_matvec_f32_kernel in rsl_mlx.metal (the metallib is compiled +
//     EMBEDDED by build.rs, loaded via newLibraryWithData:).
// On a non-Apple host none of this compiles — the crate links the generated
// no-op stub (build.rs build_mlx_stub) and device_count()==0 → CPU/SYCL/CUDA.
//
// ====================== PHASE 1b/1c TODO (DEFERRED) =====================
// Still inert (return -1 / no-op): rmsnorm_f32, the packed/quant matvecs,
// PTQ1_0/Hadamard, the forward-pass primitives (add_rmsnorm/rope/silu_mul/
// embedding_lookup), GQA flash decode/prefill (F32 + quantized-KV), and
// argmax — each a Metal compute kernel (or an mlx-c op) that is a BYTE-EXACT
// port of the CPU reference, validated by the Metal parity harness, the same
// discipline the SYCL/CUDA ports follow. They are never reached in Phase 1a
// except via an explicit device-resident dispatch, so returning -1 makes the
// caller fall back to the CPU kernel.
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

[[maybe_unused]] static id<MTLLibrary> g_mlx_library = nil;  // strong — keep the lib alive
static id<MTLComputePipelineState> g_mlx_matvec_pso = nil;   // strong
static id<MTLDevice> g_mlx_pso_device = nil;                 // strong (which dev the pso is for)
static std::mutex g_mlx_pso_mutex;

// Build (once, cached) the f32-matvec compute pipeline for `dev` from the
// embedded metallib. nil on failure (latches an error so the caller falls
// back to CPU).
static id<MTLComputePipelineState> mlx_matvec_pso(id<MTLDevice> dev) {
    std::lock_guard<std::mutex> lk(g_mlx_pso_mutex);
    if (g_mlx_matvec_pso != nil && g_mlx_pso_device == dev) return g_mlx_matvec_pso;
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
    id<MTLFunction> fn = [lib newFunctionWithName:@"rsl_mlx_matvec_f32_kernel"];
    if (fn == nil) {
        std::fprintf(stderr, "rsl_mlx: kernel rsl_mlx_matvec_f32_kernel not found\n");
        g_rsl_mlx_errors++;
        return nil;
    }
    id<MTLComputePipelineState> pso =
        [dev newComputePipelineStateWithFunction:fn error:&err];
    if (pso == nil) {
        std::fprintf(stderr, "rsl_mlx: pipeline build failed: %s\n",
                     err ? err.localizedDescription.UTF8String : "(nil)");
        g_rsl_mlx_errors++;
        return nil;
    }
    g_mlx_library = lib;
    g_mlx_matvec_pso = pso;
    g_mlx_pso_device = dev;
    return g_mlx_matvec_pso;
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
    (void)x; (void)w; (void)y; (void)n_rows; (void)d; (void)eps;
    // TODO(phase1): y = x * rsqrt(mean(x^2)+eps) * w, one threadgroup/row.
    return -1;
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

// PTQ1_0 (Bonsai ternary) + Prism Hadamard.
extern "C" int rsl_mlx_matvec_ptq1_0_packed_f32(rsl_mlx_stream *s, const void *w,
    const float *x, float *out, int M, int K) { (void)s;(void)w;(void)x;(void)out;(void)M;(void)K; RSL_MLX_STUB_KERNEL }
extern "C" int rsl_mlx_matvec_ptq1_0_packed_f32_batched(rsl_mlx_stream *s, const void *w,
    const float *x, float *out, int M, int K, int N) { (void)s;(void)w;(void)x;(void)out;(void)M;(void)K;(void)N; RSL_MLX_STUB_KERNEL }
extern "C" int rsl_mlx_hadamard_forward(rsl_mlx_stream *s, const float *x,
    const float *signs, float *out, int n_elems, int block) { (void)s;(void)x;(void)signs;(void)out;(void)n_elems;(void)block; RSL_MLX_STUB_KERNEL }

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
RSL_MLX_DEFINE_PACKED(matvec_q8_0_packed_f32)
RSL_MLX_DEFINE_PACKED(matvec_q4_k_packed_f32)
RSL_MLX_DEFINE_PACKED(matvec_q6_k_packed_f32)
RSL_MLX_DEFINE_PACKED(matvec_q5_k_packed_f32)
RSL_MLX_DEFINE_PACKED(matvec_q2_k_packed_f32)
RSL_MLX_DEFINE_PACKED(matvec_q8_k_packed_f32)
RSL_MLX_DEFINE_PACKED(matvec_q4_0_packed_f32)
RSL_MLX_DEFINE_PACKED(matvec_q5_0_packed_f32)
RSL_MLX_DEFINE_PACKED(matvec_q4_1_packed_f32)
RSL_MLX_DEFINE_PACKED(matvec_q5_1_packed_f32)
RSL_MLX_DEFINE_PACKED(matvec_iq4_nl_packed_f32)
RSL_MLX_DEFINE_PACKED(matvec_iq4_xs_packed_f32)
RSL_MLX_DEFINE_PACKED(matvec_iq2_xxs_packed_f32)
RSL_MLX_DEFINE_PACKED(matvec_iq2_xs_packed_f32)
RSL_MLX_DEFINE_PACKED(matvec_iq2_s_packed_f32)
RSL_MLX_DEFINE_PACKED(matvec_iq3_xxs_packed_f32)
RSL_MLX_DEFINE_PACKED(matvec_iq3_s_packed_f32)
RSL_MLX_DEFINE_PACKED(matvec_iq1_s_packed_f32)
RSL_MLX_DEFINE_PACKED(matvec_iq1_m_packed_f32)
RSL_MLX_DEFINE_PACKED(matvec_nvfp4_packed_f32)
RSL_MLX_DEFINE_PACKED(matvec_mxfp4_packed_f32)
RSL_MLX_DEFINE_PACKED(matvec_mxfp6_packed_f32)
RSL_MLX_DEFINE_PACKED(matvec_mxfp8_packed_f32)
RSL_MLX_DEFINE_PACKED(matvec_q3_k_packed_f32)
RSL_MLX_DEFINE_PACKED(matvec_pq2_0_packed_f32)
#undef RSL_MLX_DEFINE_PACKED

// Forward-pass primitives.
extern "C" int rsl_mlx_add_rmsnorm_f32(rsl_mlx_stream *s, float *hidden,
    const float *branch, const float *w, float *y_norm, int n_rows, int d, float eps)
    { (void)s;(void)hidden;(void)branch;(void)w;(void)y_norm;(void)n_rows;(void)d;(void)eps; RSL_MLX_STUB_KERNEL }
extern "C" int rsl_mlx_rope_f32(rsl_mlx_stream *s, float *qk, int n_heads,
    int head_dim, int pos, const float *inv_freq)
    { (void)s;(void)qk;(void)n_heads;(void)head_dim;(void)pos;(void)inv_freq; RSL_MLX_STUB_KERNEL }
extern "C" int rsl_mlx_silu_mul_f32(rsl_mlx_stream *s, const float *x,
    const float *y, float *out, int n)
    { (void)s;(void)x;(void)y;(void)out;(void)n; RSL_MLX_STUB_KERNEL }
extern "C" int rsl_mlx_embedding_lookup_f32(rsl_mlx_stream *s, const float *table,
    const int *ids, float *out, int n_ids, int d)
    { (void)s;(void)table;(void)ids;(void)out;(void)n_ids;(void)d; RSL_MLX_STUB_KERNEL }

// FlashAttention (F32 K/V). TODO(phase1): GQA online-softmax flash decode
// (one query/head) + causal prefill; port from the CUDA/SYCL flash kernels.
extern "C" int rsl_mlx_flash_attn_decode_f32(rsl_mlx_stream *s, const float *q,
    const float *k, const float *v, float *out, int n_heads, int n_kv_heads,
    int head_dim, int max_ctx, int kv_len)
    { (void)s;(void)q;(void)k;(void)v;(void)out;(void)n_heads;(void)n_kv_heads;(void)head_dim;(void)max_ctx;(void)kv_len; RSL_MLX_STUB_KERNEL }
extern "C" int rsl_mlx_flash_attn_prefill_f32(rsl_mlx_stream *s, const float *q,
    const float *k, const float *v, float *out, int n_heads, int n_kv_heads,
    int head_dim, int max_ctx, int kv_len_base, int n_new)
    { (void)s;(void)q;(void)k;(void)v;(void)out;(void)n_heads;(void)n_kv_heads;(void)head_dim;(void)max_ctx;(void)kv_len_base;(void)n_new; RSL_MLX_STUB_KERNEL }

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
RSL_MLX_DEFINE_FLASH_KV(q4_0)
RSL_MLX_DEFINE_FLASH_KV(nvfp4)
RSL_MLX_DEFINE_FLASH_KV(mxfp4)
RSL_MLX_DEFINE_FLASH_KV(mxfp6)
RSL_MLX_DEFINE_FLASH_KV(mxfp8)
#undef RSL_MLX_DEFINE_FLASH_KV

// TurboQuant KV flash carries per-row f32 scales + a `bits` selector, so it
// has its own signature (not the macro above).
extern "C" int rsl_mlx_flash_attn_decode_tq(rsl_mlx_stream *s, const float *q,
    const void *k_packed, const void *v_packed, const float *k_scales,
    const float *v_scales, int bits, float *out, int n_heads, int n_kv_heads,
    int head_dim, int max_ctx, int kv_len)
    { (void)s;(void)q;(void)k_packed;(void)v_packed;(void)k_scales;(void)v_scales;(void)bits;(void)out;(void)n_heads;(void)n_kv_heads;(void)head_dim;(void)max_ctx;(void)kv_len; RSL_MLX_STUB_KERNEL }
extern "C" int rsl_mlx_flash_attn_prefill_tq(rsl_mlx_stream *s, const float *q,
    const void *k_packed, const void *v_packed, const float *k_scales,
    const float *v_scales, int bits, float *out, int n_heads, int n_kv_heads,
    int head_dim, int max_ctx, int kv_len_base, int n_new)
    { (void)s;(void)q;(void)k_packed;(void)v_packed;(void)k_scales;(void)v_scales;(void)bits;(void)out;(void)n_heads;(void)n_kv_heads;(void)head_dim;(void)max_ctx;(void)kv_len_base;(void)n_new; RSL_MLX_STUB_KERNEL }

// Q8_0 KV flash: i8 slab + per-row f32 scale (its own signature too).
extern "C" int rsl_mlx_flash_attn_decode_q8_0(rsl_mlx_stream *s, const float *q,
    const void *k_packed, const void *v_packed, const float *k_scales,
    const float *v_scales, float *out, int n_heads, int n_kv_heads,
    int head_dim, int max_ctx, int kv_len)
    { (void)s;(void)q;(void)k_packed;(void)v_packed;(void)k_scales;(void)v_scales;(void)out;(void)n_heads;(void)n_kv_heads;(void)head_dim;(void)max_ctx;(void)kv_len; RSL_MLX_STUB_KERNEL }
extern "C" int rsl_mlx_flash_attn_prefill_q8_0(rsl_mlx_stream *s, const float *q,
    const void *k_packed, const void *v_packed, const float *k_scales,
    const float *v_scales, float *out, int n_heads, int n_kv_heads,
    int head_dim, int max_ctx, int kv_len_base, int n_new)
    { (void)s;(void)q;(void)k_packed;(void)v_packed;(void)k_scales;(void)v_scales;(void)out;(void)n_heads;(void)n_kv_heads;(void)head_dim;(void)max_ctx;(void)kv_len_base;(void)n_new; RSL_MLX_STUB_KERNEL }

// Sampling.
extern "C" int rsl_mlx_argmax_f32(rsl_mlx_stream *s, const float *logits,
    int vocab, int *out_idx)
    { (void)s;(void)logits;(void)vocab;(void)out_idx; RSL_MLX_STUB_KERNEL }

#undef RSL_MLX_STUB_KERNEL
