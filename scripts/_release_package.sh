#!/usr/bin/env bash
# ============================================================================
# Assemble a rustllama Linux release archive — runs INSIDE the build container
# (Rocky 10 x86_64 GPU image, or the Ubuntu 24.04 aarch64 image), after
# scripts/build.sh --headless. Produces the canonical layout:
#
#   release/rustllama-<version>-linux-<arch>/
#     rustllama          launcher (sets LD_LIBRARY_PATH → ./lib, execs .bin)
#     rustllama.bin      the ELF binary (all backends compiled in)
#     lib/               bundled Intel oneAPI SYCL runtime .so closure (x86_64;
#                        aarch64 ships the SYCL no-op stub → no bundled libs)
#     README.md  LICENSE-MIT  LICENSE-APACHE  docs/
#
# then `release/rustllama-<version>-linux-<arch>.tar.gz`.
#
# Self-contained: the launcher makes the bundled interdependent oneAPI libs
# resolve via LD_LIBRARY_PATH (robust to their internal cross-references,
# unlike a single RPATH). CUDA is statically linked. A GPU backend still needs
# its kernel driver on the host at run time (NVIDIA driver for CUDA; the Intel
# compute-runtime for SYCL); absent that, rustllama falls back to CPU.
#
# Usage: scripts/_release_package.sh <binary> <version> <arch>
#   <binary>  path to the built rustllama (e.g. /target/release/rustllama)
#   <version> e.g. 0.1.0
#   <arch>    x86_64 | aarch64
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

BIN="${1:?usage: _release_package.sh <binary> <version> <arch>}"
VERSION="${2:?missing version}"
ARCH="${3:?missing arch}"

[ -x "$BIN" ] || { echo "ERROR: binary not found/executable: $BIN" >&2; exit 1; }

# patchelf sets the $ORIGIN-relative RPATH that makes the single `rustllama`
# binary relocatable. Install it on demand (works whether the build image is
# Rocky/RHEL via dnf — enabling CRB/EPEL if needed — or Ubuntu via apt).
ensure_patchelf() {
  command -v patchelf >/dev/null 2>&1 && return 0
  echo ">> installing patchelf..." >&2
  if command -v apt-get >/dev/null 2>&1; then
    apt-get update -qq >/dev/null 2>&1 || true
    apt-get install -y -qq patchelf >/dev/null 2>&1 || true
  elif command -v dnf >/dev/null 2>&1; then
    dnf install -y patchelf >/dev/null 2>&1 || {
      dnf install -y dnf-plugins-core >/dev/null 2>&1 || true
      dnf config-manager --set-enabled crb >/dev/null 2>&1 || true
      dnf install -y epel-release >/dev/null 2>&1 || true
      dnf install -y patchelf >/dev/null 2>&1 || true
    }
  elif command -v yum >/dev/null 2>&1; then
    yum install -y patchelf >/dev/null 2>&1 || true
  fi
  command -v patchelf >/dev/null 2>&1
}

NAME="rustllama-${VERSION}-linux-${ARCH}"
OUT="release"
STAGE="${OUT}/${NAME}"
rm -rf "$STAGE"
mkdir -p "$STAGE/lib"

echo ">> assembling ${NAME}"

# A single, relocatable `rustllama` binary (no launcher wrapper). After
# bundling its runtime libs below we set an $ORIGIN-relative RPATH on the
# binary AND on every bundled lib (patchelf), so the loader resolves the whole
# closure from ./lib with no LD_LIBRARY_PATH and no wrapper — unpack anywhere
# and run `./rustllama`.
cp "$BIN" "$STAGE/rustllama"
chmod +x "$STAGE/rustllama"

# Resolve the binary's own directory (where the SYCL shim `librsl_kernels.so`
# lives, beside the binary) onto the search path, and source oneAPI setvars so
# the oneAPI runtime libs resolve — otherwise `ldd` can't find either and we'd
# bundle nothing, shipping a binary that won't even load (libsycl.so is a hard
# NEEDED dep). Harmless on aarch64 (no setvars, no oneAPI).
BINDIR="$(cd "$(dirname "$BIN")" && pwd)"
if [ -f /opt/intel/oneapi/setvars.sh ]; then
  set +u
  . /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
  set -u
fi
# ldd resolution only (for discovering what to bundle) — the shipped binary
# gets an $ORIGIN RPATH instead of relying on this.
export LD_LIBRARY_PATH="${BINDIR}:${LD_LIBRARY_PATH:-}"

# Bundle every shared lib the binary pulls in that ISN'T part of the base OS —
# i.e. our own `librsl_kernels.so` SYCL shim + the whole Intel oneAPI SYCL
# runtime closure. Recurse so interdependent oneAPI libs all come along. Base
# system libs (glibc/libstdc++/libgcc/the loader) + the NVIDIA driver are left
# to the host. The launcher's LD_LIBRARY_PATH=$ORIGIN/lib resolves the bundle
# at run time regardless of the libs' internal cross-references.
# Libraries to NEVER bundle: the base OS / glibc / GCC runtime + the dynamic
# loader. These are ABI-stable and present on every target Linux; bundling a
# glibc/libstdc++ from the build image can break on a host with a different
# one. Matched by BASENAME (not path) so we can still bundle the GUI's
# webkit2gtk/GTK graph, which lives in the same /usr/lib64 as glibc.
is_base_os_lib() {
  case "$(basename "$1")" in
    ld-linux*|ld64.so*|libc.so.*|libm.so.*|libmvec.so.*|libpthread.so.*) return 0 ;;
    libdl.so.*|librt.so.*|libresolv.so.*|libutil.so.*|libnsl.so.*|libanl.so.*) return 0 ;;
    libcrypt.so.*|libstdc++.so.*|libgcc_s.so.*|libgomp.so.*) return 0 ;;
    *) return 1 ;;
  esac
}
# Libraries we deliberately DON'T bundle for LICENSE reasons. libmpg123 (+ its
# out123/syn123 siblings) is GPL-2.0-or-later — an OPTIONAL MP3 decoder pulled
# transitively via GStreamer; the GUI works without it (no in-webview MP3), and
# excluding it keeps the bundle free of strong copyleft. Left to the host if a
# user wants MP3 in the embedded webview.
is_excluded_lib() {
  case "$(basename "$1")" in
    libmpg123.so*|libout123.so*|libsyn123.so*) return 0 ;;
    *) return 1 ;;
  esac
}
# GPU / graphics driver-interface libs to leave on the HOST: the GL/EGL
# dispatch (libglvnd), the kernel GPU interfaces (libdrm/libgbm), and the
# NVIDIA/Intel userspace driver .so's must MATCH the host's kernel driver, so
# bundling a build-image copy risks a mismatch. Any host running the GUI has a
# display stack (mesa/vendor driver) already. Client X11/Wayland libs are NOT
# here — those are ABI-stable and fine to bundle.
is_host_graphics_lib() {
  case "$(basename "$1")" in
    libGL.so.*|libEGL.so.*|libGLX*.so.*|libGLdispatch.so.*|libOpenGL.so.*|libGLESv2.so.*) return 0 ;;
    libdrm.so.*|libgbm.so.*|libgallium*.so.*|libGLX_*.so.*|libEGL_*.so.*) return 0 ;;
    libcuda.so.*|libnvidia-*.so.*|libze_intel_gpu.so.*|libigdrcl.so.*) return 0 ;;
    *) return 1 ;;
  esac
}
# Back-compat shim: the oneAPI-bundling loop below still calls is_system_lib.
# It now means "do NOT bundle" = base OS OR host-graphics OR license-excluded.
is_system_lib() {
  is_base_os_lib "$1" || is_host_graphics_lib "$1" || is_excluded_lib "$1"
}
# A shared object is named `<name>.so` or `<name>.so.<version>` — this filters
# out gdb pretty-printers (`*.so-gdb.py`), .cmake files, etc. that live in the
# same lib dir and aren't ELF (running ldd on them would error).
is_shared_object() {
  case "$1" in *.so | *.so.*) return 0 ;; *) return 1 ;; esac
}
bundle_deps() {
  # `|| true` so ldd on a non-ELF (or a lib with no deps) never aborts us.
  { ldd "$1" 2>/dev/null || true; } | awk '/=> \// {print $3}' | while read -r so; do
    [ -e "$so" ] || continue
    if is_system_lib "$so"; then continue; fi
    base="$(basename "$so")"
    if [ ! -e "$STAGE/lib/$base" ]; then
      cp -L "$so" "$STAGE/lib/"
      bundle_deps "$so"
    fi
  done
  return 0
}
bundle_deps "$STAGE/rustllama"

# Also bundle the Level-Zero Unified Runtime adapter that libur_loader
# dlopen()s at run time (ldd never lists it) + its non-system deps. rustllama
# prefers the Level-Zero backend for Intel GPUs, so we deliberately do NOT
# bundle the OpenCL adapter / libOpenCL / the Intel OpenCL CPU runtime
# (libintelocl + libcommon_clang, ~150 MB) — it isn't used (we have native CPU
# kernels), and an Intel GPU is reached via Level-Zero. The Intel GPU KERNEL
# driver (system libze_loader / libze_intel_gpu) must still be on the host for
# an Intel GPU to actually be used; without it, rustllama runs on CPU/CUDA.
for base in compiler/latest/lib compiler/latest/lib/x64; do
  d="/opt/intel/oneapi/${base}"
  [ -d "$d" ] || continue
  for pat in 'libur_adapter_level_zero.so*' 'libur_loader.so*'; do
    for f in "$d"/$pat; do
      [ -e "$f" ] || continue
      b="$(basename "$f")"
      is_shared_object "$b" || continue
      [ -e "$STAGE/lib/$b" ] || { cp -L "$f" "$STAGE/lib/"; bundle_deps "$f"; }
    done
  done
done

nlibs=$(find "$STAGE/lib" -type f 2>/dev/null | wc -l | tr -d ' ')
if [ "$nlibs" = "0" ]; then
  rmdir "$STAGE/lib" 2>/dev/null || true
  echo ">> no non-system runtime libs to bundle (expected on a pure-CPU/CUDA build)"
else
  echo ">> bundled ${nlibs} runtime lib(s):"
  ( cd "$STAGE/lib" && ls -1 | sed 's/^/     /' )
  # Relocatable RPATH so the single binary + its bundled libs find each other
  # via $ORIGIN with no wrapper / LD_LIBRARY_PATH. Each object gets its own
  # RPATH (RUNPATH is not transitive): the binary -> ./lib, each lib -> its own
  # dir (siblings). Also drops the libs' stale absolute /opt/intel build RPATHs.
  ensure_patchelf || { echo "ERROR: patchelf required to set the relocatable RPATH" >&2; exit 1; }
  patchelf --set-rpath '$ORIGIN/lib' "$STAGE/rustllama"
  for so in "$STAGE"/lib/*; do
    [ -f "$so" ] || continue
    patchelf --set-rpath '$ORIGIN' "$so" 2>/dev/null || true
  done
  echo ">> set \$ORIGIN RPATH (binary -> lib/, libs -> siblings)"
fi

# ---- Third-party bundled-library license NOTICES (LGPL-2.1 §4/§6) ---------
# Each bundled .so is a redistributable third-party library. For the LGPL
# ones (webkit2gtk/GTK/glib/...) LGPL-2.1 requires shipping the license text +
# a notice naming the lib + version + where its (unmodified upstream) source
# is. We emit a NOTICES table (lib -> package -> version -> license) and copy
# each owning package's own license files from the build image. Works on rpm
# (Rocky x86_64) or dpkg (Ubuntu aarch64). rustllama's OWN code stays MIT OR
# Apache-2.0; these libraries keep their own (mostly LGPL/permissive) licenses.
if [ -d "$STAGE/lib" ] && [ "$(find "$STAGE/lib" -type f 2>/dev/null | wc -l)" -gt 0 ]; then
  mkdir -p "$STAGE/licenses/third-party"
  NOTICES="$STAGE/licenses/THIRD-PARTY-NOTICES.txt"
  {
    echo "rustllama bundles the third-party shared libraries in ./lib, dynamically"
    echo "linked and replaceable. rustllama's own code is MIT OR Apache-2.0; each"
    echo "bundled library keeps its own license, listed below with its exact version."
    echo "LGPL-2.1 libraries' corresponding source is available as the named"
    echo "distribution's source package (SRPM / dsc) for that version; each owning"
    echo "package's license files are copied under ./licenses/third-party/<package>/."
    echo
    printf '%-38s %-26s %s\n' "LIBRARY (package)" "VERSION" "LICENSE"
    printf '%-38s %-26s %s\n' "----------------" "-------" "-------"
  } > "$NOTICES"
  have_rpm=0; command -v rpm  >/dev/null 2>&1 && have_rpm=1
  have_dpkg=0; command -v dpkg >/dev/null 2>&1 && have_dpkg=1
  seen_pkgs=" "
  for so in "$STAGE"/lib/*.so*; do
    [ -f "$so" ] || continue
    base="$(basename "$so")"
    syspath=""
    for d in /usr/lib64 /usr/lib /lib64 /lib \
             /opt/intel/oneapi/compiler/latest/lib /opt/intel/oneapi/compiler/latest/lib/x64; do
      [ -e "$d/$base" ] && { syspath="$d/$base"; break; }
    done
    pkg="(bundled runtime)"; ver="-"; lic="see upstream"
    if [ -n "$syspath" ] && [ "$have_rpm" = 1 ]; then
      p="$(rpm -qf --qf '%{NAME}|%{VERSION}-%{RELEASE}|%{LICENSE}\n' "$syspath" 2>/dev/null | head -1)"
      if [ -n "$p" ] && ! echo "$p" | grep -qi 'not owned'; then
        pkg="${p%%|*}"; rest="${p#*|}"; ver="${rest%%|*}"; lic="${rest#*|}"
        case "$seen_pkgs" in *" $pkg "*) : ;; *)
          seen_pkgs="$seen_pkgs$pkg "
          [ -d "/usr/share/licenses/$pkg" ] && cp -r "/usr/share/licenses/$pkg" "$STAGE/licenses/third-party/" 2>/dev/null || true
        ;; esac
      fi
    elif [ -n "$syspath" ] && [ "$have_dpkg" = 1 ]; then
      pkg="$(dpkg -S "$syspath" 2>/dev/null | head -1 | cut -d: -f1)"
      [ -z "$pkg" ] && pkg="(bundled runtime)"
      if [ "$pkg" != "(bundled runtime)" ]; then
        ver="$(dpkg-query -W -f='${Version}' "$pkg" 2>/dev/null || echo '-')"
        lic="$(awk -F': ' '/^License:/{print $2; exit}' "/usr/share/doc/$pkg/copyright" 2>/dev/null)"
        [ -z "$lic" ] && lic="see copyright"
        case "$seen_pkgs" in *" $pkg "*) : ;; *)
          seen_pkgs="$seen_pkgs$pkg "
          if [ -f "/usr/share/doc/$pkg/copyright" ]; then
            mkdir -p "$STAGE/licenses/third-party/$pkg"
            cp "/usr/share/doc/$pkg/copyright" "$STAGE/licenses/third-party/$pkg/" 2>/dev/null || true
          fi
        ;; esac
      fi
    fi
    printf '%-38s %-26s %s\n' "$base ($pkg)" "$ver" "$lic" >> "$NOTICES"
  done
  echo ">> wrote licenses/THIRD-PARTY-NOTICES.txt + per-package license files"
  # Flag (do not fail) any strong-copyleft that slipped into the bundle, so a
  # release never silently ships GPL. mpg123 is already excluded above.
  if grep -iE '(^|[^L])GPL-[0-9]' "$NOTICES" | grep -viE 'LGPL|GCC-exception|with exception|OR ' >/dev/null 2>&1; then
    echo ">> WARNING: a bundled lib reports a GPL license — review $NOTICES" >&2
    grep -iE '(^|[^L])GPL-[0-9]' "$NOTICES" | grep -viE 'LGPL|GCC-exception|with exception|OR ' >&2 || true
  fi
fi

# License + docs.
for f in README.md LICENSE-MIT LICENSE-APACHE; do
  [ -f "$f" ] && cp "$f" "$STAGE/"
done
[ -d docs ] && cp -r docs "$STAGE/docs"

# Archive: the nested top-level dir is exactly $NAME.
tar -czf "${OUT}/${NAME}.tar.gz" -C "$OUT" "$NAME"
echo ">> wrote ${OUT}/${NAME}.tar.gz"
ls -lh "${OUT}/${NAME}.tar.gz" 2>/dev/null || true
