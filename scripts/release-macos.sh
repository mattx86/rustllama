#!/usr/bin/env bash
# ============================================================================
# Build + package a rustllama macOS release archive. The Mach-O analogue of
# scripts/_release_build_linux.sh + scripts/_release_package.sh, combined into
# one script (a Mac has no Docker cross-build dance — it builds natively, and
# cross-builds the other arch with clang `-arch`). Runs ON A MAC.
#
#   scripts/release-macos.sh <arch>          arch: arm64 | x86_64
#   HEADLESS=1 scripts/release-macos.sh arm64    server-only (no GUI) build
#
# Produces the canonical layout:
#
#   release/rustllama-<version>-macos-<arch>/
#     rustllama        the relocatable Mach-O binary (all backends compiled in;
#                      SYCL + CUDA are inert stubs on macOS, CPU is real, and
#                      MLX/Metal is real on arm64 / an inert stub on x86_64)
#     lib/             bundled non-system dylibs, referenced via @rpath and an
#                      @loader_path/../lib rpath (the Mach-O analogue of the
#                      Linux $ORIGIN RPATH). On macOS this is just rustllama's
#                      own librsl_*.dylib backend shims (+ Apple's libmlx in
#                      Phase 1); the egui GL/windowing + rfd dialogs are macOS
#                      system frameworks, left on the host.
#     licenses/THIRD-PARTY-NOTICES.txt  README.md  LICENSE-MIT  LICENSE-APACHE  docs/
#
# then `release/rustllama-<version>-macos-<arch>.tar.gz`.
#
# Self-contained: unpack anywhere and run `./rustllama`. A GPU backend still
# needs its hardware at run time (Apple Silicon for MLX/Metal); absent that,
# rustllama falls back to the CPU path.
#
# Toolchain: the Xcode Command Line Tools (clang, ld, otool, install_name_tool,
# codesign, lipo) + a Rust toolchain. NO oneAPI, NO CUDA — neither exists on
# macOS, so those kernel crates build as inert stubs (see build-macos.sh).
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

ARCH="${1:?usage: release-macos.sh <arch>   (arch: arm64 | x86_64)}"
case "$ARCH" in
  arm64|aarch64) ARCH="arm64"; TRIPLE="aarch64-apple-darwin" ;;
  x86_64|intel)  ARCH="x86_64"; TRIPLE="x86_64-apple-darwin" ;;
  *) echo "ERROR: arch must be arm64 or x86_64 (got '$ARCH')" >&2; exit 2 ;;
esac

# 1. Build the binary for the requested arch via the macOS build wrapper. GUI
#    (desktop + CLI + server) by default; HEADLESS=1 builds server-only.
#    Passing --target makes the output path deterministic under the triple.
if [ "${HEADLESS:-0}" = "1" ]; then
  scripts/build-macos.sh --headless --target "$TRIPLE"
else
  scripts/build-macos.sh --target "$TRIPLE"
fi

# 2. Resolve version from the workspace manifest (same recipe as the Linux
#    packager).
VERSION="$(grep -m1 '^version = ' Cargo.toml | sed -E 's/.*"(.*)".*/\1/')"
[ -n "$VERSION" ] || { echo "ERROR: could not read version from Cargo.toml" >&2; exit 1; }

# 3. Locate the built binary (cargo nests under the triple when --target set).
BIN="${CARGO_TARGET_DIR:-target}/${TRIPLE}/release/rustllama"
[ -x "$BIN" ] || { echo "ERROR: binary not found/executable: $BIN" >&2; exit 1; }
BINDIR="$(cd "$(dirname "$BIN")" && pwd)"   # where our @rpath dylibs sit

NAME="rustllama-${VERSION}-macos-${ARCH}"
OUT="release"
STAGE="${OUT}/${NAME}"
rm -rf "$STAGE"
mkdir -p "$STAGE/lib"
echo ">> assembling ${NAME}"

cp "$BIN" "$STAGE/rustllama"
chmod u+w "$STAGE/rustllama"

# ---- Mach-O dylib bundling (the analogue of the Linux ldd + patchelf + -----
# ---- $ORIGIN machinery in _release_package.sh) -----------------------------
# Mach-O records each dependency by its *install name* (an absolute path, or an
# @rpath/@loader_path/@executable_path-relative one), not just a soname. We
# walk `otool -L`, bundle every NON-system dependency into ./lib, rewrite the
# references to @rpath/<name>, and add an @loader_path/../lib rpath to the
# binary so the loader finds ./lib beside it — no DYLD_LIBRARY_PATH, no wrapper.
#
# System dylibs/frameworks live under /usr/lib and /System/Library; they are
# macOS-provided and ABI-stable (the Mach-O analogue of is_base_os_lib), so we
# leave them on the host. That covers the entire egui/eframe GL + windowing +
# rfd graph (Cocoa/Metal/OpenGL/QuartzCore/AppKit are all system frameworks).
is_system_dylib() {
  case "$1" in
    /usr/lib/*|/System/Library/*) return 0 ;;
    *) return 1 ;;
  esac
}

# Recurse a Mach-O file's dependencies. Disk state (./lib) persists across the
# piped `while` subshell, so we test existence on disk (as the Linux bundler
# does) rather than in shell vars. `local` is essential here: the function
# recurses and then still references `$f`/`$dep`/`$base` (the `-change` in the
# absolute-path arm), so each invocation must keep its own copies — bash
# restores a caller's locals when a callee returns, preventing the child call
# from clobbering the parent's file/dep being rewritten.
bundle_macho() {
  local f="$1" self dep base src
  self="$(basename "$f")"
  # tail -n +2 drops the "<file>:" header line; awk takes the install-name path
  # (the "(compatibility version ...)" suffix is $2+).
  { otool -L "$f" 2>/dev/null || true; } | tail -n +2 | awk '{print $1}' | while read -r dep; do
    [ -n "$dep" ] || continue
    base="$(basename "$dep")"
    # Skip the file's own LC_ID_DYLIB self-reference (first dep line of a dylib).
    [ "$base" = "$self" ] && continue
    if is_system_dylib "$dep"; then continue; fi
    case "$dep" in
      @rpath/*|@loader_path/*|@executable_path/*)
        # rustllama's own inert backend shim(s): built with an @rpath install
        # name and copied BESIDE the binary by each kernel crate's build.rs.
        # Resolve from the binary's own dir; the reference is already @rpath so
        # nothing to rewrite — it just needs to land in ./lib + the rpath below.
        src="$BINDIR/$base"
        if [ ! -e "$src" ]; then
          echo ">> WARN: @rpath dependency $dep not found beside binary ($BINDIR); skipping" >&2
          continue
        fi
        if [ ! -e "$STAGE/lib/$base" ]; then
          cp -L "$src" "$STAGE/lib/"; chmod u+w "$STAGE/lib/$base" 2>/dev/null || true
          install_name_tool -id "@rpath/$base" "$STAGE/lib/$base" 2>/dev/null || true
          bundle_macho "$STAGE/lib/$base"
        fi
        ;;
      /*)
        # Absolute path to a non-system dylib (e.g. Apple's libmlx in Phase 1,
        # or a Homebrew lib). Bundle it, give it an @rpath id, and rewrite THIS
        # file's reference to the bundled copy.
        if is_system_dylib "$dep"; then continue; fi
        if [ ! -e "$STAGE/lib/$base" ]; then
          cp -L "$dep" "$STAGE/lib/"; chmod u+w "$STAGE/lib/$base" 2>/dev/null || true
          install_name_tool -id "@rpath/$base" "$STAGE/lib/$base" 2>/dev/null || true
          bundle_macho "$STAGE/lib/$base"
        fi
        install_name_tool -change "$dep" "@rpath/$base" "$f" 2>/dev/null || true
        ;;
    esac
  done
  return 0
}
bundle_macho "$STAGE/rustllama"

nlibs=$(find "$STAGE/lib" -type f 2>/dev/null | wc -l | tr -d ' ')
if [ "$nlibs" = "0" ]; then
  rmdir "$STAGE/lib" 2>/dev/null || true
  echo ">> no non-system dylibs to bundle"
else
  echo ">> bundled ${nlibs} dylib(s):"
  ( cd "$STAGE/lib" && ls -1 | sed 's/^/     /' )
  # Add an @loader_path/../lib rpath so @rpath/<name> resolves to ./lib. (The
  # build also left an @loader_path rpath on the binary — for running straight
  # out of target/release where the dylibs sit beside it; that entry is inert
  # here and harmless.)
  install_name_tool -add_rpath "@loader_path/../lib" "$STAGE/rustllama" 2>/dev/null || true
  echo ">> added @loader_path/../lib rpath (binary -> ./lib)"
fi

# ---- Ad-hoc code signing --------------------------------------------------
# Every install_name_tool edit INVALIDATES an existing code signature, and
# Apple Silicon refuses to load unsigned Mach-O. Re-sign ad-hoc (`-s -`): sign
# the leaf dylibs first, then the binary. A real Developer ID identity can
# replace `-` later (followed by notarization) for Gatekeeper-clean distribution.
if command -v codesign >/dev/null 2>&1; then
  if [ -d "$STAGE/lib" ]; then
    for so in "$STAGE"/lib/*.dylib; do
      [ -e "$so" ] || continue
      codesign --force -s - "$so" 2>/dev/null || echo ">> WARN: codesign failed for $so" >&2
    done
  fi
  if ! codesign --force --deep -s - "$STAGE/rustllama"; then
    echo ">> WARN: ad-hoc codesign of the binary failed — it may not run on Apple Silicon" >&2
  else
    echo ">> ad-hoc signed the binary + bundled dylibs (-s -)"
  fi
else
  echo ">> WARN: codesign not found — shipping UNSIGNED binary (will not run on Apple Silicon)" >&2
fi

# ---- Third-party bundled-library NOTICES ----------------------------------
# On macOS the dylibs in ./lib are rustllama's OWN inert backend shims
# (librsl_*.dylib — MIT OR Apache-2.0, same license as the binary); the egui
# GL/windowing stack + rfd file dialogs are macOS SYSTEM frameworks, provided
# by the OS and NOT bundled, so they need no notice. In Phase 1 Apple's MLX
# (libmlx, MIT) is bundled and is the one genuine third-party entry. Classify
# by basename: librsl_* = ours, anything else = third-party.
mkdir -p "$STAGE/licenses"
NOTICES="$STAGE/licenses/THIRD-PARTY-NOTICES.txt"
{
  echo "rustllama's own code is licensed MIT OR Apache-2.0 (see LICENSE-MIT /"
  echo "LICENSE-APACHE). This macOS build bundles the shared libraries in ./lib,"
  echo "dynamically linked via @rpath and replaceable."
  echo
  echo "The egui GUI's rendering, windowing and native file dialogs use macOS"
  echo "SYSTEM frameworks (Cocoa, Metal, OpenGL, QuartzCore, AppKit); those are"
  echo "provided by the operating system and are NOT bundled here."
  echo
  printf '%-32s %s\n' "BUNDLED DYLIB" "ORIGIN / LICENSE"
  printf '%-32s %s\n' "-------------" "----------------"
  third_party=0
  if [ -d "$STAGE/lib" ]; then
    for so in "$STAGE"/lib/*.dylib; do
      [ -e "$so" ] || continue
      b="$(basename "$so")"
      case "$b" in
        librsl_*) printf '%-32s %s\n' "$b" "rustllama (MIT OR Apache-2.0) — inert backend shim" ;;
        libmlx*)  printf '%-32s %s\n' "$b" "Apple MLX (MIT) — see upstream ml-explore/mlx"; third_party=1 ;;
        *)        printf '%-32s %s\n' "$b" "third-party — see upstream for license"; third_party=1 ;;
      esac
    done
  fi
  echo
  if [ "$third_party" = "0" ]; then
    echo "No third-party shared libraries are bundled: every ./lib dylib is"
    echo "rustllama's own inert backend shim."
  fi
} > "$NOTICES"
echo ">> wrote licenses/THIRD-PARTY-NOTICES.txt"

# License + docs (same set as the Linux packager).
for f in README.md LICENSE-MIT LICENSE-APACHE; do
  [ -f "$f" ] && cp "$f" "$STAGE/"
done
[ -d docs ] && cp -r docs "$STAGE/docs"

# ---- Archive --------------------------------------------------------------
# Per-arch tarball, matching the Linux per-arch approach. macOS `tar` is BSD
# tar; -czf writes a gzip tarball with the nested top-level dir = $NAME.
tar -czf "${OUT}/${NAME}.tar.gz" -C "$OUT" "$NAME"
echo ">> wrote ${OUT}/${NAME}.tar.gz"
ls -lh "${OUT}/${NAME}.tar.gz" 2>/dev/null || true

# ---- universal2 (optional) ------------------------------------------------
# We ship PER-ARCH tarballs (like Linux), which keeps each archive lean. If a
# SINGLE fat artifact is ever preferred, build BOTH arches and lipo the two
# binaries into one universal2 Mach-O, e.g.:
#
#   scripts/release-macos.sh arm64
#   scripts/release-macos.sh x86_64
#   lipo -create \
#     release/rustllama-<ver>-macos-arm64/rustllama \
#     release/rustllama-<ver>-macos-x86_64/rustllama \
#     -output rustllama-universal
#   # then re-bundle ./lib as universal too (lipo each dylib) + re-codesign.
#
# Left as a documented manual step; the default remains two per-arch tarballs.
