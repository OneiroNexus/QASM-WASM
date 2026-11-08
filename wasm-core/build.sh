#!/usr/bin/env bash
# Compiles orchestrator_core.c to a freestanding wasm32 module using
# clang's built-in wasm32-unknown-unknown backend -- no emscripten, no
# rustup/wasm-pack, no network access required. Verified working in this
# project's sandbox (clang 18, see README notes in ../AGENTS.md) after
# rustup's wasm32 target download was blocked by an organizational proxy
# policy; this toolchain has no such dependency.
set -euo pipefail
cd "$(dirname "$0")"

clang --target=wasm32-unknown-unknown -O2 -nostdlib \
  -Wl,--no-entry \
  -Wl,--export=orc_tick \
  -Wl,--export=orc_v_ptr \
  -Wl,--export=orc_i_ptr \
  -Wl,--export=orc_spiked_ptr \
  -Wl,--export=orc_max_neurons \
  -o orchestrator_core.wasm orchestrator_core.c

echo "built orchestrator_core.wasm ($(stat -c%s orchestrator_core.wasm 2>/dev/null || stat -f%z orchestrator_core.wasm) bytes)"
