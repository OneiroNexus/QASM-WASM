// run_tick_harness.mjs -- Node-based execution harness for
// orchestrator_core.wasm, using Node's native WebAssembly API only (no
// npm dependency: this sandbox's package registry access is restricted
// enough that keeping this dependency-free was a deliberate choice, not
// an oversight -- see AGENTS.md).
//
// This is also the reference host-integration example for a real
// orchestrator: it shows the actual pattern (get exported pointers once,
// view them as typed arrays over the module's own linear memory, write
// inputs / call tick / read outputs directly through those views) that
// orchestrator/ (Task 6) will use for real, not just for testing.
//
// Usage: reads one JSON scenario object from stdin:
//   { n, dt, tau, v_rest, v_reset, v_th, R, V_init, I, num_ticks }
// V_init and I are length-n arrays; I is held constant across all ticks
// (matches snn/lif_neuron.py's simulate_constant_current scenario used
// for cross-validation in tests/test_pipeline_smoke.py).
//
// Prints one JSON object to stdout:
//   { V_history: [[...] x n] x num_ticks, spiked_history: [[...] x n] x num_ticks }
// each entry is the buffer's state immediately AFTER that tick's orc_tick() call.
import { readFileSync } from "node:fs";

async function main() {
  const scenario = JSON.parse(readFileSync(0, "utf8"));
  const { n, dt, tau, v_rest, v_reset, v_th, R, V_init, I, num_ticks } = scenario;

  const wasmPath = new URL("./orchestrator_core.wasm", import.meta.url);
  const bytes = readFileSync(wasmPath);
  const { instance } = await WebAssembly.instantiate(bytes, {});
  const exports = instance.exports;

  const maxN = exports.orc_max_neurons();
  if (n > maxN) {
    throw new Error(`scenario n=${n} exceeds orc_max_neurons()=${maxN}`);
  }
  if (V_init.length !== n || I.length !== n) {
    throw new Error(`V_init/I length must equal n=${n}`);
  }

  // orchestrator_core.c uses only static arrays (no malloc, no memory.grow),
  // so the buffer never moves/detaches -- these views stay valid for the
  // life of the instance.
  const vView = new Float64Array(exports.memory.buffer, exports.orc_v_ptr(), n);
  const iView = new Float64Array(exports.memory.buffer, exports.orc_i_ptr(), n);
  const spikedView = new Float64Array(exports.memory.buffer, exports.orc_spiked_ptr(), n);

  vView.set(V_init);
  iView.set(I);

  const V_history = [];
  const spiked_history = [];
  for (let t = 0; t < num_ticks; t++) {
    const rc = exports.orc_tick(n, dt, tau, v_rest, v_reset, v_th, R);
    if (rc < 0) {
      throw new Error(`orc_tick refused n=${n} (returned ${rc}); capacity is ${maxN}`);
    }
    V_history.push(Array.from(vView));
    spiked_history.push(Array.from(spikedView));
  }

  process.stdout.write(JSON.stringify({ V_history, spiked_history }));
}

main().catch((err) => {
  process.stderr.write(String(err.stack || err) + "\n");
  process.exit(1);
});
