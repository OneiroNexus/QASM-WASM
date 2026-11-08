// M2 host: one WebAssembly instance and resident typed-array views per process.
// Each tick still supplies complete V/I vectors. Replacing BOTH active input
// ranges prevents retained memory from becoming an implicit numerical input.
// The original run_tick_harness.mjs remains the one-shot comparison baseline.
// Protocol/lifecycle and numerical checks: python/tests/test_persistent_wasm.py.
import { readFileSync } from "node:fs";
import { TextDecoder } from "node:util";

const MAX_MESSAGE_BYTES = 8 * 1024 * 1024;
const PROTOCOL = 1;
const decoder = new TextDecoder("utf-8", { fatal: true });

async function send(message) {
  const frame = Buffer.from(JSON.stringify(message) + "\n", "utf8");
  if (frame.length > MAX_MESSAGE_BYTES) throw new Error("response exceeds JSONL frame bound");
  await new Promise((resolve, reject) => {
    process.stdout.write(frame, (error) => error ? reject(error) : resolve());
  });
}

function finiteScalar(value, name) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${name} must be a finite real number`);
  }
}

function finiteVector(value, n, name) {
  if (!Array.isArray(value) || value.length !== n) {
    throw new Error(`${name} must be a length-${n} array`);
  }
  for (const element of value) finiteScalar(element, name);
}

async function main() {
  const bytes = readFileSync(new URL("./orchestrator_core.wasm", import.meta.url));
  const { instance } = await WebAssembly.instantiate(bytes, {});
  const core = instance.exports;
  const capacity = core.orc_max_neurons();
  if (!Number.isInteger(capacity) || capacity <= 0) throw new Error("invalid WASM capacity");
  // This core has static arrays and never grows memory. These views are created
  // once, cover its whole capacity, and stay resident for the process lifetime.
  const memory = core.memory.buffer;
  const voltage = new Float64Array(memory, core.orc_v_ptr(), capacity);
  const current = new Float64Array(memory, core.orc_i_ptr(), capacity);
  const spikes = new Float64Array(memory, core.orc_spiked_ptr(), capacity);
  let nextId = 1;
  await send({ type: "ready", protocol: PROTOCOL, capacity, max_message_bytes: MAX_MESSAGE_BYTES });

  async function tick(frame) {
    const request = JSON.parse(decoder.decode(frame));
    if (!request || Array.isArray(request) || typeof request !== "object"
        || request.type !== "tick" || !Number.isSafeInteger(request.id)
        || request.id !== nextId) {
      throw new Error("expected tick object with the next request ID");
    }
    const { n, dt, tau, v_rest, v_reset, v_th, R, V, I } = request;
    if (!Number.isInteger(n) || n < 0 || n > capacity) {
      throw new Error(`n must be an integer in [0, ${capacity}]`);
    }
    for (const name of ["dt", "tau", "v_rest", "v_reset", "v_th", "R"]) {
      finiteScalar(request[name], name);
    }
    if (dt < 0 || tau <= 0) throw new Error("dt must be non-negative and tau positive");
    finiteVector(V, n, "V");
    finiteVector(I, n, "I");
    if (core.memory.buffer !== memory) throw new Error("WASM memory unexpectedly changed");
    voltage.set(V, 0);
    current.set(I, 0);
    const count = core.orc_tick(n, dt, tau, v_rest, v_reset, v_th, R);
    if (!Number.isInteger(count) || count < 0 || count > n) throw new Error("orc_tick refused the request");
    const V_next = Array.from(voltage.subarray(0, n));
    const spiked = Array.from(spikes.subarray(0, n));
    finiteVector(V_next, n, "output V");
    let observedCount = 0;
    for (const value of spiked) {
      if (value !== 0 && value !== 1) throw new Error("invalid spike output");
      observedCount += value;
    }
    if (count !== observedCount) throw new Error("spike-count output mismatch");
    await send({ type: "result", id: request.id, V: V_next, spiked });
    nextId++;
  }

  // Parse frames incrementally. No unbounded readline buffer or pending-response
  // queue: await each tick/write before consuming further input chunks.
  const pending = Buffer.allocUnsafe(MAX_MESSAGE_BYTES);
  let length = 0;
  for await (const chunk of process.stdin) {
    let offset = 0;
    while (offset < chunk.length) {
      const newline = chunk.indexOf(10, offset);
      const end = newline < 0 ? chunk.length : newline;
      const part = chunk.subarray(offset, end);
      if (length + part.length + 1 > MAX_MESSAGE_BYTES) throw new Error("request exceeds JSONL frame bound");
      part.copy(pending, length);
      length += part.length;
      if (newline < 0) break;
      await tick(pending.subarray(0, length));
      length = 0;
      offset = newline + 1;
    }
  }
  if (length !== 0) throw new Error("incomplete JSONL request at EOF");
}

main().catch(async (error) => {
  const message = String(error.message || error).slice(0, 1000);
  try {
    await send({ type: "error", id: null, message });
  } catch {
    // The parent may already have closed its pipes after an expired deadline.
  }
  process.stderr.write(`persistent WASM server failed: ${message}\n`);
  process.stdin.destroy();
  process.exitCode = 1;
});
