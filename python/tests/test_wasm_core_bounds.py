"""
tests/test_wasm_core_bounds.py -- orchestrator_core.wasm at and beyond its
static capacity, calling orc_tick DIRECTLY through Node's WebAssembly API
(bypassing run_tick_harness.mjs's own guard) so that the C core's
behaviour is what is being tested, not the harness's.

Added by the independent review. VALIDATION.md recorded that nothing
exercised the engine "anywhere near its MAX_NEURONS = 65536 capacity";
the review's probe found that the original core, called with
n = MAX_NEURONS + 2, silently overwrote I[0] and I[1] (V[] and I[] are
adjacent in linear memory) and returned normally -- the sandbox stops a
module from touching the host, not from corrupting its own buffers. The
core now returns -1 and does nothing for n outside [0, MAX_NEURONS];
this file pins that, and also runs the engine at exactly MAX_NEURONS
against the Python reference.

Requires `node` and a built orchestrator_core.wasm (bash wasm-core/build.sh).
Run: python3 -m tests.test_wasm_core_bounds   (from the python/ directory)
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from snn.lif_neuron import LifParams, step

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WASM_BIN = PROJECT_ROOT / "wasm-core" / "orchestrator_core.wasm"
HARNESS = PROJECT_ROOT / "wasm-core" / "run_tick_harness.mjs"
FAILURES = []


def check(name: str, got: float, want: float, tol: float):
    ok = abs(got - want) <= tol
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: got={got!r} want={want!r} (tol={tol})")
    if not ok:
        FAILURES.append(name)


def check_true(name: str, cond: bool):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}")
    if not cond:
        FAILURES.append(name)


DIRECT_JS = r"""
import { readFileSync } from "node:fs";
const bytes = readFileSync(process.argv[2]);
const { instance } = await WebAssembly.instantiate(bytes, {});
const e = instance.exports;
const max = e.orc_max_neurons();
const iView = new Float64Array(e.memory.buffer, e.orc_i_ptr(), max);
const vView = new Float64Array(e.memory.buffer, e.orc_v_ptr(), max);
const sView = new Float64Array(e.memory.buffer, e.orc_spiked_ptr(), max);
iView.fill(0); vView.fill(0); sView.fill(0);
iView[0] = 123.456; iView[1] = 789.0; vView[max - 1] = 0.5; iView[max - 1] = 2.0;
const out = { max, memory_bytes: e.memory.buffer.byteLength };
out.rc_over_by_2 = e.orc_tick(max + 2, 0.5, 1.0, 0.0, 0.0, 1e9, 1.0);
out.I0 = iView[0]; out.I1 = iView[1]; out.V_last_after_refusal = vView[max - 1];
out.rc_negative = e.orc_tick(-1, 0.5, 1.0, 0.0, 0.0, 1e9, 1.0);
let trapped = false;
try { out.rc_huge = e.orc_tick(10_000_000, 0.5, 1.0, 0.0, 0.0, 1e9, 1.0); } catch (err) { trapped = true; }
out.trapped_huge = trapped;
out.rc_exact_max = e.orc_tick(max, 0.5, 1.0, 0.0, 0.0, 1e9, 1.0);   // dt/tau = 0.5: V[max-1] = 0.5 + 0.5*(-0.5 + 2.0) = 1.25
out.V_last_after_full_tick = vView[max - 1];
out.rc_zero = e.orc_tick(0, 0.5, 1.0, 0.0, 0.0, 1e9, 1.0);
// Return value == number of neurons that spiked this tick: 1000 neurons driven to V=1.0 (>= v_th=0.9).
vView.fill(0); iView.fill(0); sView.fill(0);
for (let k = 0; k < 1000; k++) iView[k] = 2.0;
out.rc_thousand = e.orc_tick(max, 0.5, 1.0, 0.0, 0.0, 0.9, 1.0);
let flagged = 0; for (let k = 0; k < max; k++) flagged += sView[k];
out.flagged_thousand = flagged;
process.stdout.write(JSON.stringify(out));
"""


def run_direct() -> dict:
    with tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False) as f:
        f.write(DIRECT_JS)
        path = f.name
    proc = subprocess.run(["node", path, str(WASM_BIN)], capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr)
    return json.loads(proc.stdout)


def test_core_refuses_out_of_range_n_without_touching_memory():
    print("\n-- orc_tick called directly with n outside [0, MAX_NEURONS] --")
    out = run_direct()
    check("orc_max_neurons()", float(out["max"]), 65536.0, tol=0.0)
    check("n = MAX+2 returns -1", float(out["rc_over_by_2"]), -1.0, tol=0.0)
    check("...and I[0] is untouched (was silently overwritten by the original core)", out["I0"], 123.456, tol=0.0)
    check("...and I[1] is untouched", out["I1"], 789.0, tol=0.0)
    check("...and V[MAX-1] is untouched (the refused call did no partial work)", out["V_last_after_refusal"], 0.5, tol=0.0)
    check("n = -1 returns -1", float(out["rc_negative"]), -1.0, tol=0.0)
    check_true("n = 10,000,000 is refused (returns -1) instead of trapping at the end of linear memory",
               (not out["trapped_huge"]) and out["rc_huge"] == -1)
    check("n = 0 is a legal no-op returning 0 spikes", float(out["rc_zero"]), 0.0, tol=0.0)
    check("n = MAX exactly: returns 0 spikes (threshold set at 1e9)", float(out["rc_exact_max"]), 0.0, tol=0.0)
    check("n = MAX exactly: the LAST neuron was updated (0.5 + 0.5*(-0.5 + 2.0) = 1.25)", out["V_last_after_full_tick"], 1.25, tol=0.0)
    # The mutation campaign showed nothing checked orc_tick's return value:
    # removing `spike_count++` survived.
    check("return value == number of neurons that spiked (1000 driven over threshold)", float(out["rc_thousand"]), 1000.0, tol=0.0)
    check("...and exactly those 1000 are flagged in SPIKED", float(out["flagged_thousand"]), 1000.0, tol=0.0)


def test_engine_reports_unavailable_without_the_binary():
    print("\n-- wasm_available() is False and wasm_lif_engine raises when the binary is missing (the `and` in wasm_available) --")
    from orchestrator import wasm_available, wasm_lif_engine
    hidden = WASM_BIN.with_suffix(".wasm.hidden")
    WASM_BIN.rename(hidden)
    try:
        check_true("wasm_available() is False without orchestrator_core.wasm", not wasm_available())
        try:
            wasm_lif_engine(np.zeros(2), np.zeros(2), 0.1, LifParams())
            raised = False
        except RuntimeError:
            raised = True
        check_true("wasm_lif_engine raises RuntimeError without the binary", raised)
    finally:
        hidden.rename(WASM_BIN)
    check_true("wasm_available() is True again once restored", wasm_available())


def test_full_capacity_matches_python_reference():
    print("\n-- 65,536 neurons x 20 ticks through the real harness vs snn.lif_neuron.step --")
    n, num_ticks, dt = 65536, 20, 0.02
    rng = np.random.default_rng(1)
    p = LifParams()
    V0 = rng.uniform(-0.5, 1.5, n)
    I = rng.uniform(0.0, 6.0, n)
    scn = dict(n=n, dt=dt, tau=p.tau, v_rest=p.v_rest, v_reset=p.v_reset, v_th=p.v_th, R=p.R,
               V_init=V0.tolist(), I=I.tolist(), num_ticks=num_ticks)
    proc = subprocess.run(["node", str(HARNESS)], input=json.dumps(scn), capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr)
    out = json.loads(proc.stdout)
    V = V0.copy()
    worst, flag_mismatch, spikes = 0.0, 0, 0
    for t in range(num_ticks):
        V, s = step(V, I, dt, p)
        worst = max(worst, float(np.max(np.abs(np.array(out["V_history"][t]) - V))))
        flag_mismatch += int(np.sum(np.array(out["spiked_history"][t]) != s))
        spikes += int(np.sum(s))
    print(f"  {spikes} spikes across {n * num_ticks:,} neuron-ticks")
    check_true("scenario is not vacuous", spikes > 0)
    check("max |V_wasm - V_python| at full capacity", worst, 0.0, tol=0.0)
    check("spike-flag mismatches at full capacity", float(flag_mismatch), 0.0, tol=0.0)


if __name__ == "__main__":
    if shutil.which("node") is None or not WASM_BIN.exists():
        print("node or orchestrator_core.wasm missing -- run `bash wasm-core/build.sh` first")
        sys.exit(1)
    test_core_refuses_out_of_range_n_without_touching_memory()
    test_full_capacity_matches_python_reference()
    test_engine_reports_unavailable_without_the_binary()
    print(f"\n{'ALL PASSED' if not FAILURES else f'{len(FAILURES)} FAILED: ' + ', '.join(FAILURES)}")
    sys.exit(1 if FAILURES else 0)
