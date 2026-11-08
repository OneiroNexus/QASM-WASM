"""
orchestrator/wasm_bridge.py -- adapts wasm-core/orchestrator_core.wasm (via
../../wasm-core/run_tick_harness.mjs) to exactly the same LifEngine
signature as snn.lif_neuron.step(), so pipeline.run_tick/run_pipeline can
use either engine interchangeably.

Used by tests/test_pipeline_smoke.py Part 2 to extend Part 1's proof
(WASM == Python for a fixed, externally-supplied constant current) to the
harder case: WASM == Python even when the input current is itself produced
fresh every tick by the graph-aggregation feedback loop, i.e. the two
engines are proven interchangeable as the actual hot-loop component of a
real orchestrator, not just as isolated neuron models.

This calls the Node harness once per tick (one subprocess spawn per call)
-- the right tradeoff for a correctness-proving smoke test, the wrong one
for a real hot loop. A production orchestrator would keep a single
long-lived WASM instance across all ticks (exactly like
run_tick_harness.mjs's own internal loop already does for Part 1's
single-call, many-ticks scenario) instead of reinstantiating per tick.
That performance path is noted in SCALING.md (M2) rather than built
here, since it would not change any of the numbers being validated.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from snn.lif_neuron import LifParams

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent   # .../hybrid-gnn-snn-qgnn
WASM_CORE = PROJECT_ROOT / "wasm-core"
HARNESS = WASM_CORE / "run_tick_harness.mjs"
WASM_BIN = WASM_CORE / "orchestrator_core.wasm"


def wasm_available() -> bool:
    """True only if both `node` is on PATH and the wasm binary has been
    built (run wasm-core/build.sh). Callers should skip (not fail) the
    WASM-engine comparison when this is False, with a clear printed
    reason -- an absent toolchain is an environment fact, not a bug."""
    return shutil.which("node") is not None and WASM_BIN.exists()


def wasm_lif_engine(V: np.ndarray, I: np.ndarray, dt: float, p: LifParams):
    """Same (V, I, dt, p) -> (V_next, spiked) contract as
    snn.lif_neuron.step(), backed by exactly one orc_tick() call inside
    orchestrator_core.wasm."""
    node = shutil.which("node")
    if node is None:
        raise RuntimeError("node not found on PATH -- required for wasm_lif_engine")
    if not WASM_BIN.exists():
        raise RuntimeError(f"{WASM_BIN} not built -- run `bash wasm-core/build.sh` first")
    n = len(V)
    scenario = dict(n=n, dt=dt, tau=p.tau, v_rest=p.v_rest, v_reset=p.v_reset,
                     v_th=p.v_th, R=p.R, V_init=np.asarray(V, dtype=np.float64).tolist(),
                     I=np.asarray(I, dtype=np.float64).tolist(), num_ticks=1)
    proc = subprocess.run([node, str(HARNESS)], input=json.dumps(scenario),
                           capture_output=True, text=True, timeout=30)
    if proc.returncode != 0:
        raise RuntimeError(f"node harness failed (exit {proc.returncode}):\n{proc.stderr}")
    out = json.loads(proc.stdout)
    V_next = np.array(out["V_history"][0], dtype=np.float64)
    spiked = np.array(out["spiked_history"][0], dtype=np.float64)
    return V_next, spiked
