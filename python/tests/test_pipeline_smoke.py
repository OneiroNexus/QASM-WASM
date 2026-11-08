"""
tests/test_pipeline_smoke.py

Part 1 (Task 4, this section): cross-validates wasm-core/orchestrator_core.c
-- compiled to a real wasm32 module and executed in Node's native
WebAssembly runtime -- against snn/lif_neuron.py's step(), an independent
Python/NumPy implementation of the identical LIF Euler update. Two
languages, two toolchains (clang's freestanding wasm32 backend vs. NumPy),
zero shared code path: agreement is evidence the underlying algorithm is
implemented correctly in both places, not that one copied the other's bug.
Same standard as every other cross-check in this project (naive vs.
memoized Berends-Giele, GPU vs. CPU domain-coloring texels, statevector
vs. closed-form quantum expectation values, naive vs. vectorized GNN
aggregation).

Part 2 (Task 6, below): the full QASM+SNN+GNN+fusion end-to-end pipeline
(orchestrator/) run on a toy graph -- proving the four independently-
validated layers actually COMPOSE (right tick, right node, right
pre/post-update embedding), not just that each is separately correct in
isolation. Also re-runs that same graph-coupled, feedback-driven scenario
through both LIF engines (python_lif_engine and wasm_bridge.wasm_lif_engine)
and checks they still agree exactly -- extending Part 1's constant-current
proof to the harder case where each tick's input current is itself
produced fresh by the previous tick's spikes and graph aggregation.

Requires `node` on PATH and wasm-core/orchestrator_core.wasm already built
(run wasm-core/build.sh once). Both are checked explicitly with a clear
error (Part 1) or a clearly-labeled skip (Part 2's engine comparison) rather
than an opaque crash if missing.

Run: python3 -m tests.test_pipeline_smoke   (from the python/ directory)
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from fusion import make_projection
from gnn import Graph
from orchestrator import python_lif_engine, run_pipeline, wasm_available, wasm_lif_engine
from qmodule import QasmModule
from snn.lif_neuron import LifParams, step

PYTHON_DIR = Path(__file__).resolve().parent.parent          # .../python
PROJECT_ROOT = PYTHON_DIR.parent                              # .../hybrid-gnn-snn-qgnn
WASM_CORE = PROJECT_ROOT / "wasm-core"
HARNESS = WASM_CORE / "run_tick_harness.mjs"
WASM_BIN = WASM_CORE / "orchestrator_core.wasm"
CIRCUITS = PYTHON_DIR / "qmodule" / "circuits"

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


def run_wasm_scenario(scenario: dict) -> dict:
    node = shutil.which("node")
    if node is None:
        raise RuntimeError("node not found on PATH -- required to execute orchestrator_core.wasm")
    if not WASM_BIN.exists():
        raise RuntimeError(f"{WASM_BIN} not built -- run `bash wasm-core/build.sh` first")
    proc = subprocess.run(
        [node, str(HARNESS)],
        input=json.dumps(scenario),
        capture_output=True, text=True, timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"node harness failed (exit {proc.returncode}):\n{proc.stderr}")
    return json.loads(proc.stdout)


def python_reference_trajectory(V_init, I, dt: float, num_ticks: int, p: LifParams):
    V = np.array(V_init, dtype=np.float64)
    I_arr = np.array(I, dtype=np.float64)
    V_history, spiked_history = [], []
    for _ in range(num_ticks):
        V, spiked = step(V, I_arr, dt, p)
        V_history.append(V.tolist())
        spiked_history.append(spiked.tolist())
    return V_history, spiked_history


def test_wasm_orchestrator_matches_python_lif_reference():
    print("\n-- orchestrator_core.wasm (C via Node's native WebAssembly runtime) vs. snn/lif_neuron.py step() --")
    rng = np.random.default_rng(42)
    p = LifParams()  # tau=10, v_rest=0, v_reset=0, v_th=1, R=1
    n = 25
    num_ticks = 300
    dt = 0.02
    # Deliberately mixed initial conditions -- below, at, and above
    # threshold -- so the very first tick's spike/no-spike decision is
    # exercised too, not just steady-state periodic firing.
    V_init = rng.uniform(-0.5, 1.5, size=n).tolist()
    I = rng.uniform(0.0, 6.0, size=n).tolist()

    scenario = dict(n=n, dt=dt, tau=p.tau, v_rest=p.v_rest, v_reset=p.v_reset,
                     v_th=p.v_th, R=p.R, V_init=V_init, I=I, num_ticks=num_ticks)

    wasm_out = run_wasm_scenario(scenario)
    py_V_hist, py_spiked_hist = python_reference_trajectory(V_init, I, dt, num_ticks, p)
    wasm_V_hist = wasm_out["V_history"]
    wasm_spiked_hist = wasm_out["spiked_history"]

    total = num_ticks * n
    matches_exact = 0
    max_v_err = 0.0
    mismatched_spike_flags = 0
    total_spikes_py = 0
    total_spikes_wasm = 0
    first_mismatch = None
    for t in range(num_ticks):
        for k in range(n):
            v_err = abs(wasm_V_hist[t][k] - py_V_hist[t][k])
            max_v_err = max(max_v_err, v_err)
            if v_err == 0.0:
                matches_exact += 1
            elif first_mismatch is None:
                first_mismatch = (t, k, wasm_V_hist[t][k], py_V_hist[t][k], v_err)
            if wasm_spiked_hist[t][k] != py_spiked_hist[t][k]:
                mismatched_spike_flags += 1
            total_spikes_py += int(py_spiked_hist[t][k] > 0)
            total_spikes_wasm += int(wasm_spiked_hist[t][k] > 0)

    print(f"  V agreement: {matches_exact}/{total} entries bit-exact (err == 0.0), max abs err = {max_v_err:.3e}")
    if first_mismatch is not None:
        t, k, w, py, e = first_mismatch
        print(f"  first mismatch: tick={t} neuron={k} wasm={w!r} python={py!r} err={e:.3e}")
    print(f"  spike-flag mismatches: {mismatched_spike_flags}/{total}")
    print(f"  total spikes -- python: {total_spikes_py}, wasm: {total_spikes_wasm}")
    check_true("at least one neuron actually spiked (test scenario isn't vacuously all-quiet)", total_spikes_py > 0)

    # Asserted at tol=0.0, not 1e-9: the review found the original test
    # OBSERVED bit-exactness (printed max err 0.0) but only ASSERTED 1e-9,
    # so the thesis's "bit-exact" claim was a printout, not a test. It is
    # now a test. If a legitimate change to one side's operation order ever
    # breaks this, say so in the test rather than widening it (AGENTS.md).
    check("fraction of V entries bit-exact (err == 0.0)", matches_exact / total, 1.0, tol=0.0)
    check("max |V_wasm - V_python| over all 7,500 values", max_v_err, 0.0, tol=0.0)
    check("spike-flag mismatches", float(mismatched_spike_flags), 0.0, tol=0.0)
    check("total spike count matches exactly", float(total_spikes_wasm), float(total_spikes_py), tol=0.0)


def test_wasm_tick_rejects_over_capacity_n():
    print("\n-- harness surfaces a clear error rather than silently truncating when n exceeds capacity --")
    ok = False
    try:
        run_wasm_scenario(dict(n=10_000_000, dt=0.01, tau=10.0, v_rest=0.0, v_reset=0.0,
                                v_th=1.0, R=1.0, V_init=[0.0], I=[0.0], num_ticks=1))
    except RuntimeError as e:
        ok = "exceeds" in str(e) or "length" in str(e)
    check_true("over-capacity n raised a clear RuntimeError", ok)


# ----------------------------------------------------------------------
# Part 2 (Task 6): full pipeline composition + engine-interchangeability
# under the graph-coupled feedback loop.
# ----------------------------------------------------------------------

def build_toy_graph(num_nodes: int, rng: np.random.Generator, extra_edges: int = 20) -> Graph:
    """Small deterministic-given-seed toy graph: a directed ring (so every
    node has >=1 in-edge and gated_mean_aggregate is never trivially
    starved) plus random chord edges for real structure."""
    ring_src = np.arange(num_nodes, dtype=np.int64)
    ring_dst = np.roll(ring_src, -1)
    extra_src = rng.integers(0, num_nodes, size=extra_edges).astype(np.int64)
    extra_dst = rng.integers(0, num_nodes, size=extra_edges).astype(np.int64)
    edge_src = np.concatenate([ring_src, extra_src])
    edge_dst = np.concatenate([ring_dst, extra_dst])
    return Graph(num_nodes=num_nodes, edge_src=edge_src, edge_dst=edge_dst)


def test_full_pipeline_composes_correctly_end_to_end():
    print("\n-- Part 2: full QASM+SNN+GNN+fusion pipeline on a toy graph -- composition, not just isolated correctness --")
    rng = np.random.default_rng(100)
    num_nodes, f_in = 14, 4
    graph = build_toy_graph(num_nodes, rng)
    embedding0 = rng.uniform(-1.0, 1.0, size=(num_nodes, f_in))
    proj = make_projection(f_in, seed=5)
    qasm_module = QasmModule.load(CIRCUITS / "ry_node_update.qasm")
    p = LifParams()
    dt = 0.05
    num_ticks = 300  # tau=10, so 300*0.05=15 time units -- comfortably several ISIs for the currents below
    # Comfortably supra-threshold (v_th=1.0) baseline drive guarantees
    # spontaneous spiking within the simulated window, independent of
    # whatever the (much smaller) neighbor-coupling term contributes.
    base_current = rng.uniform(2.5, 9.0, size=num_nodes)

    final_state, logs, states = run_pipeline(
        graph, embedding0, qasm_module, proj, base_current, num_ticks, dt,
        lif_params=p, coupling=0.8, qgnn_shots=300, qgnn_seed=3, qgnn_blend=1.0,
    )

    check_true("pipeline ran for the requested number of ticks", len(logs) == num_ticks and len(states) == num_ticks + 1)

    any_nan, any_spike, any_qgnn_trigger = False, False, False
    for t, log in enumerate(logs):
        # Column 3 of fused_4d (elapsed time) is legitimately +inf for a
        # node that has never yet spiked -- only the projected columns 0:3
        # must always be finite.
        if np.any(np.isnan(states[t + 1].embedding)) or np.any(np.isnan(log.fused_4d[:, :3])):
            any_nan = True
        check_true(f"tick {t}: fused_4d has correct shape", log.fused_4d.shape == (num_nodes, 4))
        any_spike = any_spike or bool(np.any(log.spiked > 0))
        any_qgnn_trigger = any_qgnn_trigger or bool(np.any(log.qgnn_triggered))

    check_true("no NaNs anywhere across the whole run", not any_nan)
    check_true("at least one spike occurred somewhere during the run", any_spike)
    check_true("at least one QGNN trigger occurred during the run", any_qgnn_trigger)

    # The strong check: re-derive, from scratch, every QGNN-triggered
    # node's post-tick embedding from the closed form already proven in
    # test_qasm_sim.py, using exactly the PRE-tick embedding
    # (states[t].embedding) as the circuit's (a, b, phi). This is what
    # would catch a wiring bug -- wrong tick's embedding fed in, one node's
    # spike triggering a different node's circuit, or a post-update
    # embedding leaking back in as if it were still pre-update.
    checked_triggers = 0
    for t, log in enumerate(logs):
        pre, post = states[t].embedding, states[t + 1].embedding
        for idx in np.nonzero(log.qgnn_triggered)[0]:
            a, b = pre[idx, 0], pre[idx, 1]
            want_z0, want_z1 = math.cos(a), math.cos(a) * math.cos(b)
            check(f"tick {t} node {idx}: post-trigger embedding[0] == cos(a) from PRE-tick (a,b,phi)",
                  post[idx, 0], want_z0, tol=1e-9)
            check(f"tick {t} node {idx}: post-trigger embedding[1] == cos(a)*cos(b) from PRE-tick (a,b,phi)",
                  post[idx, 1], want_z1, tol=1e-9)
            check(f"tick {t} node {idx}: embedding[2] (phi) unchanged by the trigger",
                  post[idx, 2], pre[idx, 2], tol=0.0)
            checked_triggers += 1
    print(f"  re-derived {checked_triggers} individual QGNN triggers from the closed form across the whole run")
    check_true("at least a handful of individual triggers were actually re-derived (test isn't vacuous)", checked_triggers >= 3)

    # Non-triggered nodes' embeddings must be byte-for-byte unchanged by
    # the QGNN step on ticks they did not spike on.
    untouched_checked, untouched_mismatches = 0, 0
    for t, log in enumerate(logs):
        pre, post = states[t].embedding, states[t + 1].embedding
        for idx in np.nonzero(~log.qgnn_triggered)[0]:
            untouched_checked += 1
            if not np.array_equal(pre[idx], post[idx]):
                untouched_mismatches += 1
    print(f"  checked {untouched_checked} non-triggered node-ticks for exact pass-through")
    check_true(f"all {untouched_checked} non-triggered node-ticks passed through with zero mismatches",
               untouched_checked > 0 and untouched_mismatches == 0)


def test_pipeline_wasm_engine_matches_python_engine_under_feedback():
    print("\n-- Part 2: wasm_lif_engine vs python_lif_engine through the SAME graph-coupled feedback loop (not just constant current) --")
    if not wasm_available():
        print("  [SKIP] node / orchestrator_core.wasm not available -- Part 1 already proved bit-exactness for")
        print("         constant current; skipping the (slower, per-tick-subprocess) feedback-loop cross-check here.")
        return

    rng = np.random.default_rng(200)
    num_nodes, f_in = 10, 3
    graph = build_toy_graph(num_nodes, rng, extra_edges=12)
    embedding0 = rng.uniform(-1.0, 1.0, size=(num_nodes, f_in))
    proj = make_projection(f_in, seed=6)
    qasm_module = QasmModule.load(CIRCUITS / "ry_node_update.qasm")
    p = LifParams()
    dt = 0.3  # larger step is fine here -- bit-exactness doesn't depend on dt, only ISI coverage does
    num_ticks = 25  # kept short: wasm_lif_engine spawns one `node` subprocess per tick; 25*0.3=7.5 time units
    base_current = rng.uniform(3.0, 9.0, size=num_nodes)

    common = dict(graph=graph, initial_embedding=embedding0, qasm_module=qasm_module, proj=proj,
                  base_current=base_current, num_ticks=num_ticks, dt=dt, lif_params=p,
                  coupling=0.8, qgnn_shots=200, qgnn_seed=9, qgnn_blend=1.0)

    _, py_logs, py_states = run_pipeline(lif_engine=python_lif_engine, **common)
    _, wasm_logs, wasm_states = run_pipeline(lif_engine=wasm_lif_engine, **common)

    max_v_err, max_embed_err, mismatched_spikes = 0.0, 0.0, 0
    for t in range(num_ticks):
        max_v_err = max(max_v_err, float(np.max(np.abs(py_states[t + 1].V - wasm_states[t + 1].V))))
        max_embed_err = max(max_embed_err, float(np.max(np.abs(py_states[t + 1].embedding - wasm_states[t + 1].embedding))))
        if not np.array_equal(py_logs[t].spiked, wasm_logs[t].spiked):
            mismatched_spikes += 1

    total_spikes = int(sum(float(np.sum(l.spiked)) for l in py_logs))
    print(f"  over {num_ticks} feedback-coupled ticks ({total_spikes} total spikes): "
          f"max|V_py-V_wasm|={max_v_err:.3e}  max|embed_py-embed_wasm|={max_embed_err:.3e}")
    check_true("at least one spike occurred (test scenario isn't vacuously all-quiet)", total_spikes > 0)
    check("max |V_python - V_wasm| across all feedback-coupled ticks", max_v_err, 0.0, tol=0.0)
    check_true("zero spike-mask mismatches across all feedback-coupled ticks", mismatched_spikes == 0)
    check("max |embedding_python - embedding_wasm| across all feedback-coupled ticks", max_embed_err, 0.0, tol=0.0)


if __name__ == "__main__":
    test_wasm_orchestrator_matches_python_lif_reference()
    test_wasm_tick_rejects_over_capacity_n()
    test_full_pipeline_composes_correctly_end_to_end()
    test_pipeline_wasm_engine_matches_python_engine_under_feedback()
    print(f"\n{'ALL PASSED' if not FAILURES else f'{len(FAILURES)} FAILED: ' + ', '.join(FAILURES)}")
    sys.exit(1 if FAILURES else 0)
