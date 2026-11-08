"""
tests/test_composition_wiring.py -- independent per-tick re-derivation of
EVERY wiring point in orchestrator/pipeline.py::run_tick.

Why this file exists (added by the independent review of the thesis):
tests/test_pipeline_smoke.py Part 2 re-derives each QGNN trigger's
arithmetic from the closed form, but it takes the SET of triggered nodes
from the pipeline's own log (log.qgnn_triggered) and never checks the
SNN->GNN current, the LIF step inside the loop, the elapsed-time clock, or
the fused 4D output against anything independent. Mutation testing showed
that the following defects all pass that suite unchanged: QGNN triggered
one tick late (M2) or on the neighbouring node (M3), spike gating dropped
from the aggregation (M14), coupling sign flipped (M15), feature-sum
instead of feature-mean readout (M18), elapsed-time clock off by one tick
(M13, M17), fused output computed from the pre-update embedding (M1), and a
LIF threshold of '>' instead of the documented '>=' (M7).

This file closes those gaps the way the rest of the project does: every
quantity run_tick produces is re-derived here from the previous tick's
state by a separately written, deliberately naive path, and compared
exactly (tol=0.0 wherever the arithmetic is identical, 1e-12 where
summation order legitimately differs).

Run: python3 -m tests.test_composition_wiring   (from the python/ directory)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from fusion import NEVER_SPIKED, make_projection
from gnn import Graph
from orchestrator import python_lif_engine, run_pipeline, wasm_available, wasm_lif_engine
from orchestrator import wasm_bridge
from qmodule import QasmModule
from snn.lif_neuron import LifParams

PYTHON_DIR = Path(__file__).resolve().parent.parent
CIRCUITS = PYTHON_DIR / "qmodule" / "circuits"

FAILURES = []


def check(name: str, got, want, tol: float):
    got_a, want_a = np.asarray(got, dtype=np.float64), np.asarray(want, dtype=np.float64)
    if got_a.shape != want_a.shape:
        ok = False
    else:
        both_inf = np.isinf(got_a) & np.isinf(want_a) & (np.sign(got_a) == np.sign(want_a))
        with np.errstate(invalid="ignore"):
            ok = bool(np.all(both_inf | (np.abs(got_a - want_a) <= tol)))
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name} (tol={tol})" + ("" if ok else f" got={got_a} want={want_a}"))
    if not ok:
        FAILURES.append(name)


def check_true(name: str, cond: bool):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}")
    if not cond:
        FAILURES.append(name)


# ----------------------------------------------------------------------
# Independent, naive re-derivations (no gather/scatter, no vectorization,
# no shared helper with the code under test).
# ----------------------------------------------------------------------

def naive_gated_mean_current(embedding, edge_src, edge_dst, num_nodes, prev_spiked, base_current, coupling):
    """Step 1+2 of run_tick, by explicit per-edge loop: for each node, the
    mean over in-edges whose SOURCE spiked last tick of the source's
    embedding, reduced to a scalar by the mean over features, times
    coupling, plus the baseline."""
    F = embedding.shape[1]
    sums = [[0.0] * F for _ in range(num_nodes)]
    cnt = [0] * num_nodes
    for s, d in zip(edge_src.tolist(), edge_dst.tolist()):
        if prev_spiked[s] > 0:
            for f in range(F):
                sums[d][f] += float(embedding[s, f])
            cnt[d] += 1
    current = np.zeros(num_nodes, dtype=np.float64)
    for i in range(num_nodes):
        if cnt[i] > 0:
            mean_vec = [sums[i][f] / cnt[i] for f in range(F)]
            current[i] = base_current[i] + coupling * (sum(mean_vec) / F)
        else:
            current[i] = base_current[i]
    return current


def naive_lif(V, current, dt, p: LifParams):
    """Step 3, one neuron at a time, same operation order as the documented rule."""
    V_next = np.zeros_like(V)
    spiked = np.zeros_like(V)
    for k in range(len(V)):
        v = V[k] + (dt / p.tau) * (-(V[k] - p.v_rest) + p.R * current[k])
        if v >= p.v_th:
            spiked[k] = 1.0
            v = p.v_reset
        V_next[k] = v
    return V_next, spiked


def naive_last_spike_time(spike_log, t, dt, num_nodes):
    """Time of each node's most recent spike up to and including tick t, from the spike log alone."""
    out = np.full(num_nodes, NEVER_SPIKED, dtype=np.float64)
    for tt in range(t + 1):
        for k in range(num_nodes):
            if spike_log[tt][k] > 0:
                out[k] = (tt + 1) * dt
    return out


def test_every_wiring_point_rederived_per_tick():
    print("\n-- every run_tick output re-derived independently from the previous tick's state, every tick --")
    rng = np.random.default_rng(100)
    num_nodes, f_in = 14, 4
    ring_src = np.arange(num_nodes, dtype=np.int64)
    ring_dst = np.roll(ring_src, -1)
    extra_src = rng.integers(0, num_nodes, size=20).astype(np.int64)
    extra_dst = rng.integers(0, num_nodes, size=20).astype(np.int64)
    graph = Graph(num_nodes=num_nodes, edge_src=np.concatenate([ring_src, extra_src]),
                  edge_dst=np.concatenate([ring_dst, extra_dst]))
    embedding0 = rng.uniform(-1.0, 1.0, size=(num_nodes, f_in))
    proj = make_projection(f_in, seed=5)
    qasm_module = QasmModule.load(CIRCUITS / "ry_node_update.qasm")
    p = LifParams()
    dt, num_ticks, coupling = 0.05, 120, 0.8
    base_current = rng.uniform(2.5, 9.0, size=num_nodes)

    final_state, logs, states = run_pipeline(graph, embedding0, qasm_module, proj, base_current,
                                             num_ticks, dt, lif_params=p, coupling=coupling,
                                             qgnn_shots=50, qgnn_seed=3, qgnn_blend=1.0)

    spike_log = [log.spiked for log in logs]
    n_triggers = 0
    worst_current = worst_V = worst_fused_xyz = worst_fused_t = worst_lst = 0.0
    mask_mismatch = spike_mismatch = 0
    prev_spiked = np.zeros(num_nodes, dtype=np.float64)
    for t, log in enumerate(logs):
        pre, post = states[t], states[t + 1]
        # steps 1+2: SNN(prev tick) -> GNN -> this tick's current
        want_current = naive_gated_mean_current(pre.embedding, graph.edge_src, graph.edge_dst, num_nodes,
                                                prev_spiked, base_current, coupling)
        worst_current = max(worst_current, float(np.max(np.abs(log.current - want_current))))
        # step 3: LIF from (pre.V, log.current)
        want_V, want_spiked = naive_lif(pre.V, log.current, dt, p)
        worst_V = max(worst_V, float(np.max(np.abs(post.V - want_V))))
        spike_mismatch += int(np.sum(want_spiked != log.spiked))
        # step 4: the trigger set must be EXACTLY this tick's spike set
        mask_mismatch += int(np.sum(log.qgnn_triggered != (log.spiked > 0)))
        n_triggers += int(np.sum(log.qgnn_triggered))
        # clock: last_spike_time from the spike log alone
        want_lst = naive_last_spike_time(spike_log, t, dt, num_nodes)
        with np.errstate(invalid="ignore"):
            lst_diff = np.where(np.isinf(post.last_spike_time) & np.isinf(want_lst), 0.0, post.last_spike_time - want_lst)
        worst_lst = max(worst_lst, float(np.max(np.abs(lst_diff))))
        # step 5: fused = [post-update embedding @ P | (t+1)*dt - last_spike_time]
        want_xyz = np.array([[sum(float(post.embedding[i, f]) * float(proj.matrix[f, j]) for f in range(f_in))
                              for j in range(3)] for i in range(num_nodes)])
        worst_fused_xyz = max(worst_fused_xyz, float(np.max(np.abs(log.fused_4d[:, :3] - want_xyz))))
        want_t = (t + 1) * dt - want_lst
        fin = np.isfinite(want_t)
        worst_fused_t = max(worst_fused_t, float(np.max(np.abs(log.fused_4d[fin, 3] - want_t[fin]))) if np.any(fin) else 0.0)
        check_true(f"tick {t}: never-spiked nodes carry +inf elapsed time, spiked nodes finite",
                   bool(np.all(np.isinf(log.fused_4d[:, 3]) == ~fin)))
        prev_spiked = log.spiked

    total_spikes = int(sum(float(np.sum(s)) for s in spike_log))
    print(f"  {num_ticks} ticks, {total_spikes} spikes, {n_triggers} QGNN triggers re-checked for wiring")
    check_true("scenario is not vacuous (spikes and triggers occurred)", total_spikes > 0 and n_triggers > 0)
    check("step 1+2: log.current == baseline + coupling * feature-mean of spike-gated in-neighbour mean (naive loop), all ticks",
          worst_current, 0.0, tol=1e-12)
    check("step 3: post.V == independent scalar LIF from (pre.V, log.current), all ticks", worst_V, 0.0, tol=0.0)
    check("step 3: spike flags == independent scalar LIF, all ticks (mismatches)", float(spike_mismatch), 0.0, tol=0.0)
    check("step 4: qgnn_triggered == (spiked > 0) on EVERY tick (mismatched entries)", float(mask_mismatch), 0.0, tol=0.0)
    check("clock: last_spike_time == (tick+1)*dt of most recent spike, from the spike log alone", worst_lst, 0.0, tol=0.0)
    check("step 5: fused[:, :3] == post-update embedding @ P (naive triple loop)", worst_fused_xyz, 0.0, tol=1e-12)
    check("step 5: fused[:, 3] == (tick+1)*dt - last_spike_time (finite entries)", worst_fused_t, 0.0, tol=0.0)


def test_exact_threshold_hit_spikes_in_both_engines():
    print("\n-- V_next landing EXACTLY on v_th must spike (documented '>=' rule), in Python and in WASM --")
    # dt/tau = 0.5 exactly; V = 0, I = 2  ->  V_next = 0 + 0.5 * (0 + 2) = 1.0 == v_th, bit-exact.
    p = LifParams(tau=2.0, v_rest=0.0, v_reset=-0.25, v_th=1.0, R=1.0)
    V = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    I = np.array([2.0, 1.999999, 2.000001], dtype=np.float64)
    V_next, spiked = python_lif_engine(V, I, 1.0, p)
    check("python: exact hit spikes, just-below does not, just-above does", spiked, [1.0, 0.0, 1.0], tol=0.0)
    check("python: exact hit resets to v_reset", V_next[0], p.v_reset, tol=0.0)
    if wasm_available():
        V_w, s_w = wasm_lif_engine(V, I, 1.0, p)
        check("wasm: same three decisions at the exact boundary", s_w, [1.0, 0.0, 1.0], tol=0.0)
        check("wasm: bit-identical V_next to python at the boundary", V_w, V_next, tol=0.0)
    else:
        print("  [SKIP] wasm engine unavailable")


def test_wasm_engine_really_invokes_the_module():
    print("\n-- wasm_lif_engine must actually run the harness (one subprocess per call), not silently fall back --")
    if not wasm_available():
        print("  [SKIP] wasm engine unavailable")
        return
    calls = []
    real_run = wasm_bridge.subprocess.run

    def counting_run(*args, **kwargs):
        calls.append(args[0])
        return real_run(*args, **kwargs)

    wasm_bridge.subprocess.run = counting_run
    try:
        V = np.array([0.2, 0.9], dtype=np.float64)
        I = np.array([3.0, 0.0], dtype=np.float64)
        V1, s1 = wasm_lif_engine(V, I, 0.02, LifParams())
        V2, s2 = wasm_lif_engine(V1, I, 0.02, LifParams())
    finally:
        wasm_bridge.subprocess.run = real_run
    check_true("harness invoked exactly once per engine call (2 calls -> 2 subprocesses)", len(calls) == 2)
    check_true("each invocation ran node on run_tick_harness.mjs",
               all(len(c) >= 2 and str(c[1]).endswith("run_tick_harness.mjs") for c in calls))
    Vp, sp = python_lif_engine(V, I, 0.02, LifParams())
    check("and its result is bit-identical to the python engine", V1, Vp, tol=0.0)


if __name__ == "__main__":
    test_every_wiring_point_rederived_per_tick()
    test_exact_threshold_hit_spikes_in_both_engines()
    test_wasm_engine_really_invokes_the_module()
    print(f"\n{'ALL PASSED' if not FAILURES else f'{len(FAILURES)} FAILED: ' + ', '.join(FAILURES)}")
    sys.exit(1 if FAILURES else 0)
