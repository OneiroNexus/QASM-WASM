"""
tests/test_properties.py -- generative (property-based) tests for every
layer, using method/props.py. Closes VALIDATION.md's oldest open gap: until
this file, every scenario in the suite was fixed or seeded-random, so only
the specific draws actually taken had been checked.

Each property below is the generalization of an existing closed-form or
cross-implementation check to RANDOM inputs -- random parameters, random
graphs, random circuits, random malformed input -- with the first failing
seed reported so a failure can be replayed exactly. Properties are chosen
so that a named injected defect would make them fail (AGENTS.md, Testing
Instructions, rule 5): the mutation campaign in method/mutate.py is what
checks that claim.

Run: python3 -m tests.test_properties   (from the python/ directory)
"""
from __future__ import annotations

import math
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from fusion import NEVER_SPIKED, fuse_to_4d, make_projection, maybe_trigger_qgnn, project_to_3d, time_since_last_spike
from gnn import Graph, GnnLayerParams, gated_mean_aggregate, gnn_layer_forward, in_degree, mean_aggregate, scatter_sum
from method import reference_sim as ref
from method.props import Gen, forall
from orchestrator import run_pipeline
from qmodule import QasmModule
from qmodule import statevector_sim as sv
from snn.lif_neuron import LifParams, analytic_rate, step

CIRCUITS = Path(__file__).resolve().parent.parent / "qmodule" / "circuits"
FAILURES = []


def report(r):
    status = "PASS" if r.ok else "FAIL"
    print(f"  [{status}] {r.summary}")
    if not r.ok:
        FAILURES.append(r.name)


# ------------------------------------------------------------------ qmodule

def test_random_circuits_match_reference():
    print("\n-- P1: random valid circuits: scaffold statevector == kron reference, exact <Z> == reference <Z> --")

    def prop(case):
        text, params = case
        circ = sv.parse_qasm(text)
        got = np.array(sv.run_statevector(circ, params))
        want = ref.run(circ, params)
        d = float(np.max(np.abs(got - want)))
        if d > 1e-12:
            return f"max amplitude diff {d:.2e}"
        for q in range(circ.n_qubits):
            if abs(sv.expectation_z(got, circ.n_qubits, q) - ref.zexp(want, circ.n_qubits, q)) > 1e-12:
                return f"<Z{q}> disagrees"
        if abs(float(np.sum(np.abs(got) ** 2)) - 1.0) > 1e-12:
            return "state not normalized"
        return None

    report(forall("P1 random circuits vs reference", Gen.circuit, prop, trials=300, seed=1))


def test_malformed_input_never_crashes_or_hangs():
    print("\n-- P2: random malformed lines: parse -> SyntaxError or success; run -> ValueError or success; never another exception, never slow --")

    def prop(text):
        t0 = time.time()
        try:
            circ = sv.parse_qasm(text)
        except SyntaxError:
            return None if time.time() - t0 < 1.0 else "parse took > 1 s"
        try:
            sv.run_statevector(circ, {})
        except ValueError:
            pass
        return None if time.time() - t0 < 1.0 else "evaluation took > 1 s"

    report(forall("P2 malformed input: SyntaxError/ValueError only, < 1 s", Gen.malformed_line, prop, trials=400, seed=2))


def test_sampled_expectation_tracks_exact():
    print("\n-- P3: random (a,b,phi), random shot counts: sampled <Z> within 6 sigma of exact; same seed -> same counts --")
    mod = QasmModule.load(CIRCUITS / "ry_node_update.qasm")

    def gen(rng):
        return (float(rng.uniform(-10, 10)), float(rng.uniform(-10, 10)), float(rng.uniform(-10, 10)),
                int(rng.choice([50, 200, 1000])), int(rng.integers(0, 10 ** 6)))

    def prop(case):
        a, b, phi, shots, seed = case
        r1 = mod.call({"a": a, "b": b, "phi": phi}, shots=shots, seed=seed)
        r2 = mod.call({"a": a, "b": b, "phi": phi}, shots=shots, seed=seed)
        if r1.counts != r2.counts:
            return "same seed gave different counts"
        if sum(r1.counts.values()) != shots:
            return "counts do not sum to shots"
        for q in (0, 1):
            ex = r1.expectation_z[q]
            sigma = math.sqrt(max(1.0 - ex * ex, 1e-12) / shots)
            if abs(r1.sampled_expectation_z[q] - ex) > 6 * sigma + 1e-12:
                return f"<Z{q}> sampled {r1.sampled_expectation_z[q]:.4f} vs exact {ex:.4f} beyond 6 sigma"
        return None

    report(forall("P3 sampled vs exact within 6 sigma; seed determinism", gen, prop, trials=200, seed=3))


# ---------------------------------------------------------------------- snn

def test_lif_random_params():
    print("\n-- P4: random LIF params and currents: vectorized step == scalar rule; closed forms hold for ANY params --")

    def gen(rng):
        p = Gen.lif_params(rng)
        n = int(rng.integers(1, 20))
        V = rng.uniform(p.v_reset - 1.0, p.v_th + 1.0, n)
        I = rng.uniform(-2.0, 8.0, n)
        dt = float(rng.uniform(0.01, p.tau * 0.5))
        return p, V, I, dt

    def prop(case):
        p, V, I, dt = case
        V_next, spiked = step(V, I, dt, p)
        for k in range(len(V)):
            raw = V[k] + (dt / p.tau) * (-(V[k] - p.v_rest) + p.R * I[k])
            want_spike = 1.0 if raw >= p.v_th else 0.0
            want_v = p.v_reset if want_spike else raw
            if spiked[k] != want_spike or V_next[k] != want_v:
                return f"neuron {k}: got ({V_next[k]}, {spiked[k]}) want ({want_v}, {want_spike})"
        # closed forms, generalized to random params
        I0 = float(I[0])
        v_inf = p.v_rest + p.R * I0
        a = 1.0 - dt / p.tau
        Vs = np.array([p.v_reset])
        if v_inf <= p.v_th:
            if analytic_rate(I0, p) != 0.0:
                return "analytic_rate nonzero below threshold"
            for _ in range(300):
                Vs, s = step(Vs, np.array([I0]), dt, p)
                if s[0] > 0:
                    return "sub-threshold neuron spiked"
        else:
            if analytic_rate(I0, p) <= 0.0:
                return "analytic_rate not positive above threshold"
            C = (v_inf - p.v_th) / (v_inf - p.v_reset)
            k_cf = math.ceil(math.log(C) / math.log(a) - 1e-9)
            # The continuous-time analytic ISI must be the dt -> 0 limit of the
            # discrete one: at dt = tau/1000 they agree to within about one
            # step plus the first-order Euler bias. (Added after the mutation
            # campaign showed analytic_rate's use of v_reset was untested:
            # every existing check used the default v_reset = 0.)
            dt_small = p.tau / 1000.0
            a_small = 1.0 - dt_small / p.tau
            isi_discrete = math.ceil(math.log(C) / math.log(a_small) - 1e-9) * dt_small
            isi_analytic = 1.0 / analytic_rate(I0, p)
            if abs(isi_discrete - isi_analytic) > 2.0 * dt_small * (1.0 + isi_analytic / p.tau):
                return f"analytic ISI {isi_analytic:.5f} vs discrete {isi_discrete:.5f} at dt=tau/1000"
            for k in range(1, 5000):
                Vs, s = step(Vs, np.array([I0]), dt, p)
                if s[0] > 0:
                    return None if k == k_cf else f"first spike at step {k}, closed form says {k_cf}"
            return "never spiked within 5000 steps although supra-threshold"
        return None

    report(forall("P4 LIF: step == scalar rule; sub/supra-threshold closed forms for random params", gen, prop, trials=250, seed=4))


# ---------------------------------------------------------------------- gnn

def _naive_mean(feat, src, dst, n, mask=None):
    F = feat.shape[1]
    sums = np.zeros((n, F))
    cnt = np.zeros(n)
    for s, d in zip(src.tolist(), dst.tolist()):
        if mask is None or mask[s] > 0:
            sums[d] += feat[s]
            cnt[d] += 1
    out = np.zeros((n, F))
    for i in range(n):
        if cnt[i] > 0:
            out[i] = sums[i] / cnt[i]
    return out


def test_aggregation_random_graphs():
    print("\n-- P5: random graphs (self-loops, duplicates, isolated nodes, E=0): aggregation == naive loop; mass conserved; degrees == Counter --")

    def gen(rng):
        n, src, dst = Gen.graph(rng)
        return n, src, dst, Gen.features(rng, n), Gen.spike_mask(rng, n)

    def prop(case):
        n, src, dst, feat, mask = case
        got = gated_mean_aggregate(feat, src, dst, n, mask)
        want = _naive_mean(feat, src, dst, n, mask)
        if not np.array_equal(got, want):
            return f"gated aggregate differs by {np.max(np.abs(got - want)):.2e}"
        if not np.array_equal(mean_aggregate(feat, src, dst, n), _naive_mean(feat, src, dst, n)):
            return "mean_aggregate differs from naive"
        msgs = feat[src] if len(src) else np.zeros((0, feat.shape[1]))
        sc = scatter_sum(msgs, dst, n)
        if np.max(np.abs(sc.sum(axis=0) - msgs.sum(axis=0))) > 1e-9 * (1 + np.max(np.abs(msgs), initial=0.0)):
            return "scatter_sum does not conserve mass"
        c = Counter(dst.tolist())
        if not np.array_equal(in_degree(dst, n), np.array([c.get(i, 0) for i in range(n)], dtype=np.float64)):
            return "in_degree differs from Counter"
        return None

    report(forall("P5 aggregation == naive loop on random graphs", gen, prop, trials=400, seed=5))


def test_layer_permutation_invariance_random():
    print("\n-- P6: gnn_layer_forward permutation-invariant on random graphs with random weights --")

    def gen(rng):
        n, src, dst = Gen.graph(rng)
        f_in, f_out = int(rng.integers(1, 6)), int(rng.integers(1, 5))
        feat = rng.normal(size=(n, f_in))
        params = GnnLayerParams(W_self=rng.normal(size=(f_in, f_out)), W_neigh=rng.normal(size=(f_in, f_out)),
                                b=rng.normal(size=(f_out,)))
        return n, src, dst, feat, params, rng.permutation(n)

    def prop(case):
        n, src, dst, feat, params, perm = case
        out1 = gnn_layer_forward(feat, src, dst, n, params)
        feat_p = np.zeros_like(feat)
        feat_p[perm] = feat
        out2 = gnn_layer_forward(feat_p, perm[src], perm[dst], n, params)
        d = float(np.max(np.abs(out2[perm] - out1))) if n else 0.0
        return None if d <= 1e-10 else f"relabeled output differs by {d:.2e}"

    report(forall("P6 permutation invariance on random graphs", gen, prop, trials=200, seed=6))


# ------------------------------------------------------------------- fusion

def test_fusion_random():
    print("\n-- P7: projection linearity, fuse_to_4d concatenation, elapsed time incl. +inf, for random inputs --")

    def gen(rng):
        n, f_in = int(rng.integers(1, 12)), int(rng.integers(3, 10))
        emb = rng.normal(size=(n, f_in)) * float(rng.choice([1e-2, 1.0, 1e2]))
        lst = rng.uniform(0.0, 5.0, n)
        lst[rng.random(n) < 0.3] = NEVER_SPIKED
        return n, f_in, emb, lst, float(rng.uniform(5.0, 9.0)), int(rng.integers(0, 10 ** 6)), float(rng.normal()), float(rng.normal())

    def prop(case):
        n, f_in, emb, lst, now, seed, a, b = case
        proj = make_projection(f_in, seed=seed)
        x, y = emb[:1], emb[-1:]
        if np.max(np.abs(project_to_3d(a * x + b * y, proj) - (a * project_to_3d(x, proj) + b * project_to_3d(y, proj)))) > 1e-9 * (1 + np.max(np.abs(emb))):
            return "projection not linear"
        fused = fuse_to_4d(emb, lst, now, proj)
        if fused.shape != (n, 4) or not np.array_equal(fused[:, :3], project_to_3d(emb, proj)):
            return "fused[:, :3] != projection"
        t = time_since_last_spike(lst, now)
        if not np.array_equal(fused[:, 3], t):
            return "fused[:, 3] != elapsed"
        never = np.isinf(lst)
        if not np.all(np.isinf(t[never])) or np.any(np.isinf(t[~never])) or np.any(t[~never] < 0):
            return "elapsed-time sentinel handling wrong"
        return None

    report(forall("P7 fusion identities on random inputs", gen, prop, trials=300, seed=7))


def test_trigger_random():
    print("\n-- P8: maybe_trigger_qgnn on random embeddings/masks/blends: trigger set, closed form, pass-through, interpolation --")
    mod = QasmModule.load(CIRCUITS / "ry_node_update.qasm")

    def gen(rng):
        n, f_in = int(rng.integers(1, 10)), int(rng.integers(3, 7))
        return rng.uniform(-10, 10, size=(n, f_in)), Gen.spike_mask(rng, n), float(rng.choice([0.0, 0.25, 0.5, 1.0])), int(rng.integers(0, 1000))

    def prop(case):
        emb, mask, blend, seed = case
        r = maybe_trigger_qgnn(emb, mask, mod, shots=20, seed=seed, blend=blend)
        if not np.array_equal(r.triggered_mask, mask > 0) or len(r.expectation_z) != int(mask.sum()):
            return "trigger set != mask"
        for i in range(len(emb)):
            if mask[i] > 0:
                a, b = emb[i, 0], emb[i, 1]
                want0 = (1 - blend) * a + blend * math.cos(a)
                want1 = (1 - blend) * b + blend * math.cos(a) * math.cos(b)
                if abs(r.updated_embedding[i, 0] - want0) > 1e-9 or abs(r.updated_embedding[i, 1] - want1) > 1e-9:
                    return f"node {i}: blended closed form mismatch"
                if not np.array_equal(r.updated_embedding[i, 2:], emb[i, 2:]):
                    return f"node {i}: components 2+ changed"
            elif not np.array_equal(r.updated_embedding[i], emb[i]):
                return f"node {i}: non-triggered row changed"
        return None

    report(forall("P8 trigger identities on random inputs", gen, prop, trials=200, seed=8))


# ------------------------------------------------------------- orchestrator

def test_pipeline_invariants_random():
    print("\n-- P9: random small pipelines: every per-tick wiring invariant holds (python engine) --")
    mod = QasmModule.load(CIRCUITS / "ry_node_update.qasm")

    def gen(rng):
        n = int(rng.integers(1, 9))
        e = int(rng.integers(0, 3 * n + 1))
        src = rng.integers(0, n, e).astype(np.int64)
        dst = rng.integers(0, n, e).astype(np.int64)
        f_in = int(rng.integers(3, 6))
        return (Graph(n, src, dst), rng.uniform(-2, 2, size=(n, f_in)), make_projection(f_in, seed=int(rng.integers(0, 100))),
                rng.uniform(0.0, 6.0, n), int(rng.integers(1, 16)), float(rng.uniform(0.02, 0.5)), Gen.lif_params(rng),
                float(rng.uniform(-1.0, 1.5)))

    def prop(case):
        graph, emb0, proj, base, ticks, dt, p, coupling = case
        _, logs, states = run_pipeline(graph, emb0, mod, proj, base, ticks, dt, lif_params=p, coupling=coupling, qgnn_shots=10)
        if len(logs) != ticks or len(states) != ticks + 1:
            return "wrong number of ticks"
        prev = np.zeros(graph.num_nodes)
        lst = np.full(graph.num_nodes, NEVER_SPIKED)
        for t, log in enumerate(logs):
            pre, post = states[t], states[t + 1]
            want_cur = base + coupling * _naive_mean(pre.embedding, graph.edge_src, graph.edge_dst, graph.num_nodes, prev).mean(axis=1)
            if np.max(np.abs(log.current - want_cur)) > 1e-12 * (1 + np.max(np.abs(want_cur))):
                return f"tick {t}: current"
            V_want, s_want = step(pre.V, log.current, dt, p)
            if not (np.array_equal(post.V, V_want) and np.array_equal(log.spiked, s_want)):
                return f"tick {t}: LIF step"
            if not np.array_equal(log.qgnn_triggered, log.spiked > 0):
                return f"tick {t}: trigger set"
            lst = np.where(log.spiked > 0, (t + 1) * dt, lst)
            if not np.array_equal(post.last_spike_time, lst):
                return f"tick {t}: clock"
            for i in range(graph.num_nodes):
                if log.spiked[i] > 0:
                    a, b = pre.embedding[i, 0], pre.embedding[i, 1]
                    if abs(post.embedding[i, 0] - math.cos(a)) > 1e-9 or abs(post.embedding[i, 1] - math.cos(a) * math.cos(b)) > 1e-9:
                        return f"tick {t} node {i}: closed form"
                elif not np.array_equal(post.embedding[i], pre.embedding[i]):
                    return f"tick {t} node {i}: pass-through"
            if not np.array_equal(log.fused_4d[:, :3], post.embedding @ proj.matrix) or np.any(np.isnan(log.fused_4d[:, :3])):
                return f"tick {t}: fused"
            prev = log.spiked
        return None

    report(forall("P9 pipeline wiring invariants on random configurations", gen, prop, trials=60, seed=9))


def test_shot_noise_advances_with_tick():
    print("\n-- P10: the same node triggered on consecutive ticks does not see identical shot noise (seed advances with the tick) --")
    mod = QasmModule.load(CIRCUITS / "ry_node_update.qasm")

    def gen(rng):
        n = int(rng.integers(1, 5))
        return (Graph(n, np.arange(n, dtype=np.int64), np.roll(np.arange(n, dtype=np.int64), -1)),
                rng.uniform(-1, 1, size=(n, 3)), make_projection(3, seed=int(rng.integers(0, 50))),
                np.full(n, 40.0), int(rng.integers(0, 1000)))

    def prop(case):
        graph, emb0, proj, base, seed = case
        # base current 40 with tau=10, dt=0.5: every node spikes every tick -> every node triggers every tick
        _, logs, _ = run_pipeline(graph, emb0, mod, proj, base, 6, 0.5, coupling=0.0, qgnn_shots=200, qgnn_seed=seed)
        for i in range(graph.num_nodes):
            samples = [tuple(sorted(log.sampled_expectation_z[i].items())) for log in logs if i in log.sampled_expectation_z]
            if len(samples) < 4:
                return f"node {i} triggered only {len(samples)} times"
            if len(set(samples)) == 1:
                return f"node {i}: identical sampled values on all {len(samples)} ticks"
        return None

    report(forall("P10 shot noise differs across ticks for the same node", gen, prop, trials=40, seed=10))


if __name__ == "__main__":
    test_random_circuits_match_reference()
    test_malformed_input_never_crashes_or_hangs()
    test_sampled_expectation_tracks_exact()
    test_lif_random_params()
    test_aggregation_random_graphs()
    test_layer_permutation_invariance_random()
    test_fusion_random()
    test_trigger_random()
    test_pipeline_invariants_random()
    test_shot_noise_advances_with_tick()
    print(f"\n{'ALL PASSED' if not FAILURES else f'{len(FAILURES)} FAILED: ' + ', '.join(FAILURES)}")
    sys.exit(1 if FAILURES else 0)
