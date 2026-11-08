"""
Adversarial validation of gnn/message_passing.py. Same standard as
tests/test_qasm_sim.py and tests/test_lif_neuron.py: every vectorized
sparse-graph operation is cross-checked against an independent, deliberately
naive per-edge Python loop, plus exact closed-form identities that a correct
implementation must satisfy (constant-feature invariance, zero-degree
exactness, permutation invariance, mass conservation) -- not just "runs
without crashing on one example graph."

Run: python3 -m tests.test_gnn_message_passing   (from the python/ directory)
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from gnn import (
    Graph,
    GnnLayerParams,
    gated_mean_aggregate,
    gnn_layer_forward,
    in_degree,
    mean_aggregate,
    scatter_sum,
)

FAILURES = []


def check(name: str, got, want, tol: float = 1e-9):
    got_a, want_a = np.asarray(got, dtype=np.float64), np.asarray(want, dtype=np.float64)
    ok = got_a.shape == want_a.shape and np.all(np.abs(got_a - want_a) <= tol)
    status = "PASS" if ok else "FAIL"
    detail = "" if ok else f" got={got_a} want={want_a}"
    print(f"  [{status}] {name} (tol={tol}){detail}")
    if not ok:
        FAILURES.append(name)


def check_true(name: str, cond: bool):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}")
    if not cond:
        FAILURES.append(name)


def random_graph(num_nodes: int, num_edges: int, rng: np.random.Generator,
                  allow_isolated: bool = True):
    if allow_isolated:
        src = rng.integers(0, num_nodes, size=num_edges).astype(np.int64)
        dst = rng.integers(0, num_nodes, size=num_edges).astype(np.int64)
    else:
        # Guarantee every node has >=1 in-edge: a directed cycle through all
        # nodes, plus random edges on top.
        cyc_src = np.arange(num_nodes, dtype=np.int64)
        cyc_dst = np.roll(cyc_src, -1)
        extra = max(0, num_edges - num_nodes)
        src = np.concatenate([cyc_src, rng.integers(0, num_nodes, size=extra).astype(np.int64)])
        dst = np.concatenate([cyc_dst, rng.integers(0, num_nodes, size=extra).astype(np.int64)])
    return src, dst


def naive_mean_aggregate(node_feat: np.ndarray, edge_src: np.ndarray, edge_dst: np.ndarray,
                          num_nodes: int) -> np.ndarray:
    """Independent reference: explicit adjacency-list build + per-node
    Python-level averaging loop. No gather/scatter/bincount vectorization
    shared with the production path -- if this and mean_aggregate agree,
    the vectorized implementation is not merely self-consistent, it agrees
    with the textbook definition of the operation."""
    F = node_feat.shape[1]
    sums = [np.zeros(F, dtype=np.float64) for _ in range(num_nodes)]
    counts = [0] * num_nodes
    for s, d in zip(edge_src.tolist(), edge_dst.tolist()):
        sums[d] = sums[d] + node_feat[s]
        counts[d] += 1
    out = np.zeros((num_nodes, F), dtype=np.float64)
    for i in range(num_nodes):
        if counts[i] > 0:
            out[i] = sums[i] / counts[i]
    return out


def test_graph_validates_inputs():
    print("\n-- Graph.__post_init__ rejects out-of-range / malformed edge lists --")
    ok = True
    try:
        Graph(num_nodes=3, edge_src=np.array([0, 5], dtype=np.int64), edge_dst=np.array([1, 1], dtype=np.int64))
        ok = False
    except AssertionError:
        pass
    check_true("out-of-range node id rejected", ok)
    ok = True
    try:
        Graph(num_nodes=3, edge_src=np.array([0], dtype=np.int32), edge_dst=np.array([1], dtype=np.int32))
        ok = False
    except AssertionError:
        pass
    check_true("non-int64 edge dtype rejected", ok)
    # Boundary and mixed cases -- added after the mutation campaign showed
    # that `and`->`or` and `<`->`<=` in the validator survived the original
    # (single-case) test.
    def rejects(src, dst, num_nodes=3):
        try:
            Graph(num_nodes=num_nodes, edge_src=np.asarray(src), edge_dst=np.asarray(dst))
            return False
        except AssertionError:
            return True
    i64 = lambda xs: np.array(xs, dtype=np.int64)  # noqa: E731
    check_true("src id == num_nodes rejected (boundary)", rejects(i64([3]), i64([0])))
    check_true("dst id == num_nodes rejected (boundary)", rejects(i64([0]), i64([3])))
    check_true("dst id out of range rejected (src fine)", rejects(i64([0]), i64([7])))
    check_true("negative src id rejected", rejects(i64([-1]), i64([0])))
    check_true("negative dst id rejected", rejects(i64([0]), i64([-1])))
    check_true("mixed dtypes (src int64, dst int32) rejected", rejects(i64([0]), np.array([1], dtype=np.int32)))
    check_true("mixed dtypes (src int32, dst int64) rejected", rejects(np.array([0], dtype=np.int32), i64([1])))
    check_true("length mismatch rejected", rejects(i64([0, 1]), i64([1])))
    g = Graph(num_nodes=3, edge_src=i64([0, 1, 2, 2]), edge_dst=i64([1, 2, 0, 0]))
    check("num_edges counts edges, duplicates included", float(g.num_edges), 4.0, tol=0.0)
    check_true("empty edge list accepted", not rejects(i64([]), i64([])))


def test_gnn_layer_forward_matches_naive_reference():
    print("\n-- gnn_layer_forward value matches a naive per-node implementation (not only invariances) --")
    # The mutation campaign showed that flipping the sign of W_neigh's term or
    # of the bias survived: permutation invariance and boundedness hold for
    # the wrong layer too. This pins the VALUE.
    rng = np.random.default_rng(21)
    num_nodes, num_edges, F_in, F_out = 12, 40, 5, 3
    src, dst = random_graph(num_nodes, num_edges, rng)
    node_feat = rng.normal(size=(num_nodes, F_in))
    params = GnnLayerParams(W_self=rng.normal(size=(F_in, F_out)), W_neigh=rng.normal(size=(F_in, F_out)),
                            b=rng.normal(size=(F_out,)))
    got = gnn_layer_forward(node_feat, src, dst, num_nodes, params)
    neigh = naive_mean_aggregate(node_feat, src, dst, num_nodes)
    want = np.zeros((num_nodes, F_out))
    for i in range(num_nodes):
        pre = np.zeros(F_out)
        for o in range(F_out):
            acc = params.b[o]
            for f in range(F_in):
                acc += node_feat[i, f] * params.W_self[f, o] + neigh[i, f] * params.W_neigh[f, o]
            pre[o] = acc
        want[i] = np.tanh(pre)
    check("layer output == naive per-node tanh(h W_self + mean(h) W_neigh + b)", got, want, tol=1e-10)


def test_scatter_sum_repeated_index_bug():
    print("\n-- scatter_sum correctly accumulates repeated destination indices --")
    # This is the exact case where `out[index] += messages` (fancy-index +=)
    # silently fails: three edges all targeting node 0.
    messages = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    index = np.array([0, 0, 0])
    got = scatter_sum(messages, index, num_nodes=2)
    check("sum of 3 repeated-index messages landing on node 0", got[0], [3.0, 3.0])
    check("node 1 (no incoming) is exact zero", got[1], [0.0, 0.0])

    naive_wrong = np.zeros((2, 2))
    naive_wrong[index] += messages  # the bug, computed explicitly for contrast
    check_true("regression guard: naive += really is wrong here (documents the bug scatter_sum avoids)",
               not np.allclose(naive_wrong[0], [3.0, 3.0]))


def test_scatter_sum_mass_conservation():
    print("\n-- scatter_sum conserves total mass: sum over nodes == sum over edges --")
    rng = np.random.default_rng(0)
    num_nodes, num_edges, F = 40, 300, 5
    _, dst = random_graph(num_nodes, num_edges, rng)
    messages = rng.normal(size=(num_edges, F))
    out = scatter_sum(messages, dst, num_nodes)
    check("total mass conserved across scatter", out.sum(axis=0), messages.sum(axis=0), tol=1e-8)


def test_in_degree_matches_naive_counter():
    print("\n-- in_degree (bincount) matches an independent Counter-based loop --")
    rng = np.random.default_rng(1)
    num_nodes, num_edges = 25, 90
    _, dst = random_graph(num_nodes, num_edges, rng)
    got = in_degree(dst, num_nodes)
    counts = Counter(dst.tolist())
    want = np.array([counts.get(i, 0) for i in range(num_nodes)], dtype=np.float64)
    check("in_degree vs Counter", got, want, tol=0.0)


def test_mean_aggregate_matches_naive_reference():
    print("\n-- mean_aggregate matches an independent naive per-edge Python loop --")
    rng = np.random.default_rng(2)
    num_nodes, num_edges, F = 35, 250, 6
    src, dst = random_graph(num_nodes, num_edges, rng)
    node_feat = rng.normal(size=(num_nodes, F))
    got = mean_aggregate(node_feat, src, dst, num_nodes)
    want = naive_mean_aggregate(node_feat, src, dst, num_nodes)
    check("mean_aggregate vs naive loop, full (N,F) array", got, want, tol=1e-10)


def test_mean_aggregate_zero_degree_is_exact_zero():
    print("\n-- isolated (zero in-degree) nodes get an exact zero vector, not NaN --")
    rng = np.random.default_rng(3)
    num_nodes, F = 10, 4
    # Only nodes 0..4 receive edges; 5..9 are isolated.
    src = rng.integers(0, 10, size=40).astype(np.int64)
    dst = rng.integers(0, 5, size=40).astype(np.int64)
    node_feat = rng.normal(size=(num_nodes, F))
    out = mean_aggregate(node_feat, src, dst, num_nodes)
    check_true("no NaNs anywhere in output", not np.any(np.isnan(out)))
    for i in range(5, 10):
        check(f"node {i} (isolated) is exact zero vector", out[i], np.zeros(F), tol=0.0)


def test_mean_aggregate_constant_feature_invariance():
    print("\n-- averaging a constant feature vector returns exactly that constant (closed-form identity) --")
    rng = np.random.default_rng(4)
    num_nodes, num_edges, F = 20, 150, 7
    src, dst = random_graph(num_nodes, num_edges, rng, allow_isolated=False)  # every node has >=1 in-edge
    c = rng.normal(size=F)
    node_feat = np.tile(c, (num_nodes, 1))
    out = mean_aggregate(node_feat, src, dst, num_nodes)
    for i in range(num_nodes):
        check(f"node {i}: mean of constant feature == constant", out[i], c, tol=1e-10)


def test_gated_aggregate_boundary_identities():
    print("\n-- gated_mean_aggregate reduces exactly to mean_aggregate (all spiking) / zero (none spiking) --")
    rng = np.random.default_rng(5)
    num_nodes, num_edges, F = 15, 80, 3
    src, dst = random_graph(num_nodes, num_edges, rng)
    node_feat = rng.normal(size=(num_nodes, F))

    all_spiked = np.ones(num_nodes)
    got_all = gated_mean_aggregate(node_feat, src, dst, num_nodes, all_spiked)
    want_all = mean_aggregate(node_feat, src, dst, num_nodes)
    check("all-nodes-spiking == plain mean_aggregate", got_all, want_all, tol=1e-10)

    none_spiked = np.zeros(num_nodes)
    got_none = gated_mean_aggregate(node_feat, src, dst, num_nodes, none_spiked)
    check("no-nodes-spiking == exact all-zero", got_none, np.zeros((num_nodes, F)), tol=0.0)

    # Partial spiking: gated result must equal mean_aggregate computed on the
    # edge-subset whose source spiked -- checked via an independent filter.
    rng2 = np.random.default_rng(6)
    spiked = (rng2.random(num_nodes) < 0.4).astype(np.float64)
    got_partial = gated_mean_aggregate(node_feat, src, dst, num_nodes, spiked)
    mask = spiked[src] > 0
    want_partial = mean_aggregate(node_feat, src[mask], dst[mask], num_nodes)
    check("partial spiking == mean_aggregate restricted to active-source edges", got_partial, want_partial, tol=1e-10)


def test_permutation_invariance():
    print("\n-- gnn_layer_forward is invariant to node relabeling (function of structure, not array position) --")
    rng = np.random.default_rng(7)
    num_nodes, num_edges, F_in, F_out = 30, 180, 5, 4
    src, dst = random_graph(num_nodes, num_edges, rng)
    node_feat = rng.normal(size=(num_nodes, F_in))
    params = GnnLayerParams(
        W_self=rng.normal(size=(F_in, F_out)) * 0.3,
        W_neigh=rng.normal(size=(F_in, F_out)) * 0.3,
        b=rng.normal(size=(F_out,)) * 0.1,
    )
    out1 = gnn_layer_forward(node_feat, src, dst, num_nodes, params)

    perm = rng.permutation(num_nodes)  # new_id = perm[old_id]
    node_feat_p = np.zeros_like(node_feat)
    node_feat_p[perm] = node_feat
    src_p, dst_p = perm[src], perm[dst]
    out2 = gnn_layer_forward(node_feat_p, src_p, dst_p, num_nodes, params)

    check("relabeled-then-unpermuted output matches original, node-for-node", out2[perm], out1, tol=1e-10)


def test_gnn_layer_forward_shape_and_no_nans():
    print("\n-- gnn_layer_forward: output shape correct, no NaNs, activation bound respected (tanh) --")
    rng = np.random.default_rng(8)
    num_nodes, num_edges, F_in, F_out = 50, 400, 8, 6
    src, dst = random_graph(num_nodes, num_edges, rng)
    node_feat = rng.normal(size=(num_nodes, F_in))
    params = GnnLayerParams(
        W_self=rng.normal(size=(F_in, F_out)),
        W_neigh=rng.normal(size=(F_in, F_out)),
        b=rng.normal(size=(F_out,)),
    )
    out = gnn_layer_forward(node_feat, src, dst, num_nodes, params)
    check_true("output shape == (num_nodes, F_out)", out.shape == (num_nodes, F_out))
    check_true("no NaNs", not np.any(np.isnan(out)))
    check_true("tanh activation bounded in [-1, 1]", bool(np.all(out >= -1.0) and np.all(out <= 1.0)))


if __name__ == "__main__":
    test_graph_validates_inputs()
    test_scatter_sum_repeated_index_bug()
    test_scatter_sum_mass_conservation()
    test_in_degree_matches_naive_counter()
    test_mean_aggregate_matches_naive_reference()
    test_mean_aggregate_zero_degree_is_exact_zero()
    test_mean_aggregate_constant_feature_invariance()
    test_gated_aggregate_boundary_identities()
    test_permutation_invariance()
    test_gnn_layer_forward_shape_and_no_nans()
    test_gnn_layer_forward_matches_naive_reference()
    print(f"\n{'ALL PASSED' if not FAILURES else f'{len(FAILURES)} FAILED: ' + ', '.join(FAILURES)}")
    sys.exit(1 if FAILURES else 0)
