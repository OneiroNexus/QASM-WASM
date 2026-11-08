"""
gnn/message_passing.py -- toy graph message-passing layer, built for the
scale this project actually targets (up to ~10^10 nodes): every operation
here is a sparse edge-list gather/scatter, never a dense N x N adjacency
matrix, which would be physically impossible to allocate at that node count
(a dense float32 adjacency matrix for 2*10^10 nodes would need ~4*10^20
entries). This module is a correctness-first reference for the *algorithm*
a real distributed engine (see SCALING.md, M3) would shard across
machines -- not a claim that plain NumPy scatter-add itself runs at that
scale. Every function is validated in tests/test_gnn_message_passing.py
against an independent, deliberately naive per-edge Python loop, the same
"naive vs. optimized must agree" standard used for the Berends-Giele
recursion and the QASM statevector simulator elsewhere in this project.

This is also the layer that receives the SNN layer's spike mask: a node's
outgoing message exists only on timesteps it spikes (gated_mean_aggregate),
which is the actual mechanism -- not a metaphor -- that keeps a graph's
per-timestep work proportional to its *firing* fraction rather than its
total size. See qmodule/circuits/ry_node_update.qasm and
qmodule/qasm_module.py for the third layer this connects to: a node whose
aggregated neighbor message crosses the fusion layer's trigger condition
(Task in fusion/) gets its QGNN node-update circuit called, turning a
classical aggregated feature into a genuinely quantum-derived one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Graph:
    """Directed edge list. For an undirected graph, include both (u, v) and
    (v, u) explicitly -- keeping the representation directed-only avoids
    silently doubling (or halving) messages in code downstream that forgets
    which convention is in force."""
    num_nodes: int
    edge_src: np.ndarray   # (E,) int64
    edge_dst: np.ndarray   # (E,) int64

    def __post_init__(self):
        assert self.edge_src.shape == self.edge_dst.shape, "edge_src/edge_dst length mismatch"
        assert self.edge_src.dtype == np.int64 and self.edge_dst.dtype == np.int64, \
            "edge arrays must be int64 (scaffold-wide convention, matches wasm-core's i64 node ids)"
        if self.edge_src.size:
            assert self.edge_src.min() >= 0 and self.edge_src.max() < self.num_nodes
            assert self.edge_dst.min() >= 0 and self.edge_dst.max() < self.num_nodes

    @property
    def num_edges(self) -> int:
        return int(self.edge_src.shape[0])


def in_degree(edge_dst: np.ndarray, num_nodes: int) -> np.ndarray:
    """Count of incoming edges per node, via bincount -- the vectorized,
    C-level primitive that scatter_sum below is also built on."""
    return np.bincount(edge_dst, minlength=num_nodes).astype(np.float64)


def scatter_sum(messages: np.ndarray, index: np.ndarray, num_nodes: int) -> np.ndarray:
    """messages: (E, F), one row per edge. index: (E,) destination node id
    per edge. Returns (num_nodes, F): sum of messages landing on each node.

    Uses np.add.at, not `out[index] += messages`: plain fancy-index += is
    silently wrong under repeated indices (it drops all but one write to
    each repeated destination within a single call), which is exactly the
    bug a first, un-adversarially-tested attempt at this function makes.
    See tests/test_gnn_message_passing.py::test_scatter_sum_repeated_index_bug
    for a regression test that pins this down directly.
    """
    F = messages.shape[1]
    out = np.zeros((num_nodes, F), dtype=np.float64)
    np.add.at(out, index, messages)
    return out


def mean_aggregate(node_feat: np.ndarray, edge_src: np.ndarray, edge_dst: np.ndarray,
                    num_nodes: int) -> np.ndarray:
    """Classic GNN neighbor-mean aggregation: for each node, the average of
    its in-neighbors' features. Zero-in-degree nodes get an exact zero
    vector -- the divide-by-degree is guarded with a where() rather than an
    additive epsilon, so the zero-degree case is exact (0.0), not merely
    "very small."""
    messages = node_feat[edge_src]                        # (E, F) gather
    summed = scatter_sum(messages, edge_dst, num_nodes)    # (N, F) scatter
    deg = in_degree(edge_dst, num_nodes)                   # (N,)
    # One guarded divide: zero-degree rows are never divided and stay at the
    # exact 0.0 of `out`. (The original used a where()-substituted divisor
    # AND an overwrite; the mutation campaign showed the two guards were
    # redundant with each other, so three mutants of either were invisible.)
    return np.divide(summed, deg[:, None], out=np.zeros_like(summed), where=deg[:, None] > 0)


def gated_mean_aggregate(node_feat: np.ndarray, edge_src: np.ndarray, edge_dst: np.ndarray,
                          num_nodes: int, spiked: np.ndarray) -> np.ndarray:
    """Event-driven variant: identical to mean_aggregate, except an edge
    carries a message only on timesteps its source node spiked (`spiked` is
    the {0,1} array snn.lif_neuron.step() returns). This is the literal
    wiring point between the SNN and GNN layers -- silence here is not an
    approximation, it is the mechanism that keeps per-timestep work
    proportional to the number of firing nodes, not the size of the graph.

    Two exact boundary identities pin this down (checked in the test file):
    spiked == all-ones must reduce to plain mean_aggregate exactly, and
    spiked == all-zeros must return an exact all-zero array.
    """
    active_edges = spiked[edge_src] > 0
    if not np.any(active_edges):
        return np.zeros((num_nodes, node_feat.shape[1]), dtype=np.float64)
    return mean_aggregate(node_feat, edge_src[active_edges], edge_dst[active_edges], num_nodes)


@dataclass(frozen=True)
class GnnLayerParams:
    W_self: np.ndarray    # (F_in, F_out)
    W_neigh: np.ndarray   # (F_in, F_out)
    b: np.ndarray         # (F_out,)


def gnn_layer_forward(node_feat: np.ndarray, edge_src: np.ndarray, edge_dst: np.ndarray,
                       num_nodes: int, params: GnnLayerParams,
                       activation: Callable[[np.ndarray], np.ndarray] = np.tanh) -> np.ndarray:
    """One GraphSAGE-style message-passing layer:

        h' = activation( h @ W_self + mean_neighbor(h) @ W_neigh + b )

    Weights are untrained scaffold placeholders -- the deliverable here is
    a verified-correct *architecture* (in particular: a function of graph
    structure and features alone, provably independent of node-labeling --
    see test_permutation_invariance), not a trained model. Training is
    explicitly out of scope for the scaffold; see AGENTS.md Boundaries.
    """
    neigh = mean_aggregate(node_feat, edge_src, edge_dst, num_nodes)
    pre_activation = node_feat @ params.W_self + neigh @ params.W_neigh + params.b
    return activation(pre_activation)
