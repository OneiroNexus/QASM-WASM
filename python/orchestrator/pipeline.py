"""
orchestrator/pipeline.py -- wires qmodule (QGNN), snn (SNN), gnn (GNN), and
fusion (shared 4D space) into one end-to-end per-tick pipeline over a toy
graph. This is Task 6: proving the four previously isolated,
independently-validated layers actually compose correctly, not just that
each one is separately correct in a vacuum.

One tick, in this order:

  1. gated_mean_aggregate: every node that spiked on the PREVIOUS tick
     sends its current GNN embedding to its out-neighbors -- SNN -> GNN.
  2. That aggregated neighbor signal is reduced to a scalar per node (a
     fixed readout) and added to a fixed per-node baseline drive current,
     giving this tick's LIF input current -- GNN -> SNN.
  3. The LIF layer steps every neuron by one dt, through a pluggable
     engine (python_lif_engine, the default, or wasm_bridge.wasm_lif_engine
     -- see that module). Both are proven bit-exact for constant current in
     tests/test_pipeline_smoke.py Part 1, and re-checked here under the
     full graph-coupled feedback loop in Part 2.
  4. maybe_trigger_qgnn fires ry_node_update.qasm for every node that just
     spiked THIS tick, folding the resulting <Z0>, <Z1> back into that
     node's embedding -- SNN -> QGNN -> GNN.
  5. fuse_to_4d projects the (possibly quantum-updated) embeddings plus
     each node's elapsed-time-since-last-spike into the shared 4D
     coordinate.

Every wiring point above already has an isolated unit test elsewhere
(gnn/, snn/, fusion/, qmodule/ + their tests/*). This module and
tests/test_pipeline_smoke.py's Part 2 test only the COMPOSITION: e.g. that
a spike reaches maybe_trigger_qgnn on the exact tick it happened (not one
tick late or early), and that the embedding QGNN reads is the one that was
current entering that tick, not one a step already mutated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np

from fusion import NEVER_SPIKED, ProjectionParams, QgnnTriggerResult, fuse_to_4d, maybe_trigger_qgnn
from gnn import Graph, gated_mean_aggregate
from qmodule import QasmModule
from snn.lif_neuron import LifParams
from snn.lif_neuron import step as python_lif_step

LifEngine = Callable[[np.ndarray, np.ndarray, float, LifParams], Tuple[np.ndarray, np.ndarray]]


def python_lif_engine(V: np.ndarray, I: np.ndarray, dt: float, p: LifParams):
    """Default LifEngine: snn.lif_neuron.step(), in-process, no subprocess
    overhead -- the right choice for a toy/smoke-test orchestrator. See
    orchestrator/wasm_bridge.py for the wasm-core-backed alternative used
    only to prove engine-interchangeability, not as the default."""
    return python_lif_step(V, I, dt, p)


def readout_scalar(aggregated: np.ndarray) -> np.ndarray:
    """Reduces an (N, F) aggregated-neighbor-embedding array to (N,) scalar
    LIF input currents. Plain mean over feature dimensions -- deliberately
    the simplest possible reduction, so the pipeline's behavior is easy to
    reason about and to re-derive by hand in tests."""
    return aggregated.mean(axis=1)


@dataclass
class PipelineState:
    embedding: np.ndarray        # (N, F) current GNN node embedding
    V: np.ndarray                 # (N,) LIF membrane potential
    last_spike_time: np.ndarray   # (N,) seconds; NEVER_SPIKED sentinel
    tick: int = 0


@dataclass
class TickLog:
    spiked: np.ndarray             # (N,)
    current: np.ndarray            # (N,) this tick's LIF input current
    qgnn_triggered: np.ndarray     # (N,) bool
    fused_4d: np.ndarray           # (N, 4)
    sampled_expectation_z: dict    # node -> shot-sampled {0:.., 1:..} for triggered nodes (logged, never consumed)


def run_tick(state: PipelineState, graph: Graph, qasm_module: QasmModule,
             proj: ProjectionParams, base_current: np.ndarray, prev_spiked: np.ndarray,
             dt: float, lif_params: LifParams, coupling: float = 1.0,
             lif_engine: LifEngine = python_lif_engine, qgnn_shots: int = 500,
             qgnn_seed: int = 0, qgnn_blend: float = 1.0) -> Tuple[PipelineState, TickLog]:
    # 1 + 2: SNN(prev tick) -> GNN aggregation -> this tick's current
    neighbor_signal = gated_mean_aggregate(state.embedding, graph.edge_src, graph.edge_dst,
                                            graph.num_nodes, prev_spiked)
    current = base_current + coupling * readout_scalar(neighbor_signal)

    # 3: LIF step (pluggable engine)
    V_next, spiked = lif_engine(state.V, current, dt, lif_params)

    new_time = (state.tick + 1) * dt
    last_spike_time = np.where(spiked > 0, new_time, state.last_spike_time)

    # 4: SNN(this tick) -> QGNN -> GNN embedding update.
    # Reads state.embedding -- the value entering this tick, BEFORE any
    # update -- deliberately: the quantum node-update circuit encodes what
    # the node believed about itself and its neighbors going into the tick
    # it spiked, not a value some earlier step in this same tick already
    # changed.
    # The shot-sampling seed advances with the tick. maybe_trigger_qgnn seeds
    # each node's sampler as (seed + node index); passing a constant seed
    # from here would give every node bit-identical shot noise on every
    # tick -- invisible today (only the exact expectation_z field is
    # consumed) but a trap for any future consumer of sampled_expectation_z
    # (independent review finding F). 1_000_003 is a prime larger than any
    # toy-scale node count, so (tick, node) pairs never collide.
    qgnn_result: QgnnTriggerResult = maybe_trigger_qgnn(
        state.embedding, spiked, qasm_module, shots=qgnn_shots,
        seed=qgnn_seed + state.tick * 1_000_003, blend=qgnn_blend)

    # 5: -> shared 4D space
    fused = fuse_to_4d(qgnn_result.updated_embedding, last_spike_time, new_time, proj)

    next_state = PipelineState(embedding=qgnn_result.updated_embedding, V=V_next,
                                last_spike_time=last_spike_time, tick=state.tick + 1)
    log = TickLog(spiked=spiked, current=current, qgnn_triggered=qgnn_result.triggered_mask, fused_4d=fused,
                  sampled_expectation_z=qgnn_result.sampled_expectation_z)
    return next_state, log


def run_pipeline(graph: Graph, initial_embedding: np.ndarray, qasm_module: QasmModule,
                  proj: ProjectionParams, base_current: np.ndarray, num_ticks: int, dt: float,
                  lif_params: LifParams = LifParams(), coupling: float = 1.0,
                  lif_engine: LifEngine = python_lif_engine, qgnn_shots: int = 500,
                  qgnn_seed: int = 0, qgnn_blend: float = 1.0,
                  ) -> Tuple[PipelineState, List[TickLog], List[PipelineState]]:
    """Runs num_ticks of the full pipeline from a cold start (no prior
    spikes, LIF potentials at v_reset). Returns (final_state, logs, states)
    where states[t] is the state ENTERING tick t (states[0] is the cold
    start, states[num_ticks] == final_state) and logs[t] is that tick's
    result -- kept so tests can re-derive any tick's QGNN input/output
    directly from states[t]/states[t+1] rather than only from the summary
    in logs[t]."""
    n = graph.num_nodes
    state = PipelineState(
        embedding=initial_embedding.copy(),
        V=np.full(n, lif_params.v_reset, dtype=np.float64),
        last_spike_time=np.full(n, NEVER_SPIKED, dtype=np.float64),
        tick=0,
    )
    prev_spiked = np.zeros(n, dtype=np.float64)  # nothing has spiked before tick 0
    states = [state]
    logs: List[TickLog] = []
    for _ in range(num_ticks):
        state, log = run_tick(state, graph, qasm_module, proj, base_current, prev_spiked,
                               dt, lif_params, coupling, lif_engine, qgnn_shots, qgnn_seed, qgnn_blend)
        logs.append(log)
        states.append(state)
        prev_spiked = log.spiked
    return state, logs, states
