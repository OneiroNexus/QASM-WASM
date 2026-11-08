"""Run a small graph-feedback demonstration with the persistent WASM LIF engine."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from fusion import make_projection
from gnn import Graph
from orchestrator import PersistentWasmLifEngine, run_pipeline
from qmodule import QasmModule
from snn.lif_neuron import LifParams


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=16)
    parser.add_argument("--ticks", type=int, default=50)
    args = parser.parse_args()
    if not 1 <= args.nodes <= 65536 or not 1 <= args.ticks <= 1000:
        parser.error("Use 1..65536 nodes and 1..1000 ticks; large simulated circuits may be slow.")
    rng = np.random.default_rng(20261108)
    src = np.arange(args.nodes, dtype=np.int64)
    graph = Graph(num_nodes=args.nodes, edge_src=src, edge_dst=(src + 1) % args.nodes)
    circuit = Path(__file__).resolve().parents[1] / "qmodule" / "circuits" / "ry_node_update.qasm"
    with PersistentWasmLifEngine() as engine:
        pid = engine.pid
        state, logs, _ = run_pipeline(
            graph=graph, initial_embedding=rng.uniform(-1.0, 1.0, (args.nodes, 3)),
            qasm_module=QasmModule.load(circuit), proj=make_projection(3, seed=5),
            base_current=rng.uniform(3.0, 9.0, args.nodes),
            num_ticks=args.ticks, dt=0.3, lif_params=LifParams(),
            coupling=0.8, lif_engine=engine, qgnn_shots=100, qgnn_seed=9,
        )
        summary = dict(
            nodes=args.nodes, ticks=state.tick,
            spikes=int(sum(np.sum(log.spiked) for log in logs)),
            circuit_triggers=int(sum(np.sum(log.qgnn_triggered) for log in logs)),
            same_worker_throughout=pid == engine.pid, capacity=engine.capacity,
            final_voltage_range=[float(np.min(state.V)), float(np.max(state.V))],
            note="Classical statevector simulation; no quantum hardware, training or quantum advantage.",
        )
    summary["worker_closed"] = engine.closed
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

