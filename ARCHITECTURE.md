> Historical scaffold document. Read [REVIEW_STATUS.md](REVIEW_STATUS.md) and Revision 4 for current corrections, evidence limits and rerun status. Original text follows.

# Architecture

## One tick, end to end

Every node has a GNN embedding (`R^F`), a LIF membrane potential, and a last-spike time. One tick of `orchestrator/pipeline.py::run_tick` does exactly this, in this order:

```
                     ┌─────────────────────────────────────────┐
                     │figure this tick's LIF input current      │
prev tick's  ───────▶│  gated_mean_aggregate(embedding, edges,   │
spike mask           │    prev_spiked)  -- only spiking sources  │
                     │    contribute a message  [SNN -> GNN]     │
                     │  + fixed per-node base_current            │
                     └───────────────────┬───────────────────────┘
                                          ▼
                     ┌─────────────────────────────────────────┐
                     │ step every neuron one dt (LifEngine)      │
                     │  python_lif_engine or wasm_bridge's       │
                     │  wasm_lif_engine -- proven interchangeable│
                     └───────────────────┬───────────────────────┘
                                          ▼ spiked (this tick)
                     ┌─────────────────────────────────────────┐
                     │ maybe_trigger_qgnn: for every node that   │
                     │  spiked THIS tick, call                   │
                     │  ry_node_update.qasm with (a,b,phi) =     │
                     │  that node's PRE-tick embedding[0:3],     │
                     │  fold <Z0>,<Z1> back into embedding[0:2]  │
                     │  [SNN -> QGNN -> GNN]                     │
                     └───────────────────┬───────────────────────┘
                                          ▼
                     ┌─────────────────────────────────────────┐
                     │ fuse_to_4d: project embedding -> R^3,     │
                     │  append elapsed-time-since-last-spike     │
                     │  -> (N, 4) shared coordinate               │
                     └─────────────────────────────────────────┘
```

Two details matter more than they look like they should, and both are directly, individually unit-tested:

- **The QGNN circuit reads the embedding as it was *entering* the tick, not as some earlier step in the same tick already changed it.** `run_tick` passes `state.embedding` (unmodified) into `maybe_trigger_qgnn`, before `state` is replaced. `test_pipeline_smoke.py`'s end-to-end test re-derives, from the closed form, exactly what each triggered node's post-tick embedding must be from its *pre*-tick embedding — the check that catches a wrong-*value* bug (wrong embedding fed in, result written to the wrong row). The independent review showed by mutation that it does **not** catch a wrong-*selection* bug (QGNN fired one tick late, or on the neighbouring node), because it takes the set of triggered nodes from the pipeline's own log; `tests/test_composition_wiring.py` closes that by re-deriving the trigger set, the SNN→GNN current, the LIF step, the clock and the fused output independently on every tick.
- **A node's outgoing message exists only on ticks it spiked**, not "recently" or "on average" — `gated_mean_aggregate` filters edges by the exact previous-tick spike mask, nothing fuzzier. This is what makes per-tick cost scale with firing activity rather than graph size (see SCALING.md).

## The four layers

### QGNN (`python/qmodule/`)

A from-scratch OpenQASM 2.0-subset parser (`statevector_sim.py::parse_qasm`) and statevector simulator (`run_statevector`, `sample_shots`, `expectation_z`) supporting `h, x, y, z, rx, ry, rz`, the general single-qubit unitary `u(theta,phi,lambda)`, and `cx`, with parameterized gate angles bound at call time. `qasm_module.py::QasmModule` wraps this in a two-phase interface deliberately shaped like a WASM runtime:

```python
module = QasmModule.load("circuits/ry_node_update.qasm")   # parse once (like instantiate)
result = module.call({"a": 0.7, "b": 1.3, "phi": 0.4}, shots=2000)  # cheap, repeatable (like a call)
```

**Where the WASM analogy holds:** the load/call split genuinely mirrors compile-once/invoke-many. **Where it deliberately breaks:** a WASM export call is deterministic — same inputs, same output, always. A quantum circuit is not: one execution samples one bitstring from a distribution. `call()` therefore returns a `QasmResult` carrying both the *exact* expectation values (`expectation_z`, computed directly from the statevector — what an ideal, infinite-shot execution would give you) and the *sampled* ones (`sampled_expectation_z`, derived from `shots`-many simulated measurements — what real hardware actually hands you). Every place in this codebase that folds a quantum result back into classical state (`fusion/shared_space.py::maybe_trigger_qgnn`) uses the exact value, which is a scaffold-stage simplification worth flagging explicitly: it lets the rest of the pipeline be checked against closed-form math without also absorbing shot noise into every downstream number. A production version would need to decide, deliberately, how much of that shot noise the classical layers above it are supposed to see — that decision has not been made here.

`ry_node_update.qasm` is the actual QGNN primitive: `ry(a) q[0]; ry(b) q[1]; cx q[0],q[1]; rz(phi) q[1];` — q0 encodes the node's own aggregated feature, q1 the incoming neighbor message, and `cx` is what makes this a *graph* primitive (entangling the two) rather than two independent qubits. Its closed form, `<Z0>=cos(a)`, `<Z1>=cos(a)*cos(b)`, is derived in `tests/test_qasm_sim.py` and re-used as an oracle three more times downstream (in `test_shared_space.py` and twice in `test_pipeline_smoke.py`) — the same proof gets re-checked in an increasingly realistic context each time, rather than trusted once and forgotten.

### SNN (`python/snn/lif_neuron.py`)

Standard leaky-integrate-and-fire, forward-Euler discretized:

```
V[t+1] = V[t] + (dt/tau) * ( -(V[t] - v_rest) + R*I[t] )
spike if V[t+1] >= v_th, then V[t+1] = v_reset
```

vectorized over all nodes at once (`step`). This is the layer that makes activity, not node count, the thing that sets per-tick cost: a node that never spikes never triggers a QGNN call and never contributes a message to `gated_mean_aggregate`. `analytic_rate` gives the continuous steady-state firing-rate closed form (used only as a cross-check — see VALIDATION.md); the *exact discrete* spike-timing formula the tests actually pin every spike against is independently re-derived in `tests/test_lif_neuron.py`'s own docstring from the Euler recurrence's closed-form solution.

### GNN (`python/gnn/message_passing.py`)

Sparse, edge-list-only message passing — `Graph` is a directed edge list (`edge_src`, `edge_dst`), never a dense `N x N` adjacency matrix, because a dense matrix is not just slow but *physically unallocatable* at the node counts this project targets (see SCALING.md). `mean_aggregate` gathers neighbor features and scatter-sums them with `np.add.at` (correct under repeated indices, unlike `out[index] += messages` — see the regression test in `test_gnn_message_passing.py` for exactly this gotcha). `gnn_layer_forward` is one GraphSAGE-style layer (`h' = activation(h @ W_self + mean_neighbor(h) @ W_neigh + b)`) with untrained, fixed weights (`GnnLayerParams`) — the deliverable is a verified-correct architecture, not a trained model. `gated_mean_aggregate` is the SNN-gated variant `run_tick` actually uses.

### Fusion (`python/fusion/shared_space.py`)

Two jobs live here:

1. **The shared 4D coordinate.** `project_to_3d` is a fixed Gaussian random projection (`R^F -> R^3`, scaled by `1/sqrt(3)` so `E[||Px||^2] = ||x||^2` — a Johnson-Lindenstrauss-style, defensible-not-arbitrary dimensionality reduction). `time_since_last_spike` is the node's real elapsed time since it last fired (`+inf`, honestly, if it never has). `fuse_to_4d` is nothing but their concatenation. **Read the module's own docstring for why this is described as "3 projected structural dimensions + 1 genuine temporal dimension," not "4D spacetime"** — the framing is deliberate and the distinction matters: three of the coordinates are a lossy, arbitrary-up-to-rotation embedding convenience, and calling the whole tuple a unified geometric space would overclaim what it is.
2. **The QGNN trigger.** `maybe_trigger_qgnn` is where the SNN's spike signal (not a separate heuristic) decides *when* `ry_node_update.qasm` fires, and where the GNN embedding's own components 0/1/2 become that circuit's `(a, b, phi)`. The returned `<Z0>, <Z1>` blend back into embedding components 0/1; component 2 (`phi`) is deliberately never overwritten, because the closed form already proves `rz(phi)` is invisible to any Z-basis measurement (`test_qasm_sim.py::test_rz_invisible_to_z_measurement`) — writing a "quantum-derived" value into it anyway would be fabricating a signal that provably cannot exist.

### Orchestrator (`python/orchestrator/`)

`pipeline.py` is the tick loop described above. `LifEngine` is a `Protocol`-shaped seam (`(V, I, dt, params) -> (V_next, spiked)`) with two interchangeable implementations: `python_lif_engine` (in-process, the default) and `wasm_bridge.py::wasm_lif_engine` (one call into `orchestrator_core.wasm` per tick, via a subprocess to `node`). Both are proven to produce bit-identical trajectories, including under the full graph-coupled feedback loop where each tick's input current depends on the *previous* tick's spikes (`tests/test_pipeline_smoke.py` Part 2) — not just for an externally-fixed constant current (Part 1). The subprocess-per-tick approach is a correctness-proving convenience, explicitly not a performance design; see SCALING.md M2.

### wasm-core (`wasm-core/orchestrator_core.c`)

The classical, always-on control loop, as a second, from-scratch implementation of the identical LIF update rule — no shared code path with `snn/lif_neuron.py`, on purpose, so agreement between them is evidence about the algorithm rather than about one copying the other's bug. Compiled with clang's built-in `wasm32-unknown-unknown` backend (`-nostdlib`, no emscripten): no libc is needed because the LIF update uses only `+ - * /`, no libm dependency to manage in a freestanding build. State lives in fixed-size static arrays (`MAX_NEURONS = 65536`), exposed to the host via pointer-returning exports (`orc_v_ptr`, `orc_i_ptr`, `orc_spiked_ptr`) that JS reads/writes directly through the module's linear memory — `run_tick_harness.mjs` is both the test harness and the reference example of that integration pattern. This is the literal realization of "treat QASM like WASM": a sandboxed, low-latency, always-on classical loop (this file), separate from the on-demand, sampled, higher-latency quantum call (`qmodule`) — mirroring the actual production split used by real quantum error-correction control stacks (WASM hosting real-time decoder/control logic co-located with the hardware; the quantum circuit itself invoked as a distinct, explicitly non-deterministic operation).

## Why "toy graph," precisely

Every test in this repository runs on graphs with somewhere between 2 and ~30 nodes. That is not a stand-in for "large graph, just smaller" — it is the actual scale at which every check here (closed-form comparison, naive-loop cross-check, permutation-invariance re-labeling, bit-exact engine comparison) is cheap enough to run exhaustively and print every individual result. Nothing about the algorithms is toy-scale-specific (the GNN layer already refuses dense adjacency for exactly this reason), but nothing here has been *run*, let alone validated, at the 10^8-10^10 node scale that motivated the project. See SCALING.md for what changes, staged, to get there.
