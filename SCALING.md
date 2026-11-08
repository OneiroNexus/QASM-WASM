> Historical scaffold document. Read [REVIEW_STATUS.md](REVIEW_STATUS.md) and Revision 4 for current corrections, evidence limits and rerun status. Original text follows.

# Scaling: from this toy scaffold to 140M–20B nodes

This is a **proposed, staged roadmap**, not a scoped engineering plan — no timelines or cost estimates are given, because none would be credible without far more information about the target deployment (available hardware, latency requirements, which parts of the graph are dense vs. sparse in practice). The 140M–20B node figures come from the original request that motivated this project, not from a capacity analysis performed here. Treat this document as a starting point for a conversation with the user about what to build next, not a committed backlog.

Each milestone follows the convention already used elsewhere in this project's validation work: a milestone is "complete" only once it has its own closed-form or cross-implementation validation at its own scale — the same standard M0 is held to in VALIDATION.md, not a weaker one just because the numbers get bigger.

## M0 — this scaffold (done)

Single process, single machine, tens of nodes, every layer validated (see VALIDATION.md: 1039/1039 checks passing). Sparse edge-list representation already in place in `gnn/message_passing.py` (never dense adjacency) — this was a deliberate M0 decision specifically so M1 doesn't start with a rewrite.

## M1 — single-machine scale-up (millions of nodes)

**Goal:** the GNN/SNN/fusion layers running on one machine, but with millions of nodes and edges instead of tens, at a speed that isn't Python-loop-bound.

**What changes:**
- `mean_aggregate`'s `np.add.at`-based scatter-sum is correct but not fast at this scale (`np.add.at` does not vectorize as well as numpy's other ufuncs). The natural upgrade: for a *fixed* edge set, mean-aggregation is exactly a sparse matrix-vector product — precompute a row-normalized sparse adjacency matrix once (`scipy.sparse.csr_matrix`) and replace per-tick aggregation with `A @ node_feat`. This is the same trick real GNN frameworks (PyG, DGL) use internally.
- **Open question, not yet resolved:** `gated_mean_aggregate`'s edge set is *not* fixed — it's re-filtered every tick by which source nodes spiked, and (importantly) the averaging denominator (in-degree) is computed only over the *active* edges, not all edges. A fixed sparse matrix `A` doesn't capture a per-tick-changing denominator for free. Two honest options: (a) recompute a per-tick active-degree via its own fast sparse operation (`A_mask @ spiked`, itself cheap), or (b) redefine the aggregation semantics to average over *all* neighbors with non-firing ones contributing zero (changes what "neighbor average" means — dilutes it by silence — and would need its own closed-form re-derivation and re-validation, not a silent swap). Pick one deliberately; don't let performance work quietly change the math.
- Replace the Python-loop parts of `qmodule`'s parser/simulator with something that doesn't run once per node per tick in pure Python — batch many small circuit evaluations (see M4) or cache parsed circuits more aggressively (already partially done: `QasmModule.load()` parses once).

**Validation criterion:** the sparse-matrix aggregation reproduces `test_gnn_message_passing.py`'s existing naive-loop cross-check exactly at small scale, plus a new benchmark confirming it's actually faster at large scale (a correctness win that isn't also a speed win hasn't accomplished M1's goal).

## M2 — a real persistent classical control loop

**Goal:** realize the actual "QASM is to WASM" production shape this project is named for — a sandboxed, low-latency, *always-resident* classical loop — instead of M0's correctness-proving-but-slow one-subprocess-per-tick pattern (`orchestrator/wasm_bridge.py`'s own docstring already flags this explicitly).

**What changes:**
- Replace per-tick `node run_tick_harness.mjs` subprocess spawns with a long-lived process (or embedded runtime) that keeps `orchestrator_core.wasm`'s instance — and its linear memory, so neuron state stays resident — alive across many ticks, communicating over a fast channel (a persistent Node process with a line-delimited stdin/stdout protocol is the smallest change from what M0 already proved works; an embedded runtime like `wasmtime` accessed from Python would avoid the cross-process/cross-language hop entirely, but `pip install wasmtime` failed in the sandbox this scaffold was built in — worth retrying in whatever environment does M2, since that may well have been an environment-specific restriction, not a fundamental one).
- If the target deployment genuinely needs the hardware co-location Quantinuum's real decoder stack has (WASM control logic physically next to the quantum hardware), that's an infrastructure/deployment decision, not a code change — flag it as a question for the user rather than assuming it here.

**Validation criterion:** re-run `test_pipeline_smoke.py`'s Part 2 feedback-loop comparison (WASM engine vs. Python engine, bit-exact under live feedback) against the new persistent engine — it must still be bit-exact; only the plumbing around it should have changed.

## M3 — distributed graph, multiple machines

**Goal:** a graph too large for one machine's memory, sharded across many.

**What changes:**
- **Don't build a bespoke distributed graph framework from scratch.** Real ones exist (DistDGL, PyG's remote/distributed backends, AliGraph, and others) and solve the hard parts of this — cross-shard message delivery, fault tolerance, load balancing — that this project has no comparative advantage in re-deriving. The custom, actually-novel compute (the SNN gating rule, the QGNN trigger) should sit *on top of* an adopted distributed graph substrate, not reimplement its plumbing.
- **Partitioning strategy is a real open question, not a default.** Graph-topology-aware partitioning (METIS-style edge-cut minimization) keeps message-passing mostly local but is expensive to compute and needs recomputing as the graph changes. `fusion/shared_space.py`'s 4D coordinate is explicitly designed to double as a cheap, already-computed shard key (e.g. via a space-filling curve like Morton/Z-order over the 4D coordinate, a standard spatial-indexing technique) — but be honest about its limitation: two nodes close in that 4D coordinate are *not* guaranteed to be graph-neighbors (three of its four dimensions are a random projection, which preserves rough distances on average, not graph adjacency), so coordinate-based sharding trades partition quality for being nearly free to compute. Which tradeoff is right depends on the actual graph's structure and isn't decidable without real data — benchmark both before committing.

**Validation criterion:** the same naive-vs-optimized and permutation-invariance checks from `test_gnn_message_passing.py`, re-run against the sharded implementation on a graph split across (at least) 2 shards, confirming sharding changes performance and not results.

## M4 — a real quantum backend behind `qmodule`

**Goal:** `QasmModule` backed by something other than a from-scratch classical simulator — real hardware, or a simulator that scales further than this one.

**This is the milestone that most needs a skeptical reading, not an optimistic one.** At `10^8`–`10^10` nodes, even with the SNN layer keeping *active* (spiking) nodes to a small fraction of the total, "small fraction of 10^10" can still be millions of QGNN triggers in a single tick. Calling a real QPU once per triggered node per tick is not a scaling problem to engineer around — current (2026) QPU job submission has queueing and compilation overhead on the order of tens of milliseconds to seconds per job, many orders of magnitude too slow for a real-time per-node-per-tick trigger model at that volume. `qasm_module.py`'s docstring already flags that a WASM call is deterministic and a quantum call isn't; this is the same kind of thing — the analogy's limits, made concrete at scale, not a detail to paper over.

**What a realistic version of this milestone looks like:**
- **Batch, don't call-per-node.** Pack many nodes' independent small circuits into one wider job (disjoint qubit ranges of one larger circuit, or a vendor's native parametric/vectorized-circuit batching API where available). This changes `QasmModule.call()`'s signature non-trivially — from one `(a,b,phi) -> result` call per node to one `[(a,b,phi), ...] -> [result, ...]` call per *batch* of triggered nodes per tick — and is real design work, not a drop-in swap.
- **Consider a scalable classical simulator before real hardware.** If the goal is "quantum-inspired computation at scale" rather than literally running on a QPU, a GPU-accelerated simulator (e.g. NVIDIA cuQuantum, Google's qsim) sidesteps the QPU queueing problem entirely and may be the more realistic near-term path, at the cost of no longer being "real" quantum execution — an explicit tradeoff to put in front of the user, not a decision to make silently.

**Validation criterion:** the exact `<Z0>=cos(a)`, `<Z1>=cos(a)*cos(b)` closed-form check from `test_qasm_sim.py`, re-run against whatever backend replaces the from-scratch simulator, at whatever batch size M4 settles on. If the backend is genuinely a QPU, this check now has real hardware noise in it — that's the point of re-running it, not a reason to skip it.

## M5 — the actual smoke test at target scale

Only meaningful once M1–M4 (or whichever subset the user actually wants) exist. A concrete, falsifiable success criterion, in the same spirit as `test_pipeline_smoke.py`'s current one: one full tick over `N` nodes distributed across `K` machines completes within a stated time budget, *and* the same closed-form/cross-implementation validation methodology used throughout this project is re-run — not skipped — against a representative sample of that run, at that scale. "It should work" is a hypothesis; this milestone is what turns it into a result.
