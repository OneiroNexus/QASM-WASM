# Hybrid GNN + SNN + QGNN Scaffold

A working, adversarially-tested scaffold combining three kinds of graph computation over one shared node set, plus the classical control core a real deployment would need to run them together in real time.

- **GNN** (`python/gnn/`) — classical structural message-passing over a sparse edge list.
- **SNN** (`python/snn/`) — leaky-integrate-and-fire spiking neurons that decide *when* a node's message actually moves, so activity (not graph size) sets the per-tick cost.
- **QGNN** (`python/qmodule/`) — a real OpenQASM 2.0 circuit, executed on a from-scratch statevector simulator, invoked once per node exactly on the tick it spikes; its quantum expectation values fold back into that node's classical embedding.
- **Fusion** (`python/fusion/`) — combines the above into one 4D per-node coordinate: 3 dimensions are a projection of the GNN embedding, the 4th is the node's real elapsed time since its last spike (see [ARCHITECTURE.md](ARCHITECTURE.md) for why this is described as 3+1, not "4D spacetime").
- **Orchestrator** (`python/orchestrator/`) — wires all of the above into one per-tick pipeline, with a pluggable LIF engine.
- **wasm-core** (`wasm-core/`) — the always-on classical control loop (the LIF integration), compiled to a real wasm32 module with no libc/emscripten/network dependency, proven bit-exact against the Python reference engine.

This scaffold was built to answer a specific framing: **treat a `.qasm` file the way you'd treat a `.wasm` module** — parse/validate once, call cheaply and repeatedly — and use that same instantiate/call shape, alongside a literal (not metaphorical) WASM classical control core, to combine a GNN, an SNN, and a QGNN into one system. See `python/qmodule/qasm_module.py`'s docstring and [ARCHITECTURE.md](ARCHITECTURE.md) for exactly where that analogy holds and where it deliberately breaks.

## If you are an AI coding agent picking this up

Read **[AGENTS.md](AGENTS.md)** first. It has build/test commands, the code-style and testing conventions actually followed here, and an explicit Boundaries section on what not to change unilaterally.

## Quick start

```bash
bash run_tests.sh
```

Builds `wasm-core/orchestrator_core.wasm` and runs all 10 test modules in dependency order. Requires `clang` (with its built-in wasm32-unknown-unknown target — no emscripten needed), `node`, and `python3` with `numpy`. Nothing else: see [AGENTS.md](AGENTS.md) for why that's deliberate.

Current state: **1286/1286 checks passing, 0 failures, across all 10 test modules**, and **437/446 mechanically generated mutants killed (98.0%)** with every survivor triaged — see [BUILD_INDEX.md](BUILD_INDEX.md) for the full claim-by-claim, mutant-by-mutant index and [VALIDATION.md](VALIDATION.md) for the history (1039 checks in the original build; +4 modules from the independent review; +generative tests, a mutation engine and a claim registry from the systematized method). See [VALIDATION.md](VALIDATION.md) for the breakdown and what each check actually proves.

## What's actually done vs. proposed

This is a correctness-proven *reference implementation at toy scale* (tens of nodes, one process, one machine) — not a system that has been run anywhere near the 140M–20B node scale that motivated it. Concretely, done and validated:

- A from-scratch OpenQASM 2.0-subset parser and statevector simulator (H/X/Y/Z/RX/RY/RZ/general-U/CX, parameterized circuits, shot sampling, exact *and* sampled expectation values), checked against 3 independently hand-derived closed forms.
- A vectorized LIF spiking layer, checked against an independently-derived exact closed-form discrete spike-timing formula.
- A sparse GNN message-passing layer (never a dense adjacency matrix — see [SCALING.md](SCALING.md) for why that matters), checked against a naive per-edge Python reference plus four exact invariants (constant-feature invariance, zero-degree exactness, permutation invariance, mass conservation).
- A fusion layer producing the 4D coordinate and gating QGNN triggers off real SNN spikes, checked against the QASM closed form reused as an oracle.
- `orchestrator_core.c`, compiled to real wasm32, proven **bit-exact** against the Python LIF reference — both in isolation and under the full graph-coupled feedback loop.
- An end-to-end orchestrator, validated by re-deriving over a hundred individual QGNN triggers from closed-form math directly out of a live multi-tick run on a toy graph.

Explicitly **not** done — see [SCALING.md](SCALING.md) for the staged, honest roadmap: nothing here is distributed or sharded; there is no real QPU/Aer backend behind `qmodule` (though the interface was designed as a seam for one); no weights anywhere are trained (`GnnLayerParams`, `ProjectionParams` are fixed/random by design); and the WASM engine currently used by the orchestrator spawns one subprocess per tick, which is correct but not how a real hot loop should work.

## Directory map

```
hybrid-gnn-snn-qgnn/
├── AGENTS.md              <- read this first if you are an AI coding agent
├── ARCHITECTURE.md        <- what connects to what, and why
├── SCALING.md             <- toy -> 140M-20B node roadmap, staged and honest about what's unsolved
├── VALIDATION.md          <- what has actually been tested, with real numbers
├── README.md              <- this file
├── run_tests.sh           <- builds wasm-core, runs all 10 test modules
├── run_method.sh          <- suite + mutation campaign + BUILD_INDEX.md regeneration (~3 min)
├── BUILD_INDEX.md         <- generated: modules, claims, tests, mutation scores, survivor verdicts
├── python/
│   ├── qmodule/            <- OpenQASM 2.0 parser + statevector sim (the QGNN layer)
│   │   ├── statevector_sim.py
│   │   ├── qasm_module.py    <- the "instantiate/call, WASM-shaped" interface
│   │   └── circuits/*.qasm
│   ├── snn/lif_neuron.py   <- leaky integrate-and-fire spiking layer
│   ├── gnn/message_passing.py <- sparse graph message-passing layer
│   ├── fusion/shared_space.py <- 3 projected dims + 1 real time dim -> 4D; SNN-gated QGNN trigger logic
│   ├── orchestrator/
│   │   ├── pipeline.py       <- per-tick wiring of all four layers
│   │   └── wasm_bridge.py    <- drop-in WASM-backed alternative to the Python LIF engine
│   └── tests/               <- one file per layer, plus the full-pipeline smoke test
└── wasm-core/
    ├── orchestrator_core.c   <- the always-on classical control loop, compiled to real wasm32
    ├── build.sh
    ├── orchestrator_core.wasm  (built artifact)
    └── run_tick_harness.mjs  <- Node execution harness / reference host-integration example
```
