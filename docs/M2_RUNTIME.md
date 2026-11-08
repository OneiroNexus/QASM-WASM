# M2: persistent WASM runtime

Software version **0.2.0** adds a persistent host around the existing C/WASM LIF kernel. The reviewed baseline is commit `371578d144d30aa412b30fb839cdd2daeea9fb88`. The Revision 4 paper describes that baseline; this document records the later software change.

## Runtime contract

`PersistentWasmLifEngine(timeout=10.0)` implements the same `(V, I, dt, LifParams) -> (V_next, spiked)` callable used by `run_pipeline`. One Node process instantiates the supplied WASM binary once and retains its linear memory and typed-array views until closed. The existing `wasm_lif_engine` still starts a fresh Node process per call and remains available for comparison.

Python supplies complete V and I arrays on **every call**. The server overwrites both active input ranges before stepping, so changing sizes or alternating independent trajectories cannot silently reuse stale values. Outputs are copied back to Python. This is persistent execution, **not zero-copy transport**: JSON encoding, pipe traffic, validation and array copies remain. Graph aggregation, circuit simulation and fusion remain in Python.

Use a context manager so the worker is reaped even when the caller raises:

```python
from orchestrator import PersistentWasmLifEngine, run_pipeline

with PersistentWasmLifEngine(timeout=10.0) as engine:
    final_state, logs, states = run_pipeline(
        # supply the ordinary graph, embedding, circuit and other parameters
        lif_engine=engine,
        **pipeline_parameters,
    )
```

For a complete runnable example:

```bash
cd python
python -m orchestrator.demo
```

The new backend accepts matching one-dimensional finite real V/I arrays, including empty arrays, up to the module-reported capacity (65,536 in the supplied binary). Parameters must be finite; tau must be positive and dt nonnegative. These are input-domain checks, not a guarantee that every accepted timestep is physically appropriate or numerically stable.

Each request has a sequential ID. Messages are UTF-8 JSON lines bounded at 8 MiB. Startup waits for a protocol/capacity handshake. One I/O worker handles pipe writes and replies; a second drains stderr into a bounded tail. Per-engine calls are serialized. The deadline covers handshake or pipe exchange, including blocked writes; it excludes local validation/serialization, waiting for another caller's lock, and bounded shutdown time.

Caller validation raises `ValueError` without poisoning an otherwise healthy engine. Invalid replies, EOF, process failure or non-finite output raise `RuntimeError`; an expired exchange raises `TimeoutError`. Terminal transport failures close the worker. There is no implicit restart or fallback. Create another instance explicitly to recover. `close()` is idempotent. Context management or explicit close is required; destructor cleanup is only best effort.

The code assumes the bundled trusted WASM module with fixed exported buffers. It is not a sandbox for arbitrary untrusted WASM modules or QASM programs.

## Verification and performance

The M2 claims are O12-O16 in `python/method/claims.py`, checked by `python/tests/test_persistent_wasm.py`. They cover independent scalar and cross-engine trajectories, graph-coupled feedback, capacity and changing sizes, worker reuse, input isolation and injected transport failures.

Run all modules without replacing the historical experiment ledgers:

```bash
python -m pip install -r requirements.txt
python tools/verify.py --output out/my-verification
```

The output directory must be new. Node.js must be on PATH and the WASM binary must exist. For a fresh compiler rebuild, install clang and lld, then run `bash run_tests.sh`. The GitHub workflow also rebuilds before running the suite.

Run the matched LIF-only host benchmark:

```bash
cd python
python -m benchmarks.persistent_wasm --output ../out/my-benchmark.json
```

The benchmark records Node startup separately, measures full host roundtrips, rotates engine order and checks trajectories against Python outside the measured calls. These are local measurements of the LIF transport path, not full-graph throughput, kernel-only timing, quantum advantage or energy savings. See the versioned JSON evidence files accompanying this document for actual measurements.

The C source and supplied WASM bytes are unchanged by M2. The historical mutation ledgers remain evidence about the original implementation. They do not cover the new bridge/server. M2 uses targeted fault injection and behavioral tests; no new exhaustive mutation score is claimed.

JSON transport preserves the tested finite numerical values but does not preserve all floating-point bit patterns (notably signed zero). Equality reported here is numerical equality, not universal bit identity.

## Remaining work

An embedded runtime or binary/shared-memory protocol could reduce serialization costs. Fully resident trajectory state with current-only updates would need a different state-ownership API and explicit reset/session semantics. These are future work. This milestone does not implement distributed graphs, training, a real QPU backend or a production deployment.


## Measured November 8, 2026 result

The final local verification ran all eleven modules: **1,536 PASS entries, zero failures, zero skips, zero nonzero exits and empty stderr**. This includes the unchanged original ten modules (1,286 entries) plus 250 M2 entries. These are assertion markers, not independent empirical studies. Python 3.12.14, NumPy 2.3.5, Node v24.22.1, Windows; supplied binary used without recompilation.

- [Full-suite summary](m2-verification-2026-11-08.json)
- [Benchmark data, boundaries and source fingerprints](m2-benchmark-2026-11-08.json)

Median milliseconds per measured LIF call:

| Neurons | Python | One-shot WASM | Persistent WASM | Persistent startup |
|---:|---:|---:|---:|---:|
| 25 | 0.0151 | 159.199 | 0.860 | 183.477 |
| 1,024 | 0.0197 | 160.975 | 4.040 | 191.354 |

The persistent host was about 185x and 40x faster than one-shot WASM for these two workloads, respectively, but about 57x and 205x slower than in-process Python/NumPy. Startup is excluded from those per-call ratios and shown separately. Each workload used three repetitions of 30 measured ticks after three warmup ticks, with rotating engine order. Every warmup and measured voltage/spike array matched the reference numerically.

This demonstrates removal of repeated process startup in this host design. It establishes neither a novel numerical algorithm nor a performance advantage over the best available classical implementation. No confidence intervals, energy measurements or quantum-hardware results are claimed.
