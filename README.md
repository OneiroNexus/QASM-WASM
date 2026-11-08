# Treating QASM as WASM

An experimental reference implementation connecting graph aggregation, spiking control and a quantum-circuit interface, with tests that check their interactions.

**Aaron Michael Weis** · [ORCID](https://orcid.org/0009-0002-1906-148X)

**Version 0.2.0 / M2:** one persistent Node process now keeps a WebAssembly LIF instance and its memory alive across ticks. The original one-process-per-tick engine remains available. [Runtime contract and evidence](docs/M2_RUNTIME.md).

The QASM/WASM connection is a software-interface analogy: load an artifact, supply parameters, invoke an operation and inspect a defined result. This is **not a QASM-to-WASM compiler**. The quantum circuit runs in a classical Python statevector simulator.

## Try it

Use Python 3.12+, Node.js 24 and NumPy. The supplied WASM binary permits a demonstration without a compiler.

```bash
python -m pip install -r requirements.txt
cd python
python -m orchestrator.demo
```

The demo runs 16 nodes for 50 ticks, reports spikes and circuit triggers, then closes the worker. All weights are fixed or random; it does not train a model.

```python
from orchestrator import PersistentWasmLifEngine, run_pipeline

with PersistentWasmLifEngine() as engine:
    result = run_pipeline(lif_engine=engine, **pipeline_parameters)
```

Each tick still transfers full voltage/current arrays as JSON. Persistence removes repeated process/module initialization; it does not remove serialization or run the whole graph in WASM.

## Verify and benchmark

From the repository root:

```bash
python tools/verify.py --output out/verification
```

This runs the original ten modules plus M2 tests, requires Node and the binary, rejects skipped checks, and saves logs in a new directory. Historical experiment records remain untouched.

For a compiler rebuild, install clang and lld with the WASM target, then run `bash run_tests.sh`. The GitHub workflow also rebuilds before testing. Local November 8 evidence uses the supplied binary without rebuilding; see [review status](REVIEW_STATUS.md) and the workflow result for exact build status.

From the `python` directory:

```bash
python -m benchmarks.persistent_wasm
```

This compares Python, one-shot WASM and persistent WASM on matched changing-current trajectories. Startup and full per-call transport time are reported separately; numerical checks occur outside timing. Results are machine-specific, not full-graph throughput or energy measurements.

## Read the research

- [Revision 4 paper](docs/Treating-QASM-as-WASM-rev4.pdf) · [editable manuscript](docs/Treating-QASM-as-WASM-rev4.docx)
- [M2 software supplement](docs/M2_RUNTIME.md) · [review status](REVIEW_STATUS.md)
- [Architecture](ARCHITECTURE.md) · [scaling roadmap](SCALING.md)
- [Original mutation ledger](python/method/mutation_results.json) · [claim registry](python/method/claims.py)

Revision 4 describes the baseline at [commit 371578d](https://github.com/OneiroNexus/QASM-WASM/tree/371578d144d30aa412b30fb839cdd2daeea9fb88) and remains unchanged. The M2 supplement documents the later runtime change. Historical mutation ledgers do not validate newly added code.

The baseline rerun completed ten modules with 1,286 PASS entries, including property summaries over 2,350 seeded trials. A Python-only mutation rerun reproduced all 434 outcomes: 425 kills and nine survivors. Twelve C mutants were not rerun locally because the compiler was unavailable. Mutation kill counts are not accuracy or reliability scores.

## Evidence boundaries

The graph aggregate influences LIF current and spike timing. A triggered node sends its own pre-tick embedding into the two-qubit circuit. The pipeline consumes exact expectations; sampled output is separate. For the current Z readout the consumed values are `cos(a)` and `cos(a)*cos(b)`, giving a direct classical baseline.

Every neuron is stepped each tick. Weighted GraphSAGE is tested separately, not inside the composed tick. No training, real-QPU result, quantum advantage, billion-node deployment, energy savings or production latency is established.

Known frontend finite-value, seed-domain and sampler-boundary limitations remain in [REVIEW_STATUS.md](REVIEW_STATUS.md). M2 validates its own transport inputs; it does not repair the separate QASM parser. The server assumes the trusted bundled WASM kernel.

## Repository layout

- `python/qmodule`: supported OpenQASM subset and statevector simulator
- `python/snn`, `gnn`, `fusion`: numerical components
- `python/orchestrator`: composition, both WASM bridges and demo
- `wasm-core`: C kernel, supplied WASM and both Node hosts
- `python/tests`: standalone verification modules
- `python/benchmarks`: reproducible LIF host comparison
- `python/method`: claim registry and historical experiment records
- `docs`: paper, M2 supplement and sanitized evidence reports

## Attribution and reuse

Code: [MIT](LICENSE). Paper and its figures: [CC BY 4.0](docs/PAPER_LICENSE.md).

The manuscript discloses AI assistance separately from human authorship. No independent human peer review is claimed. `CITATION.cff` identifies the software; `.zenodo.json` supplies software metadata for GitHub archiving. A software DOI and any later paper DOI identify distinct records. A draft release is not a published archive.
