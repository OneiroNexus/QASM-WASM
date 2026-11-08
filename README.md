# Treating QASM as WASM

A small research scaffold for asking whether graph aggregation, spiking control and a quantum-circuit interface behave correctly when connected.

**Aaron Michael Weis** · [ORCID](https://orcid.org/0009-0002-1906-148X)

The QASM/WASM connection is a software-interface analogy: load an artifact, supply parameters, invoke an operation and inspect a defined result. This repository includes a Python statevector simulator and an actual C-to-WASM LIF kernel. It is not a QASM-to-WASM compiler.

## Start here

- [Revision 4 paper](docs/Treating-QASM-as-WASM-rev4.pdf) · [editable manuscript](docs/Treating-QASM-as-WASM-rev4.docx)
- [Current review and reproducibility status](REVIEW_STATUS.md)
- [Architecture](ARCHITECTURE.md), [tests and historical validation](VALIDATION.md), [proposed scaling work](SCALING.md)
- [Original experiment ledger](python/method/mutation_results.json) and [claim registry](python/method/claims.py)

## What runs

The graph aggregate influences LIF current and spike timing. Each triggered node sends its own pre-tick embedding components into the two-qubit circuit. The current pipeline consumes exact statevector expectations; its sampled output is a separate API field. The weighted GraphSAGE layer is tested separately and does not run inside the composed tick.

All weights are fixed or random. No training, quantum-hardware result, quantum advantage or billion-node deployment is demonstrated. For the current Z readout, the consumed circuit outputs are simply `cos(a)` and `cos(a)*cos(b)`, providing a direct classical baseline. Every neuron is still stepped on each tick, even when few nodes spike.

## Run the tests

With Python and NumPy, Node.js, and clang plus a WASM linker available:

```bash
bash run_tests.sh
```

To run the supplied binary without rebuilding, put Node on PATH, open the `python` directory, and run:

```bash
python -m method.run_suite
```

That command writes a new `python/method/suite_results.json`; preserve the historical copy before rerunning. The independent November 8 run instead kept raw logs outside the source tree. It completed all ten modules with **1,286 passing check entries and zero failures**, including ten property summaries over 2,350 seeded trials. It used Python 3.12.14, NumPy 2.3.5, Node 24.22.1 and the supplied 560-byte WASM binary on Windows. The C binary was not rebuilt in that run.

The source mutation runner is intended for its documented POSIX environment:

```bash
bash run_method.sh
```

Its temporary-worker cleanup uses fixed names; use an isolated working copy and inspect the runner before executing it in a shared temporary directory. A fresh Windows Python-only campaign reproduced all 434 Python-mutant outcomes: 425 kills and nine survivors. Twelve C mutants were not rerun because the compiler was unavailable. See REVIEW_STATUS.md for scope and runner adaptations. The original final ledger records **437/446 kills**, including crashes, with nine triaged survivors. That percentage is not an accuracy or reliability score. The C mutants are already included in the six operator totals.

## Research directions

Useful next contributions include finite-parameter contract tests, seed-domain and sampler-boundary checks, clean compiler rebuilds, a persistent WASM host benchmark, and richer circuit workloads compared with the direct classical baseline. The current finite-value validator has a known fast-path gap documented in REVIEW_STATUS.md. The original implementation is preserved here so that the reported experiment records remain interpretable.

## Attribution and reuse

Code: [MIT](LICENSE). Paper and paper figures: [CC BY 4.0](docs/PAPER_LICENSE.md).
The manuscript discloses AI assistance separately from human authorship. There is no claim of independent human peer review. Use CITATION.cff to credit this work; no Zenodo DOI has been issued by this repository preparation.
