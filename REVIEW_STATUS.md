# M2 software update — November 8, 2026

Version 0.2.0 implements a persistent Node/WASM LIF host while retaining the original one-shot engine. The final local suite passed all eleven modules: 1,536 PASS entries (250 M2), no failures/skips/nonzero exits or stderr. See [M2 runtime contract and measured benchmark](docs/M2_RUNTIME.md).

The numerical C kernel, supplied binary, Revision 4 paper and historical mutation ledgers are unchanged. The newer Python bridge, server and tests are covered by M2 behavioral/cross-implementation tests, not the historical mutation score. The original ten test files were not edited. Source-to-binary rebuild evidence must come from the separate compiler workflow; the local run used the supplied binary.

Below is the preserved baseline review report. Its statements about unchanged production code and ten modules refer to baseline commit 371578d, not the M2 addition.

---

# Review and reproducibility status

Updated November 8, 2026. This release preserves the supplied implementation and original experiment records, with a revised manuscript and current release notes. The original explanatory documents below are historical and must be read alongside these corrections.

## Executed in this review

All ten test modules passed unchanged: 1,286 PASS markers, zero FAIL markers, zero nonzero exits, zero skips and no stderr. The ten property markers summarize 2,350 seeded trials. Python 3.12.14, NumPy 2.3.5, Node 24.22.1; Windows; supplied WASM binary used without rebuilding. All 47 input files remained byte-identical during this run. Full-capacity tests exercised 65,536 neurons for 20 ticks with zero numerical voltage difference and no spike-mask mismatch. Exact numeric equality does not establish raw floating-point bit identity.

A separate mathematical check tested 1,000 angle triples against the two-qubit closed forms (maximum difference 4.44e-16) and 200 monotone Euler-LIF first-crossing cases (maximum trajectory difference 5.33e-15). These checks are independent formula calculations, not a replacement for the original implementation tests.

## Original mutation evidence

The supplied September 14 ledger contains 446 unique mutants: 437 kills, nine survivors and no timeouts. Kills comprise 238 assertion-failure outcomes and 199 crashes. The six operator-family totals already include the twelve C mutants. Revision 3 displayed that C subset as a separate row without making its nonadditivity clear; Revision 4 corrects the label. No empirical result was invented to reconcile the tables.

The reported 464-mutant baseline and original eighteen-mutant review harness are not included. Their historical comparisons have not been independently rerun here. Survivor labels are recorded judgments, not proofs that no defect remains.

## Known scope and open issues

- `_eval_param` returns from literal/bound-parameter fast paths before its general `isfinite` check. The blanket finite-parameter claim is too strong; explicitly rejecting non-finite values on every path needs targeted tests.
- Quantum evolution references share the production parser or parameter evaluator. Independence applies to evolution algorithms, not the entire frontend.
- P10 observes changing sampled outputs while embeddings also change. It does not directly prove distinct seeds. Seed domains, conversions and independence need separate analysis.
- The final-CDF guard survivor has a testable boundary behavior; the claimed universal probability near 10^-16 is not established.
- Relabeling tests check node equivariance, not invariance. Three-dimensional random projections have unbiased squared norms over random matrices, not a demonstrated low-distortion guarantee for one fixed projection.
- General OpenQASM measurement semantics, arbitrary untrusted-input safety, concurrent execution, training, real QPUs, large scale, energy advantage and production latency are not established.

No production numerical code or test assertion was changed during preparation of this baseline. Further Magica simulator fragments are excluded until specifically supplied and cleared for this public release.

## Fresh Python mutation rerun

On November 8, 434 Python mutants were executed with two workers: 425 were killed (226 assertion failures and 199 crashes), nine survived, and none timed out or were invalid. The mutant ID set and every status match the supplied ledger's Python subset. The run took 842.89 seconds and recorded 700 selected test invocations. See [machine-readable summary](docs/mutation-rerun-python-2026-11-08.json).

The twelve C mutants were excluded because clang/wasm-ld was unavailable. This is a partial rerun, not reproduction of the full 446-mutant campaign. The copied runner was adapted for task-owned Windows worker directories, guarded cleanup, Python-only selection and complete log capture. Production code, tests and the supplied WASM binary remained unchanged. Original full ledgers are preserved separately; local execution logs include machine paths and are not part of this public upload.
