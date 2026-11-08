> Historical scaffold document. Read [REVIEW_STATUS.md](REVIEW_STATUS.md) and Revision 4 for current corrections, evidence limits and rerun status. Original text follows.

# AGENTS.md

Instructions for any AI coding agent (this scaffold was built to be picked up and continued by one) working in this repository.

## Build & Test Commands

```bash
bash run_tests.sh                              # builds wasm-core, runs everything, in dependency order
bash wasm-core/build.sh                        # rebuild orchestrator_core.wasm after editing orchestrator_core.c
cd python && python3 -m tests.test_qasm_sim
cd python && python3 -m tests.test_qasm_gate_coverage        # added by the review: every gate vs an independent kron-built reference
cd python && python3 -m tests.test_qasm_parser_robustness    # added by the review: malformed .qasm is rejected at load()
cd python && python3 -m tests.test_lif_neuron
cd python && python3 -m tests.test_gnn_message_passing
cd python && python3 -m tests.test_shared_space
cd python && python3 -m tests.test_pipeline_smoke
cd python && python3 -m tests.test_composition_wiring        # added by the review: every run_tick wiring point re-derived per tick
cd python && python3 -m tests.test_wasm_core_bounds          # added by the review: orc_tick at and beyond MAX_NEURONS, called directly
cd python && python3 -m tests.test_properties                # v1.2: generative (property-based) tests for every layer, seeded, no Hypothesis
bash run_method.sh                                           # v1.2: suite + operator-based mutation campaign + BUILD_INDEX.md (~3 minutes)
cd python && python3 -m method.mutate --only gnn/message_passing.py   # mutate one file
cd python && python3 -m method.build_index                   # regenerate the index from the last results
```

**The method (`python/method/`) is part of the build.** `method/claims.py` is the registry of every validated property; `method/mutate.py` generates and runs mutants over every source module; `method/triage.json` holds a verdict for every surviving mutant; `method/build_index.py` writes `BUILD_INDEX.md`. `run_method.sh` refuses to regenerate the index while a survivor is untriaged. When you add or change a numerical function: add its claim to the registry, name its test, run `bash run_method.sh`, and triage anything new that survives — a survivor with no verdict is an open question, not a statistic.

Every test module is a standalone script (`if __name__ == "__main__":`), not a pytest suite — run it directly with `python3 -m tests.<name>` from the `python/` directory (the `-m` matters: it puts `python/` on `sys.path` the way the modules expect). Each prints one `[PASS]`/`[FAIL]` line per assertion and ends with `ALL PASSED` or a list of failures, exiting 0 or 1 accordingly.

**Toolchain required:** `clang` with its built-in `wasm32-unknown-unknown` backend (no emscripten), `node` (native `WebAssembly` API, no npm package), `python3` with `numpy`. Nothing else. This is deliberate, not an oversight: `qiskit` and `wasmtime` were both tried during this scaffold's construction and could not be installed in the sandbox it was built in (see VALIDATION.md's toolchain notes) — rather than block on that, `qmodule` is a from-scratch statevector simulator and `wasm-core` uses clang's freestanding backend directly. A real deployment may reasonably want a real QPU SDK or a production WASM runtime; adding one is a legitimate next step (see SCALING.md), but it changes this project's dependency footprint and should be a visible, deliberate change — not something that happens quietly inside an unrelated commit. Also note: `rustup target add wasm32-unknown-unknown` was tried first and was blocked by an organizational network policy (403 on `static.rust-lang.org`) in the sandbox this was built in — that is why clang's own backend is used instead of the more common Rust-toolchain path. That may or may not apply in your environment; either toolchain produces a valid wasm32 module.

## Code Style Rules

- `from __future__ import annotations` at the top of every module — keeps modern type hints (`int | None`, `tuple[np.ndarray, np.ndarray]`) valid without a Python-version fight.
- Parameter and result bundles are `@dataclass` (usually `frozen=True` for parameters), never bare dicts or tuples with positional meaning.
- Every module docstring explains **why** the module exists and where its non-obvious design choices came from — not just what functions it contains. Keep doing this; it is what makes the code legible to the next agent (you, later, or a different one) without archaeology through git history.
- Any function with a non-obvious numerical or structural property names, in its own docstring, the test that checks that property (e.g. "see test_shared_space.py::test_projection_is_linear"). If you add such a function, add both halves together.
- Explicit `dtype` (`np.float64`, `np.int64`) on every array you construct — never leave it to numpy's default inference. Node ids are `int64` project-wide (matches `wasm-core`'s use of C `int`/pointer-sized indices); floating data is `float64` project-wide (matches `orchestrator_core.c`'s use of C `double`) — this consistency is what makes the WASM/Python bit-exactness checks in `test_pipeline_smoke.py` meaningful rather than lucky.
- No bare `except:`. Precondition/environment failures (missing `node`, unbuilt `.wasm`, oversized input) raise `RuntimeError` with a message that says what to do about it, not just that something went wrong.
- Prefer a plain, explicit loop or a well-named helper over a clever one-liner when the clever version would be harder to independently re-derive in a test. This codebase's validation strategy depends on every nontrivial computation being re-derivable by a second, differently-written path (see Testing Instructions) — code that's too clever to re-derive independently is code that can't actually be adversarially tested, however elegant it looks.

## Project Structure

See README.md's directory map for the full tree. Layer ownership, briefly:

| Directory | Owns |
|---|---|
| `python/qmodule/` | OpenQASM 2.0 parsing + statevector simulation (the QGNN layer's execution engine) |
| `python/snn/` | The LIF spiking neuron update rule (Python reference implementation) |
| `python/gnn/` | Sparse graph message-passing (gather/scatter, never dense adjacency) |
| `python/fusion/` | The 4D shared coordinate + the logic deciding when a node's QGNN circuit fires |
| `python/orchestrator/` | Per-tick composition of the four layers above; the pluggable LIF-engine seam |
| `wasm-core/` | The classical control loop's *other* implementation (C, compiled to wasm32), proven equivalent to `snn/lif_neuron.py` |
| `python/method/` | The validation method as code: claim registry, mutation engine, property harness, reference simulator, index generator, survivor triage |
| `python/tests/` | One file per layer (`test_qasm_sim.py`, `test_lif_neuron.py`, `test_gnn_message_passing.py`, `test_shared_space.py`), `test_pipeline_smoke.py` covering the WASM/Python cross-check and the full end-to-end composition, plus the four modules the independent review added: `test_qasm_gate_coverage.py`, `test_qasm_parser_robustness.py`, `test_composition_wiring.py`, `test_wasm_core_bounds.py` (see VALIDATION.md, "Independent review") |

## Testing Instructions

This project's validation standard, applied consistently across every layer so far — keep applying it:

1. **Every new numerical function needs an exact closed-form check, an independent cross-implementation check, or both.** A test that only checks "it ran without crashing" or "the output has the right shape" is a legitimate *supplement* but never a *substitute* for one of these. Examples already in this codebase: `test_qasm_sim.py` checks the Bell-state and RY/CX/RZ circuits against hand-derived `cos`/`cos*cos` formulas; `test_lif_neuron.py` checks every discrete spike time against an independently-derived closed form of the Euler recurrence itself (not just the continuous analytic rate); `test_gnn_message_passing.py` checks the vectorized aggregation against a naive per-edge Python loop with zero shared code path; `test_pipeline_smoke.py` checks the WASM engine against the Python engine bit-for-bit, including under live feedback.
2. **Statistical checks must say so.** If a check depends on shot noise or Monte Carlo sampling (e.g. `test_shared_space.py`'s projection-unbiasedness check), print that it's statistical, print the observed value and error, and justify the tolerance against the actual sampling behavior — don't pick a tolerance by feel and don't let a statistical check masquerade as an exact one.
3. **When two independent implementations of the same thing disagree, that is a real bug in one of them, not a rounding artifact to wave away** — until you've actually shown otherwise (e.g. traced the discrepancy to a genuine, justified difference in floating-point operation order). `test_pipeline_smoke.py` found bit-exact (not just "close") agreement between C/wasm32 and Python/NumPy specifically because both implement the identical arithmetic in the identical order; if you change one side's operation order, expect the tolerance there to need revisiting, and say why in the test.
4. **Run `bash run_tests.sh` before considering any change done.** If your change is expected to affect a specific layer's numbers (e.g. a new gate type changes `qmodule`'s closed forms), update that layer's derivation in the relevant test file's docstring/comments — don't just widen the tolerance until it passes.
5. **A test that takes its expected value from the code under test proves nothing.** The review's mutation testing (VALIDATION.md, "Independent review") found that the original composition test re-derived each QGNN trigger correctly but took the *set* of triggered nodes from the pipeline's own log, so a trigger fired one tick late or on the wrong node passed it. Re-derive the expected value — including *which* nodes, *which* tick — from the previous state by a separate path, never from the output being checked. When you add a test, ask which injected bug would make it fail; if you cannot name one, it is a smoke test, not a validation.
6. **A property not in `method/claims.py` is not validated, whatever a docstring says; a mutant without a verdict in `method/triage.json` is not understood.** Both files are inputs to `BUILD_INDEX.md`, which is the one place that states what this tree proves. Prefer a refactor that removes an equivalent-mutant site (a redundant guard, a hand-written search) over a verdict that explains it away — the v1.2 refactors of `mean_aggregate`, `_apply_cx`, `sample_shots` and `expectation_z` are the pattern.
7. Test files have zero framework dependency (no pytest) by design, matching the project's dependency-free ethos. You may introduce pytest if it clearly earns its keep, but keep each module runnable standalone the way it is now, or call out the migration explicitly rather than silently dropping the old entry points.

## Git Workflow

This scaffold was built without a git repository. If you're continuing it, `git init` and make one initial commit of the tree as-is (it is a coherent, fully-passing baseline — see VALIDATION.md) before making further changes, so subsequent work has a real diff history. After that:

- One logical change per commit. "Add distributed scatter-sum for M1" and "fix GNN permutation-invariance test tolerance" are two commits, not one.
- Reference which layer/milestone a commit belongs to in its message (e.g. `gnn: replace np.add.at scatter-sum with scipy.sparse (M1)`) — this project's roadmap is staged (see SCALING.md's M0/M1/... labeling), and commit messages that say which stage they belong to make that roadmap's actual progress legible later.
- Don't rewrite history on shared branches; don't force-push.

## Boundaries

- **Don't add a training loop or optimizer for `GnnLayerParams` or `ProjectionParams` without checking with the user first.** Every weight in this scaffold is fixed/random by explicit design (see each module's docstring) — that's a scope decision, not a gap to quietly fill in.
- **Don't silently swap in `qiskit`, `cirq`, or a real QPU backend for `qmodule`.** `QasmModule`/`QasmResult` were specifically shaped (see that module's docstring) to be a drop-in seam for this, so making the swap is welcome and expected eventually (see SCALING.md M4) — but it changes the project's dependency footprint and, per SCALING.md M4, raises a real unsolved latency/batching problem at scale. Surface it as its own explicit change, not something buried in an unrelated commit.
- **Don't delete or weaken an existing test to make a change pass.** If a change legitimately invalidates a previously-derived closed form, re-derive it and update the test file's docstring/comments to show the new derivation — don't just loosen the tolerance or remove the check.
- **Don't claim a scale that hasn't actually been run and measured.** Follow the M0/M1/... convention in SCALING.md: a milestone is "complete" only once its own validation (closed-form and/or cross-implementation, per the Testing Instructions above) passes at that milestone's scale, the same way M0 (this scaffold) is validated at toy scale in VALIDATION.md. "Should work at 10^9 nodes" is a hypothesis, not a result.
- **The 140M–20B node figures come from the original request, not from any capacity analysis performed in this repository.** Treat SCALING.md as a proposed roadmap to discuss with the user, not a committed plan with implied timelines or cost estimates — none were given because none would be credible without more information about the target deployment.
