> Historical scaffold document. Read [REVIEW_STATUS.md](REVIEW_STATUS.md) and Revision 4 for current corrections, evidence limits and rerun status. Original text follows.

# Validation

What has actually been tested, how, and with what result — current as of this scaffold's initial build. Every number below is from an actual run of the test suite (`bash run_tests.sh`), not an estimate.

**Current totals: 1286/1286 checks passing, 0 failures, across 10 test modules, all exiting 0; 437/446 generated mutants killed (98.0%), 9 survivors, each with a written verdict.** (The original build had 1039 checks in 5 modules; the independent review added four modules; the systematized method (v1.2, last section of this file) added generative tests, a mutation engine, a claim registry and `BUILD_INDEX.md`.) For the current claim-by-claim and mutant-by-mutant state see `BUILD_INDEX.md`; this file is the narrative.

| Test module | Checks | Result | Origin |
|---|---:|---|---|
| `tests/test_qasm_sim.py` | 25 | ALL PASSED | original (+5 in v1.2) |
| `tests/test_qasm_gate_coverage.py` | 30 | ALL PASSED | review (+2 in v1.2) |
| `tests/test_qasm_parser_robustness.py` | 37 | ALL PASSED | review (+7 in v1.2) |
| `tests/test_lif_neuron.py` | 306 | ALL PASSED | original |
| `tests/test_gnn_message_passing.py` | 52 | ALL PASSED | original (+11 in v1.2) |
| `tests/test_shared_space.py` | 24 | ALL PASSED | original |
| `tests/test_pipeline_smoke.py` | 649 | ALL PASSED | original (+1 assertion, 2 tightened to exact) |
| `tests/test_composition_wiring.py` | 135 | ALL PASSED | review |
| `tests/test_wasm_core_bounds.py` | 18 | ALL PASSED | review (+5 in v1.2) |
| `tests/test_properties.py` | 10 properties (2,350 random trials) | ALL PASSED | v1.2 |
| **Total** | **1286** | | |

A note on what the count measures: it is a count of `[PASS]` lines, and it is dominated by repeated instances of a few check types — 300 of `test_pipeline_smoke.py`'s checks are the per-tick `fused_4d` shape check, 333 are the three per-trigger closed-form checks over 111 triggers, 210 of `test_lif_neuron.py`'s are per-spike time comparisons. The number of distinct *properties* checked is a few dozen, across 39 test functions. Both numbers are honest; the second is the better measure of coverage.

## Methodology

Three kinds of check are used throughout, and every module docstring says which kind applies to which claim:

- **Exact closed-form.** A result is compared to a value derived by hand, independently of the code under test, usually to `1e-9` or exact (`tol=0.0`) where floating-point should make that achievable. Used wherever the math allows it — it's the strongest evidence available, because there's no "reference implementation" to have copied a bug from.
- **Independent cross-implementation.** Two implementations of the same operation, written without a shared code path, are compared against each other. Used where a closed form doesn't exist or isn't practical to derive (e.g. GNN aggregation on an arbitrary random graph). Agreement is evidence about the algorithm; it does not rule out both implementations sharing the same *conceptual* misunderstanding.
- **Statistical / Monte Carlo.** Used only where the quantity being checked is inherently statistical (shot-sampled quantum measurement, a random-projection unbiasedness property) — always labeled as such in the test's own print output, with the observed value, error, and tolerance shown, not hidden behind a bare pass/fail.

## Per-module results

### `test_qasm_sim.py` — QGNN / statevector simulator (20 checks)

- Bell state (`h; cx`): exact `<Z0>=<Z1>=0`, exact zero probability of the `01`/`10` outcomes — checked from the full statevector, not sampled.
- `ry_node_update.qasm` (`ry(a); ry(b); cx; rz(phi)`) at 4 parameter points, each checked two ways: **exact** expectation values against `<Z0>=cos(a)`, `<Z1>=cos(a)*cos(b)` (`tol=1e-9`), and **sampled** expectation values from 50,000 shots against the same closed form (`tol=0.03`, sized for shot noise at that count, not arbitrary).
- `rz(phi)` invisibility: swept `phi` across 4 values spanning more than a full period and confirmed `<Z1>`'s spread across all of them is exactly `0.0` — i.e. a Z-basis measurement provably cannot see a phase rotation. This exact fact is what licenses `fusion/shared_space.py::maybe_trigger_qgnn` to never overwrite the `phi` embedding component.

**Gate coverage note:** `statevector_sim.py`'s `GATE_MATRIX` implements `h, x, y, z, s, t, rx, ry, rz` (plus `cx`), but only `h`, `cx`, `ry`, and `rz` are exercised by any circuit file or test in this repository. `x, y, z, s, t, rx` are implemented and presumably correct (they follow the same `_apply_single`/`_u` code path already validated for `h`/`ry`/`rz`) but have **no dedicated closed-form check of their own**. Adding one for each (or at minimum for `rx`, the only other parameterized single-qubit gate) is a reasonable, cheap next step before relying on them.

### `test_lif_neuron.py` — SNN (306 checks)

- **Sub-threshold exactness:** for `I` in `{0.0, 0.3, 0.6, 0.9, 1.0}` (all `<= v_th`), confirmed `analytic_rate` returns exactly `0.0` and a 500-time-unit simulation produces exactly `0` spikes — including the `I == v_th` boundary case.
- **Exact discrete spike-timing match:** for `I` in `{1.05, 1.2, 1.5, 2, 3, 5, 10, 50}` at `dt=0.01`, every individual spike time (not just the first, not just an average rate) was checked against a closed-form solution of the *discrete* Euler recurrence itself (derived independently in the test file's docstring, not reused from `analytic_rate`), to exact integer-step precision. This is the bulk of the 306 checks.
- **Discretization-order sanity check:** at a fixed supra-threshold current, discretization error between the exact-discrete formula and the continuous analytic ISI was measured at `dt = 0.2, 0.02, 0.002` and observed to shrink monotonically (`0.0685 -> 0.00853 -> 0.000528`), consistent with forward Euler's expected first-order convergence — a real numerical-methods property, not just "close enough."
- **Vectorization correctness:** 6 neurons (mixed sub/supra-threshold, including the `I == v_th` boundary) stepped together in one batched call, cross-checked spike-for-spike and time-for-time against the same neurons run one at a time — exact match confirms no cross-neuron leakage in the vectorized code.

### `test_gnn_message_passing.py` — GNN (41 checks)

- `scatter_sum` regression test for the specific, real NumPy gotcha it exists to avoid (`out[index] += messages` silently drops repeated indices; `np.add.at` does not) — plus an exact mass-conservation identity (total scattered mass equals total input mass) on a 40-node/300-edge random graph.
- `mean_aggregate` cross-checked against an independent, naive per-edge Python loop on a 35-node/250-edge random graph (full array match, `tol=1e-10`); zero-in-degree nodes confirmed to return an exact zero vector (never `NaN`); averaging a constant feature vector confirmed to return exactly that constant, at every node, on a 20-node graph.
- `gated_mean_aggregate` boundary identities: all-nodes-spiking reduces exactly to `mean_aggregate`; no-nodes-spiking returns exact all-zero; partial spiking matches `mean_aggregate` restricted to the active-source edge subset.
- `gnn_layer_forward` **permutation invariance**: a 30-node/180-edge graph was relabeled under a random permutation and the layer re-run; outputs matched the original run exactly (`tol=1e-10`) node-for-node after undoing the relabeling — evidence the layer is a genuine function of graph structure and features, not of array-position artifacts.

### `test_shared_space.py` — Fusion (24 checks)

- `project_to_3d` exact linearity (`tol=1e-10`).
- **Statistical:** the random projection's unbiasedness (`E[||Px||^2] = ||x||^2`, the property the `1/sqrt(3)` scaling exists to guarantee) was estimated via 4,000 independent Monte Carlo trials and observed at **2.088% relative error** against a `10%` tolerance — a clean result, printed and labeled as statistical in the test output rather than treated as exact.
- `time_since_last_spike` exact arithmetic, including the never-spiked `+inf` case and selective `inf_fallback` substitution.
- `fuse_to_4d` confirmed to do no independent computation of its own — its output is an exact concatenation of `project_to_3d` and `time_since_last_spike`'s outputs.
- `maybe_trigger_qgnn`: triggering exactly matches the spike mask; per-triggered-node results matched the `cos(a)`/`cos(a)*cos(b)` closed form (`tol=1e-9`, reusing the oracle proven in `test_qasm_sim.py`); `phi` (embedding component 2) confirmed untouched; `blend` boundary identities (`0.0` = untouched, `0.5` = exact linear interpolation between the `0.0` and `1.0` results) confirmed exactly.

### `test_pipeline_smoke.py` — WASM/Python engine equivalence + full composition (648 checks)

**Part 1 — isolated neurons, externally-fixed current** (25 neurons, 300 ticks, `dt=0.02`, mixed sub/at/supra-threshold initial conditions): `orchestrator_core.wasm` (C, compiled to real wasm32, executed via Node's native `WebAssembly` API) vs. `snn/lif_neuron.py::step()` (Python/NumPy) — **7,500/7,500 membrane-potential values matched to `0.000e+00` max absolute error (bit-exact), 0/7,500 spike-flag mismatches, and identical total spike counts (37 = 37)** between the two independently-written implementations. Since the review, this is *asserted* at `tol=0.0` (the original test printed the 0.0 but asserted only `<= 1e-9`). Plus a negative test confirming an over-capacity neuron count raises a clear, actionable error rather than silently truncating.

**Part 2 — full pipeline composition** (14-node toy graph — a directed ring plus random chords, guaranteeing every node has an in-edge — 300 ticks, `dt=0.05`): confirmed no `NaN`s anywhere, spikes and QGNN triggers actually occurred (the test would fail loudly if the scenario were vacuously silent — this caught and required fixing an initially-too-short simulated time window during development, see the git history once one exists), and then **re-derived 111 individual live QGNN triggers directly from the `cos(a)`/`cos(a)*cos(b)` closed form**, each checked against the *pre-tick* embedding that fed the circuit (catching same-tick ordering bugs, not just eventual-consistency ones), plus **4,089 non-triggered node-ticks confirmed byte-for-byte unchanged** by the QGNN step.

**Part 2 — engine equivalence under feedback** (10-node graph, 25 ticks, `dt=0.3`, 38 total spikes): re-ran the *same* graph-coupled scenario — where each tick's LIF input current is itself produced by the previous tick's spikes via GNN aggregation, not held externally constant — through both `python_lif_engine` and `wasm_bridge.py::wasm_lif_engine`, and found **exact agreement propagating through the entire feedback loop**: `max|V_python - V_wasm| = 0.0`, zero spike-mask mismatches, `max|embedding_python - embedding_wasm| = 0.0` (asserted at `tol=0.0` since the review). This is the stronger claim Part 1 alone doesn't establish — that the two engines stay interchangeable once their outputs feed back into everything else, not just in isolation.

## Toolchain notes (why the dependency list is what it is)

- `pip install qiskit` and `pip install wasmtime` both failed in the sandbox this scaffold was built in (`Could not find a version that satisfies the requirement ... (from versions: none)`), most likely a restricted package index rather than a hard network block — plain PyPI installs of other packages (`numpy`, `networkx`) succeeded. Rather than leave this ambiguous, `qmodule` was built from scratch instead: self-contained, auditable, and it removes the dependency question entirely rather than working around it once.
- `rustup target add wasm32-unknown-unknown` failed with an explicit HTTP 403 organizational policy denial on `static.rust-lang.org`. Per that proxy's own stated policy, this was treated as a real "no," not retried. `clang`'s built-in `wasm32-unknown-unknown` backend (`--target=wasm32-unknown-unknown -nostdlib`) has no such dependency and was verified end-to-end (compile, load via Node's native `WebAssembly` API, call, get the right answer) before being adopted for `wasm-core/`.

Neither of these is necessarily true of whatever environment picks this scaffold up next — they're recorded here so that environment's own toolchain probing doesn't have to rediscover them from scratch, and so "we used a from-scratch simulator instead of qiskit" reads as a documented decision rather than an unexplained gap.

### `test_qasm_gate_coverage.py` — every gate vs an independent reference (28 checks, added by the review)

An independent simulator (full `2^n x 2^n` unitaries built with `numpy.kron` from textbook matrices, no code shared with `statevector_sim.py`) is compared against the scaffold's simulator on the **full complex statevector, phases included** — a stronger check than the Z-basis-only checks in `test_qasm_sim.py`. Covers all nine single-qubit gates alone and inside `h;G;h` (which makes phase gates visible), `rx/ry/rz` swept over `[-2π, 2π]` (41 angles) with closed forms `<Z> = cos θ`, `ry_node_update.qasm` on 247 `(a,b,φ)` points including unbounded embeddings, GHZ on three qubits, and `cx` with the control index *greater* than the target's. Result: max amplitude difference `0.0` everywhere. This closes the "six untested gates" gap, and gives `rz` its first *positive* test — the original `rz` check (that it does not move `<Z1>`) is also passed by an `rz` implemented as the identity.

### `test_qasm_parser_robustness.py` — malformed input rejected at load (30 checks, added by the review)

Twenty-three malformed or unsupported constructs (unknown gates, out-of-range qubit and clbit indices, `cx q[0],q[0]`, wrong operand counts, parameters on parameter-free gates and vice versa, second `qreg`, undeclared registers, statements before `qreg`, custom `gate`/`opaque`/`if`/`reset`, missing semicolons and commas) are each confirmed to raise `SyntaxError` at `parse_qasm` time, and the parameter-expression bomb `9**9**9**9` is confirmed refused in under a millisecond. The review found the original parser accepted every one of these silently: some failed later inside `call()` with `ValueError: negative shift count`, and several returned a wrong answer without any error (`cx q[0],q[0]` was a no-op; a third `cx` operand was ignored; `h(0.3)` ignored its parameter; a second `qreg` silently replaced the first; a reference to an undeclared register was applied to the declared one; `9**9**9**9` hung the process).

### `test_composition_wiring.py` — every wiring point re-derived per tick (135 checks, added by the review)

For 120 ticks of the full pipeline, every quantity `run_tick` produces is re-derived from the previous tick's state by a separately written naive path and compared exactly: this tick's current (per-edge loop over the spike-gated in-neighbour mean, feature mean, coupling, baseline; `tol=1e-12`), the LIF step and spike flags (scalar loop; `tol=0.0`), the QGNN trigger set (`== spiked > 0`, every tick), `last_spike_time` (from the spike log alone; `tol=0.0`), and the fused output (post-update embedding times the projection by triple loop; elapsed time from the log; `+inf` exactly where a node has never spiked). Also: an input that lands *exactly* on `v_th` must spike (the documented `>=`), checked in both engines bit-identically; and `wasm_lif_engine` is confirmed to actually spawn the Node harness once per call. See "Independent review" below for why each of these exists.

### `test_wasm_core_bounds.py` — the C core at and beyond capacity (13 checks, added by the review)

`orc_tick` is called *directly* through Node's `WebAssembly` API (bypassing the harness's guard) with `n = MAX_NEURONS + 2`, `-1`, `10,000,000`, `0`, and exactly `MAX_NEURONS`: out-of-range `n` now returns `-1` and touches no memory (the original core silently updated `I[0]` and `I[1]` as if they were neurons, because `V[]` and `I[]` are adjacent in linear memory; only `n = 10,000,000` reached the end of linear memory and trapped). At exactly 65,536 neurons, 20 ticks through the real harness are bit-exact against the Python reference (`1,310,720` neuron-ticks, 19,264 spikes, max error `0.0`).

## Known gaps — not covered by the current suite

- Property-style sweeps exist only where the review added them (500 random graphs for `gated_mean_aggregate`, in the review's own probe script rather than this suite; 247 random parameter points in `test_qasm_gate_coverage.py`). There is still no Hypothesis-style generative testing, and the remaining random-scenario tests use fixed seeds.
- The composed pipeline never applies `gnn_layer_forward` (`W_self`, `W_neigh`, `tanh`): `run_tick` uses only `gated_mean_aggregate` and a feature-mean readout. The GraphSAGE layer is validated in isolation but does not run in the tick — the "GNN" in the composition is the spike-gated aggregation primitive, not the learned-layer transform.
- `measure` statements are parsed and validated but `call()` always samples every qubit; the classical-register mapping is not modelled.
- No test exercises more than three qubits, or more than a few dozen graph nodes in the composed pipeline.
- No concurrency/parallelism testing of any kind — everything here runs single-threaded, single-process.
- The bit-exact WASM/Python result depends on both sides using IEEE-754 double with identical operation order and on the JSON round trip through the harness being shortest-round-trip on both sides (Python's `repr`, V8's `JSON.stringify`); neither is asserted separately.

## Independent review (mutation testing and what it changed)

The scaffold and its thesis were reviewed by a second model, whose brief was to test the thesis's claims rather than restate them. Three kinds of evidence were gathered.

**Reproduction.** From a clean extraction of the delivered archive, every reported number reproduced exactly: 20/306/41/24/648 = 1039 checks, 7,500/7,500 bit-exact with 37 = 37 spikes, 111 re-derived triggers plus 4,089 exact pass-throughs, 25 feedback ticks with 38 spikes at 0.0 error, 2.088% Johnson–Lindenstrauss error. Independent re-derivations (a `numpy.kron` reference simulator, the discrete LIF closed form re-solved, a 200,000-trial JL estimate at 0.077% error) agreed with every closed form the thesis states.

**Mutation testing.** Eighteen defects were injected one at a time into a fresh copy of the tree and the original suite run. It killed 7 of 18. Every *value-level* mutation was caught (result written to the wrong node's row, `out[index] +=`, dropped `1/sqrt(3)`, removed LIF reset, `ry` at half angle, C core in single precision, gating on the destination node). Every *wiring-level* mutation inside `run_tick` survived: QGNN fired one tick late, or on the neighbouring node; spike gating removed from the aggregation; coupling sign flipped; feature-sum instead of feature-mean readout; elapsed-time clock off by one tick (two variants); fused output computed from the pre-update embedding. Also surviving: `rz` replaced by the identity (its only check was a null test), LIF `>=` changed to `>` (no test landed exactly on threshold), and `wasm_lif_engine` silently running the Python step (nothing asserted the harness was invoked). The reason for the wiring survivals is structural: `test_pipeline_smoke.py` re-derives each trigger's *arithmetic* correctly but takes the *set* of triggered nodes, and every other per-tick quantity, from the pipeline's own log. With the four review modules added, the score is **18/18 killed**.

| # | Injected defect | Original suite | With review modules |
|---|---|---|---|
| M1 | fused output computed from the pre-update embedding | survived | killed |
| M2 | QGNN triggered from the previous tick's spike mask | survived | killed |
| M3 | spike mask rolled by one node before QGNN | survived | killed |
| M4 | QGNN result written to the wrong node's row | killed | killed |
| M5 | `out[index] += messages` instead of `np.add.at` | killed | killed |
| M6 | random projection without `1/sqrt(3)` | killed | killed |
| M7 | LIF threshold `>=` changed to `>` | survived | killed |
| M8 | post-spike reset removed | killed | killed |
| M9 | `ry` applies half the angle | killed | killed |
| M10 | `rz` replaced by the identity | survived | killed |
| M11 | C core computes in single precision | killed | killed |
| M12 | gating on destination instead of source spike | killed (unit test only) | killed |
| M13 | elapsed-time clock off by one tick | survived | killed |
| M14 | spike gating dropped from the pipeline's aggregation | survived | killed |
| M15 | GNN→SNN coupling sign flipped | survived | killed |
| M16 | `wasm_lif_engine` silently runs the Python step | survived | killed |
| M17 | `last_spike_time` updated from the previous tick's mask | survived | killed |
| M18 | readout sums features instead of averaging | survived | killed |

**Probing the admitted gaps.** The parser and the C core were exercised on the inputs the original gap list said were unspecified; the findings are recorded in the `test_qasm_parser_robustness.py` and `test_wasm_core_bounds.py` sections above. Both were fixed in the code (load-time validation and the `**` refusal in `statevector_sim.py`; a bounds guard in `orc_tick`), and the fixes are pinned by those modules. Two further code changes: the QGNN shot-sampling seed now advances with the tick (the original reused `seed + node` on every tick, so any future consumer of `sampled_expectation_z` would have seen identical shot noise every tick), and three module docstrings that referred to files outside the repository (`../scaling/SCALING.md`, `quantum-spec.md`) were corrected.

Nothing in the review contradicted a numerical claim in the original VALIDATION.md. What it changed is the *strength* of what the suite proves about the composition, and the precision of two claims: "bit-exact" is now asserted rather than observed, and "parses and validates once" is now true of `QasmModule.load()`.

## The method, systematized (v1.2)

The review's eighteen hand-chosen mutants were a probe. v1.2 turns the probe into a measurement and makes the measurement part of the build (`python/method/`, `bash run_method.sh`, `BUILD_INDEX.md`).

**Mutation engine (`method/mutate.py`).** Six AST-level operators — arithmetic (AOR), relational (ROR), logical (LCR), constant (CRP), statement deletion (SDL), argument swap (AGS) — applied to every eligible site in every source module, plus twelve explicit textual mutants of the C core. Defaults in signatures and dataclass fields are never mutated (configuration, not logic). Each mutant runs in a fresh copy of the tree; the test modules that transitively import the mutated file (computed from the import graph, not listed by hand) run fastest-first and stop at the first failure; a mutant that hangs a test for 30 s is a *timeout*, counted separately from *killed*. Mutant ids are content hashes, so verdicts survive line-number changes. Sources are snapshotted at generation: the engine's first run read the live tree while it was being edited and produced seven "site not found" mutants — the engine's own `invalid` column caught the engine's own bug.

**Claim registry and index.** `method/claims.py` lists every validated property with its mode, its test function(s) and the version that added it (46 claims). `method/build_index.py` writes `BUILD_INDEX.md` from the registry, the last suite run and the last campaign, including every survivor with its verdict from `method/triage.json`; `run_method.sh` refuses to regenerate the index while any survivor is untriaged.

**Generative tests (`method/props.py`, `tests/test_properties.py`).** A seeded property harness with no dependency (Hypothesis is not installable in every environment this scaffold targets): ten properties, each the generalization of an existing closed-form or cross-implementation check to random parameters, graphs, circuits or malformed input, reporting the first failing seed. The first run of P2 (random malformed lines must raise SyntaxError or ValueError and nothing else) found that `_eval_param` let Python's own `SyntaxError` and `ZeroDivisionError` escape from `eval` for expressions such as `1..2` and `1/0`; now wrapped.

**Baseline campaign, v1.1 suite (9 modules, 1,246 checks):** 464 mutants generated, 410 killed (88.4%), 47 survived, 7 timed out. The survivors sorted into seven classes, most of them invisible to the review's hand-chosen probe:

| Class | Count | Example | Why the suite could not see it |
|---|---:|---|---|
| Gates never applied to a state with a \|1> component | 8 | `U(θ,φ,λ)`'s second column mutated; `rx`'s λ = 2π; `base + off` → `base − off` in the stride loop | every `rx`/`ry`/`rz` in the suite acted on \|0>, and every single-qubit gate on a qubit with stride > 1 acted on a basis state, so the matrix's second column and the loop's second pair were never exercised |
| Default parameters hide arithmetic | 2 | `v_inf − v_reset` → `+`; `−(V − v_rest)` → `+` | every LIF test used `v_reset = v_rest = 0` |
| Validator boundaries and connectives | 6 | `<` → `<=` on node ids; `and` → `or` on dtypes; `>=` → `>` on measure indices | one out-of-range case and one dtype case each; no boundary (`id == n`), no mixed case |
| Redundant guards (equivalent mutants) | 10 | `where(deg > 0, …)` plus `mean[deg == 0] = 0`; `if i < j` in CX; `× ±1` in ⟨Z⟩; `1.0 + 0j` | two guards for one condition, or an operation with no observable effect: mutating either is invisible by construction |
| Unchecked outputs and untested specifications | 13 | `gnn_layer_forward`'s bias and W_neigh signs; `orc_tick`'s return; `n_clbits`; sampled-⟨Z⟩ accumulator; `num_edges`; `source_path`; renormalization; `params=None` | invariance and boundedness were checked, the value was not; or the documented behaviour had no test |
| Hand-written binary search | 7 (timeouts) | `lo < hi` → `<=`; `mid + 1` → `mid − 1` | every mutant is an infinite loop |
| Conventions and configuration | 8 | seed arithmetic (4); timeout 30 → 31; length limit 200 → 201 and its boundary; `eval` argument order | not defect classes; recorded as *accepted* / *config* |

**Refactor.** Six sites were rewritten so the equivalent mutant cannot exist: `mean_aggregate` uses one guarded `np.divide`; `_apply_cx` visits each swapped pair once from its target-clear member; `sample_shots` uses `bisect` instead of a hand-written search (removing all seven infinite-loop sites); `expectation_z` adds `±p` instead of multiplying by `±1`; the initial amplitude is `complex(1.0)`; `_eval_param` no longer strips. Thirty checks were added for the real gaps (gate coverage on non-basis states; layer value vs a naive loop; validator boundaries and mixed dtypes; measure-index boundaries, `n_clbits`, `params=None`; sampled ⟨Z⟩ recomputed from counts; renormalization of an unnormalized state; `orc_tick`'s return value; `wasm_available()` without the binary), and P4 now checks `analytic_rate` against the discrete closed form for random `v_reset`.

**Final campaign, v1.2 suite (10 modules, 1,286 checks):** 446 mutants generated, 437 killed (98.0%), 9 survived, 0 timed out. Every survivor has a verdict: four are the shot-sampling seed derivation (any tick- and node-distinct derivation satisfies the specification, which `test_properties::P10` pins), three are configuration constants (a subprocess timeout, a length limit and its boundary), one is the `eval` sandbox's defense-in-depth argument order (unobservable through the character whitelist), and one is a defensive line against a ~1e-16 float-drift event. None is a gap.

**What the systematized method found that the review did not.** The review's eighteen probes were all wiring and specification errors; the campaign's forty-seven survivors were dominated by two blind spots no reviewer had thought to probe — parameterized gates never applied to a non-basis state, and default parameter values masking arithmetic — plus a class of hand-written code (the binary search) where every possible mutant was a hang. All were found by mechanism, not insight, which is the point.
