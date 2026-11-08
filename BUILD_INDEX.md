> M2 update: the persistent runtime is implemented in software version 0.2.0. See [docs/M2_RUNTIME.md](docs/M2_RUNTIME.md) for its current contract, tests and benchmark. The baseline text and historical counts below remain preserved; they do not include M2 evidence.

> Historical scaffold document. Read [REVIEW_STATUS.md](REVIEW_STATUS.md) and Revision 4 for current corrections, evidence limits and rerun status. Original text follows.

# BUILD INDEX

Generated 2026-09-14 21:55:13 by `python3 -m method.build_index`. Do not edit by hand; edit `method/claims.py`, `method/triage.json`, the code, or the tests, and regenerate with `bash run_method.sh`.

**Suite:** 1286 checks passing, 0 failing, across 10 modules (run 2026-09-14 21:51:08).
**Mutation campaign:** 437/446 mutants killed (98.0%), 9 survived, 0 timed out, over 8 files (run 2026-09-14 21:53:12, 123.0s).
Survivor verdicts: accepted 6, config 3.
**Claims registered:** 46 (see section 3).

## 1. Source modules

| File | Purpose | Lines | Functions / classes | Mutants killed / valid | Survived |
|---|---|---:|---|---:|---:|
| `qmodule/statevector_sim.py` | OpenQASM 2.0 subset parser + statevector simulator | 330 | _u, GateOp, MeasureOp, ParsedCircuit, parse_qasm, _apply_single, _apply_cx, _eval_param, run_statevector, sample_shots, expectation_z | 227 / 231 | 4 |
| `qmodule/qasm_module.py` | QasmModule: load once / call many; QasmResult separates exact from sampled | 74 | QasmResult, QasmModule | 21 / 21 | 0 |
| `snn/lif_neuron.py` | Vectorized LIF step; continuous-time analytic rate | 74 | LifParams, analytic_rate, step, simulate_constant_current | 40 / 40 | 0 |
| `gnn/message_passing.py` | Sparse gather/scatter aggregation; gated variant; GraphSAGE-style layer | 137 | Graph, in_degree, scatter_sum, mean_aggregate, gated_mean_aggregate, GnnLayerParams, gnn_layer_forward | 37 / 37 | 0 |
| `fusion/shared_space.py` | 3D random projection + elapsed time; the QGNN trigger | 146 | ProjectionParams, make_projection, project_to_3d, time_since_last_spike, fuse_to_4d, QgnnTriggerResult, maybe_trigger_qgnn | 47 / 48 | 1 |
| `orchestrator/pipeline.py` | The per-tick composition; pluggable LifEngine | 155 | python_lif_engine, readout_scalar, PipelineState, TickLog, run_tick, run_pipeline | 36 / 39 | 3 |
| `orchestrator/wasm_bridge.py` | wasm_lif_engine: one harness subprocess per call | 68 | wasm_available, wasm_lif_engine | 17 / 18 | 1 |
| `wasm-core/orchestrator_core.c` | The LIF core in C, compiled to wasm32; bounds-guarded | 99 | orc_max_neurons, orc_v_ptr, orc_i_ptr, orc_spiked_ptr, orc_tick | 12 / 12 | 0 |

## 2. Test modules

| Module | Checks | Sections | Seconds | Claims served |
|---|---:|---:|---:|---|
| `test_composition_wiring.py` | 135 | 3 | 0.32 | O5, O6, O7, O8, S5 |
| `test_gnn_message_passing.py` | 52 | 11 | 0.12 | G1, G2, G3, G4, G5, G6, G7 |
| `test_lif_neuron.py` | 306 | 4 | 1.14 | S1, S2, S3, S4 |
| `test_pipeline_smoke.py` | 649 | 4 | 1.47 | O1, O2, O3, O4, Q2 |
| `test_properties.py` | 10 | 10 | 0.85 | F1, F3, F4, F5, G1, G2, G3, G4, G5, O5, O6, O7, O9, O10, Q3, Q7, Q9, Q10, Q11, S1, S2, S4, S6 |
| `test_qasm_gate_coverage.py` | 30 | 5 | 0.19 | Q4, Q5, Q6, Q13 |
| `test_qasm_parser_robustness.py` | 37 | 3 | 0.1 | Q8, Q9, Q14 |
| `test_qasm_sim.py` | 25 | 5 | 0.13 | Q1, Q2, Q3, Q4, Q12, Q14 |
| `test_shared_space.py` | 24 | 6 | 0.2 | F1, F2, F3, F4, F5, Q2 |
| `test_wasm_core_bounds.py` | 18 | 3 | 1.3 | O11, W1, W2, W3 |

## 3. Claims

Every validated property, the mode of validation, the test(s) that check it, and the version that added it. A property not listed here is not validated.

### qmodule

| Id | Claim | Mode | Checked by | Since |
|---|---|---|---|---|
| Q1 | Bell state (h; cx): <Z0> = <Z1> = 0 exactly; outcomes 01/10 have exactly zero probability | closed-form | `test_qasm_sim::test_bell_state_exact` | v1.0 |
| Q2 | ry(a) q0; ry(b) q1; cx q0,q1; rz(phi) q1 gives <Z0> = cos a and <Z1> = cos a cos b (exact, to 1e-9) | closed-form | `test_qasm_sim::test_ry_node_update_closed_form`<br>`test_shared_space::test_qgnn_trigger_gates_on_spikes_and_matches_closed_form`<br>`test_pipeline_smoke::test_full_pipeline_composes_correctly_end_to_end` | v1.0 |
| Q3 | Shot-sampled expectation values track the exact ones within the statistical bound implied by the shot count | statistical | `test_qasm_sim::test_ry_node_update_closed_form`<br>`test_properties::test_sampled_expectation_tracks_exact` | v1.0 |
| Q4 | rz(phi) cannot change a Z-basis expectation value (null test) -- and rz is not the identity (positive test h;rz;h -> cos theta) | closed-form | `test_qasm_sim::test_rz_invisible_to_z_measurement`<br>`test_qasm_gate_coverage::test_parameterized_gates_over_a_full_period` | v1.1 |
| Q5 | Every gate (h,x,y,z,s,t,rx,ry,rz,cx) agrees with an independent Kronecker-product reference on the full complex statevector, phases included | cross-implementation | `test_qasm_gate_coverage::test_every_single_qubit_gate_matches_reference_statevector`<br>`test_qasm_gate_coverage::test_parameterized_gates_over_a_full_period`<br>`test_qasm_gate_coverage::test_ry_node_update_full_statevector_including_phases` | v1.1 |
| Q6 | Three-qubit circuits, including cx with control index > target index, agree with the reference | cross-implementation | `test_qasm_gate_coverage::test_three_qubits_and_cx_orientation` | v1.1 |
| Q7 | Random valid circuits over the supported subset agree with the reference; the state stays normalized | generative | `test_properties::test_random_circuits_match_reference` | v1.2 |
| Q8 | Malformed or unsupported QASM is rejected at load() with a SyntaxError naming the line; nothing malformed is silently accepted | specification | `test_qasm_parser_robustness::test_malformed_input_is_rejected_at_load` | v1.1 |
| Q9 | A parameter expression can only evaluate to a finite float or raise ValueError; exponentiation is refused; nothing hangs | specification | `test_qasm_parser_robustness::test_parameter_expression_bomb_is_refused_quickly`<br>`test_properties::test_malformed_input_never_crashes_or_hangs` | v1.2 |
| Q10 | Random malformed lines produce SyntaxError (parse) or ValueError (evaluation) and no other exception, in under a second | generative | `test_properties::test_malformed_input_never_crashes_or_hangs` | v1.2 |
| Q11 | The same seed reproduces the same shot counts; counts sum to the shot count | specification | `test_properties::test_sampled_expectation_tracks_exact` | v1.2 |
| Q12 | sampled_expectation_z is exactly (N0 - N1)/shots recomputed from the returned counts | closed-form | `test_qasm_sim::test_sampled_expectation_is_exactly_the_count_average` | v1.2 |
| Q13 | Parameterized gates applied to non-basis states (after h, x, s, t, and on entangled 3-qubit states) agree with the reference on the full statevector | cross-implementation | `test_qasm_gate_coverage::test_parameterized_gates_on_non_basis_states` | v1.2 |
| Q14 | sample_shots renormalizes an unnormalized input state; measure indices at the register boundary are rejected; n_clbits is recorded; a parameterized circuit called without params raises ValueError | specification | `test_qasm_sim::test_sample_shots_renormalizes_an_unnormalized_state`<br>`test_qasm_parser_robustness::test_malformed_input_is_rejected_at_load`<br>`test_qasm_parser_robustness::test_supported_constructs_still_accepted` | v1.2 |

### snn

| Id | Claim | Mode | Checked by | Since |
|---|---|---|---|---|
| S1 | Sub-threshold constant current never spikes and the analytic rate is exactly 0 | closed-form | `test_lif_neuron::test_subthreshold_never_spikes`<br>`test_properties::test_lif_random_params` | v1.0 |
| S2 | Every spike of a supra-threshold neuron lands on the integer step predicted by the closed form of the discrete Euler recurrence | closed-form | `test_lif_neuron::test_discrete_spike_train_matches_closed_form`<br>`test_properties::test_lif_random_params` | v1.0 |
| S3 | The discrete inter-spike interval converges to the continuous analytic one as dt -> 0 (first-order) | closed-form | `test_lif_neuron::test_discrete_formula_converges_to_continuous_analytic_rate` | v1.0 |
| S4 | The vectorized step treats neurons independently (batched == one-at-a-time, exactly) | cross-implementation | `test_lif_neuron::test_vectorized_step_matches_scalar_loop`<br>`test_properties::test_lif_random_params` | v1.0 |
| S5 | An input landing exactly on v_th spikes (the documented >=), in both engines, and resets to v_reset | specification | `test_composition_wiring::test_exact_threshold_hit_spikes_in_both_engines` | v1.1 |
| S6 | analytic_rate agrees with the dt -> 0 limit of the discrete closed form for RANDOM v_rest, v_reset, v_th, R, tau (not only the defaults) | generative | `test_properties::test_lif_random_params` | v1.2 |

### gnn

| Id | Claim | Mode | Checked by | Since |
|---|---|---|---|---|
| G1 | scatter_sum accumulates repeated destination indices (the out[index] += bug is avoided) and conserves total mass | closed-form | `test_gnn_message_passing::test_scatter_sum_repeated_index_bug`<br>`test_gnn_message_passing::test_scatter_sum_mass_conservation`<br>`test_properties::test_aggregation_random_graphs` | v1.0 |
| G2 | in_degree equals an independent Counter | cross-implementation | `test_gnn_message_passing::test_in_degree_matches_naive_counter`<br>`test_properties::test_aggregation_random_graphs` | v1.0 |
| G3 | mean_aggregate equals a naive per-edge loop; zero-degree nodes get an exact zero; a constant feature averages to itself | cross-implementation | `test_gnn_message_passing::test_mean_aggregate_matches_naive_reference`<br>`test_gnn_message_passing::test_mean_aggregate_zero_degree_is_exact_zero`<br>`test_gnn_message_passing::test_mean_aggregate_constant_feature_invariance`<br>`test_properties::test_aggregation_random_graphs` | v1.0 |
| G4 | gated_mean_aggregate reduces to mean_aggregate when all spike, to zero when none spike, and to the active-source edge subset otherwise | closed-form | `test_gnn_message_passing::test_gated_aggregate_boundary_identities`<br>`test_properties::test_aggregation_random_graphs` | v1.0 |
| G5 | gnn_layer_forward is invariant to node relabeling (a function of structure and features only) | closed-form | `test_gnn_message_passing::test_permutation_invariance`<br>`test_properties::test_layer_permutation_invariance_random` | v1.0 |
| G6 | Graph rejects out-of-range, boundary (id == num_nodes) and negative node ids, mixed and non-int64 dtypes, and length mismatches; num_edges counts duplicates | specification | `test_gnn_message_passing::test_graph_validates_inputs` | v1.0 |
| G7 | gnn_layer_forward's VALUE equals a naive per-node tanh(h W_self + mean(h) W_neigh + b) (not only its invariances) | cross-implementation | `test_gnn_message_passing::test_gnn_layer_forward_matches_naive_reference` | v1.2 |

### fusion

| Id | Claim | Mode | Checked by | Since |
|---|---|---|---|---|
| F1 | project_to_3d is exactly linear | closed-form | `test_shared_space::test_projection_is_linear`<br>`test_properties::test_fusion_random` | v1.0 |
| F2 | E[||Px||^2] = ||x||^2 over the projection's randomness (Johnson-Lindenstrauss scaling), measured | statistical | `test_shared_space::test_random_projection_preserves_expected_norm` | v1.0 |
| F3 | time_since_last_spike is exact, with +inf for never-spiked nodes and an optional finite fallback | closed-form | `test_shared_space::test_time_since_last_spike_exact`<br>`test_properties::test_fusion_random` | v1.0 |
| F4 | fuse_to_4d is the exact concatenation of project_to_3d and time_since_last_spike | closed-form | `test_shared_space::test_fuse_to_4d_is_exact_concatenation`<br>`test_properties::test_fusion_random` | v1.0 |
| F5 | maybe_trigger_qgnn fires exactly for the spike mask, writes the closed form into components 0,1, leaves component 2 and non-triggered rows untouched, and blends linearly | closed-form | `test_shared_space::test_qgnn_trigger_gates_on_spikes_and_matches_closed_form`<br>`test_shared_space::test_qgnn_trigger_blend_boundary_identities`<br>`test_properties::test_trigger_random` | v1.0 |

### orchestrator

| Id | Claim | Mode | Checked by | Since |
|---|---|---|---|---|
| O1 | The WASM core and the Python step agree bit-exactly on 25 neurons x 300 ticks of externally-fixed current | cross-implementation | `test_pipeline_smoke::test_wasm_orchestrator_matches_python_lif_reference` | v1.0 |
| O2 | The harness refuses an over-capacity neuron count with a clear error | specification | `test_pipeline_smoke::test_wasm_tick_rejects_over_capacity_n` | v1.0 |
| O3 | Every QGNN trigger in a 300-tick run re-derives from the closed form of its pre-tick state; non-triggered node-ticks pass through unchanged | closed-form | `test_pipeline_smoke::test_full_pipeline_composes_correctly_end_to_end` | v1.0 |
| O4 | The WASM and Python engines agree bit-exactly through the graph-coupled feedback loop | cross-implementation | `test_pipeline_smoke::test_pipeline_wasm_engine_matches_python_engine_under_feedback` | v1.0 |
| O5 | On every tick: current == baseline + coupling * feature-mean of the spike-gated in-neighbour mean (naive loop) | cross-implementation | `test_composition_wiring::test_every_wiring_point_rederived_per_tick`<br>`test_properties::test_pipeline_invariants_random` | v1.1 |
| O6 | On every tick: the LIF step inside the loop equals an independent scalar rule; the trigger set equals the spike set | cross-implementation | `test_composition_wiring::test_every_wiring_point_rederived_per_tick`<br>`test_properties::test_pipeline_invariants_random` | v1.1 |
| O7 | On every tick: last_spike_time follows from the spike log alone; fused output equals the post-update embedding times P and the elapsed time | cross-implementation | `test_composition_wiring::test_every_wiring_point_rederived_per_tick`<br>`test_properties::test_pipeline_invariants_random` | v1.1 |
| O8 | wasm_lif_engine actually invokes the Node harness once per call | specification | `test_composition_wiring::test_wasm_engine_really_invokes_the_module` | v1.1 |
| O9 | Random small pipelines satisfy every per-tick wiring invariant | generative | `test_properties::test_pipeline_invariants_random` | v1.2 |
| O10 | A node triggered on consecutive ticks does not see identical shot noise (the sampling seed advances with the tick); sampled values are logged per tick | generative | `test_properties::test_shot_noise_advances_with_tick` | v1.2 |
| O11 | wasm_available() is False and wasm_lif_engine raises RuntimeError when the module binary is absent | specification | `test_wasm_core_bounds::test_engine_reports_unavailable_without_the_binary` | v1.2 |

### wasm-core

| Id | Claim | Mode | Checked by | Since |
|---|---|---|---|---|
| W1 | orc_tick refuses a neuron count outside [0, MAX_NEURONS] (returns -1) without touching memory | specification | `test_wasm_core_bounds::test_core_refuses_out_of_range_n_without_touching_memory` | v1.1 |
| W2 | At exactly MAX_NEURONS = 65,536 neurons the core is bit-exact against the Python reference | cross-implementation | `test_wasm_core_bounds::test_full_capacity_matches_python_reference` | v1.1 |
| W3 | orc_tick's return value equals the number of neurons that spiked, and exactly those are flagged in SPIKED | specification | `test_wasm_core_bounds::test_core_refuses_out_of_range_n_without_touching_memory` | v1.2 |

## 4. Mutation campaign

Operators: AOR (arithmetic), ROR (relational), LCR (logical), CRP (constants), SDL (statement deletion), AGS (argument swap); explicit textual mutants for the C core. One fresh copy of the tree per mutant; the test modules that transitively import the mutated file run fastest-first and stop at the first failure.

| File | Killed | Survived | Timeout | Invalid | Score |
|---|---:|---:|---:|---:|---:|
| `fusion/shared_space.py` | 47 | 1 | 0 | 0 | 97.9% |
| `gnn/message_passing.py` | 37 | 0 | 0 | 0 | 100.0% |
| `orchestrator/pipeline.py` | 36 | 3 | 0 | 0 | 92.3% |
| `orchestrator/wasm_bridge.py` | 17 | 1 | 0 | 0 | 94.4% |
| `qmodule/qasm_module.py` | 21 | 0 | 0 | 0 | 100.0% |
| `qmodule/statevector_sim.py` | 227 | 4 | 0 | 0 | 98.3% |
| `snn/lif_neuron.py` | 40 | 0 | 0 | 0 | 100.0% |
| `wasm-core/orchestrator_core.c` | 12 | 0 | 0 | 0 | 100.0% |

| Operator | Killed | Survived | Timeout | Invalid |
|---|---:|---:|---:|---:|
| AGS | 37 | 1 | 0 | 0 |
| AOR | 89 | 3 | 0 | 0 |
| CRP | 124 | 3 | 0 | 0 |
| LCR | 18 | 0 | 0 | 0 |
| ROR | 38 | 1 | 0 | 0 |
| SDL | 131 | 1 | 0 | 0 |

First test module to kill (fastest-first order, so this credits the cheapest sufficient test): `test_qasm_parser_robustness` 164, `test_composition_wiring` 73, `test_shared_space` 46, `test_qasm_gate_coverage` 45, `test_qasm_sim` 39, `test_gnn_message_passing` 37, `test_lif_neuron` 15, `test_properties` 14, `test_wasm_core_bounds` 4.

### Survivors and timeouts, with verdicts

| Id | File:line | Operator | Original | Mutant | Status | Verdict | Note |
|---|---|---|---|---|---|---|---|
| AOR-6d0b6d6bdc | `fusion/shared_space.py:140` | AOR | `seed + int(idx)` | `seed - int(idx)` | survived | accepted | fusion: seed + idx -> seed - idx. The specification is that distinct nodes get distinct shot-sampling seeds (and, via the pipeline, distinct ticks do too -- pinned by test_properties::P10). Any injective derivation satisfies it; pinning the arithmetic would over-specify a convention. |
| AOR-70ab1b0d8c | `orchestrator/pipeline.py:114` | AOR | `state.tick * 1_000_003` | `state.tick / 1000003` | survived | accepted | pipeline: tick * 1_000_003 -> tick / 1000003. Still tick-distinct (P10 holds); the specification is distinctness, not the formula. |
| AOR-e870f1e708 | `orchestrator/pipeline.py:114` | AOR | `qgnn_seed + state.tick * 1_000_003` | `qgnn_seed - state.tick * 1000003` | survived | accepted | pipeline: seed + tick*p -> seed - tick*p. Still tick-distinct (P10 holds). |
| CRP-d4f55b9b2d | `orchestrator/pipeline.py:114` | CRP | `1_000_003` | `1000004` | survived | accepted | pipeline: the prime 1_000_003 -> 1000004. Still tick-distinct for every toy-scale node count (P10 holds); the prime is a convenience, not a requirement the tests should pin. |
| CRP-e953ce3811 | `orchestrator/wasm_bridge.py:62` | CRP | `30` | `31` | survived | config | wasm_bridge: subprocess timeout 30 s -> 31 s. A configuration constant; no test should depend on it. |
| CRP-fed10ae8e4 | `qmodule/statevector_sim.py:254` | CRP | `200` | `201` | survived | config | statevector_sim: parameter-expression length limit 200 -> 201. A defensive limit; its exact value is configuration. |
| ROR-41ee2f28ed | `qmodule/statevector_sim.py:254` | ROR | `len(safe) > 200` | `len(safe) >= 200` | survived | config | statevector_sim: length limit > 200 vs >= 200. Same defensive limit; boundary of a configuration constant. |
| AGS-51b2622aea | `qmodule/statevector_sim.py:257` | AGS | `eval(safe, {"__builtins__": {}}, {})` | `eval(safe, {}, {'__builtins__': {}})` | survived | accepted | statevector_sim: eval(safe, {'__builtins__': {}}, {}) with globals/locals swapped. The character whitelist is the sandbox (no name can be spelled from [0-9eE+-*/(). ]); the empty __builtins__ is defense in depth and its removal is not observable through the whitelist. Recorded so a future widening of the whitelist re-examines it. |
| SDL-04e25c9f53 | `qmodule/statevector_sim.py:310` | SDL | `cumulative[-1] = 1.0` | `pass` | survived | accepted | statevector_sim: removing `cumulative[-1] = 1.0`. The line guards a ~1e-16 float-drift event (the last cumulative probability rounding to just under 1.0 while the random draw lands above it); not reachable by a test with any useful probability. Kept as a documented defensive line. |

## 5. How to regenerate

```bash
bash run_method.sh          # suite -> mutation campaign -> this index (about 3 minutes on 2 cores)
bash run_tests.sh           # the suite alone (seconds)
cd python && python3 -m method.mutate --only qmodule/statevector_sim.py   # one file
```

Verdict vocabulary for `method/triage.json`: **equivalent** (the mutant cannot change observable behaviour), **config** (a configuration constant, not logic), **accepted** (a behaviour change that no test should pin, with the reason), **gap** (a real blind spot: a test must be added, or the code refactored so the mutant cannot exist).
