"""
method/claims.py -- the claim registry: every property this scaffold is
validated for, which test checks it, by which mode, and since which
version. BUILD_INDEX.md is generated from this file plus the last suite run
and the last mutation campaign (method/build_index.py).

A claim that is not in this registry is not validated, whatever a docstring
says. A test that checks no registered claim is a smoke test. Keep both
directions honest: when you add a numerical property, add its claim here
and name the test; when you add a test, name the claim it serves.

Modes:
  closed-form          compared to a value derived by hand, independently of the code
  cross-implementation compared to an independently written implementation
  statistical          a Monte Carlo / shot-sampled quantity, reported with its error
  specification        a behavioural contract (rejection, error type, guard), pinned exactly
  generative           a closed-form or cross-implementation check over random inputs
Versions: v1.0 original build; v1.1 independent review; v1.2 the method systematized.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Claim:
    id: str
    layer: str
    statement: str
    mode: str
    tests: tuple
    since: str


CLAIMS = [
    # ---------------------------------------------------------------- qmodule
    Claim("Q1", "qmodule", "Bell state (h; cx): <Z0> = <Z1> = 0 exactly; outcomes 01/10 have exactly zero probability", "closed-form", ("test_qasm_sim::test_bell_state_exact",), "v1.0"),
    Claim("Q2", "qmodule", "ry(a) q0; ry(b) q1; cx q0,q1; rz(phi) q1 gives <Z0> = cos a and <Z1> = cos a cos b (exact, to 1e-9)", "closed-form", ("test_qasm_sim::test_ry_node_update_closed_form", "test_shared_space::test_qgnn_trigger_gates_on_spikes_and_matches_closed_form", "test_pipeline_smoke::test_full_pipeline_composes_correctly_end_to_end"), "v1.0"),
    Claim("Q3", "qmodule", "Shot-sampled expectation values track the exact ones within the statistical bound implied by the shot count", "statistical", ("test_qasm_sim::test_ry_node_update_closed_form", "test_properties::test_sampled_expectation_tracks_exact"), "v1.0"),
    Claim("Q4", "qmodule", "rz(phi) cannot change a Z-basis expectation value (null test) -- and rz is not the identity (positive test h;rz;h -> cos theta)", "closed-form", ("test_qasm_sim::test_rz_invisible_to_z_measurement", "test_qasm_gate_coverage::test_parameterized_gates_over_a_full_period"), "v1.1"),
    Claim("Q5", "qmodule", "Every gate (h,x,y,z,s,t,rx,ry,rz,cx) agrees with an independent Kronecker-product reference on the full complex statevector, phases included", "cross-implementation", ("test_qasm_gate_coverage::test_every_single_qubit_gate_matches_reference_statevector", "test_qasm_gate_coverage::test_parameterized_gates_over_a_full_period", "test_qasm_gate_coverage::test_ry_node_update_full_statevector_including_phases"), "v1.1"),
    Claim("Q6", "qmodule", "Three-qubit circuits, including cx with control index > target index, agree with the reference", "cross-implementation", ("test_qasm_gate_coverage::test_three_qubits_and_cx_orientation",), "v1.1"),
    Claim("Q7", "qmodule", "Random valid circuits over the supported subset agree with the reference; the state stays normalized", "generative", ("test_properties::test_random_circuits_match_reference",), "v1.2"),
    Claim("Q8", "qmodule", "Malformed or unsupported QASM is rejected at load() with a SyntaxError naming the line; nothing malformed is silently accepted", "specification", ("test_qasm_parser_robustness::test_malformed_input_is_rejected_at_load",), "v1.1"),
    Claim("Q9", "qmodule", "A parameter expression can only evaluate to a finite float or raise ValueError; exponentiation is refused; nothing hangs", "specification", ("test_qasm_parser_robustness::test_parameter_expression_bomb_is_refused_quickly", "test_properties::test_malformed_input_never_crashes_or_hangs"), "v1.2"),
    Claim("Q10", "qmodule", "Random malformed lines produce SyntaxError (parse) or ValueError (evaluation) and no other exception, in under a second", "generative", ("test_properties::test_malformed_input_never_crashes_or_hangs",), "v1.2"),
    Claim("Q11", "qmodule", "The same seed reproduces the same shot counts; counts sum to the shot count", "specification", ("test_properties::test_sampled_expectation_tracks_exact",), "v1.2"),
    Claim("Q12", "qmodule", "sampled_expectation_z is exactly (N0 - N1)/shots recomputed from the returned counts", "closed-form", ("test_qasm_sim::test_sampled_expectation_is_exactly_the_count_average",), "v1.2"),
    Claim("Q13", "qmodule", "Parameterized gates applied to non-basis states (after h, x, s, t, and on entangled 3-qubit states) agree with the reference on the full statevector", "cross-implementation", ("test_qasm_gate_coverage::test_parameterized_gates_on_non_basis_states",), "v1.2"),
    Claim("Q14", "qmodule", "sample_shots renormalizes an unnormalized input state; measure indices at the register boundary are rejected; n_clbits is recorded; a parameterized circuit called without params raises ValueError", "specification", ("test_qasm_sim::test_sample_shots_renormalizes_an_unnormalized_state", "test_qasm_parser_robustness::test_malformed_input_is_rejected_at_load", "test_qasm_parser_robustness::test_supported_constructs_still_accepted"), "v1.2"),
    # -------------------------------------------------------------------- snn
    Claim("S1", "snn", "Sub-threshold constant current never spikes and the analytic rate is exactly 0", "closed-form", ("test_lif_neuron::test_subthreshold_never_spikes", "test_properties::test_lif_random_params"), "v1.0"),
    Claim("S2", "snn", "Every spike of a supra-threshold neuron lands on the integer step predicted by the closed form of the discrete Euler recurrence", "closed-form", ("test_lif_neuron::test_discrete_spike_train_matches_closed_form", "test_properties::test_lif_random_params"), "v1.0"),
    Claim("S3", "snn", "The discrete inter-spike interval converges to the continuous analytic one as dt -> 0 (first-order)", "closed-form", ("test_lif_neuron::test_discrete_formula_converges_to_continuous_analytic_rate",), "v1.0"),
    Claim("S4", "snn", "The vectorized step treats neurons independently (batched == one-at-a-time, exactly)", "cross-implementation", ("test_lif_neuron::test_vectorized_step_matches_scalar_loop", "test_properties::test_lif_random_params"), "v1.0"),
    Claim("S5", "snn", "An input landing exactly on v_th spikes (the documented >=), in both engines, and resets to v_reset", "specification", ("test_composition_wiring::test_exact_threshold_hit_spikes_in_both_engines",), "v1.1"),
    Claim("S6", "snn", "analytic_rate agrees with the dt -> 0 limit of the discrete closed form for RANDOM v_rest, v_reset, v_th, R, tau (not only the defaults)", "generative", ("test_properties::test_lif_random_params",), "v1.2"),
    # -------------------------------------------------------------------- gnn
    Claim("G1", "gnn", "scatter_sum accumulates repeated destination indices (the out[index] += bug is avoided) and conserves total mass", "closed-form", ("test_gnn_message_passing::test_scatter_sum_repeated_index_bug", "test_gnn_message_passing::test_scatter_sum_mass_conservation", "test_properties::test_aggregation_random_graphs"), "v1.0"),
    Claim("G2", "gnn", "in_degree equals an independent Counter", "cross-implementation", ("test_gnn_message_passing::test_in_degree_matches_naive_counter", "test_properties::test_aggregation_random_graphs"), "v1.0"),
    Claim("G3", "gnn", "mean_aggregate equals a naive per-edge loop; zero-degree nodes get an exact zero; a constant feature averages to itself", "cross-implementation", ("test_gnn_message_passing::test_mean_aggregate_matches_naive_reference", "test_gnn_message_passing::test_mean_aggregate_zero_degree_is_exact_zero", "test_gnn_message_passing::test_mean_aggregate_constant_feature_invariance", "test_properties::test_aggregation_random_graphs"), "v1.0"),
    Claim("G4", "gnn", "gated_mean_aggregate reduces to mean_aggregate when all spike, to zero when none spike, and to the active-source edge subset otherwise", "closed-form", ("test_gnn_message_passing::test_gated_aggregate_boundary_identities", "test_properties::test_aggregation_random_graphs"), "v1.0"),
    Claim("G5", "gnn", "gnn_layer_forward is invariant to node relabeling (a function of structure and features only)", "closed-form", ("test_gnn_message_passing::test_permutation_invariance", "test_properties::test_layer_permutation_invariance_random"), "v1.0"),
    Claim("G6", "gnn", "Graph rejects out-of-range, boundary (id == num_nodes) and negative node ids, mixed and non-int64 dtypes, and length mismatches; num_edges counts duplicates", "specification", ("test_gnn_message_passing::test_graph_validates_inputs",), "v1.0"),
    Claim("G7", "gnn", "gnn_layer_forward's VALUE equals a naive per-node tanh(h W_self + mean(h) W_neigh + b) (not only its invariances)", "cross-implementation", ("test_gnn_message_passing::test_gnn_layer_forward_matches_naive_reference",), "v1.2"),
    # ----------------------------------------------------------------- fusion
    Claim("F1", "fusion", "project_to_3d is exactly linear", "closed-form", ("test_shared_space::test_projection_is_linear", "test_properties::test_fusion_random"), "v1.0"),
    Claim("F2", "fusion", "E[||Px||^2] = ||x||^2 over the projection's randomness (Johnson-Lindenstrauss scaling), measured", "statistical", ("test_shared_space::test_random_projection_preserves_expected_norm",), "v1.0"),
    Claim("F3", "fusion", "time_since_last_spike is exact, with +inf for never-spiked nodes and an optional finite fallback", "closed-form", ("test_shared_space::test_time_since_last_spike_exact", "test_properties::test_fusion_random"), "v1.0"),
    Claim("F4", "fusion", "fuse_to_4d is the exact concatenation of project_to_3d and time_since_last_spike", "closed-form", ("test_shared_space::test_fuse_to_4d_is_exact_concatenation", "test_properties::test_fusion_random"), "v1.0"),
    Claim("F5", "fusion", "maybe_trigger_qgnn fires exactly for the spike mask, writes the closed form into components 0,1, leaves component 2 and non-triggered rows untouched, and blends linearly", "closed-form", ("test_shared_space::test_qgnn_trigger_gates_on_spikes_and_matches_closed_form", "test_shared_space::test_qgnn_trigger_blend_boundary_identities", "test_properties::test_trigger_random"), "v1.0"),
    # ------------------------------------------------------------ orchestrator
    Claim("O1", "orchestrator", "The WASM core and the Python step agree bit-exactly on 25 neurons x 300 ticks of externally-fixed current", "cross-implementation", ("test_pipeline_smoke::test_wasm_orchestrator_matches_python_lif_reference",), "v1.0"),
    Claim("O2", "orchestrator", "The harness refuses an over-capacity neuron count with a clear error", "specification", ("test_pipeline_smoke::test_wasm_tick_rejects_over_capacity_n",), "v1.0"),
    Claim("O3", "orchestrator", "Every QGNN trigger in a 300-tick run re-derives from the closed form of its pre-tick state; non-triggered node-ticks pass through unchanged", "closed-form", ("test_pipeline_smoke::test_full_pipeline_composes_correctly_end_to_end",), "v1.0"),
    Claim("O4", "orchestrator", "The WASM and Python engines agree bit-exactly through the graph-coupled feedback loop", "cross-implementation", ("test_pipeline_smoke::test_pipeline_wasm_engine_matches_python_engine_under_feedback",), "v1.0"),
    Claim("O5", "orchestrator", "On every tick: current == baseline + coupling * feature-mean of the spike-gated in-neighbour mean (naive loop)", "cross-implementation", ("test_composition_wiring::test_every_wiring_point_rederived_per_tick", "test_properties::test_pipeline_invariants_random"), "v1.1"),
    Claim("O6", "orchestrator", "On every tick: the LIF step inside the loop equals an independent scalar rule; the trigger set equals the spike set", "cross-implementation", ("test_composition_wiring::test_every_wiring_point_rederived_per_tick", "test_properties::test_pipeline_invariants_random"), "v1.1"),
    Claim("O7", "orchestrator", "On every tick: last_spike_time follows from the spike log alone; fused output equals the post-update embedding times P and the elapsed time", "cross-implementation", ("test_composition_wiring::test_every_wiring_point_rederived_per_tick", "test_properties::test_pipeline_invariants_random"), "v1.1"),
    Claim("O8", "orchestrator", "wasm_lif_engine actually invokes the Node harness once per call", "specification", ("test_composition_wiring::test_wasm_engine_really_invokes_the_module",), "v1.1"),
    Claim("O9", "orchestrator", "Random small pipelines satisfy every per-tick wiring invariant", "generative", ("test_properties::test_pipeline_invariants_random",), "v1.2"),
    Claim("O10", "orchestrator", "A node triggered on consecutive ticks does not see identical shot noise (the sampling seed advances with the tick); sampled values are logged per tick", "generative", ("test_properties::test_shot_noise_advances_with_tick",), "v1.2"),
    Claim("O11", "orchestrator", "wasm_available() is False and wasm_lif_engine raises RuntimeError when the module binary is absent", "specification", ("test_wasm_core_bounds::test_engine_reports_unavailable_without_the_binary",), "v1.2"),
    # -------------------------------------------------------------- wasm-core
    Claim("W1", "wasm-core", "orc_tick refuses a neuron count outside [0, MAX_NEURONS] (returns -1) without touching memory", "specification", ("test_wasm_core_bounds::test_core_refuses_out_of_range_n_without_touching_memory",), "v1.1"),
    Claim("W2", "wasm-core", "At exactly MAX_NEURONS = 65,536 neurons the core is bit-exact against the Python reference", "cross-implementation", ("test_wasm_core_bounds::test_full_capacity_matches_python_reference",), "v1.1"),
    Claim("W3", "wasm-core", "orc_tick's return value equals the number of neurons that spiked, and exactly those are flagged in SPIKED", "specification", ("test_wasm_core_bounds::test_core_refuses_out_of_range_n_without_touching_memory",), "v1.2"),
]


def by_layer() -> dict[str, list[Claim]]:
    out: dict[str, list[Claim]] = {}
    for c in CLAIMS:
        out.setdefault(c.layer, []).append(c)
    return out
