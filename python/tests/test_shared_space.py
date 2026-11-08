"""
Adversarial validation of fusion/shared_space.py. Same standard as the rest
of this project: exact closed-form identities wherever derivable (the
concatenation structure of fuse_to_4d, the spike-gating boundary cases, the
cos(a)/cos(a)*cos(b) closed form already proven in test_qasm_sim.py reused
here as an oracle for the fused pipeline), plus one honest statistical
check (the projection's unbiasedness) clearly labeled as statistical rather
than dressed up as exact.

Run: python3 -m tests.test_shared_space   (from the python/ directory)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from fusion import (
    NEVER_SPIKED,
    fuse_to_4d,
    make_projection,
    maybe_trigger_qgnn,
    project_to_3d,
    time_since_last_spike,
)
from qmodule import QasmModule

CIRCUITS = Path(__file__).resolve().parent.parent / "qmodule" / "circuits"

FAILURES = []


def check(name: str, got, want, tol: float):
    got_a, want_a = np.asarray(got, dtype=np.float64), np.asarray(want, dtype=np.float64)
    finite_ok = np.isinf(got_a) == np.isinf(want_a)  # inf must match inf in the same places
    with np.errstate(invalid="ignore"):
        diffs_ok = np.where(np.isinf(want_a), True, np.abs(got_a - want_a) <= tol)
    ok = bool(np.all(finite_ok) and np.all(diffs_ok)) and got_a.shape == want_a.shape
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name} (tol={tol})" + ("" if ok else f" got={got_a} want={want_a}"))
    if not ok:
        FAILURES.append(name)


def check_true(name: str, cond: bool):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}")
    if not cond:
        FAILURES.append(name)


def test_projection_is_linear():
    print("\n-- project_to_3d is an exact linear map: proj(a*x + b*y) == a*proj(x) + b*proj(y) --")
    rng = np.random.default_rng(10)
    f_in = 12
    proj = make_projection(f_in, seed=1)
    x = rng.normal(size=(1, f_in))
    y = rng.normal(size=(1, f_in))
    a, b = 2.3, -1.7
    lhs = project_to_3d(a * x + b * y, proj)
    rhs = a * project_to_3d(x, proj) + b * project_to_3d(y, proj)
    check("linearity of projection", lhs, rhs, tol=1e-10)


def test_random_projection_preserves_expected_norm():
    print("\n-- statistical check: E[||Px||^2] over the projection's OWN randomness equals ||x||^2 exactly --")
    print("   (this is the Johnson-Lindenstrauss-style unbiasedness the 1/sqrt(k) scaling is FOR; it is a")
    print("    Monte Carlo estimate, called out explicitly as statistical rather than an exact check)")
    f_in = 40
    rng = np.random.default_rng(123)
    x = rng.normal(size=f_in)
    want = float(np.sum(x ** 2))
    num_trials = 4000
    total = 0.0
    for trial in range(num_trials):
        proj = make_projection(f_in, seed=trial)
        px = project_to_3d(x[None, :], proj)[0]
        total += float(np.sum(px ** 2))
    got = total / num_trials
    rel_err = abs(got - want) / want
    print(f"  ||x||^2={want:.4f}  Monte Carlo E[||Px||^2] over {num_trials} projections={got:.4f}  rel_err={rel_err:.3%}")
    check_true("Monte Carlo E[||Px||^2] within 10% of exact ||x||^2", rel_err < 0.10)


def test_time_since_last_spike_exact():
    print("\n-- time_since_last_spike: exact arithmetic, including the never-spiked (+inf) case --")
    last_spike_time = np.array([3.0, 5.0, NEVER_SPIKED, 4.5])
    current_time = 5.0
    got = time_since_last_spike(last_spike_time, current_time)
    want = np.array([2.0, 0.0, np.inf, 0.5])
    check("elapsed times, incl. spike-on-this-tick == 0 and never-spiked == +inf", got, want, tol=1e-12)

    got_fallback = time_since_last_spike(last_spike_time, current_time, inf_fallback=999.0)
    want_fallback = np.array([2.0, 0.0, 999.0, 0.5])
    check("inf_fallback replaces only the +inf entry, others untouched", got_fallback, want_fallback, tol=1e-12)


def test_fuse_to_4d_is_exact_concatenation():
    print("\n-- fuse_to_4d does no independent computation: cols 0:3 == project_to_3d, col 3 == time_since_last_spike --")
    rng = np.random.default_rng(11)
    n, f_in = 8, 6
    embedding = rng.normal(size=(n, f_in))
    last_spike_time = rng.uniform(0.0, 4.0, size=n)
    last_spike_time[2] = NEVER_SPIKED
    current_time = 5.0
    proj = make_projection(f_in, seed=2)

    fused = fuse_to_4d(embedding, last_spike_time, current_time, proj)
    check_true("fused shape is (n, 4)", fused.shape == (n, 4))
    check("fused[:, :3] == project_to_3d(embedding)", fused[:, :3], project_to_3d(embedding, proj), tol=0.0)
    check("fused[:, 3] == time_since_last_spike(...)", fused[:, 3],
          time_since_last_spike(last_spike_time, current_time), tol=0.0)


def test_qgnn_trigger_gates_on_spikes_and_matches_closed_form():
    print("\n-- maybe_trigger_qgnn: fires only on spiking nodes; result matches the proven <Z0>=cos(a), <Z1>=cos(a)cos(b) closed form --")
    mod = QasmModule.load(CIRCUITS / "ry_node_update.qasm")
    rng = np.random.default_rng(12)
    n, f_in = 6, 4
    embedding = rng.uniform(-2.0, 2.0, size=(n, f_in))
    spiked = np.array([1, 0, 1, 0, 0, 1], dtype=np.float64)

    result = maybe_trigger_qgnn(embedding, spiked, mod, shots=1000, seed=0, blend=1.0)

    check("triggered_mask exactly equals spiked>0", result.triggered_mask, spiked > 0, tol=0.0)
    check_true("QASM called exactly once per spiking node", len(result.expectation_z) == int(spiked.sum()))

    for idx in range(n):
        a, b = embedding[idx, 0], embedding[idx, 1]
        if spiked[idx] > 0:
            want_z0, want_z1 = math.cos(a), math.cos(a) * math.cos(b)
            check(f"node {idx} (spiked): updated embedding[0] == cos(a) (exact closed form)",
                  result.updated_embedding[idx, 0], want_z0, tol=1e-9)
            check(f"node {idx} (spiked): updated embedding[1] == cos(a)*cos(b) (exact closed form)",
                  result.updated_embedding[idx, 1], want_z1, tol=1e-9)
            check(f"node {idx} (spiked): embedding[2] (phi) left untouched -- no Z-measurement can justify changing it",
                  result.updated_embedding[idx, 2], embedding[idx, 2], tol=0.0)
        else:
            check(f"node {idx} (did not spike): embedding row passed through bit-identical",
                  result.updated_embedding[idx], embedding[idx], tol=0.0)


def test_qgnn_trigger_blend_boundary_identities():
    print("\n-- blend=0.0 leaves embedding untouched even when triggered; blend=0.5 is exact linear interpolation --")
    mod = QasmModule.load(CIRCUITS / "ry_node_update.qasm")
    rng = np.random.default_rng(13)
    n, f_in = 4, 3
    embedding = rng.uniform(-1.5, 1.5, size=(n, f_in))
    spiked = np.ones(n)

    result0 = maybe_trigger_qgnn(embedding, spiked, mod, shots=500, seed=1, blend=0.0)
    check("blend=0.0: updated embedding exactly equals original, even though all nodes triggered",
          result0.updated_embedding, embedding, tol=0.0)

    result1 = maybe_trigger_qgnn(embedding, spiked, mod, shots=500, seed=1, blend=1.0)
    result_half = maybe_trigger_qgnn(embedding, spiked, mod, shots=500, seed=1, blend=0.5)
    want_half = 0.5 * embedding + 0.5 * result1.updated_embedding
    # phi (col 2) is never touched regardless of blend, so compare only cols 0,1 to the interpolation
    # and confirm col 2 separately.
    check("blend=0.5 cols[0:2] == exact midpoint of blend=0.0 and blend=1.0 results",
          result_half.updated_embedding[:, :2], want_half[:, :2], tol=1e-9)
    check("blend=0.5 col[2] (phi) untouched same as blend=0.0/1.0", result_half.updated_embedding[:, 2], embedding[:, 2], tol=0.0)


if __name__ == "__main__":
    test_projection_is_linear()
    test_random_projection_preserves_expected_norm()
    test_time_since_last_spike_exact()
    test_fuse_to_4d_is_exact_concatenation()
    test_qgnn_trigger_gates_on_spikes_and_matches_closed_form()
    test_qgnn_trigger_blend_boundary_identities()
    print(f"\n{'ALL PASSED' if not FAILURES else f'{len(FAILURES)} FAILED: ' + ', '.join(FAILURES)}")
    sys.exit(1 if FAILURES else 0)
