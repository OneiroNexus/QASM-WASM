"""
Adversarial validation of qmodule against hand-derived closed-form results —
not just "does it run," but "does it produce the exact number the math
predicts." Same standard as every other piece of this project.

Run: python3 -m tests.test_qasm_sim   (from the python/ directory)
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qmodule import QasmModule

CIRCUITS = Path(__file__).resolve().parent.parent / "qmodule" / "circuits"

FAILURES = []


def check(name: str, got: float, want: float, tol: float = 1e-9):
    ok = abs(got - want) <= tol
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: got={got!r} want={want!r} (tol={tol})")
    if not ok:
        FAILURES.append(name)


def check_true(name: str, cond: bool):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}")
    if not cond:
        FAILURES.append(name)


def check_close(name: str, got: float, want: float, tol: float):
    ok = abs(got - want) <= tol
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: got={got:.4f} want={want:.4f} (+/- {tol})")
    if not ok:
        FAILURES.append(name)


def test_bell_state_exact():
    print("\n-- bell.qasm: exact statevector amplitudes --")
    mod = QasmModule.load(CIRCUITS / "bell.qasm")
    result = mod.call(shots=20000, seed=1)
    # Exact expectation values from the full statevector (no sampling noise).
    check("Bell <Z0>", result.expectation_z[0], 0.0)
    check("Bell <Z1>", result.expectation_z[1], 0.0)
    print(f"  counts (20000 shots): {result.counts}")
    odd = result.counts.get("01", 0) + result.counts.get("10", 0)
    check("Bell P(01)+P(10) exact", 0.0 if odd == 0 else 1.0, 0.0)


def test_ry_node_update_closed_form():
    print("\n-- ry_node_update.qasm: <Z0>=cos(a), <Z1>=cos(a)*cos(b) --")
    mod = QasmModule.load(CIRCUITS / "ry_node_update.qasm")
    test_points = [
        (0.0, 0.0, 0.0),
        (0.7, 1.3, 0.4),
        (math.pi / 2, math.pi / 3, 1.1),
        (2.1, -0.9, 3.0),
    ]
    for a, b, phi in test_points:
        result = mod.call({"a": a, "b": b, "phi": phi}, shots=50000, seed=7)
        want_z0 = math.cos(a)
        want_z1 = math.cos(a) * math.cos(b)
        check(f"exact <Z0> a={a:.3f}", result.expectation_z[0], want_z0, tol=1e-9)
        check(f"exact <Z1> a={a:.3f} b={b:.3f}", result.expectation_z[1], want_z1, tol=1e-9)
        # Sampled (shot-based) estimate should be statistically close — this
        # is the number real hardware would actually hand you.
        check_close(f"sampled <Z0> a={a:.3f}", result.sampled_expectation_z[0], want_z0, tol=0.03)
        check_close(f"sampled <Z1> a={a:.3f} b={b:.3f}", result.sampled_expectation_z[1], want_z1, tol=0.03)


def test_sampled_expectation_is_exactly_the_count_average():
    print("\n-- sampled_expectation_z is exactly (N0 - N1)/shots recomputed from counts (not only statistically close) --")
    # The mutation campaign showed that seeding the accumulator at 1.0
    # instead of 0.0 (an error of 1/shots) was invisible to the statistical
    # tolerance. This recomputes the estimate from the returned counts.
    mod = QasmModule.load(CIRCUITS / "ry_node_update.qasm")
    result = mod.call({"a": 0.7, "b": 1.3, "phi": 0.4}, shots=777, seed=3)
    for q in (0, 1):
        n0 = sum(c for bits, c in result.counts.items() if bits[q] == "0")
        n1 = sum(c for bits, c in result.counts.items() if bits[q] == "1")
        check(f"sampled <Z{q}> == (N0 - N1)/shots from counts", result.sampled_expectation_z[q], (n0 - n1) / 777, tol=0.0)
    check("counts sum to shots", float(sum(result.counts.values())), 777.0, tol=0.0)


def test_sample_shots_renormalizes_an_unnormalized_state():
    print("\n-- sample_shots renormalizes: an unnormalized (1, 1) state samples both outcomes ~equally --")
    # Pins the documented 'renormalize against float drift' behaviour; the
    # mutation campaign showed p/total -> p*total was invisible on
    # normalized inputs.
    from qmodule import statevector_sim as sv
    counts = sv.sample_shots([1.0 + 0j, 1.0 + 0j], n=1, shots=4000, seed=5)
    check_true("both outcomes present", set(counts) == {"0", "1"})
    check_close("P(0) from an unnormalized state", counts["0"] / 4000, 0.5, tol=0.05)


def test_rz_invisible_to_z_measurement():
    print("\n-- confirms RZ(phi) provably cannot move <Z1> (phase is invisible to Z-basis measurement) --")
    mod = QasmModule.load(CIRCUITS / "ry_node_update.qasm")
    a, b = 0.9, -0.4
    z1_values = []
    for phi in (0.0, 1.0, 3.14, -2.0):
        result = mod.call({"a": a, "b": b, "phi": phi}, shots=1)  # shots irrelevant, only exact value checked
        z1_values.append(result.expectation_z[1])
    spread = max(z1_values) - min(z1_values)
    check("max spread of <Z1> across phi values", spread, 0.0, tol=1e-9)


if __name__ == "__main__":
    test_bell_state_exact()
    test_ry_node_update_closed_form()
    test_sampled_expectation_is_exactly_the_count_average()
    test_sample_shots_renormalizes_an_unnormalized_state()
    test_rz_invisible_to_z_measurement()
    print(f"\n{'ALL PASSED' if not FAILURES else f'{len(FAILURES)} FAILED: ' + ', '.join(FAILURES)}")
    sys.exit(1 if FAILURES else 0)
