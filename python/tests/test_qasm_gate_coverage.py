"""
tests/test_qasm_gate_coverage.py -- closes the gate-coverage gap recorded in
VALIDATION.md: x, y, z, s, t and rx had no check of their own, and rz's only
check was a null test (that it does NOT move <Z1>), which an rz implemented
as the identity would also pass (mutation M10 in the review survived the
original suite for exactly that reason).

Method: an independent reference simulator that shares no code with
qmodule/statevector_sim.py -- it builds the full 2^n x 2^n unitary with
numpy.kron from textbook matrices (big-endian, qubit 0 = leftmost factor,
the convention statevector_sim.py documents) and multiplies it into the
state. The scaffold's simulator applies gates with a stride-based in-place
update on a Python list, so agreement between the two on the FULL complex
statevector -- phases included, not only Z-basis probabilities -- validates
gate matrices, gate application, CX index handling and endianness at once.

Run: python3 -m tests.test_qasm_gate_coverage   (from the python/ directory)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from qmodule import statevector_sim as sv

CIRCUITS = Path(__file__).resolve().parent.parent / "qmodule" / "circuits"
FAILURES = []


def check(name: str, got: float, want: float, tol: float):
    got, want = float(got), float(want)
    ok = abs(got - want) <= tol
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: got={got!r} want={want!r} (tol={tol})")
    if not ok:
        FAILURES.append(name)


# ---- reference: textbook matrices, typed independently ----------------------
I2 = np.eye(2, dtype=complex)
REF = {
    "h": lambda: np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2),
    "x": lambda: np.array([[0, 1], [1, 0]], dtype=complex),
    "y": lambda: np.array([[0, -1j], [1j, 0]]),
    "z": lambda: np.array([[1, 0], [0, -1]], dtype=complex),
    "s": lambda: np.array([[1, 0], [0, 1j]]),
    "t": lambda: np.array([[1, 0], [0, np.exp(1j * math.pi / 4)]]),
    "rx": lambda th: np.array([[math.cos(th / 2), -1j * math.sin(th / 2)], [-1j * math.sin(th / 2), math.cos(th / 2)]]),
    "ry": lambda th: np.array([[math.cos(th / 2), -math.sin(th / 2)], [math.sin(th / 2), math.cos(th / 2)]]),
    "rz": lambda th: np.array([[np.exp(-1j * th / 2), 0], [0, np.exp(1j * th / 2)]]),
}


def ref_single(n: int, q: int, M: np.ndarray) -> np.ndarray:
    out = np.array([[1.0 + 0j]])
    for k in range(n):
        out = np.kron(out, M if k == q else I2)
    return out


def ref_cx(n: int, c: int, t: int) -> np.ndarray:
    P0 = np.array([[1, 0], [0, 0]], dtype=complex)
    P1 = np.array([[0, 0], [0, 1]], dtype=complex)
    X = REF["x"]()
    a = np.array([[1.0 + 0j]])
    b = np.array([[1.0 + 0j]])
    for k in range(n):
        a = np.kron(a, P0 if k == c else I2)
        b = np.kron(b, P1 if k == c else (X if k == t else I2))
    return a + b


def ref_run(circuit: sv.ParsedCircuit, params: dict) -> np.ndarray:
    n = circuit.n_qubits
    psi = np.zeros(2 ** n, dtype=complex)
    psi[0] = 1.0
    for op in circuit.ops:
        if isinstance(op, sv.MeasureOp):
            continue
        if op.name == "cx":
            psi = ref_cx(n, op.qubits[0], op.qubits[1]) @ psi
        elif op.name in ("rx", "ry", "rz"):
            psi = ref_single(n, op.qubits[0], REF[op.name](sv._eval_param(op.param_expr, params))) @ psi
        else:
            psi = ref_single(n, op.qubits[0], REF[op.name]()) @ psi
    return psi


def ref_zexp(psi: np.ndarray, n: int, q: int) -> float:
    p = np.abs(psi) ** 2
    return float(sum(p[i] * (1.0 if ((i >> (n - q - 1)) & 1) == 0 else -1.0) for i in range(len(p))))


def max_amp_diff(text: str, params: dict | None = None) -> float:
    circ = sv.parse_qasm(text)
    got = np.array(sv.run_statevector(circ, params))
    want = ref_run(circ, params or {})
    return float(np.max(np.abs(got - want)))


HDR1 = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\ncreg c[1];\n'
HDR3 = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\ncreg c[3];\n'


def test_every_single_qubit_gate_matches_reference_statevector():
    print("\n-- each of the 9 single-qubit gates, alone and inside h-G-h (which makes phase gates visible to Z) --")
    for g in ("h", "x", "y", "z", "s", "t"):
        check(f"{g}: |amp diff| vs kron reference", max_amp_diff(HDR1 + f"{g} q[0];"), 0.0, tol=1e-12)
        check(f"h;{g};h: |amp diff| vs kron reference", max_amp_diff(HDR1 + f"h q[0];\n{g} q[0];\nh q[0];"), 0.0, tol=1e-12)
    # Closed forms a Z-basis measurement CAN see, derived by hand:
    #   h z h = x            -> <Z> = -1
    #   h s h                -> <Z> = cos(pi/2) = 0
    #   h t h                -> <Z> = cos(pi/4)
    for g, want in (("z", -1.0), ("s", 0.0), ("t", math.cos(math.pi / 4))):
        psi = np.array(sv.run_statevector(sv.parse_qasm(HDR1 + f"h q[0];\n{g} q[0];\nh q[0];")))
        check(f"h;{g};h: <Z> closed form", ref_zexp(psi, 1, 0), want, tol=1e-12)


def test_parameterized_gates_over_a_full_period():
    print("\n-- rx, ry, rz swept over [-2pi, 2pi]: statevector vs reference, plus closed forms --")
    thetas = np.linspace(-2 * math.pi, 2 * math.pi, 41)
    worst = {"rx": 0.0, "ry": 0.0, "rz": 0.0, "rx<Z>": 0.0, "ry<Z>": 0.0, "h;rz;h<Z>": 0.0}
    for th in thetas:
        prm = {"theta": float(th)}
        for g in ("rx", "ry", "rz"):
            worst[g] = max(worst[g], max_amp_diff(HDR1 + f"{g}(theta) q[0];", prm))
        # <Z> after rx(theta) or ry(theta) on |0> is cos(theta); after h;rz(theta);h it is cos(theta) too.
        for g in ("rx", "ry"):
            psi = np.array(sv.run_statevector(sv.parse_qasm(HDR1 + f"{g}(theta) q[0];"), prm))
            worst[g + "<Z>"] = max(worst[g + "<Z>"], abs(ref_zexp(psi, 1, 0) - math.cos(th)))
        psi = np.array(sv.run_statevector(sv.parse_qasm(HDR1 + "h q[0];\nrz(theta) q[0];\nh q[0];"), prm))
        worst["h;rz;h<Z>"] = max(worst["h;rz;h<Z>"], abs(ref_zexp(psi, 1, 0) - math.cos(th)))
    for k, v in worst.items():
        check(f"{k}: worst over 41 angles", v, 0.0, tol=1e-12)


def test_parameterized_gates_on_non_basis_states():
    print("\n-- rx, ry, rz applied AFTER h / x / s (a state with a |1> component): full statevector vs reference --")
    # The mutation campaign showed that every parameterized gate in the
    # suite was applied to |0>, so mutants of the matrix's second column
    # (e.g. U(theta, phi, lambda) with lambda replaced) were invisible.
    thetas = (-2.3, -0.4, 0.9, 2.8)
    worst = 0.0
    for prep in ("h q[0];", "x q[0];", "h q[0];\ns q[0];", "x q[0];\nh q[0];\nt q[0];"):
        for g in ("rx", "ry", "rz"):
            for th in thetas:
                worst = max(worst, max_amp_diff(HDR1 + prep + f"\n{g}(theta) q[0];", {"theta": th}))
    check("worst |amp diff| over 4 preparations x 3 gates x 4 angles", worst, 0.0, tol=1e-12)
    # and on a qubit that is not the last one (stride > 1), after entangling
    worst = 0.0
    for th in thetas:
        worst = max(worst, max_amp_diff(HDR3 + "h q[0];\ncx q[0],q[1];\nry(theta) q[0];\nrx(theta) q[1];\nrz(theta) q[2];\ncx q[2],q[0];", {"theta": th}))
    check("worst |amp diff| for parameterized gates on entangled 3-qubit states", worst, 0.0, tol=1e-12)


def test_ry_node_update_full_statevector_including_phases():
    print("\n-- ry_node_update.qasm: FULL complex statevector (not only <Z>) vs reference over a parameter grid --")
    text = (CIRCUITS / "ry_node_update.qasm").read_text()
    rng = np.random.default_rng(2026)
    worst = 0.0
    pts = [(a, b, phi) for a in np.linspace(-math.pi, math.pi, 7) for b in np.linspace(-math.pi, math.pi, 7) for phi in (0.0, 0.7, -2.4)]
    pts += [tuple(map(float, rng.uniform(-10, 10, 3))) for _ in range(100)]
    for a, b, phi in pts:
        worst = max(worst, max_amp_diff(text, {"a": a, "b": b, "phi": phi}))
    check(f"worst |amp diff| over {len(pts)} (a,b,phi) points incl. unbounded embeddings", worst, 0.0, tol=1e-12)


def test_three_qubits_and_cx_orientation():
    print("\n-- three qubits: GHZ, CX with control index > target index, mixed circuit --")
    check("GHZ(3) |amp diff|", max_amp_diff(HDR3 + "h q[0];\ncx q[0],q[1];\ncx q[1],q[2];"), 0.0, tol=1e-12)
    psi = np.array(sv.run_statevector(sv.parse_qasm(HDR3 + "h q[0];\ncx q[0],q[1];\ncx q[1],q[2];")))
    check("GHZ(3) P(000)", abs(psi[0]) ** 2, 0.5, tol=1e-12)
    check("GHZ(3) P(111)", abs(psi[7]) ** 2, 0.5, tol=1e-12)
    check("cx q[2],q[0] on |001> -> |101| (control to the RIGHT of target)",
          max_amp_diff(HDR3 + "x q[2];\ncx q[2],q[0];"), 0.0, tol=1e-12)
    psi = np.array(sv.run_statevector(sv.parse_qasm(HDR3 + "x q[2];\ncx q[2],q[0];")))
    check("...and the populated basis index is 5 (= 0b101)", float(int(np.argmax(np.abs(psi)))), 5.0, tol=0.0)
    check("mixed 3-qubit circuit |amp diff|",
          max_amp_diff(HDR3 + "ry(a) q[1];\ncx q[1],q[2];\nrz(p) q[2];\ncx q[0],q[1];\nt q[0];", {"a": 0.9, "p": 1.3}), 0.0, tol=1e-12)


if __name__ == "__main__":
    test_every_single_qubit_gate_matches_reference_statevector()
    test_parameterized_gates_over_a_full_period()
    test_parameterized_gates_on_non_basis_states()
    test_ry_node_update_full_statevector_including_phases()
    test_three_qubits_and_cx_orientation()
    print(f"\n{'ALL PASSED' if not FAILURES else f'{len(FAILURES)} FAILED: ' + ', '.join(FAILURES)}")
    sys.exit(1 if FAILURES else 0)
