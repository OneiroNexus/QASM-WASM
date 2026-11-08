"""
method/reference_sim.py -- an independent statevector simulator used ONLY as
a reference in tests. It shares no code with qmodule/statevector_sim.py:
it builds the full 2^n x 2^n unitary of every gate with numpy.kron from
textbook matrices (big-endian: qubit 0 is the leftmost tensor factor, the
convention statevector_sim.py documents) and multiplies it into the state.
Agreement between the two on the FULL complex statevector -- phases
included -- validates gate matrices, gate application, CX index handling
and endianness at once, which no Z-basis quantity alone can do.

Deliberately slow and obvious (dense unitaries): if the two simulators ever
disagree, the bug is far more likely to be in the clever one.
"""
from __future__ import annotations

import math

import numpy as np

from qmodule import statevector_sim as sv

I2 = np.eye(2, dtype=complex)

GATES = {
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


def single(n: int, q: int, M: np.ndarray) -> np.ndarray:
    out = np.array([[1.0 + 0j]])
    for k in range(n):
        out = np.kron(out, M if k == q else I2)
    return out


def cx(n: int, c: int, t: int) -> np.ndarray:
    P0 = np.array([[1, 0], [0, 0]], dtype=complex)
    P1 = np.array([[0, 0], [0, 1]], dtype=complex)
    X = GATES["x"]()
    a = np.array([[1.0 + 0j]])
    b = np.array([[1.0 + 0j]])
    for k in range(n):
        a = np.kron(a, P0 if k == c else I2)
        b = np.kron(b, P1 if k == c else (X if k == t else I2))
    return a + b


def run(circuit: sv.ParsedCircuit, params: dict | None = None) -> np.ndarray:
    """Final statevector of a parsed circuit, by dense unitary products."""
    params = params or {}
    n = circuit.n_qubits
    psi = np.zeros(2 ** n, dtype=complex)
    psi[0] = 1.0
    for op in circuit.ops:
        if isinstance(op, sv.MeasureOp):
            continue
        if op.name == "cx":
            psi = cx(n, op.qubits[0], op.qubits[1]) @ psi
        elif op.name in ("rx", "ry", "rz"):
            psi = single(n, op.qubits[0], GATES[op.name](sv._eval_param(op.param_expr, params))) @ psi
        else:
            psi = single(n, op.qubits[0], GATES[op.name]()) @ psi
    return psi


def zexp(psi: np.ndarray, n: int, q: int) -> float:
    p = np.abs(psi) ** 2
    return float(sum(p[i] * (1.0 if ((i >> (n - q - 1)) & 1) == 0 else -1.0) for i in range(len(p))))
