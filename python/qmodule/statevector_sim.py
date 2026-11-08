"""
statevector_sim.py — a from-scratch, dependency-free OpenQASM 2.0-subset
statevector simulator.

Why this exists instead of a dependency on qiskit/cirq/etc: this scaffold
treats a .qasm file as "a module" in the WASM sense — something you load
once and call many times with different parameters. That abstraction is
cleaner to reason about (and to hand to another coding agent) when the
runtime underneath it is small enough to read in one sitting. Swapping this
for a real hardware backend later means replacing QasmModule._simulate()
with a call to a cloud QPU/Aer backend — the QasmModule.call() contract
does not need to change.

CONVENTION: qubit 0 is the *leftmost* tensor factor, i.e. the state is
    |psi> = sum_x amp[x] |x_0 x_1 ... x_{n-1}>
with x_0 the most-significant bit of the integer index `x`. This matches
Nielsen & Chuang's textbook convention. It does NOT match Qiskit's
little-endian statevector ordering — if a future implementer swaps this
simulator for real Qiskit, they must account for that, and this comment
is the flag to do it.
"""
from __future__ import annotations

import bisect
import cmath
import math
import random
import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------- gates ---

def _u(theta: float, phi: float, lam: float) -> list[list[complex]]:
    """General single-qubit unitary U(theta, phi, lambda) — the OpenQASM 2.0
    primitive every other single-qubit gate is defined in terms of."""
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return [
        [c, -cmath.exp(1j * lam) * s],
        [cmath.exp(1j * phi) * s, cmath.exp(1j * (phi + lam)) * c],
    ]


_SQRT1_2 = 1 / math.sqrt(2)

GATE_MATRIX = {
    "h": lambda: [[_SQRT1_2, _SQRT1_2], [_SQRT1_2, -_SQRT1_2]],
    "x": lambda: [[0, 1], [1, 0]],
    "y": lambda: [[0, -1j], [1j, 0]],
    "z": lambda: [[1, 0], [0, -1]],
    "s": lambda: [[1, 0], [0, 1j]],
    "t": lambda: [[1, 0], [0, cmath.exp(1j * math.pi / 4)]],
    "rx": lambda theta: _u(theta, -math.pi / 2, math.pi / 2),
    "ry": lambda theta: _u(theta, 0, 0),
    "rz": lambda theta: [[cmath.exp(-1j * theta / 2), 0], [0, cmath.exp(1j * theta / 2)]],
}


# --------------------------------------------------------------- parser ---

@dataclass
class GateOp:
    name: str
    qubits: list[int]
    param_expr: str | None = None          # raw text, e.g. "theta" or "1.5707963"


@dataclass
class MeasureOp:
    qubit: int
    clbit: int


@dataclass
class ParsedCircuit:
    n_qubits: int
    n_clbits: int
    ops: list[GateOp | MeasureOp] = field(default_factory=list)


_GATE_LINE = re.compile(
    r"^\s*([a-zA-Z][a-zA-Z0-9_]*)\s*(?:\(([^)]*)\))?\s+(.+?);\s*$"
)
_QREG = re.compile(r"^\s*qreg\s+(\w+)\s*\[\s*(\d+)\s*\]\s*;\s*$")
_CREG = re.compile(r"^\s*creg\s+(\w+)\s*\[\s*(\d+)\s*\]\s*;\s*$")
_MEASURE = re.compile(r"^\s*measure\s+(\w+)\s*\[\s*(\d+)\s*\]\s*->\s*(\w+)\s*\[\s*(\d+)\s*\]\s*;\s*$")
_QUBIT_REF = re.compile(r"(\w+)\s*\[\s*(\d+)\s*\]")


_PARAMETERIZED = ("rx", "ry", "rz")


def parse_qasm(text: str) -> ParsedCircuit:
    """Parses the OpenQASM 2.0 subset this scaffold needs: ONE qreg and one
    creg declaration, single-qubit gates (h,x,y,z,s,t,rx,ry,rz), cx,
    barrier (accepted and ignored -- it is a no-op for a statevector) and
    measure. `include "qelib1.inc";` and comments are accepted and ignored.

    Everything the simulator cannot execute is rejected HERE, at load time,
    with a SyntaxError naming the offending line -- unknown gate names,
    qubit indices outside the declared register, wrong operand counts,
    `cx` with identical control and target, a parameter on a gate that
    takes none, a missing parameter on rx/ry/rz, a reference to a register
    that was not declared, or a second qreg. This is what makes
    QasmModule.load() the "parse and validate once" step the WASM analogy
    promises. (The independent review of this scaffold found that the
    original parser accepted all of the above silently and either failed
    at call() time with an unhelpful error or, worse, returned a wrong
    answer; see tests/test_qasm_parser_robustness.py.)

    Not supported, and rejected rather than approximated: custom `gate`
    definitions, `opaque`, `if`, `reset`, multiple quantum registers.

    Measurement semantics: `measure` lines are parsed and validated but
    call() always samples every qubit in the computational basis; the
    classical-register mapping is not modelled. This is a documented
    scaffold simplification, not an OpenQASM-conformant measurement."""
    n_qubits = n_clbits = 0
    qreg_name = creg_name = None
    ops: list[GateOp | MeasureOp] = []

    for raw in text.splitlines():
        line = raw.split("//", 1)[0].strip()
        if not line:
            continue
        if line.startswith(("OPENQASM", "include")):
            continue

        m = _QREG.match(line)
        if m:
            if qreg_name is not None:
                raise SyntaxError(f"second qreg declaration; this simulator supports one quantum register: {raw!r}")
            qreg_name, n_qubits = m.group(1), int(m.group(2))
            if n_qubits == 0:
                raise SyntaxError(f"qreg of size 0: {raw!r}")
            continue
        m = _CREG.match(line)
        if m:
            creg_name, n_clbits = m.group(1), int(m.group(2))
            continue
        if qreg_name is None:
            raise SyntaxError(f"statement before any `qreg` declaration: {raw!r}")

        m = _MEASURE.match(line)
        if m:
            if m.group(1) != qreg_name:
                raise SyntaxError(f"measure references undeclared quantum register {m.group(1)!r}: {raw!r}")
            if creg_name is None or m.group(3) != creg_name:
                raise SyntaxError(f"measure references undeclared classical register {m.group(3)!r}: {raw!r}")
            q, c = int(m.group(2)), int(m.group(4))
            if q >= n_qubits:
                raise SyntaxError(f"qubit index {q} outside qreg {qreg_name}[{n_qubits}]: {raw!r}")
            if c >= n_clbits:
                raise SyntaxError(f"clbit index {c} outside creg {creg_name}[{n_clbits}]: {raw!r}")
            ops.append(MeasureOp(qubit=q, clbit=c))
            continue

        m = _GATE_LINE.match(line)
        if not m:
            raise SyntaxError(f"could not parse QASM line: {raw!r}")
        name, param_expr, qargs_text = m.group(1).lower(), m.group(2), m.group(3)
        refs = _QUBIT_REF.findall(qargs_text)
        if not refs or "".join(qargs_text.split()) != ",".join(f"{r}[{q}]" for r, q in refs):
            raise SyntaxError(f"malformed qubit operand list: {raw!r}")
        for reg, _ in refs:
            if reg != qreg_name:
                raise SyntaxError(f"reference to undeclared register {reg!r}: {raw!r}")
        qubits = [int(q) for _, q in refs]
        for q in qubits:
            if q >= n_qubits:
                raise SyntaxError(f"qubit index {q} outside qreg {qreg_name}[{n_qubits}]: {raw!r}")

        if name == "barrier":
            continue  # no-op for a statevector simulation
        if name == "cx":
            if len(qubits) != 2:
                raise SyntaxError(f"cx takes exactly two qubit operands: {raw!r}")
            if qubits[0] == qubits[1]:
                raise SyntaxError(f"cx control and target must differ: {raw!r}")
            if param_expr is not None:
                raise SyntaxError(f"cx takes no parameter: {raw!r}")
        elif name in GATE_MATRIX:
            if len(qubits) != 1:
                raise SyntaxError(f"{name} takes exactly one qubit operand: {raw!r}")
            if name in _PARAMETERIZED and param_expr is None:
                raise SyntaxError(f"{name} requires a parameter: {raw!r}")
            if name not in _PARAMETERIZED and param_expr is not None:
                raise SyntaxError(f"{name} takes no parameter: {raw!r}")
        else:
            raise SyntaxError(f"unsupported gate {name!r} (supported: {', '.join(sorted(GATE_MATRIX))}, cx, barrier): {raw!r}")
        ops.append(GateOp(name=name, qubits=qubits, param_expr=param_expr))

    if n_qubits == 0:
        raise SyntaxError("no `qreg` declaration found")
    return ParsedCircuit(n_qubits=n_qubits, n_clbits=n_clbits or n_qubits, ops=ops)


# ------------------------------------------------------------ simulator ---

def _apply_single(state: list[complex], n: int, qubit: int, mat: list[list[complex]]) -> list[complex]:
    """Applies a 2x2 unitary to one qubit of an n-qubit statevector.
    `qubit` counts from the left (see module docstring)."""
    dim = 1 << n
    stride = 1 << (n - qubit - 1)          # distance between the |...0...> and |...1...> partners
    out = state[:]
    block = stride * 2
    for base in range(0, dim, block):
        for off in range(stride):
            i0, i1 = base + off, base + off + stride
            a0, a1 = state[i0], state[i1]
            out[i0] = mat[0][0] * a0 + mat[0][1] * a1
            out[i1] = mat[1][0] * a0 + mat[1][1] * a1
    return out


def _apply_cx(state: list[complex], n: int, control: int, target: int) -> list[complex]:
    dim = 1 << n
    c_bit = 1 << (n - control - 1)
    t_bit = 1 << (n - target - 1)
    out = state[:]
    for i in range(dim):
        # Visit each swapped pair once, from its member with the target bit
        # clear (control set, target clear) -- there is no second visit to
        # guard against, so no `if i < j` and no site for a mutant to hide.
        if (i & c_bit) and not (i & t_bit):
            j = i | t_bit
            out[i], out[j] = state[j], state[i]
    return out


def _eval_param(expr: str | None, params: dict[str, float]) -> float | None:
    if expr is None:
        return None
    try:
        return float(expr)
    except ValueError:
        pass
    if expr in params:
        return float(params[expr])
    # tiny expression support: "pi/4", "2*theta", etc., with `pi` and bound
    # parameter names as the only identifiers allowed — deliberately not a
    # general eval() to keep an externally-supplied .qasm file inert.
    safe = re.sub(r"\bpi\b", repr(math.pi), expr)
    for k, v in params.items():
        safe = re.sub(rf"\b{re.escape(k)}\b", repr(float(v)), safe)
    if not re.fullmatch(r"[0-9eE+\-*/(). ]+", safe):
        raise ValueError(f"unbound or unsafe parameter expression: {expr!r}")
    if "**" in safe:
        # The character whitelist above admits '*', and therefore '**'. Left
        # unchecked, `ry(9**9**9**9)` in an externally supplied .qasm file
        # makes call() hang while Python builds a number with ~10^9 digits
        # (found by the independent review; see tests/test_qasm_parser_robustness.py).
        raise ValueError(f"exponentiation is not allowed in parameter expressions: {expr!r}")
    if len(safe) > 200:
        raise ValueError(f"parameter expression too long: {expr!r}")
    try:
        value = float(eval(safe, {"__builtins__": {}}, {}))  # noqa: S307 — sandboxed by the checks above
    except (SyntaxError, ZeroDivisionError, OverflowError, TypeError, ValueError) as e:
        # e.g. "1..2" (eval's own SyntaxError), "1/0", "(1" -- found by the
        # generative test tests/test_properties.py::P2, which requires that a
        # bad parameter surfaces as ValueError and nothing else.
        raise ValueError(f"malformed parameter expression {expr!r}: {e}") from None
    if not math.isfinite(value):
        raise ValueError(f"parameter expression is not finite: {expr!r}")
    return value


def run_statevector(circuit: ParsedCircuit, params: dict[str, float] | None = None) -> list[complex]:
    """Runs the circuit exactly (no sampling) and returns the final
    statevector — the ground truth every shot-based measurement below is
    checked against."""
    if params is None:
        params = {}
    n = circuit.n_qubits
    state = [0j] * (1 << n)
    state[0] = complex(1.0)
    for op in circuit.ops:
        if isinstance(op, MeasureOp):
            continue  # measurement handled separately for shot sampling
        if op.name == "cx":
            state = _apply_cx(state, n, op.qubits[0], op.qubits[1])
            continue
        if op.name not in GATE_MATRIX:
            raise NotImplementedError(f"unsupported gate: {op.name}")
        fn = GATE_MATRIX[op.name]
        if op.name in ("rx", "ry", "rz"):
            theta = _eval_param(op.param_expr, params)
            if theta is None:
                raise ValueError(f"gate {op.name} requires a parameter")
            mat = fn(theta)
        else:
            mat = fn()
        state = _apply_single(state, n, op.qubits[0], mat)
    return state


def sample_shots(state: list[complex], n: int, shots: int, seed: int | None = None) -> dict[str, int]:
    """Samples `shots` computational-basis measurements from the exact
    statevector — this is the statistical step real quantum hardware (or a
    faithful simulator) actually performs, and the reason a QASM 'call'
    returns a distribution rather than a single value (see qasm_module.py)."""
    rng = random.Random(seed)
    probs = [abs(a) ** 2 for a in state]
    total = sum(probs)
    cumulative = []
    acc = 0.0
    for p in probs:
        acc += p / total                    # renormalized against float drift
        cumulative.append(acc)
    cumulative[-1] = 1.0                    # r in [0, 1) is then always < cumulative[-1], so bisect never runs off the end
    counts: dict[str, int] = {}
    for _ in range(shots):
        # Inverse-CDF sampling via the standard library's bisect: the
        # original hand-written binary search had six mutable sites, every
        # one of which turned a mutant into an infinite loop.
        idx = bisect.bisect_right(cumulative, rng.random())
        bitstring = format(idx, f"0{n}b")
        counts[bitstring] = counts.get(bitstring, 0) + 1
    return counts


def expectation_z(state: list[complex], n: int, qubit: int) -> float:
    """Exact <Z_qubit> from the statevector (not sampled) — used as the
    analytic ground truth in tests/test_qasm_sim.py."""
    val = 0.0
    for i, amp in enumerate(state):
        bit = (i >> (n - qubit - 1)) & 1
        p = abs(amp) ** 2
        val += p if bit == 0 else -p
    return val
