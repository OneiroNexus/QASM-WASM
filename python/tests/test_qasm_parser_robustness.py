"""
tests/test_qasm_parser_robustness.py -- pins the parser's behaviour on
malformed, unsupported and adversarial input, which VALIDATION.md
previously recorded as "unspecified, not just untested".

Added by the independent review, which probed the original parser and
found that it accepted every one of the inputs below silently and either
failed later inside call() with an unhelpful error ("negative shift
count"), or -- worse -- returned a wrong answer: `cx q[0],q[0]` was a
silent no-op, a third operand on cx was ignored, `h(0.3)` ignored its
parameter, a second qreg silently replaced the first, a reference to an
undeclared register was applied to the declared one, and the parameter
expression `9**9**9**9` hung the process (the whitelist admitted '**').

Every case below must now be REJECTED AT LOAD TIME with a SyntaxError
(or, for the expression bomb, a ValueError at evaluation time, in well
under a second) -- so that QasmModule.load() really is the parse-and-
validate-once step the WASM analogy promises, and so that a bad .qasm
file can never produce a plausible-looking wrong number.

Run: python3 -m tests.test_qasm_parser_robustness   (from the python/ directory)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from qmodule import QasmModule
from qmodule import statevector_sim as sv

CIRCUITS = Path(__file__).resolve().parent.parent / "qmodule" / "circuits"
HDR = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\n'
FAILURES = []


def check_true(name: str, cond: bool):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}")
    if not cond:
        FAILURES.append(name)


def rejected_at_load(text: str) -> tuple[bool, str]:
    try:
        sv.parse_qasm(text)
        return False, "accepted"
    except SyntaxError as e:
        return True, str(e)


def test_malformed_input_is_rejected_at_load():
    print("\n-- every malformed / unsupported construct is rejected by parse_qasm with a SyntaxError --")
    cases = [
        ("unknown gate name", HDR + "foo q[0];"),
        ("qubit index beyond register (h q[5])", HDR + "h q[5];"),
        ("qubit index == register size (h q[2] on qreg[2])", HDR + "h q[2];"),
        ("cx with identical control and target", HDR + "cx q[0],q[0];"),
        ("cx with three operands", HDR + "cx q[0],q[1],q[0];"),
        ("cx with one operand", HDR + "cx q[0];"),
        ("single-qubit gate with two operands", HDR + "h q[0],q[1];"),
        ("parameter on a non-parameterized gate (h(0.3))", HDR + "h(0.3) q[0];"),
        ("missing parameter on ry", HDR + "ry q[0];"),
        ("second qreg declaration", 'OPENQASM 2.0;\nqreg a[1];\nqreg b[3];\nh a[0];\n'),
        ("reference to an undeclared register", 'OPENQASM 2.0;\nqreg a[2];\ncreg c[2];\nh b[0];\n'),
        ("measure from an undeclared quantum register", HDR + "measure r[0] -> c[0];"),
        ("measure into an undeclared classical register", HDR + "measure q[0] -> zz[0];"),
        ("measure clbit index beyond creg", HDR + "measure q[0] -> c[7];"),
        ("measure qubit index == register size", HDR + "measure q[2] -> c[0];"),
        ("measure clbit index == creg size", HDR + "measure q[0] -> c[2];"),
        ("gate before any qreg", 'OPENQASM 2.0;\nh q[0];\nqreg q[1];\n'),
        ("qreg of size zero", 'OPENQASM 2.0;\nqreg q[0];\n'),
        ("empty file", ""),
        ("missing semicolon", HDR + "h q[0]"),
        ("operands without a comma", HDR + "cx q[0] q[1];"),
        ("custom gate definition", HDR + "gate mygate a { h a; }\nmygate q[0];"),
        ("opaque declaration", HDR + "opaque foo q;"),
        ("classical conditional", HDR + "if(c==1) x q[0];"),
        ("reset statement", HDR + "reset q[0];"),
    ]
    for name, text in cases:
        ok, msg = rejected_at_load(text)
        check_true(f"{name}: rejected at load ({msg[:60]}...)" if ok else f"{name}: NOT rejected at load", ok)


def test_parameter_expression_bomb_is_refused_quickly():
    print("\n-- the exponent bomb 9**9**9**9 is refused, not evaluated --")
    circ = sv.parse_qasm(HDR + "ry(9**9**9**9) q[0];")   # syntactically a parameter; the check is at evaluation
    t0 = time.time()
    refused = False
    try:
        sv.run_statevector(circ, {})
    except ValueError as e:
        refused = "exponentiation" in str(e)
    dt = time.time() - t0
    check_true(f"refused with a ValueError naming exponentiation, in {dt*1000:.1f} ms", refused and dt < 1.0)


def test_supported_constructs_still_accepted():
    print("\n-- the constructs the project relies on still load and run --")
    for name in ("bell.qasm", "ry_node_update.qasm"):
        mod = QasmModule.load(CIRCUITS / name)
        check_true(f"{name} loads (n_qubits={mod.n_qubits})", mod.n_qubits == 2)
    r = QasmModule.load(CIRCUITS / "ry_node_update.qasm").call({"a": 0.7, "b": 1.3, "phi": 0.4}, shots=10, seed=0)
    check_true("ry_node_update.qasm still evaluates to cos(a) after the parser changes", abs(r.expectation_z[0] - np.cos(0.7)) < 1e-12)
    circ = sv.parse_qasm(HDR + "h q[0];\nbarrier q[0],q[1];\ncx q[0],q[1];")
    check_true("barrier accepted and dropped (no-op)", [op.name for op in circ.ops] == ["h", "cx"])
    circ = sv.parse_qasm(HDR + "ry(pi/2) q[0];\nrz(2*theta) q[1];\nrx(spin) q[0];")
    vals = [sv._eval_param(op.param_expr, {"theta": 0.25, "spin": 1.5}) for op in circ.ops]
    check_true("pi/2, 2*theta and a parameter whose name contains 'pi' all evaluate correctly",
               abs(vals[0] - np.pi / 2) < 1e-12 and abs(vals[1] - 0.5) < 1e-12 and abs(vals[2] - 1.5) < 1e-12)
    check_true("spacing inside operands tolerated: 'cx q[ 0 ], q[1] ;'",
               [op.qubits for op in sv.parse_qasm(HDR + "cx q[ 0 ], q[1] ;").ops] == [[0, 1]])
    circ = sv.parse_qasm((CIRCUITS / "ry_node_update.qasm").read_text())
    kinds = [(op.name if isinstance(op, sv.GateOp) else f"measure {op.qubit}->{op.clbit}") for op in circ.ops]
    check_true("parsed op list is exactly [ry, ry, cx, rz, measure 0->0, measure 1->1]",
               kinds == ["ry", "ry", "cx", "rz", "measure 0->0", "measure 1->1"])
    check_true("creg size is recorded (creg c[3] with qreg q[2] -> n_clbits 3)",
               sv.parse_qasm('OPENQASM 2.0;\nqreg q[2];\ncreg c[3];\nh q[0];\n').n_clbits == 3)
    check_true("without a creg, n_clbits defaults to n_qubits",
               sv.parse_qasm('OPENQASM 2.0;\nqreg q[2];\nh q[0];\n').n_clbits == 2)
    mod = QasmModule.load(CIRCUITS / "ry_node_update.qasm")
    check_true("QasmModule records its source path and repr names it",
               mod.source_path.endswith("ry_node_update.qasm") and "ry_node_update.qasm" in repr(mod))
    try:
        mod.call(shots=1)
        unbound_ok = False
    except ValueError:
        unbound_ok = True
    except Exception:
        unbound_ok = False
    check_true("calling a parameterized circuit with no params raises ValueError (not TypeError)", unbound_ok)


if __name__ == "__main__":
    test_malformed_input_is_rejected_at_load()
    test_parameter_expression_bomb_is_refused_quickly()
    test_supported_constructs_still_accepted()
    print(f"\n{'ALL PASSED' if not FAILURES else f'{len(FAILURES)} FAILED: ' + ', '.join(FAILURES)}")
    sys.exit(1 if FAILURES else 0)
