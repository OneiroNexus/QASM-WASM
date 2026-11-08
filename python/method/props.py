"""
method/props.py -- a seeded, dependency-free property-testing harness.

VALIDATION.md's oldest open gap was "no property-based testing anywhere;
every scenario is fixed or seeded-random". Hypothesis would be the natural
tool, and it is not available in every environment this scaffold has to
run in (see AGENTS.md on dependencies), so this is the minimal thing that
gives the same discipline: a property is a function that returns a
failure message (or None), a generator draws inputs from a seeded RNG, and
`forall` runs the property over many draws and reports the FIRST failing
seed so the case can be replayed exactly.

There is no shrinking. The generators are written to produce small cases
by default, which is most of what shrinking buys.

Usage:
    from method.props import forall, Gen
    r = forall("scatter conserves mass", Gen.graph, lambda g: None if ok(g) else "mass lost", trials=300, seed=1)
    check_true(r.summary, r.ok)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


@dataclass
class Result:
    name: str
    trials: int
    failures: int
    first_failure: Optional[str] = None
    first_seed: Optional[int] = None
    seconds: float = 0.0
    notes: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.failures == 0

    @property
    def summary(self) -> str:
        s = f"{self.name}: {self.trials} trials, {self.failures} failures ({self.seconds:.2f}s)"
        if self.first_failure is not None:
            s += f" -- first failure at seed {self.first_seed}: {self.first_failure}"
        return s


def forall(name: str, gen: Callable[[np.random.Generator], object], prop: Callable[[object], Optional[str]],
           trials: int = 200, seed: int = 0) -> Result:
    """Runs `prop(gen(rng))` for `trials` independent seeds derived from
    `seed`. prop returns None on success or a message on failure; an
    exception counts as a failure with the exception text."""
    t0 = time.time()
    failures = 0
    first: Optional[str] = None
    first_seed: Optional[int] = None
    for i in range(trials):
        case_seed = seed * 1_000_003 + i
        rng = np.random.default_rng(case_seed)
        try:
            msg = prop(gen(rng))
        except Exception as e:  # noqa: BLE001 -- a property must never raise
            msg = f"{type(e).__name__}: {e}"
        if msg is not None:
            failures += 1
            if first is None:
                first, first_seed = msg, case_seed
    return Result(name=name, trials=trials, failures=failures, first_failure=first, first_seed=first_seed,
                  seconds=time.time() - t0)


class Gen:
    """Generators for this project's data. Each takes an np.random.Generator."""

    @staticmethod
    def graph(rng: np.random.Generator, max_nodes: int = 30, max_edges: int = 90) -> tuple[int, np.ndarray, np.ndarray]:
        n = int(rng.integers(1, max_nodes + 1))
        e = int(rng.integers(0, max_edges + 1))
        src = rng.integers(0, n, e).astype(np.int64)
        dst = rng.integers(0, n, e).astype(np.int64)
        if e and rng.random() < 0.3:      # self-loops
            k = int(rng.integers(1, e + 1))
            src[:k] = dst[:k]
        if e > 1 and rng.random() < 0.3:  # duplicate edges
            src[1] = src[0]
            dst[1] = dst[0]
        return n, src, dst

    @staticmethod
    def features(rng: np.random.Generator, n: int, max_f: int = 6) -> np.ndarray:
        f = int(rng.integers(1, max_f + 1))
        scale = float(rng.choice([1e-3, 1.0, 1e3]))
        return rng.normal(size=(n, f)) * scale

    @staticmethod
    def spike_mask(rng: np.random.Generator, n: int) -> np.ndarray:
        frac = float(rng.choice([0.0, 0.1, 0.5, 0.9, 1.0]))
        return (rng.random(n) < frac).astype(np.float64)

    @staticmethod
    def lif_params(rng: np.random.Generator):
        from snn.lif_neuron import LifParams
        tau = float(rng.uniform(1.0, 50.0))
        v_rest = float(rng.uniform(-1.0, 1.0))
        v_reset = float(rng.uniform(-1.0, v_rest + 0.5))
        v_th = float(v_rest + rng.uniform(0.2, 3.0))
        R = float(rng.uniform(0.2, 5.0))
        return LifParams(tau=tau, v_rest=v_rest, v_reset=v_reset, v_th=v_th, R=R)

    @staticmethod
    def circuit(rng: np.random.Generator, max_qubits: int = 3, max_gates: int = 8) -> tuple[str, dict]:
        """A random VALID circuit over the supported subset, plus bound parameters."""
        n = int(rng.integers(1, max_qubits + 1))
        lines = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{n}];", f"creg c[{n}];"]
        params: dict[str, float] = {}
        gates = ["h", "x", "y", "z", "s", "t", "rx", "ry", "rz", "cx", "barrier"]
        for _ in range(int(rng.integers(0, max_gates + 1))):
            g = str(rng.choice(gates))
            if g == "cx":
                if n < 2:
                    continue
                c, t = rng.choice(n, size=2, replace=False)
                lines.append(f"cx q[{c}],q[{t}];")
            elif g == "barrier":
                lines.append("barrier " + ",".join(f"q[{i}]" for i in range(n)) + ";")
            elif g in ("rx", "ry", "rz"):
                q = int(rng.integers(0, n))
                if rng.random() < 0.5:
                    name = f"p{len(params)}"
                    params[name] = float(rng.uniform(-12, 12))
                    lines.append(f"{g}({name}) q[{q}];")
                else:
                    lines.append(f"{g}({rng.uniform(-12, 12):.6f}) q[{q}];")
            else:
                lines.append(f"{g} q[{int(rng.integers(0, n))}];")
        if rng.random() < 0.7:
            for i in range(n):
                lines.append(f"measure q[{i}] -> c[{i}];")
        return "\n".join(lines) + "\n", params

    @staticmethod
    def malformed_line(rng: np.random.Generator) -> str:
        """A random line that is NOT valid in the supported subset, appended
        to a valid 2-qubit header."""
        n = 2
        bad = [
            f"foo q[0];", f"h q[{n}];", f"h q[{n + 5}];", "cx q[0],q[0];", "cx q[0];", "cx q[0],q[1],q[0];",
            "h q[0],q[1];", "h(0.5) q[0];", "ry q[0];", "h b[0];", "measure q[0] -> zz[0];", "measure q[0] -> c[9];",
            "qreg r[3];", "gate g a { h a; }", "opaque o q;", "if(c==1) x q[0];", "reset q[0];", "h q[0]",
            "cx q[0] q[1];", "rx(1..2) q[0];", "rx(theta*) q[0];", "rx(9**9**9) q[0];", "rx(__x__) q[0];",
            "h q[-1];", "h q[0]; h q[1];", "measure q[0] -> c[0]", "barrier;", "x;",
        ]
        alphabet = "hxyzstrc()[],;q0123 \tab*+-/."
        if rng.random() < 0.6:
            line = str(rng.choice(bad))
        else:
            line = "".join(rng.choice(list(alphabet)) for _ in range(int(rng.integers(1, 25))))
        return 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\n' + line + "\n"
