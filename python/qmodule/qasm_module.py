"""
qasm_module.py — the "treat a .qasm file like a .wasm module" interface.

    module = QasmModule.load("circuits/ry_node_update.qasm")   # instantiate once
    result = module.call({"theta": 0.7}, shots=2000)            # call many times
    result.expectation_z[0]                                     # -> ~cos(0.7)

This is deliberately shaped like a WebAssembly runtime's instantiate/call
split: parsing and validating the circuit happens once at load() time
(the expensive, do-it-once step, analogous to WASM module compilation),
and call() is the cheap, repeatable part (analogous to invoking an
exported function) — except see WHERE THE ANALOGY BREAKS below.

WHERE THE ANALOGY BREAKS, ON PURPOSE:
A WASM export call is deterministic: same inputs, same output, every time.
A quantum circuit is not — executing it once yields one sampled bitstring
drawn from a probability distribution, not "the answer." call() therefore
does not return a single value; it returns a QasmResult holding the raw
shot counts and the derived expectation values, and takes a `shots`
parameter that has no WASM equivalent at all. Any scaffolding that hides
this distinction (e.g. by having call() silently return only the most
common outcome) would be lying about what quantum execution actually is.
Keep that shape when this is later backed by a real QPU or Aer instead of
the statevector simulator below — the shot-based return type is not an
implementation detail of the simulator, it is what quantum measurement is.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import statevector_sim as sv


@dataclass
class QasmResult:
    shots: int
    counts: dict[str, int]
    expectation_z: dict[int, float]          # exact, from the statevector
    sampled_expectation_z: dict[int, float]  # estimated from `counts` — what real hardware gives you


class QasmModule:
    """A parsed, ready-to-run OpenQASM circuit. Immutable after load()."""

    def __init__(self, circuit: sv.ParsedCircuit, source_path: str):
        self._circuit = circuit
        self.source_path = source_path
        self.n_qubits = circuit.n_qubits

    @classmethod
    def load(cls, path: str | Path) -> "QasmModule":
        text = Path(path).read_text()
        circuit = sv.parse_qasm(text)
        return cls(circuit, str(path))

    def call(self, params: dict[str, float] | None = None, shots: int = 1024,
              seed: int | None = None) -> QasmResult:
        state = sv.run_statevector(self._circuit, params)
        counts = sv.sample_shots(state, self.n_qubits, shots, seed)

        exact = {q: sv.expectation_z(state, self.n_qubits, q) for q in range(self.n_qubits)}

        sampled: dict[int, float] = {q: 0.0 for q in range(self.n_qubits)}
        for bitstring, n in counts.items():
            for q, bit in enumerate(bitstring):
                sampled[q] += (1 if bit == "0" else -1) * n
        sampled = {q: v / shots for q, v in sampled.items()}

        return QasmResult(shots=shots, counts=counts, expectation_z=exact,
                           sampled_expectation_z=sampled)

    def __repr__(self) -> str:
        return f"QasmModule(source={self.source_path!r}, n_qubits={self.n_qubits})"
