// ry_node_update.qasm — the QGNN "node update" primitive.
//
// This is the piece a real quantum-graph-neural-network layer calls once
// per node, per message-passing round: q0 carries the node's own
// (already-aggregated) feature, encoded as a rotation angle `a`; q1 carries
// the incoming neighbor message, encoded as `b`. CX(0,1) is the part that
// makes this a *graph* primitive rather than two independent qubits — it
// entangles the node's own state with what its neighbors sent it, which is
// the quantum analogue of a classical GNN's message-aggregation step.
// RZ(phi) is a free, trainable phase — included deliberately even though
// it provably cannot change a Z-basis measurement outcome (see
// tests/test_qasm_sim.py), because a real parameterized circuit designer
// needs to know that fact, not have it hidden by leaving the gate out.
//
// Closed form this circuit is checked against (derived in
// tests/test_qasm_sim.py's docstring):
//     <Z0> = cos(a)
//     <Z1> = cos(a) * cos(b)
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
ry(a) q[0];
ry(b) q[1];
cx q[0],q[1];
rz(phi) q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
