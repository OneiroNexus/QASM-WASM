"""
fusion/shared_space.py

Combines the three layers -- GNN structural embedding, SNN spike timing,
QGNN quantum expectation values -- into one 4-dimensional per-node
coordinate, and owns the actual trigger logic deciding when a node's QASM
node-update circuit gets called.

Being explicit about what "4D" means here, because it would be easy to
oversell it: three of the four coordinates are a fixed linear projection
of the GNN layer's embedding down to R^3 -- a spatial-indexing
convenience. Random projections approximately preserve pairwise distances
(the Johnson-Lindenstrauss guarantee), so this is a real, defensible
dimensionality reduction rather than an arbitrary one; see
tests/test_shared_space.py::test_random_projection_preserves_expected_norm
for an empirical check of exactly the property that makes it defensible.
It is untrained by design -- see AGENTS.md Boundaries.

The fourth coordinate is NOT another projected embedding dimension dressed
up to look geometric: it is the node's actual elapsed time since its last
spike, a real physical quantity the SNN layer produces for free. Calling
the 4-tuple "4D spacetime" would be exactly the kind of unearned physical
framing this project deliberately avoids. So: 3 projected structural
dimensions + 1 genuine temporal dimension -- honest, cheap, and still a
useful bounded key for the sharding scheme in SCALING.md (M3).

This module is also the literal integration point of all three subsystems:
the SNN layer's spike signal decides WHEN a node's QGNN circuit fires (not
a separate heuristic), the GNN layer's embedding supplies WHAT gets encoded
into that circuit's rotation angles, and qmodule's statevector simulator
supplies the quantum-derived expectation values folded back into the
node's classical embedding.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qmodule import QasmModule

NEVER_SPIKED = -np.inf


@dataclass(frozen=True)
class ProjectionParams:
    """Fixed (untrained) Gaussian random projection, F_in -> 3. Build with
    make_projection() rather than by hand, so the 1/sqrt(k) scaling stays
    consistent with what the JL-style unbiasedness check assumes."""
    matrix: np.ndarray  # (F_in, 3)


def make_projection(f_in: int, seed: int = 0) -> ProjectionParams:
    """Standard Gaussian random projection scaled by 1/sqrt(k), k=3. This
    specific scaling is what makes E[||Px||^2] = ||x||^2 hold exactly over
    the projection's own randomness (see test file) -- an unscaled random
    matrix would systematically shrink or grow every embedding's norm."""
    rng = np.random.default_rng(seed)
    k = 3
    matrix = rng.normal(size=(f_in, k)) / np.sqrt(k)
    return ProjectionParams(matrix=matrix)


def project_to_3d(embeddings: np.ndarray, proj: ProjectionParams) -> np.ndarray:
    """(N, F_in) -> (N, 3). A plain linear map -- see
    test_shared_space.py::test_projection_is_linear."""
    return embeddings @ proj.matrix


def time_since_last_spike(last_spike_time: np.ndarray, current_time: float,
                           inf_fallback: float | None = None) -> np.ndarray:
    """last_spike_time: (N,) array, NEVER_SPIKED (-inf) sentinel for nodes
    that have not fired yet. current_time must be in the same units as
    last_spike_time (i.e. tick*dt, not a raw tick count) -- that is what
    keeps this a real elapsed time rather than a step count wearing a
    time-shaped costume.

    A node that has never spiked gets +inf elapsed time by construction
    (current_time - (-inf) = +inf). That is not a bug to paper over with an
    arbitrary default -- it is the mathematically correct answer to "how
    long since an event that hasn't happened." Pass inf_fallback to replace
    +inf with a finite sentinel only where a downstream consumer genuinely
    cannot handle inf (e.g. a fixed-width spatial index); the default of
    None leaves the honest value in place.
    """
    elapsed = current_time - last_spike_time
    if inf_fallback is not None:
        elapsed = np.where(np.isinf(elapsed), inf_fallback, elapsed)
    return elapsed


def fuse_to_4d(gnn_embedding: np.ndarray, last_spike_time: np.ndarray, current_time: float,
               proj: ProjectionParams, inf_fallback: float | None = None) -> np.ndarray:
    """(N, F_in) GNN embedding + (N,) spike-time state -> (N, 4). Columns
    0:3 are exactly project_to_3d's output; column 3 is exactly
    time_since_last_spike's output. This function does no independent
    computation of its own, by design -- see
    test_shared_space.py::test_fuse_to_4d_is_exact_concatenation."""
    xyz = project_to_3d(gnn_embedding, proj)
    t = time_since_last_spike(last_spike_time, current_time, inf_fallback)
    return np.concatenate([xyz, t[:, None]], axis=1)


@dataclass(frozen=True)
class QgnnTriggerResult:
    triggered_mask: np.ndarray     # (N,) bool -- which nodes actually called the circuit this tick
    updated_embedding: np.ndarray  # (N, F_in) -- bit-identical to input for non-triggered nodes
    expectation_z: dict            # node_index -> {0: <Z0>, 1: <Z1>} (exact, statevector-derived)
    sampled_expectation_z: dict    # node_index -> {0: ..., 1: ...} (shot-sampled; what real hardware would return)


def maybe_trigger_qgnn(gnn_embedding: np.ndarray, spiked: np.ndarray, qasm_module: QasmModule,
                        shots: int = 2000, seed: int = 0,
                        blend: float = 1.0) -> QgnnTriggerResult:
    """The SNN spike signal -- not a separate heuristic -- gates the QGNN
    call: ry_node_update.qasm is invoked exactly for nodes that spiked this
    tick, once each. Embedding components 0, 1, 2 are used directly as that
    circuit's (a, b, phi) rotation angles: RY and RZ are both periodic in
    their argument, so the embedding's (unbounded) real values need no
    squashing/clamping for this encoding to be well-defined.

    The returned <Z0>, <Z1> expectation values are blended into embedding
    components 0 and 1 (blend=1.0 replaces them outright, blend=0.0 leaves
    the embedding untouched, values between linearly interpolate).
    Component 2 (phi) is deliberately left alone: ry_node_update.qasm's own
    closed form proves RZ(phi) is invisible to any Z-basis measurement (see
    tests/test_qasm_sim.py::test_rz_invisible_to_z_measurement), so there is
    no Z-measurement-derived signal that could ever justify writing a new
    value into it -- doing so anyway would be manufacturing a fake update
    that looks quantum-derived but provably isn't.
    """
    n, f_in = gnn_embedding.shape
    assert f_in >= 3, "embedding needs >=3 components to supply (a, b, phi)"
    updated = gnn_embedding.copy()
    triggered_mask = spiked > 0
    expectation_z: dict[int, dict[int, float]] = {}
    sampled: dict[int, dict[int, float]] = {}
    for idx in np.nonzero(triggered_mask)[0]:
        a, b, phi = (float(gnn_embedding[idx, 0]), float(gnn_embedding[idx, 1]), float(gnn_embedding[idx, 2]))
        result = qasm_module.call({"a": a, "b": b, "phi": phi}, shots=shots, seed=seed + int(idx))
        expectation_z[int(idx)] = result.expectation_z
        sampled[int(idx)] = result.sampled_expectation_z
        updated[idx, 0] = (1 - blend) * gnn_embedding[idx, 0] + blend * result.expectation_z[0]
        updated[idx, 1] = (1 - blend) * gnn_embedding[idx, 1] + blend * result.expectation_z[1]
    return QgnnTriggerResult(triggered_mask=triggered_mask, updated_embedding=updated,
                              expectation_z=expectation_z, sampled_expectation_z=sampled)
