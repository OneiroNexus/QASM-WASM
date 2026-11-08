"""
lif_neuron.py — leaky integrate-and-fire neurons, vectorized over many
graph nodes at once. This is the layer that makes a huge graph survivable:
a node only sends a message to its neighbors (and only triggers its QASM
node-update circuit) on the timestep it spikes, not on every timestep —
so the *classical* graph can be enormous while the *active* fraction of it
at any instant stays small.

Discrete update rule (forward-Euler on the standard LIF ODE
tau*dV/dt = -(V - v_rest) + R*I):

    V[t+1] = V[t] + (dt/tau) * ( -(V[t] - v_rest) + R*I[t] )
    if V[t+1] >= v_th:  spike=1, V[t+1] = v_reset
    else:                spike=0

This exact rule is re-implemented in wasm-core/orchestrator_core.c — the
two are cross-checked against each other in tests/test_pipeline_smoke.py,
not just each checked in isolation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LifParams:
    tau: float = 10.0        # membrane time constant
    v_rest: float = 0.0
    v_reset: float = 0.0
    v_th: float = 1.0
    R: float = 1.0            # membrane resistance


def analytic_rate(I: float, p: LifParams = LifParams()) -> float:
    """Closed-form steady-state firing rate for constant input current I
    (Gerstner & Kistler; Dayan & Abbott). Returns 0.0 for sub-threshold I,
    where the membrane potential asymptotes below v_th and the neuron
    never fires at all -- also an exact, checkable prediction, not just
    the "rate" formula's domain restriction."""
    v_inf = p.v_rest + p.R * I           # steady-state potential if never reset
    if v_inf <= p.v_th:
        return 0.0
    isi = p.tau * math.log((v_inf - p.v_reset) / (v_inf - p.v_th))
    return 1.0 / isi


def step(V: np.ndarray, I: np.ndarray, dt: float, p: LifParams = LifParams()) -> tuple[np.ndarray, np.ndarray]:
    """One vectorized Euler step for N neurons at once.
    Returns (V_next, spiked) where `spiked` is a {0,1} float array."""
    V_next = V + (dt / p.tau) * (-(V - p.v_rest) + p.R * I)
    spiked = (V_next >= p.v_th).astype(np.float64)
    V_next = np.where(spiked > 0, p.v_reset, V_next)
    return V_next, spiked


def simulate_constant_current(I: float, duration: float, dt: float,
                                p: LifParams = LifParams()) -> np.ndarray:
    """Single neuron, constant input current, for `duration` time units.
    Returns the array of spike times -- used only to validate `step()`
    against `analytic_rate()` in tests/test_lif_neuron.py."""
    n_steps = int(duration / dt)
    V = np.array([p.v_rest])
    I_arr = np.array([I])
    spike_times = []
    t = 0.0
    for _ in range(n_steps):
        V, spiked = step(V, I_arr, dt, p)
        if spiked[0] > 0:
            spike_times.append(t)
        t += dt
    return np.array(spike_times)
