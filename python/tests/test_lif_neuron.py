"""
Adversarial validation of snn/lif_neuron.py against an independently
derived, exact closed-form solution of the *discrete* forward-Euler
recurrence -- not just "spikes at roughly the right rate," but "spikes at
exactly the integer timestep the recursion predicts," checked to near
machine precision. Same standard as tests/test_qasm_sim.py.

Derivation: the Euler update
    V_{k+1} = a*V_k + (1-a)*v_inf,     a = 1 - dt/tau,  v_inf = v_rest + R*I
has the closed-form solution
    V_k = v_inf + (v_reset - v_inf) * a^k        (V_0 = v_reset)
Solving V_k >= v_th for the smallest integer k gives an exact, closed-form
predicted spike step -- no simulation loop needed to derive it independently:
    a^k <= C,  C = (v_inf - v_th) / (v_inf - v_reset)   (0 < C < 1 supra-threshold)
    k_spike = ceil( ln(C) / ln(a) )
Since every post-spike reset lands on the same v_reset and I is constant,
this same k_spike is also the exact inter-spike interval (in steps) for
*every* subsequent spike, so one formula predicts the entire spike train.

Note on simulate_constant_current's time bookkeeping: it records `t` BEFORE
applying step() and increments after, so a spike produced by the k-th call
to step() is logged at time (k-1)*dt, not k*dt. This constant bookkeeping
offset cancels in any inter-spike interval (ISI) -- it only has to be
accounted for when checking the very first spike's absolute time.

Run: python3 -m tests.test_lif_neuron   (from the python/ directory)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from snn.lif_neuron import LifParams, analytic_rate, simulate_constant_current, step

FAILURES = []


def check(name: str, got: float, want: float, tol: float):
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


def exact_discrete_spike_step(I: float, dt: float, p: LifParams = LifParams()) -> int | None:
    """Independent closed-form derivation (see module docstring) of the exact
    integer step k at which the discrete Euler recurrence first satisfies
    V_k >= v_th, starting from V_0 = v_reset. None if it never spikes."""
    v_inf = p.v_rest + p.R * I
    if v_inf <= p.v_th:
        return None
    a = 1.0 - dt / p.tau
    C = (v_inf - p.v_th) / (v_inf - p.v_reset)
    k_c = math.log(C) / math.log(a)
    return math.ceil(k_c - 1e-9)  # epsilon guards against fp noise landing just above an integer


def simulate_population(I_arr: np.ndarray, duration: float, dt: float,
                         p: LifParams = LifParams()):
    """Independent multi-neuron loop (mirrors simulate_constant_current's own
    loop structure, batched) -- used only to check that step()'s vectorization
    treats each neuron independently, with no cross-neuron leakage."""
    n_steps = int(duration / dt)
    n = len(I_arr)
    V = np.full(n, p.v_rest, dtype=np.float64)
    spike_times = [[] for _ in range(n)]
    t = 0.0
    for _ in range(n_steps):
        V, spiked = step(V, I_arr, dt, p)
        for i in range(n):
            if spiked[i] > 0:
                spike_times[i].append(t)
        t += dt
    return spike_times


def test_subthreshold_never_spikes():
    print("\n-- sub-threshold current: exactly zero spikes, exactly zero analytic rate --")
    p = LifParams()
    for I in (0.0, 0.3, 0.6, 0.9, 1.0):
        v_inf = p.v_rest + p.R * I
        check_true(f"I={I}: v_inf={v_inf} <= v_th={p.v_th} (sub-threshold by construction)", v_inf <= p.v_th)
        check("analytic_rate exact zero", analytic_rate(I, p), 0.0, tol=0.0)
        spikes = simulate_constant_current(I, duration=500.0, dt=0.05, p=p)
        check("simulated spike count exact zero", float(len(spikes)), 0.0, tol=0.0)


def test_discrete_spike_train_matches_closed_form():
    print("\n-- supra-threshold: EVERY spike time matches the exact discrete closed form --")
    p = LifParams()
    dt = 0.01
    target_spikes = 8
    for I in (1.05, 1.2, 1.5, 2.0, 3.0, 5.0, 10.0, 50.0):
        k_spike = exact_discrete_spike_step(I, dt, p)
        assert k_spike is not None
        isi_approx = 1.0 / analytic_rate(I, p)
        duration = isi_approx * (target_spikes + 2) + 10 * dt
        spikes = simulate_constant_current(I, duration=duration, dt=dt, p=p)
        check_true(f"I={I}: at least {target_spikes} spikes observed to check spacing", len(spikes) >= target_spikes)
        # First recorded spike time is logged one step "early" (see module docstring).
        first_k = round(spikes[0] / dt)
        check(f"I={I}: first spike step (recorded convention: k_spike-1)", float(first_k), float(k_spike - 1), tol=0.0)
        # Every ISI (in steps) should equal k_spike exactly -- the recurrence
        # restarts identically from v_reset after every spike, and the
        # recording-offset is constant so it cancels in any interval.
        for j in range(1, min(len(spikes), target_spikes)):
            isi_k = round((spikes[j] - spikes[j - 1]) / dt)
            check(f"I={I}: ISI step #{j}", float(isi_k), float(k_spike), tol=0.0)


def test_discrete_formula_converges_to_continuous_analytic_rate():
    print("\n-- sanity: as dt -> 0, exact discrete spike step * dt -> continuous analytic ISI --")
    p = LifParams()
    I = 2.0
    want_isi = 1.0 / analytic_rate(I, p)
    prev_err = None
    discrete_isi = None
    for dt in (0.2, 0.02, 0.002):
        k_spike = exact_discrete_spike_step(I, dt, p)
        discrete_isi = k_spike * dt
        err = abs(discrete_isi - want_isi)
        print(f"  dt={dt}: discrete_isi={discrete_isi:.6f} analytic_isi={want_isi:.6f} err={err:.6f}")
        if prev_err is not None:
            check_true(f"dt={dt}: discretization error shrank vs previous (10x finer dt)", err < prev_err)
        prev_err = err
    check("finest dt: discrete ISI within 1% of continuous analytic ISI", discrete_isi, want_isi, tol=0.01 * want_isi)


def test_vectorized_step_matches_scalar_loop():
    print("\n-- step() batched over many neurons matches independent single-neuron loops exactly --")
    p = LifParams()
    dt = 0.02
    I_arr = np.array([0.5, 1.0, 1.3, 2.0, 4.0, 9.0])  # mix of sub- and supra-threshold, incl. boundary I=v_th
    duration = 150.0
    batch_spikes = simulate_population(I_arr, duration, dt, p)
    for i, I in enumerate(I_arr):
        single_spikes = simulate_constant_current(float(I), duration, dt, p)
        check(f"I={I}: spike count matches single-neuron run", float(len(batch_spikes[i])), float(len(single_spikes)), tol=0.0)
        for j in range(len(single_spikes)):
            check(f"I={I}: spike #{j} time matches single-neuron run", batch_spikes[i][j], single_spikes[j], tol=1e-12)


if __name__ == "__main__":
    test_subthreshold_never_spikes()
    test_discrete_spike_train_matches_closed_form()
    test_discrete_formula_converges_to_continuous_analytic_rate()
    test_vectorized_step_matches_scalar_loop()
    print(f"\n{'ALL PASSED' if not FAILURES else f'{len(FAILURES)} FAILED: ' + ', '.join(FAILURES)}")
    sys.exit(1 if FAILURES else 0)
