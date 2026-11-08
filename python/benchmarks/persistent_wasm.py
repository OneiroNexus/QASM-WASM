"""Compare Python, one-shot WASM, and persistent WASM LIF roundtrips.

Run from python/: python -m benchmarks.persistent_wasm --output REPORT.json

The workload is a synthetic LIF sequence, not a graph-pipeline, QPU, or
energy benchmark. Inputs are precomputed; voltage evolves from each engine's
own preceding output. Every warmup and measured output is compared with a
Python reference outside the timed regions. Each measured region contains
only one engine call, including that engine's serialization and array
conversion costs. History copies, comparisons, fixture generation, startup,
and shutdown are excluded. No performance threshold is asserted.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np

from orchestrator import wasm_bridge
from orchestrator.pipeline import python_lif_engine
from snn.lif_neuron import LifParams, step

BENCHMARK_VERSION = "1.0"
ENGINE_NAMES = ("python", "one_shot_wasm", "persistent_wasm")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_hash(*arrays: np.ndarray) -> str:
    """Hash canonical little-endian float64 arrays, including their shapes."""
    digest = hashlib.sha256()
    for value in arrays:
        canonical = np.ascontiguousarray(value, dtype="<f8")
        digest.update(json.dumps(list(canonical.shape)).encode("ascii"))
        digest.update(canonical.tobytes())
    return digest.hexdigest()


def _fixture(nodes: int, total_ticks: int, seed: int, params: LifParams, delta_t: float):
    rng = np.random.Generator(np.random.PCG64(seed + nodes))
    initial = np.asarray(rng.uniform(-0.1, 0.7, size=nodes), dtype=np.float64)
    currents = np.asarray(rng.uniform(0.4, 2.4, size=(total_ticks, nodes)), dtype=np.float64)
    expected_v = np.empty((total_ticks, nodes), dtype=np.float64)
    expected_spikes = np.empty_like(expected_v)
    voltage = initial.copy()
    for tick in range(total_ticks):
        voltage, spiked = step(voltage, currents[tick], delta_t, params)
        expected_v[tick] = voltage
        expected_spikes[tick] = spiked
    return initial, currents, expected_v, expected_spikes


def _exercise(engine: Callable, initial: np.ndarray, currents: np.ndarray,
              expected_v: np.ndarray, expected_spikes: np.ndarray,
              params: LifParams, delta_t: float, warmup: int) -> dict:
    """Time calls only; retain snapshots and check every tick after timing.

    Voltage returned by a call feeds the next call. Copying a returned array
    into the history occurs after its timer stops, so comparison/storage
    cannot inflate measured WASM or Python call durations.
    """
    observed_v = np.empty_like(expected_v)
    observed_spikes = np.empty_like(expected_spikes)
    voltage = initial.copy()
    elapsed_ns = []
    for tick in range(len(currents)):
        if tick < warmup:
            voltage, spiked = engine(voltage, currents[tick], delta_t, params)
        else:
            started = time.perf_counter_ns()
            voltage, spiked = engine(voltage, currents[tick], delta_t, params)
            stopped = time.perf_counter_ns()
            elapsed_ns.append(stopped - started)
        if np.shape(voltage) != initial.shape or np.shape(spiked) != initial.shape:
            raise RuntimeError(f"Engine returned an incorrect array shape at tick {tick}")
        observed_v[tick] = voltage
        observed_spikes[tick] = spiked

    # These comparisons are deliberately outside every measured region.
    if not np.all(np.isfinite(observed_v)) or not np.all(np.isfinite(observed_spikes)):
        raise RuntimeError("An engine produced non-finite output")
    if not np.array_equal(observed_v, expected_v):
        tick, node = np.argwhere(observed_v != expected_v)[0]
        raise RuntimeError(f"Voltage differs from Python reference at tick {tick}, node {node}")
    if not np.array_equal(observed_spikes, expected_spikes):
        tick, node = np.argwhere(observed_spikes != expected_spikes)[0]
        raise RuntimeError(f"Spike mask differs from Python reference at tick {tick}, node {node}")
    measured_seconds = [value / 1e9 for value in elapsed_ns]
    total_seconds = sum(elapsed_ns) / 1e9
    return {
        "measured_call_seconds": measured_seconds,
        "measured_total_seconds": total_seconds,
        "mean_seconds_per_tick": total_seconds / len(elapsed_ns),
        "all_ticks_equal_python_reference": True,
        "max_abs_voltage_difference": float(np.max(np.abs(observed_v - expected_v))),
        "spike_mask_mismatches": int(np.count_nonzero(observed_spikes != expected_spikes)),
        "checked_ticks_including_warmup": len(currents),
        "final_state_sha256": _array_hash(observed_v[-1], observed_spikes[-1]),
        "total_spikes_including_warmup": int(observed_spikes.sum()),
    }


def _case(nodes: int, ticks: int, warmup: int, repeats: int, seed: int,
          params: LifParams, delta_t: float) -> dict:
    initial, currents, expected_v, expected_spikes = _fixture(
        nodes, ticks + warmup, seed, params, delta_t)
    samples = {name: [] for name in ENGINE_NAMES}
    orders = []
    for repetition in range(repeats):
        offset = repetition % len(ENGINE_NAMES)
        order = ENGINE_NAMES[offset:] + ENGINE_NAMES[:offset]
        orders.append(list(order))
        for name in order:
            startup_seconds = None
            capacity = None
            process_id = None
            same_process = None
            if name == "persistent_wasm":
                started = time.perf_counter_ns()
                persistent = wasm_bridge.PersistentWasmLifEngine()
                startup_seconds = (time.perf_counter_ns() - started) / 1e9
                with persistent:
                    capacity = persistent.capacity
                    process_id = persistent.pid
                    if nodes > capacity:
                        raise ValueError(f"Requested {nodes} nodes exceeds persistent capacity {capacity}")
                    result = _exercise(persistent, initial, currents, expected_v,
                                       expected_spikes, params, delta_t, warmup)
                    same_process = persistent.pid == process_id
                    if not same_process:
                        raise RuntimeError("Persistent engine changed process during the measured sequence")
            else:
                engine = python_lif_engine if name == "python" else wasm_bridge.wasm_lif_engine
                result = _exercise(engine, initial, currents, expected_v,
                                   expected_spikes, params, delta_t, warmup)
            samples[name].append({"repeat": repetition + 1, "startup_seconds": startup_seconds,
                                  "capacity": capacity, "pid": process_id,
                                  "same_process_for_all_ticks": same_process, **result})

    summaries = {}
    for name, runs in samples.items():
        per_tick = [run["mean_seconds_per_tick"] for run in runs]
        startup = [run["startup_seconds"] for run in runs if run["startup_seconds"] is not None]
        summaries[name] = {
            "median_measured_total_seconds": statistics.median(run["measured_total_seconds"] for run in runs),
            "median_mean_seconds_per_tick": statistics.median(per_tick),
            "min_mean_seconds_per_tick": min(per_tick),
            "max_mean_seconds_per_tick": max(per_tick),
            "median_startup_seconds": statistics.median(startup) if startup else None,
        }
    python_time = summaries["python"]["median_mean_seconds_per_tick"]
    one_shot_time = summaries["one_shot_wasm"]["median_mean_seconds_per_tick"]
    persistent_time = summaries["persistent_wasm"]["median_mean_seconds_per_tick"]
    return {
        "nodes": nodes,
        "fixture_sha256": _array_hash(initial, currents),
        "engine_order_per_repeat": orders,
        "summary": summaries,
        "persistent_speedup_vs_one_shot": one_shot_time / persistent_time,
        "persistent_time_relative_to_python": persistent_time / python_time,
        "persistent_vs_python": "slower" if persistent_time > python_time else "faster" if persistent_time < python_time else "equal",
        "samples": samples,
    }


def run_benchmark(nodes: list[int], ticks: int, warmup: int, repeats: int, seed: int) -> dict:
    node = shutil.which("node")
    if node is None or not wasm_bridge.WASM_BIN.is_file():
        raise RuntimeError("Node on PATH and an existing WASM binary are required")
    node_version = subprocess.run([node, "--version"], capture_output=True, text=True,
                                  check=True, timeout=10).stdout.strip()
    params = LifParams(tau=7.5, v_rest=0.1, v_reset=-0.05, v_th=0.9, R=1.1)
    delta_t = 0.5
    return {
        "schema_version": 1,
        "benchmark_version": BENCHMARK_VERSION,
        "author": "Aaron Michael Weis",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "node": node_version, "os": platform.system(), "os_release": platform.release(),
                        "architecture": platform.machine(), "logical_cpu_count": os.cpu_count()},
        "fingerprints": {"wasm_sha256": _sha256(wasm_bridge.WASM_BIN),
                         "bridge_sha256": _sha256(Path(wasm_bridge.__file__)),
                         "one_shot_host_sha256": _sha256(wasm_bridge.HARNESS),
                         "persistent_host_sha256": _sha256(wasm_bridge.PERSISTENT_HARNESS),
                         "benchmark_sha256": _sha256(Path(__file__))},
        "configuration": {"nodes": nodes, "measured_ticks": ticks, "warmup_ticks": warmup,
                          "repeats": repeats, "seed": seed, "rng": "NumPy PCG64(seed + nodes)",
                          "delta_t": delta_t, "lif_parameters": asdict(params)},
        "timing_contract": {
            "clock": "time.perf_counter_ns",
            "startup": "Persistent constructor, process launch, WASM instantiation and ready handshake; reported separately.",
            "warmup": f"Ticks 0 through {warmup - 1} are unmeasured; voltage evolves and continues into measured ticks." if warmup else "No warmup ticks.",
            "measured": f"Ticks {warmup} through {warmup + ticks - 1}; sum of separate callable-entry-to-return regions.",
            "included": "Engine work, serialization, pipe roundtrip, JSON parsing and array conversion; one-shot engine additionally launches/instantiates/shuts down its Node process per call.",
            "excluded": "Fixture/reference generation, history copies, correctness checks, persistent startup/shutdown and outer-loop bookkeeping.",
            "timer_overhead": "One perf_counter_ns pair per measured call; no overhead subtraction.",
            "aggregation": "Median across repetitions of each repetition's mean per-call duration; speedup is the ratio of those medians.",
        },
        "correctness_contract": "Every warmup and measured voltage/spike array is compared to a precomputed Python reference after timing. Exact numeric equality is checked, not raw floating-point bit identity.",
        "limitations": ["Synthetic changing-current LIF workload only; no full graph, circuit, energy or production-latency claim.",
                        "Persistent JSON serialization and array copies are retained; this is not a zero-copy kernel benchmark.",
                        "Few repetitions describe this run; they are not confidence intervals or a cross-machine performance guarantee.",
                        "The benchmark uses and hashes the existing WASM binary; it does not rebuild it or prove source-to-binary correspondence.",
                        "A speedup over one-shot WASM does not imply a speedup over the in-process Python/NumPy implementation."],
        "cases": [_case(count, ticks, warmup, repeats, seed, params, delta_t) for count in nodes],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", nargs="+", type=int, default=[25, 1024])
    parser.add_argument("--ticks", type=int, default=30, help="measured ticks per repetition (default: 30)")
    parser.add_argument("--warmup", type=int, default=3, help="unmeasured evolving-state ticks (default: 3)")
    parser.add_argument("--repeats", type=int, default=3, help="at least three repetitions (default: 3)")
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--output", type=Path, help="JSON destination; existing parent required; defaults to stdout")
    parser.add_argument("--overwrite", action="store_true", help="explicitly allow replacing the output file")
    args = parser.parse_args(argv)
    if any(count <= 0 for count in args.nodes) or len(set(args.nodes)) != len(args.nodes):
        parser.error("--nodes must contain distinct positive integers")
    if args.ticks < 1 or args.warmup < 0 or args.repeats < 3 or args.seed < 0:
        parser.error("ticks >= 1, warmup >= 0, repeats >= 3 and seed >= 0 are required")
    if args.output and (not args.output.parent.is_dir() or (args.output.exists() and not args.overwrite)):
        parser.error("output parent must exist and output must be new unless --overwrite is explicit")
    report = run_benchmark(args.nodes, args.ticks, args.warmup, args.repeats, args.seed)
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("w" if args.overwrite else "x", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized)
        for case in report["cases"]:
            persistent_ms = 1000 * case["summary"]["persistent_wasm"]["median_mean_seconds_per_tick"]
            print(f"{case['nodes']} nodes: persistent {persistent_ms:.3f} ms/tick; "
                  f"{case['persistent_speedup_vs_one_shot']:.2f}x speedup vs one-shot; "
                  f"{case['persistent_time_relative_to_python']:.2f}x Python time "
                  f"({case['persistent_vs_python']} than Python).")
        print("All warmup and measured outputs matched the Python reference.")
    else:
        sys.stdout.write(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
