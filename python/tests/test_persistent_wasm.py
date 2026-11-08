"""Independent M2 checks for the resident Node/WASM LIF engine.

Numerical checks use the written Euler rule, the Python implementation,
and the preserved one-shot WASM implementation. Process checks observe
real child processes; temporary JSONL servers exercise protocol failures
without modifying the production server or the supplied WASM binary.
Exact equality below means numerical array equality, not raw-bit identity.

Run from python/: python -m tests.test_persistent_wasm
The module fails explicitly when Node/WASM is unavailable; it never skips
the runtime tests and reports success from an unexecuted comparison.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import time
import traceback
from unittest.mock import patch

import numpy as np

from fusion import make_projection
from gnn import Graph
from orchestrator import python_lif_engine, run_pipeline
from orchestrator import wasm_bridge
from orchestrator.wasm_bridge import PersistentWasmLifEngine, wasm_lif_engine
from qmodule import QasmModule
from snn.lif_neuron import LifParams


FAILURES: list[str] = []
PASS_COUNT = 0
PYTHON_DIR = Path(__file__).resolve().parent.parent
READY = {"type": "ready", "protocol": 1, "capacity": 65536,
         "max_message_bytes": 8388608}


def check(name: str, condition: bool) -> None:
    global PASS_COUNT
    if bool(condition):
        PASS_COUNT += 1
        print(f"  [PASS] {name}")
    else:
        FAILURES.append(name)
        print(f"  [FAIL] {name}")


def equal(name: str, actual, expected) -> None:
    actual_array, expected_array = np.asarray(actual), np.asarray(expected)
    check(name, actual_array.shape == expected_array.shape
          and np.array_equal(actual_array, expected_array))


def raises(name: str, exception: type[Exception], operation) -> None:
    try:
        operation()
    except exception as error:
        check(name, bool(str(error)))
    except Exception as error:
        check(f"{name}: got {type(error).__name__}, wanted {exception.__name__}", False)
    else:
        check(f"{name}: no exception", False)


@contextmanager
def record_children():
    """Observe actual Popen objects, not merely a self-reported PID."""
    children = []
    original = wasm_bridge.subprocess.Popen

    def start(*args, **kwargs):
        child = original(*args, **kwargs)
        children.append(child)
        return child

    with patch.object(wasm_bridge.subprocess, "Popen", side_effect=start):
        try:
            yield children
        finally:
            # A failing cleanup assertion must not leave a hung test fixture.
            for child in children:
                if child.poll() is None:
                    child.kill()
                    child.wait(timeout=5)


def scalar_lif(V, I, dt, p):
    """Independent scalar transcription of the documented Euler map."""
    voltage, spikes = [], []
    for v, current in zip(V, I):
        updated = float(v) + (float(dt) / float(p.tau)) * (
            -(float(v) - float(p.v_rest)) + float(p.R) * float(current))
        fired = updated >= p.v_th
        voltage.append(p.v_reset if fired else updated)
        spikes.append(1.0 if fired else 0.0)
    return (np.asarray(voltage, dtype=np.float64),
            np.asarray(spikes, dtype=np.float64))


def test_one_process_scalar_rule_and_cleanup():
    print("\n-- one resident process, scalar rule and lifetime --")
    p = LifParams(tau=2.0, v_reset=-0.25)
    V = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    I = np.array([2.0, 1.999999, 2.000001], dtype=np.float64)
    with record_children() as children:
        engine = PersistentWasmLifEngine()
        try:
            check("ready reports supplied core's 65536-neuron capacity", engine.capacity == 65536)
            check("ready engine is open", not engine.closed)
            check("reported PID belongs to the live child", len(children) == 1
                  and children[0].poll() is None and engine.pid == children[0].pid)
            pid = engine.pid
            result, spikes = engine(V, I, 1.0, p)
            equal("threshold equality, just below, just above", spikes, [1.0, 0.0, 1.0])
            equal("threshold hit resets to explicit reset value", result, [-0.25, 0.9999995, -0.25])
            rng = np.random.default_rng(6817)
            all_voltages = all_spikes = True
            for tick in range(40):
                n = 1 + (tick * 7) % 29
                voltage = rng.uniform(-1.2, 1.8, n).astype(np.float64)
                current = rng.uniform(-3.0, 6.0, n).astype(np.float64)
                params = LifParams(tau=2.5 + tick % 3, v_rest=-0.125,
                                   v_reset=-0.375, v_th=0.75, R=-0.5 if tick % 2 else 1.25)
                expected = scalar_lif(voltage, current, 0.25, params)
                actual = engine(voltage, current, 0.25, params)
                all_voltages &= np.array_equal(actual[0], expected[0])
                all_spikes &= np.array_equal(actual[1], expected[1])
            check("40 changing-input calls agree with scalar voltage oracle", all_voltages)
            check("40 changing-input calls agree with scalar spike oracle", all_spikes)
            check("41 calls created exactly one Node child", len(children) == 1)
            check("PID remains live and unchanged across calls", engine.pid == pid
                  and children[0].poll() is None)
        finally:
            engine.close()
        check("close marks the runtime closed", engine.closed)
        check("close reaps the actual child", children[0].poll() is not None)
        check("close releases all three child pipe handles", all(
            stream.closed for stream in (children[0].stdin, children[0].stdout, children[0].stderr)))
        engine.close()
        check("second close is idempotent", engine.closed and children[0].poll() is not None)
        raises("closed runtime rejects another tick", RuntimeError, lambda: engine(V, I, 1.0, p))
        check("closed runtime never silently respawns", len(children) == 1)

    class ContextProbe(Exception):
        pass

    with record_children() as children:
        try:
            with PersistentWasmLifEngine() as engine:
                raise ContextProbe("exercise exceptional context exit")
        except ContextProbe:
            pass
        check("exceptional context exit closes and reaps child", engine.closed
              and len(children) == 1 and children[0].poll() is not None)


def test_cross_implementation_trajectories():
    print("\n-- Python / one-shot / persistent numerical agreement --")
    rng = np.random.default_rng(7741)
    voltage = rng.uniform(-0.5, 1.5, 37).astype(np.float64)
    with PersistentWasmLifEngine() as engine:
        for tick in range(8):
            current = rng.uniform(-2.0, 8.0, 37).astype(np.float64)
            params = LifParams(tau=1.5 + tick % 3, v_rest=-0.2,
                               v_reset=-0.1, v_th=0.8, R=0.75)
            python = python_lif_engine(voltage, current, 0.5, params)
            legacy = wasm_lif_engine(voltage, current, 0.5, params)
            resident = engine(V=voltage, I=current, dt=0.5, p=params)
            equal(f"trajectory {tick}: resident voltage equals Python", resident[0], python[0])
            equal(f"trajectory {tick}: resident spikes equal Python", resident[1], python[1])
            equal(f"trajectory {tick}: resident voltage equals one-shot", resident[0], legacy[0])
            equal(f"trajectory {tick}: resident spikes equal one-shot", resident[1], legacy[1])
            voltage = python[0]


def test_capacity_sizes_and_array_isolation():
    print("\n-- empty/full capacity, changing sizes and caller-owned arrays --")
    rng = np.random.default_rng(7939)
    p = LifParams(tau=3.0, v_rest=-0.125, v_reset=-0.5, v_th=0.75, R=1.5)
    with PersistentWasmLifEngine() as engine:
        for index, n in enumerate([engine.capacity, 3, 0, engine.capacity - 1, 1, engine.capacity]):
            V = rng.uniform(-1.0, 1.5, n).astype(np.float64)
            I = rng.uniform(-2.0, 4.0, n).astype(np.float64)
            original_v, original_i = V.copy(), I.copy()
            expected = python_lif_engine(V, I, 0.125, p)
            actual = engine(V, I, 0.125, p)
            equal(f"size transition {index} n={n}: voltage", actual[0], expected[0])
            equal(f"size transition {index} n={n}: spikes", actual[1], expected[1])
            check(f"size transition {index}: outputs are owned float64 vectors",
                  all(a.dtype == np.float64 and a.shape == (n,) for a in actual)
                  and not np.shares_memory(actual[0], actual[1]))
            check(f"size transition {index}: inputs not mutated",
                  np.array_equal(V, original_v) and np.array_equal(I, original_i))
        # Strided, read-only arrays expose accidental in-place writes and
        # assumptions that caller arrays are contiguous WASM memory views.
        V = (np.arange(18, dtype=np.float64) / 16.0)[::2]
        I = (np.arange(18, dtype=np.float64) / 4.0)[1::2]
        V.setflags(write=False)
        I.setflags(write=False)
        first = engine(V, I, 0.25, p)
        saved = tuple(a.copy() for a in first)
        second = engine(np.zeros(9, dtype=np.float64), np.zeros(9, dtype=np.float64), 0.25, p)
        equal("later call cannot overwrite earlier voltage output", first[0], saved[0])
        equal("later call cannot overwrite earlier spike output", first[1], saved[1])
        first[0].fill(123.0)
        first[1].fill(1.0)
        third = engine(V, I, 0.25, p)
        expected = scalar_lif(V, I, 0.25, p)
        equal("mutating prior output cannot poison next voltage", third[0], expected[0])
        equal("mutating prior output cannot poison next spikes", third[1], expected[1])
        equal("different supplied input replaces resident neuron state", second[0],
              python_lif_engine(np.zeros(9, dtype=np.float64), np.zeros(9, dtype=np.float64), 0.25, p)[0])
        zero_dt = engine(np.array([0.25, 1.0], dtype=np.float64), [4, 4], 0.0, LifParams())
        equal("dt=0 still applies documented threshold/reset", zero_dt[0], [0.25, 0.0])
        equal("dt=0 threshold flags", zero_dt[1], [0.0, 1.0])


def test_invalid_input_does_not_poison_runtime():
    print("\n-- explicit parameter/domain validation and recovery --")
    with patch.object(wasm_bridge.shutil, "which", return_value=None), record_children() as children:
        raises("missing Node fails explicitly", RuntimeError, PersistentWasmLifEngine)
        check("missing Node cannot start or silently fall back", not children)
    with tempfile.TemporaryDirectory(prefix="qasm_m2_missing_") as tmp:
        with patch.object(wasm_bridge, "WASM_BIN", Path(tmp) / "absent.wasm"), record_children() as children:
            raises("missing WASM fails explicitly", RuntimeError, PersistentWasmLifEngine)
            check("missing WASM cannot start or silently fall back", not children)
    for timeout in [0.0, -1.0, float("nan"), float("inf")]:
        with record_children() as children:
            raises(f"invalid timeout {timeout!r}", ValueError,
                   lambda timeout=timeout: PersistentWasmLifEngine(timeout=timeout))
            check(f"invalid timeout {timeout!r} starts no child", not children)

    V = np.array([0.0, 0.25], dtype=np.float64)
    I = np.array([2.0, -1.0], dtype=np.float64)
    p = LifParams()
    with record_children() as children:
        with PersistentWasmLifEngine() as engine:
            pid = engine.pid
            cases = [
                ("mismatched lengths", V, I[:1], 0.1, p),
                ("2D V", V.reshape(1, 2), I, 0.1, p),
                ("2D I", V, I.reshape(1, 2), 0.1, p),
                ("scalar V", np.array(0.0, dtype=np.float64), I, 0.1, p),
                ("scalar I", V, np.array(0.0, dtype=np.float64), 0.1, p),
                ("complex V", np.array([1j, 0], dtype=np.complex128), I, 0.1, p),
                ("complex I", V, np.array([1j, 0], dtype=np.complex128), 0.1, p),
                ("over capacity", np.zeros(engine.capacity + 1, dtype=np.float64),
                 np.zeros(engine.capacity + 1, dtype=np.float64), 0.1, p),
                ("negative dt", V, I, -0.01, p),
                ("boolean dt", V, I, True, p),
                ("zero tau", V, I, 0.1, replace(p, tau=0.0)),
                ("negative tau", V, I, 0.1, replace(p, tau=-1.0)),
                ("empty still validates tau", np.array([], dtype=np.float64),
                 np.array([], dtype=np.float64), 0.1, replace(p, tau=0.0)),
            ]
            for value in [float("nan"), float("inf"), -float("inf")]:
                cases.extend([
                    (f"nonfinite V {value!r}", np.array([value, 0.0], dtype=np.float64), I, 0.1, p),
                    (f"nonfinite I {value!r}", V, np.array([value, 0.0], dtype=np.float64), 0.1, p),
                    (f"nonfinite dt {value!r}", V, I, value, p),
                ])
                for field in ["tau", "v_rest", "v_reset", "v_th", "R"]:
                    cases.append((f"nonfinite {field} {value!r}", V, I, 0.1, replace(p, **{field: value})))
            for field in ["tau", "v_rest", "v_reset", "v_th", "R"]:
                cases.append((f"boolean {field}", V, I, 0.1, replace(p, **{field: True})))
            for name, voltage, current, dt, params in cases:
                raises(name, ValueError, lambda voltage=voltage, current=current, dt=dt, params=params:
                       engine(voltage, current, dt, params))
            check("all rejected calls leave the same child alive", engine.pid == pid
                  and not engine.closed and len(children) == 1 and children[0].poll() is None)
            actual = engine(V, I, 0.1, p)
            expected = scalar_lif(V, I, 0.1, p)
            equal("valid voltage works after rejected calls", actual[0], expected[0])
            equal("valid spikes work after rejected calls", actual[1], expected[1])


def test_pipeline_feedback_interchangeability():
    print("\n-- live graph feedback, QGNN triggers and fusion --")
    rng = np.random.default_rng(9013)
    n = 8
    src = np.arange(n, dtype=np.int64)
    graph = Graph(num_nodes=n, edge_src=np.concatenate([src, src]),
                  edge_dst=np.concatenate([np.roll(src, -1), np.roll(src, -3)]))
    initial = rng.uniform(-0.9, 0.9, (n, 3)).astype(np.float64)
    base = rng.uniform(3.0, 9.0, n).astype(np.float64)
    kwargs = dict(graph=graph, initial_embedding=initial,
                  qasm_module=QasmModule.load(PYTHON_DIR / "qmodule" / "circuits" / "ry_node_update.qasm"),
                  proj=make_projection(3, seed=17), base_current=base,
                  num_ticks=24, dt=0.3, lif_params=LifParams(), coupling=0.8,
                  qgnn_shots=31, qgnn_seed=19, qgnn_blend=0.75)
    python = run_pipeline(lif_engine=python_lif_engine, **kwargs)
    # Reusing this engine for the whole graph run is the actual M2 seam.
    with record_children() as children:
        with PersistentWasmLifEngine() as engine:
            resident = run_pipeline(lif_engine=engine, **kwargs)
            check("24 feedback ticks use one live Node process", len(children) == 1
                  and children[0].poll() is None)
        check("pipeline context reaps its runtime", children[0].poll() is not None)
    total_spikes = int(sum(np.sum(log.spiked) for log in python[1]))
    check(f"feedback scenario is active ({total_spikes} spikes)", total_spikes > 0)
    check("spike-dependent currents differ from baseline",
          any(not np.array_equal(log.current, base) for log in python[1]))
    for attr in ["V", "embedding", "last_spike_time"]:
        check(f"every pipeline state agrees with Python: {attr}",
              len(resident[2]) == len(python[2]) and all(
                  np.array_equal(getattr(a, attr), getattr(b, attr))
                  for a, b in zip(resident[2], python[2])))
    for attr in ["spiked", "current", "qgnn_triggered", "fused_4d"]:
        check(f"every pipeline log agrees with Python: {attr}",
              len(resident[1]) == len(python[1]) and all(
                  np.array_equal(getattr(a, attr), getattr(b, attr))
                  for a, b in zip(resident[1], python[1])))
    check("sampled QGNN logs agree under the same explicit seeds", all(
        a.sampled_expectation_z == b.sampled_expectation_z for a, b in zip(resident[1], python[1])))
    check("final pipeline tick count is correct", resident[0].tick == 24)


def fixture_script(ready: str, response: str | None) -> str:
    """Create a tiny fault source, independent of production server code."""
    script = "import readline from 'node:readline';\n" + ready + "\n"
    if response is not None:
        script += ("readline.createInterface({input:process.stdin,crlfDelay:Infinity})"
                   ".on('line', line => { const request=JSON.parse(line);\n"
                   + response + "\n});\n")
    return script


def test_handshake_failures_reap_process():
    print("\n-- malformed/missing startup handshake --")
    cases = [
        ("startup malformed JSON", "console.log('not-json');setInterval(()=>{},1000);", RuntimeError),
        ("startup wrong protocol", "console.log(JSON.stringify(" + json.dumps({**READY, "protocol": 999}) + "));setInterval(()=>{},1000);", RuntimeError),
        ("startup invalid capacity", "console.log(JSON.stringify(" + json.dumps({**READY, "capacity": 0}) + "));setInterval(()=>{},1000);", RuntimeError),
        ("startup EOF", "process.exit(7);", RuntimeError),
        ("startup timeout", "setInterval(()=>{},1000);", TimeoutError),
    ]
    with tempfile.TemporaryDirectory(prefix="qasm_m2_ready_") as tmp:
        for index, (name, ready, exception) in enumerate(cases):
            server = Path(tmp) / f"ready_{index}.mjs"
            server.write_text(fixture_script(ready, None), encoding="utf-8")
            with patch.object(wasm_bridge, "PERSISTENT_HARNESS", server), record_children() as children:
                start = time.monotonic()
                raises(name, exception, lambda: PersistentWasmLifEngine(timeout=1.0))
                check(f"{name}: child reaped before return", len(children) == 1 and children[0].poll() is not None)
                check(f"{name}: returns within bounded deadline", time.monotonic() - start < 8.0)


def test_reply_failures_are_terminal():
    print("\n-- malformed replies, EOF, remote errors and deadline --")
    ready = "console.log(" + json.dumps(json.dumps(READY)) + ");"
    # Each valid-looking result must be checked before it reaches callers.
    cases = [
        ("reply malformed JSON", "console.log('not-json');", RuntimeError),
        ("reply wrong type", "console.log(JSON.stringify({type:'ready',id:request.id,V:[0,0],spiked:[0,0]}));", RuntimeError),
        ("reply wrong request id", "console.log(JSON.stringify({type:'result',id:request.id+1,V:[0,0],spiked:[0,0]}));", RuntimeError),
        ("reply short voltage", "console.log(JSON.stringify({type:'result',id:request.id,V:[0],spiked:[0,0]}));", RuntimeError),
        ("reply short spikes", "console.log(JSON.stringify({type:'result',id:request.id,V:[0,0],spiked:[0]}));", RuntimeError),
        ("reply nonbinary spikes", "console.log(JSON.stringify({type:'result',id:request.id,V:[0,0],spiked:[0,0.5]}));", RuntimeError),
        ("reply null voltage", "console.log(JSON.stringify({type:'result',id:request.id,V:[null,0],spiked:[0,0]}));", RuntimeError),
        ("reply boolean voltage", "console.log(JSON.stringify({type:'result',id:request.id,V:[true,0],spiked:[0,0]}));", RuntimeError),
        ("reply nonfinite voltage", "console.log('{\"type\":\"result\",\"id\":'+request.id+',\"V\":[NaN,0],\"spiked\":[0,0]}');", RuntimeError),
        ("reply remote error", "console.log(JSON.stringify({type:'error',id:request.id,message:'injected failure'}));", RuntimeError),
        ("reply EOF", "process.exit(9);", RuntimeError),
        ("reply timeout", "setInterval(()=>{},1000);", TimeoutError),
        ("reply oversized line", "console.log('x'.repeat(8388609));", RuntimeError),
    ]
    V = np.zeros(2, dtype=np.float64)
    I = np.ones(2, dtype=np.float64)
    with tempfile.TemporaryDirectory(prefix="qasm_m2_reply_") as tmp:
        for index, (name, response, exception) in enumerate(cases):
            server = Path(tmp) / f"reply_{index}.mjs"
            server.write_text(fixture_script(ready, response), encoding="utf-8")
            with patch.object(wasm_bridge, "PERSISTENT_HARNESS", server), record_children() as children:
                engine = PersistentWasmLifEngine(timeout=1.0)
                try:
                    start = time.monotonic()
                    raises(name, exception, lambda: engine(V, I, 0.1, LifParams()))
                    check(f"{name}: runtime becomes closed", engine.closed)
                    check(f"{name}: child reaped before return", len(children) == 1 and children[0].poll() is not None)
                    check(f"{name}: returns within bounded deadline", time.monotonic() - start < 8.0)
                    raises(f"{name}: next call cannot reuse damaged session", RuntimeError,
                           lambda: engine(V, I, 0.1, LifParams()))
                    check(f"{name}: failure cannot silently restart", len(children) == 1)
                finally:
                    engine.close()
    _check_stderr_backpressure()


def _check_stderr_backpressure():
    """A noisy child's stderr must be drained while stdout awaits a result."""
    ready = "console.log(" + json.dumps(json.dumps(READY)) + ");"
    response = ("process.stderr.write('x'.repeat(1048576),()=>"
                "console.log(JSON.stringify({type:'result',id:request.id,V:[0.01,0.01],spiked:[0,0]}))); ")
    with tempfile.TemporaryDirectory(prefix="qasm_m2_stderr_") as tmp:
        server = Path(tmp) / "noisy_stderr.mjs"
        server.write_text(fixture_script(ready, response), encoding="utf-8")
        with patch.object(wasm_bridge, "PERSISTENT_HARNESS", server), record_children() as children:
            with PersistentWasmLifEngine(timeout=2.0) as engine:
                result, spikes = engine(np.zeros(2, dtype=np.float64),
                                        np.ones(2, dtype=np.float64), 0.1, LifParams())
                check("1 MiB stderr before reply cannot deadlock exchange",
                      np.array_equal(result, [0.01, 0.01]) and np.array_equal(spikes, [0.0, 0.0]))
                check("noisy child remains open after successful reply", not engine.closed
                      and len(children) == 1 and children[0].poll() is None)
            check("noisy child is reaped on context exit", engine.closed and children[0].poll() is not None)


def test_finite_input_overflow_is_explicit():
    print("\n-- finite-input overflow is never silently converted to zero --")
    with record_children() as children:
        engine = PersistentWasmLifEngine()
        try:
            # Negative overflow avoids the normal positive-threshold reset.
            raises("nonfinite computed voltage fails explicitly", RuntimeError,
                   lambda: engine(np.array([0.0], dtype=np.float64),
                                  np.array([-1e308], dtype=np.float64), 1.0,
                                  LifParams(tau=1e-308)))
            check("overflow failure closes and reaps runtime", engine.closed
                  and len(children) == 1 and children[0].poll() is not None)
        finally:
            engine.close()


def test_timeout_covers_blocked_stdin_write():
    print("\n-- deadline also covers a full stdin pipe --")
    # The fixture emits ready but deliberately never reads stdin. Its own
    # five-second exit bounds this test even if client write deadlines fail.
    ready = "console.log(" + json.dumps(json.dumps(READY)) + ");setTimeout(()=>process.exit(0),5000);"
    with tempfile.TemporaryDirectory(prefix="qasm_m2_pipe_") as tmp:
        server = Path(tmp) / "unread_stdin.mjs"
        server.write_text(fixture_script(ready, None), encoding="utf-8")
        with patch.object(wasm_bridge, "PERSISTENT_HARNESS", server), record_children() as children:
            engine = PersistentWasmLifEngine(timeout=1.0)
            try:
                start = time.monotonic()
                raises("full-capacity write to unread stdin times out", TimeoutError,
                       lambda: engine(np.zeros(65536, dtype=np.float64),
                                      np.ones(65536, dtype=np.float64), 0.1, LifParams()))
                check("blocked write deadline precedes fixture's five-second exit",
                      time.monotonic() - start < 4.0)
                check("blocked-write timeout closes and reaps runtime", engine.closed
                      and len(children) == 1 and children[0].poll() is not None)
            finally:
                engine.close()


def main() -> int:
    if not wasm_bridge.wasm_available():
        check("Node on PATH and supplied WASM binary are required (no skip)", False)
    else:
        tests = [test_one_process_scalar_rule_and_cleanup,
                 test_cross_implementation_trajectories,
                 test_capacity_sizes_and_array_isolation,
                 test_invalid_input_does_not_poison_runtime,
                 test_pipeline_feedback_interchangeability,
                 test_handshake_failures_reap_process,
                 test_reply_failures_are_terminal,
                 test_finite_input_overflow_is_explicit,
                 test_timeout_covers_blocked_stdin_write]
        for test in tests:
            try:
                test()
            except Exception as error:
                check(f"{test.__name__} raised {type(error).__name__}: {error}", False)
                traceback.print_exc()
    print(f"\n{PASS_COUNT} PASS markers; {len(FAILURES)} FAIL markers; zero skips")
    print("ALL PASSED" if not FAILURES else "FAILED: " + ", ".join(FAILURES))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
