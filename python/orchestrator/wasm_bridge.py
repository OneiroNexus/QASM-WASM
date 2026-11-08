"""
orchestrator/wasm_bridge.py -- adapts wasm-core/orchestrator_core.wasm (via
../../wasm-core/run_tick_harness.mjs) to exactly the same LifEngine
signature as snn.lif_neuron.step(), so pipeline.run_tick/run_pipeline can
use either engine interchangeably.

Used by tests/test_pipeline_smoke.py Part 2 to extend Part 1's proof
(WASM == Python for a fixed, externally-supplied constant current) to the
harder case: WASM == Python even when the input current is itself produced
fresh every tick by the graph-aggregation feedback loop, i.e. the two
engines are proven interchangeable as the actual hot-loop component of a
real orchestrator, not just as isolated neuron models.

wasm_lif_engine retains the original one-process-per-tick baseline.
PersistentWasmLifEngine adds M2's long-lived Node/WASM host without
changing the numerical core or the pipeline's engine signature. Its
linear memory is resident, but V and I are explicitly supplied and
copied on every call; JSON serialization is still part of every tick.
The bounded request/response protocol has an I/O deadline and fails
closed instead of silently changing backends. Cross-engine agreement,
input isolation, and lifecycle failures are checked by
tests/test_persistent_wasm.py.
"""
from __future__ import annotations

import json
import math
import numbers
import queue
import shutil
import subprocess
import threading
from pathlib import Path

import numpy as np

from snn.lif_neuron import LifParams

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent   # .../hybrid-gnn-snn-qgnn
WASM_CORE = PROJECT_ROOT / "wasm-core"
HARNESS = WASM_CORE / "run_tick_harness.mjs"
WASM_BIN = WASM_CORE / "orchestrator_core.wasm"
PERSISTENT_HARNESS = WASM_CORE / "persistent_tick_server.mjs"
MAX_MESSAGE_BYTES = 8 * 1024 * 1024
_PROTOCOL_VERSION = 1


def wasm_available() -> bool:
    """True only if both `node` is on PATH and the wasm binary has been
    built (run wasm-core/build.sh). Callers should skip (not fail) the
    WASM-engine comparison when this is False, with a clear printed
    reason -- an absent toolchain is an environment fact, not a bug."""
    return shutil.which("node") is not None and WASM_BIN.exists()


def wasm_lif_engine(V: np.ndarray, I: np.ndarray, dt: float, p: LifParams):
    """Same (V, I, dt, p) -> (V_next, spiked) contract as
    snn.lif_neuron.step(), backed by exactly one orc_tick() call inside
    orchestrator_core.wasm."""
    node = shutil.which("node")
    if node is None:
        raise RuntimeError("node not found on PATH -- required for wasm_lif_engine")
    if not WASM_BIN.exists():
        raise RuntimeError(f"{WASM_BIN} not built -- run `bash wasm-core/build.sh` first")
    n = len(V)
    scenario = dict(n=n, dt=dt, tau=p.tau, v_rest=p.v_rest, v_reset=p.v_reset,
                     v_th=p.v_th, R=p.R, V_init=np.asarray(V, dtype=np.float64).tolist(),
                     I=np.asarray(I, dtype=np.float64).tolist(), num_ticks=1)
    proc = subprocess.run([node, str(HARNESS)], input=json.dumps(scenario),
                           capture_output=True, text=True, timeout=30)
    if proc.returncode != 0:
        raise RuntimeError(f"node harness failed (exit {proc.returncode}):\n{proc.stderr}")
    out = json.loads(proc.stdout)
    V_next = np.array(out["V_history"][0], dtype=np.float64)
    spiked = np.array(out["spiked_history"][0], dtype=np.float64)
    return V_next, spiked


def _reject_json_constant(value: str):
    raise ValueError(f"non-finite JSON constant: {value}")


def _read_message(stream) -> dict:
    """Read one bounded UTF-8 JSONL frame, including its terminating LF."""
    line = stream.readline(MAX_MESSAGE_BYTES + 1)
    if not line:
        raise RuntimeError("persistent WASM child closed stdout unexpectedly")
    if len(line) > MAX_MESSAGE_BYTES or not line.endswith(b"\n"):
        raise RuntimeError("persistent WASM response exceeds the frame bound or lacks a newline")
    try:
        message = json.loads(line.decode("utf-8"), parse_constant=_reject_json_constant)
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise RuntimeError("persistent WASM child returned invalid JSON") from exc
    if not isinstance(message, dict):
        raise RuntimeError("persistent WASM response must be a JSON object")
    return message


def _exchange_worker(proc, requests: queue.Queue, replies: queue.Queue) -> None:
    """Own pipe I/O so a deadline also covers a blocked Windows pipe write."""
    try:
        replies.put(_read_message(proc.stdout))
        while True:
            request = requests.get()
            if request is None:
                return
            remaining = memoryview(request)
            while remaining:
                written = proc.stdin.write(remaining)
                if written is None or written <= 0:
                    raise RuntimeError("persistent WASM stdin write failed")
                remaining = remaining[written:]
            proc.stdin.flush()
            replies.put(_read_message(proc.stdout))
    except Exception as exc:
        # At most one exchange is outstanding; do not block cleanup if a reply
        # was already delivered immediately before the child failed.
        try:
            replies.put_nowait(exc)
        except queue.Full:
            pass


def _drain_stderr(stream, tail: bytearray, lock: threading.Lock) -> None:
    """Drain continuously without allowing a noisy child to consume unbounded RAM."""
    try:
        while True:
            chunk = stream.read(4096)
            if not chunk:
                return
            with lock:
                tail.extend(chunk)
                del tail[:-4096]
    except (OSError, ValueError):
        return  # Closing the child and its pipes is an expected lifecycle event.


def _finite_scalar(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_vector(value, name: str) -> np.ndarray:
    try:
        array = np.asarray(value)
        if array.ndim != 1 or array.dtype.kind not in "iuf":
            raise ValueError(f"{name} must be a one-dimensional real numeric array")
        # Copy before sending: caller inputs may be read-only or strided and
        # must never alias returned state or the resident WASM buffers.
        array = np.array(array, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a one-dimensional real numeric array") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


class PersistentWasmLifEngine:
    """One Node child and one WASM instance, callable as ``(V, I, dt, p)``.

    Construction waits for a capacity handshake. ``timeout`` bounds each
    handshake or pipe exchange, including writes; serialization and local input
    validation precede that deadline. Calls are serialized by a lock. Caller
    validation errors leave the engine usable; child, protocol, and timeout
    failures close and reap it, with no restart or fallback. Use a context
    manager or explicitly call ``close()``. See tests/test_persistent_wasm.py
    for per-call input isolation and cross-engine/lifecycle checks.

    This JSON transport preserves finite numerical values, not signed-zero
    bit patterns. It does not make a universal bitwise-portability guarantee.
    """

    def __init__(self, timeout: float = 10.0):
        self._timeout = _finite_scalar(timeout, "timeout")
        if self._timeout <= 0:
            raise ValueError("timeout must be positive")
        self._lock = threading.RLock()
        self._closed = False
        self._proc = None
        self._worker = None
        self._stderr_worker = None
        self._requests = queue.Queue(maxsize=1)
        self._replies = queue.Queue(maxsize=1)
        self._stderr_tail = bytearray()
        self._stderr_lock = threading.Lock()
        self._next_id = 1
        self._capacity = 0
        node = shutil.which("node")
        if node is None:
            self._closed = True
            raise RuntimeError("node not found on PATH -- required for PersistentWasmLifEngine")
        if not WASM_BIN.is_file() or not Path(PERSISTENT_HARNESS).is_file():
            self._closed = True
            raise RuntimeError("persistent WASM files missing -- build wasm-core and retain persistent_tick_server.mjs")
        try:
            self._proc = subprocess.Popen(
                [node, str(PERSISTENT_HARNESS)], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            self._stderr_worker = threading.Thread(
                target=_drain_stderr,
                args=(self._proc.stderr, self._stderr_tail, self._stderr_lock),
                daemon=True, name="persistent-wasm-stderr",
            )
            self._worker = threading.Thread(
                target=_exchange_worker, args=(self._proc, self._requests, self._replies),
                daemon=True, name="persistent-wasm-io",
            )
            self._stderr_worker.start()
            self._worker.start()
            ready = self._receive("startup")
            capacity = ready.get("capacity")
            if (ready.get("type") != "ready"
                    or type(ready.get("protocol")) is not int
                    or ready["protocol"] != _PROTOCOL_VERSION
                    or type(capacity) is not int or not 0 < capacity <= 2**31 - 1
                    or type(ready.get("max_message_bytes")) is not int
                    or ready["max_message_bytes"] != MAX_MESSAGE_BYTES):
                raise RuntimeError("invalid persistent WASM ready handshake")
            self._capacity = capacity
        except BaseException:
            self.close()
            raise

    @property
    def capacity(self) -> int:
        """Neuron capacity reported by the instantiated module."""
        return self._capacity

    @property
    def pid(self) -> int:
        """Child process ID; retained after close for lifecycle diagnostics."""
        return self._proc.pid

    @property
    def closed(self) -> bool:
        return self._closed

    def _receive(self, phase: str) -> dict:
        try:
            reply = self._replies.get(timeout=self._timeout)
        except queue.Empty as exc:
            raise TimeoutError(f"persistent WASM {phase} timed out after {self._timeout:g}s") from exc
        if isinstance(reply, Exception):
            with self._stderr_lock:
                detail = bytes(self._stderr_tail).decode("utf-8", errors="replace").strip()
            suffix = f": {detail}" if detail else ""
            raise RuntimeError(f"persistent WASM {phase} failed: {reply}{suffix}") from reply
        return reply

    def __call__(self, V: np.ndarray, I: np.ndarray, dt: float, p: LifParams):
        with self._lock:
            if self._closed:
                raise RuntimeError("persistent WASM engine is closed; create a new instance explicitly")
            voltage = _finite_vector(V, "V")
            current = _finite_vector(I, "I")
            n = voltage.size
            if current.shape != voltage.shape:
                raise ValueError("V and I must have the same one-dimensional shape")
            if n > self._capacity:
                raise ValueError(f"n={n} exceeds persistent WASM capacity {self._capacity}")
            timestep = _finite_scalar(dt, "dt")
            if timestep < 0:
                raise ValueError("dt must be non-negative")
            if not isinstance(p, LifParams):
                raise ValueError("p must be LifParams")
            scalars = {name: _finite_scalar(getattr(p, name), name)
                       for name in ("tau", "v_rest", "v_reset", "v_th", "R")}
            if scalars["tau"] <= 0:
                raise ValueError("tau must be positive")
            request_id = self._next_id
            request = dict(type="tick", id=request_id, n=n, dt=timestep,
                           **scalars, V=voltage.tolist(), I=current.tolist())
            frame = (json.dumps(request, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
            if len(frame) > MAX_MESSAGE_BYTES:
                raise ValueError(f"request exceeds {MAX_MESSAGE_BYTES}-byte JSONL frame bound")
            try:
                self._requests.put_nowait(frame)
                reply = self._receive("tick")
                if reply.get("type") == "error":
                    message = reply.get("message")
                    if not isinstance(message, str):
                        message = "malformed error response"
                    raise RuntimeError(f"persistent WASM child rejected tick: {message[:1000]}")
                if (reply.get("type") != "result" or type(reply.get("id")) is not int
                        or reply["id"] != request_id):
                    raise RuntimeError("persistent WASM response type or request ID mismatch")
                # Validate JSON types before conversion: null and booleans must
                # not silently become NaN/0/1 through NumPy coercion.
                for name in ("V", "spiked"):
                    values = reply.get(name)
                    if (not isinstance(values, list) or len(values) != n
                            or any(type(value) not in (int, float) for value in values)):
                        raise RuntimeError(f"invalid persistent WASM {name} response shape or values")
                try:
                    next_voltage = _finite_vector(reply["V"], "response V")
                    spiked = _finite_vector(reply["spiked"], "response spiked")
                except ValueError as exc:
                    raise RuntimeError("persistent WASM returned non-finite or invalid output") from exc
                if not np.all((spiked == 0.0) | (spiked == 1.0)):
                    raise RuntimeError("persistent WASM spike outputs must be 0 or 1")
                self._next_id += 1
                return next_voltage, spiked
            except BaseException:
                self.close()
                raise

    def close(self) -> None:
        """Idempotently terminate/reap the child and release its pipes/threads.

        The protocol carries no user state to flush; explicit termination also
        unblocks an I/O worker stuck in either pipe direction. Closing does not
        implicitly execute another tick. See tests/test_persistent_wasm.py.
        """
        lock = getattr(self, "_lock", None)
        if lock is None:
            return
        with lock:
            if self._closed:
                return
            self._closed = True
            proc = self._proc
            if proc is not None:
                if proc.poll() is None:
                    try:
                        proc.terminate()
                    except ProcessLookupError:
                        pass
                try:
                    proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=1.0)
            try:
                self._requests.put_nowait(None)
            except queue.Full:
                pass
            for worker in (self._worker, self._stderr_worker):
                if worker is not None and worker.ident is not None:
                    worker.join(timeout=1.0)
            if proc is not None:
                for stream in (proc.stdin, proc.stdout, proc.stderr):
                    if stream is not None:
                        try:
                            stream.close()
                        except (BrokenPipeError, OSError):
                            pass

    def __enter__(self):
        if self._closed:
            raise RuntimeError("persistent WASM engine is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass  # Explicit close/context management is the supported lifecycle.
