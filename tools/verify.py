"""Run every standalone test without overwriting historical experiment ledgers."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ORDER = [
    "test_qasm_sim", "test_qasm_gate_coverage", "test_qasm_parser_robustness",
    "test_lif_neuron", "test_gnn_message_passing", "test_shared_space",
    "test_pipeline_smoke", "test_composition_wiring", "test_wasm_core_bounds",
    "test_properties", "test_persistent_wasm",
]

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="New directory for raw logs and a machine-readable summary")
    args = parser.parse_args()
    if shutil.which("node") is None:
        parser.error("Node.js must be on PATH; WASM checks may not be skipped")
    binary = ROOT / "wasm-core" / "orchestrator_core.wasm"
    if not binary.is_file():
        parser.error("WASM binary missing; run bash wasm-core/build.sh")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    test_dir = ROOT / "python" / "tests"
    modules = ORDER + sorted(p.stem for p in test_dir.glob("test_*.py")
                             if p.stem not in ORDER)
    results = []
    for module in modules:
        started = time.perf_counter()
        try:
            run = subprocess.run(
                [sys.executable, "-m", "tests." + module],
                cwd=ROOT / "python", env=env, capture_output=True,
                text=True, encoding="utf-8", errors="replace", timeout=300,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            stdout, stderr, code = run.stdout, run.stderr, run.returncode
        except subprocess.TimeoutExpired as exc:
            def decode(value):
                return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else (value or "")
            stdout, stderr, code = decode(exc.stdout), decode(exc.stderr), -1
            stderr += "\nTest module exceeded the 300-second deadline.\n"
        (output / (module + ".stdout.txt")).write_text(stdout, encoding="utf-8")
        (output / (module + ".stderr.txt")).write_text(stderr, encoding="utf-8")
        skipped = sum("[SKIP]" in line.upper() for line in (stdout + stderr).splitlines())
        fail_markers = stdout.count("[FAIL]")
        result = dict(module=module, exit_code=code, elapsed_seconds=time.perf_counter()-started,
                      pass_markers=stdout.count("[PASS]"), fail_markers=fail_markers,
                      skip_lines=skipped, stderr_bytes=len(stderr.encode("utf-8")),
                      passed=code == 0 and fail_markers == 0 and skipped == 0)
        results.append(result)
        print(f"{module}: {'PASS' if result['passed'] else 'FAIL'} ({result['elapsed_seconds']:.2f}s)", flush=True)
    import numpy as np
    summary = dict(
        status="PASS" if all(x["passed"] for x in results) else "FAIL",
        python=platform.python_version(), numpy=np.__version__,
        node=subprocess.check_output(["node", "--version"], text=True).strip(),
        platform=platform.system(), wasm_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        wasm_rebuilt_by_this_command=False, modules=results,
        total_pass_markers=sum(x["pass_markers"] for x in results),
        total_fail_markers=sum(x["fail_markers"] for x in results),
        total_skip_lines=sum(x["skip_lines"] for x in results),
        note="PASS markers summarize checks, including seeded properties; they are not independent experiments. Historical ledgers are untouched.",
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    print(summary["status"], flush=True)
    return 0 if summary["status"] == "PASS" else 1

if __name__ == "__main__":
    raise SystemExit(main())

