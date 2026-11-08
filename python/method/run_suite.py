"""
method/run_suite.py -- runs every test module, records per-module check
counts and durations to method/suite_results.json (the build index reads
it), and exits nonzero if anything failed. `bash run_tests.sh` remains the
human-facing entry point; this is the machine-facing one.

Run from python/:  python3 -m method.run_suite
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

PYTHON_DIR = Path(__file__).resolve().parent.parent
OUT = PYTHON_DIR / "method" / "suite_results.json"


def main() -> int:
    modules = sorted(t.stem for t in (PYTHON_DIR / "tests").glob("test_*.py"))
    results = {}
    total_pass = total_fail = 0
    for m in modules:
        t0 = time.time()
        p = subprocess.run([sys.executable, "-m", f"tests.{m}"], cwd=PYTHON_DIR, capture_output=True, text=True)
        passes, fails = p.stdout.count("[PASS]"), p.stdout.count("[FAIL]")
        functions = [ln.strip() for ln in p.stdout.splitlines() if ln.startswith("-- ")]
        results[m] = dict(passes=passes, fails=fails, rc=p.returncode, seconds=round(time.time() - t0, 2),
                          sections=len(functions))
        total_pass += passes
        total_fail += fails
        print(f"  {m:32} {passes:5} pass {fails:3} fail  rc={p.returncode}  {results[m]['seconds']:.1f}s")
    payload = dict(generated_at=time.strftime("%Y-%m-%d %H:%M:%S"), modules=results,
                   total_pass=total_pass, total_fail=total_fail)
    OUT.write_text(json.dumps(payload, indent=1))
    print(f"  {'TOTAL':32} {total_pass:5} pass {total_fail:3} fail  -> {OUT}")
    return 1 if (total_fail or any(r["rc"] != 0 for r in results.values())) else 0


if __name__ == "__main__":
    sys.exit(main())
