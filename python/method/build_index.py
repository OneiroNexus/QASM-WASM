"""
method/build_index.py -- generates BUILD_INDEX.md at the repository root:
one page that says, for this exact tree, what exists, what is claimed,
which test checks each claim, what the last suite run found, and what the
last mutation campaign could and could not see.

Inputs (all produced by `bash run_method.sh`):
  method/claims.py             the claim registry (hand-written)
  method/suite_results.json    python3 -m method.run_suite
  method/mutation_results.json python3 -m method.mutate
  method/triage.json           hand-written verdict on every surviving mutant:
                               {"<mutant id>": {"verdict": "equivalent|gap|config|accepted", "note": "..."}}

Run from python/:  python3 -m method.build_index
"""
from __future__ import annotations

import ast
import json
import sys
import time
from pathlib import Path

from method.claims import CLAIMS, by_layer

PYTHON_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PYTHON_DIR.parent
OUT = PROJECT_ROOT / "BUILD_INDEX.md"

SOURCE_FILES = [
    ("qmodule/statevector_sim.py", "OpenQASM 2.0 subset parser + statevector simulator"),
    ("qmodule/qasm_module.py", "QasmModule: load once / call many; QasmResult separates exact from sampled"),
    ("snn/lif_neuron.py", "Vectorized LIF step; continuous-time analytic rate"),
    ("gnn/message_passing.py", "Sparse gather/scatter aggregation; gated variant; GraphSAGE-style layer"),
    ("fusion/shared_space.py", "3D random projection + elapsed time; the QGNN trigger"),
    ("orchestrator/pipeline.py", "The per-tick composition; pluggable LifEngine"),
    ("orchestrator/wasm_bridge.py", "wasm_lif_engine: one harness subprocess per call"),
    ("wasm-core/orchestrator_core.c", "The LIF core in C, compiled to wasm32; bounds-guarded"),
]


def _functions(path: Path) -> list[str]:
    if path.suffix != ".py":
        return ["orc_max_neurons", "orc_v_ptr", "orc_i_ptr", "orc_spiked_ptr", "orc_tick"]
    tree = ast.parse(path.read_text())
    return [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]


def _load(name: str):
    p = PYTHON_DIR / "method" / name
    return json.loads(p.read_text()) if p.exists() else None


def main() -> int:
    suite = _load("suite_results.json")
    mut = _load("mutation_results.json")
    triage = _load("triage.json") or {}
    lines: list[str] = []
    w = lines.append

    w("# BUILD INDEX")
    w("")
    w(f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')} by `python3 -m method.build_index`. Do not edit by hand; edit "
      "`method/claims.py`, `method/triage.json`, the code, or the tests, and regenerate with `bash run_method.sh`.")
    w("")
    if suite:
        w(f"**Suite:** {suite['total_pass']} checks passing, {suite['total_fail']} failing, across {len(suite['modules'])} modules "
          f"(run {suite['generated_at']}).")
    if mut:
        s = mut["summary"]
        valid = sum(v["killed"] + v["survived"] + v["timeout"] for v in s.values())
        killed = sum(v["killed"] for v in s.values())
        surv = sum(v["survived"] for v in s.values())
        tmo = sum(v["timeout"] for v in s.values())
        w(f"**Mutation campaign:** {killed}/{valid} mutants killed ({100.0 * killed / max(valid, 1):.1f}%), {surv} survived, "
          f"{tmo} timed out, over {len(s)} files (run {mut['generated_at']}, {mut['elapsed_seconds']}s).")
        if surv:
            verdicts = {}
            for m in mut["mutants"]:
                if m["status"] == "survived":
                    v = triage.get(m["id"], {}).get("verdict", "UNTRIAGED")
                    verdicts[v] = verdicts.get(v, 0) + 1
            w("Survivor verdicts: " + ", ".join(f"{k} {v}" for k, v in sorted(verdicts.items())) + ".")
    w(f"**Claims registered:** {len(CLAIMS)} (see section 3).")
    w("")

    # ---------------------------------------------------------------- 1. modules
    w("## 1. Source modules")
    w("")
    w("| File | Purpose | Lines | Functions / classes | Mutants killed / valid | Survived |")
    w("|---|---|---:|---|---:|---:|")
    for rel, purpose in SOURCE_FILES:
        path = (PROJECT_ROOT / rel) if rel.startswith("wasm-core") else (PYTHON_DIR / rel)
        n_lines = len(path.read_text().splitlines())
        fns = ", ".join(_functions(path))
        score = ""
        surv = ""
        if mut and rel in mut["summary"]:
            s = mut["summary"][rel]
            valid = s["killed"] + s["survived"] + s["timeout"]
            score = f"{s['killed']} / {valid}"
            surv = str(s["survived"])
        w(f"| `{rel}` | {purpose} | {n_lines} | {fns} | {score} | {surv} |")
    w("")

    # ---------------------------------------------------------------- 2. tests
    w("## 2. Test modules")
    w("")
    w("| Module | Checks | Sections | Seconds | Claims served |")
    w("|---|---:|---:|---:|---|")
    served: dict[str, set[str]] = {}
    for c in CLAIMS:
        for t in c.tests:
            served.setdefault(t.split("::")[0], set()).add(c.id)
    modules = sorted(t.stem for t in (PYTHON_DIR / "tests").glob("test_*.py"))
    for m in modules:
        r = (suite or {}).get("modules", {}).get(m, {})
        w(f"| `{m}.py` | {r.get('passes', '')} | {r.get('sections', '')} | {r.get('seconds', '')} | "
          f"{', '.join(sorted(served.get(m, set()), key=lambda x: (x[0], int(x[1:]))))} |")
    w("")

    # ---------------------------------------------------------------- 3. claims
    w("## 3. Claims")
    w("")
    w("Every validated property, the mode of validation, the test(s) that check it, and the version that added it. "
      "A property not listed here is not validated.")
    w("")
    for layer, claims in by_layer().items():
        w(f"### {layer}")
        w("")
        w("| Id | Claim | Mode | Checked by | Since |")
        w("|---|---|---|---|---|")
        for c in claims:
            tests = "<br>".join(f"`{t}`" for t in c.tests)
            w(f"| {c.id} | {c.statement} | {c.mode} | {tests} | {c.since} |")
        w("")

    # ---------------------------------------------------------------- 4. mutation
    w("## 4. Mutation campaign")
    w("")
    if not mut:
        w("_No campaign results found. Run `python3 -m method.mutate`._")
    else:
        w("Operators: AOR (arithmetic), ROR (relational), LCR (logical), CRP (constants), SDL (statement deletion), "
          "AGS (argument swap); explicit textual mutants for the C core. One fresh copy of the tree per mutant; the "
          "test modules that transitively import the mutated file run fastest-first and stop at the first failure.")
        w("")
        w("| File | Killed | Survived | Timeout | Invalid | Score |")
        w("|---|---:|---:|---:|---:|---:|")
        for f, s in mut["summary"].items():
            valid = s["killed"] + s["survived"] + s["timeout"]
            w(f"| `{f}` | {s['killed']} | {s['survived']} | {s['timeout']} | {s['invalid']} | "
              f"{100.0 * s['killed'] / max(valid, 1):.1f}% |")
        w("")
        by_op: dict[str, dict[str, int]] = {}
        for m in mut["mutants"]:
            d = by_op.setdefault(m["operator"], {"killed": 0, "survived": 0, "timeout": 0, "invalid": 0})
            d[m["status"]] += 1
        w("| Operator | Killed | Survived | Timeout | Invalid |")
        w("|---|---:|---:|---:|---:|")
        for op, d in sorted(by_op.items()):
            w(f"| {op} | {d['killed']} | {d['survived']} | {d['timeout']} | {d['invalid']} |")
        w("")
        killers: dict[str, int] = {}
        for m in mut["mutants"]:
            if m["status"] == "killed":
                killers[m["killed_by"]] = killers.get(m["killed_by"], 0) + 1
        w("First test module to kill (fastest-first order, so this credits the cheapest sufficient test): "
          + ", ".join(f"`{k}` {v}" for k, v in sorted(killers.items(), key=lambda kv: -kv[1])) + ".")
        w("")
        w("### Survivors and timeouts, with verdicts")
        w("")
        w("| Id | File:line | Operator | Original | Mutant | Status | Verdict | Note |")
        w("|---|---|---|---|---|---|---|---|")
        for m in mut["mutants"]:
            if m["status"] in ("survived", "timeout"):
                t = triage.get(m["id"], {})
                esc = lambda s: s.replace("|", "\\|").replace("\n", " ")  # noqa: E731
                w(f"| {m['id']} | `{m['file']}:{m['lineno']}` | {m['operator']} | `{esc(m['original'][:60])}` | "
                  f"`{esc(m['replacement'][:60])}` | {m['status']} | {t.get('verdict', 'UNTRIAGED')} | {esc(t.get('note', ''))} |")
        w("")

    # ---------------------------------------------------------------- 5. method
    w("## 5. How to regenerate")
    w("")
    w("```bash")
    w("bash run_method.sh          # suite -> mutation campaign -> this index (about 3 minutes on 2 cores)")
    w("bash run_tests.sh           # the suite alone (seconds)")
    w("cd python && python3 -m method.mutate --only qmodule/statevector_sim.py   # one file")
    w("```")
    w("")
    w("Verdict vocabulary for `method/triage.json`: **equivalent** (the mutant cannot change observable behaviour), "
      "**config** (a configuration constant, not logic), **accepted** (a behaviour change that no test should pin, with the reason), "
      "**gap** (a real blind spot: a test must be added, or the code refactored so the mutant cannot exist).")
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT} ({len(lines)} lines)")
    if "--require-triage" in sys.argv and mut:
        untriaged = [m["id"] for m in mut["mutants"] if m["status"] in ("survived", "timeout") and m["id"] not in triage]
        if untriaged:
            print(f"UNTRIAGED survivors ({len(untriaged)}): " + ", ".join(untriaged))
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
