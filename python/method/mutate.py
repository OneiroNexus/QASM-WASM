"""
method/mutate.py -- operator-based mutation testing for the whole scaffold.

Why this exists: the independent review (VALIDATION.md, thesis Chapter 7)
showed that a suite of 1,039 passing assertions let eleven of eighteen
hand-chosen defects through. Eighteen hand-chosen defects is a probe, not a
measurement. This module generates mutants mechanically, from a fixed set
of operators applied to every eligible expression in every source module,
so that the mutation score is a property of the suite rather than of the
reviewer's imagination. It is the fourth validation mode of thesis Section
4.2, made systematic.

Operators (Python, AST-based, one mutant per site):
  AOR  arithmetic operator replacement      + <-> -, * <-> /, unary minus dropped
  ROR  relational operator replacement      >= -> >, > -> >=, <= -> <, < -> <=, == <-> !=
  LCR  logical connector replacement        and <-> or, `not x` -> x
  CRP  constant replacement                 0 -> 1, 1 -> 0, 0.0 -> 1.0, 1.0 -> 0.0, c -> c+1
  SDL  statement deletion                   an assignment / augmented assignment /
                                            expression statement inside a function -> pass
  AGS  argument swap                        the last two positional arguments of a call
Operators (C, textual, listed explicitly in C_MUTANTS): the update rule's
arithmetic, the threshold comparison, the loop bound, the bounds guard.

Defaults in function signatures and dataclass fields are never mutated:
they are configuration, and a test that derives its expectation from the
same parameter object cannot see them change -- that is not a defect class.

Each mutant is applied to a fresh copy of the tree; the test modules that
(transitively) import the mutated module are run fastest-first and stop at
the first failure. Outcomes: killed (a test failed or crashed), timeout
(a test exceeded its budget -- counted separately, not as killed),
survived. Results are written to method/mutation_results.json keyed by a
content hash of (file, operator, original text, replacement, occurrence),
so verdicts in method/triage.json survive line-number changes.

Run from python/:  python3 -m method.mutate [--workers 2] [--only qmodule/statevector_sim.py]
"""
from __future__ import annotations

import ast
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

PYTHON_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PYTHON_DIR.parent
RESULTS_PATH = PYTHON_DIR / "method" / "mutation_results.json"

SOURCE_FILES = [
    "qmodule/statevector_sim.py",
    "qmodule/qasm_module.py",
    "snn/lif_neuron.py",
    "gnn/message_passing.py",
    "fusion/shared_space.py",
    "orchestrator/pipeline.py",
    "orchestrator/wasm_bridge.py",
]

# Textual mutants for the C core: (id suffix, original, replacement)
C_FILE = "wasm-core/orchestrator_core.c"
C_MUTANTS = [
    ("C-AOR-1", "double v_next = v + (dt / tau) * (-(v - v_rest) + R * i_in);", "double v_next = v - (dt / tau) * (-(v - v_rest) + R * i_in);"),
    ("C-AOR-2", "double v_next = v + (dt / tau) * (-(v - v_rest) + R * i_in);", "double v_next = v + (dt * tau) * (-(v - v_rest) + R * i_in);"),
    ("C-AOR-3", "double v_next = v + (dt / tau) * (-(v - v_rest) + R * i_in);", "double v_next = v + (dt / tau) * ((v - v_rest) + R * i_in);"),
    ("C-AOR-4", "double v_next = v + (dt / tau) * (-(v - v_rest) + R * i_in);", "double v_next = v + (dt / tau) * (-(v - v_rest) + R / i_in);"),
    ("C-ROR-1", "if (v_next >= v_th) {", "if (v_next > v_th) {"),
    ("C-ROR-2", "for (int k = 0; k < n; k++) {", "for (int k = 0; k <= n; k++) {"),
    ("C-ROR-3", "for (int k = 0; k < n; k++) {", "for (int k = 1; k < n; k++) {"),
    ("C-ROR-4", "if (n < 0 || n > MAX_NEURONS) {", "if (n < 0 || n >= MAX_NEURONS) {"),
    ("C-SDL-1", "            v_next = v_reset;\n", "\n"),
    ("C-SDL-2", "            spike_count++;\n", "\n"),
    ("C-SDL-3", "        SPIKED[k] = spiked;\n", "\n"),
    ("C-CRP-1", "        double spiked = 0.0;\n", "        double spiked = 1.0;\n"),
]

# Measured with `bash run_tests.sh`; used only to order test modules fastest-first.
TEST_COST = {
    "test_qasm_parser_robustness": 0.1, "test_gnn_message_passing": 0.1, "test_qasm_sim": 0.2,
    "test_shared_space": 0.2, "test_qasm_gate_coverage": 0.3, "test_composition_wiring": 0.4,
    "test_properties": 0.8, "test_lif_neuron": 1.2, "test_pipeline_smoke": 1.7, "test_wasm_core_bounds": 2.0,
}
PER_TEST_TIMEOUT = 30


@dataclass
class Mutant:
    id: str
    file: str
    operator: str
    lineno: int
    original: str
    replacement: str
    occurrence: int
    span: tuple = (0, 0, 0, 0)       # (lineno, col, end_lineno, end_col) in the source this was generated from
    status: str = "pending"          # killed | survived | timeout | invalid
    killed_by: str = ""
    tests_run: list = field(default_factory=list)
    seconds: float = 0.0
    detail: str = ""


# ---------------------------------------------------------------------------
# Import graph -> which test modules exercise which source file
# ---------------------------------------------------------------------------

def _imports_of(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    pkgs = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                pkgs.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            pkgs.add(node.module.split(".")[0])
    return pkgs


def build_selection() -> dict[str, list[str]]:
    """Maps each mutable file (Python or C) to the ordered list of test
    modules that transitively depend on it."""
    packages = sorted({f.split("/")[0] for f in SOURCE_FILES})
    pkg_deps: dict[str, set[str]] = {}
    for p in packages:
        deps = set()
        for f in (PYTHON_DIR / p).glob("*.py"):
            deps |= {d for d in _imports_of(f) if d in packages and d != p}
        pkg_deps[p] = deps

    def closure(p: str) -> set[str]:
        seen, stack = set(), [p]
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            stack.extend(pkg_deps.get(x, ()))
        return seen

    tests = sorted(t.stem for t in (PYTHON_DIR / "tests").glob("test_*.py"))
    test_pkgs = {t: {d for d in _imports_of(PYTHON_DIR / "tests" / f"{t}.py") if d in packages} for t in tests}
    test_uses_wasm = {t: ("orchestrator_core.wasm" in (PYTHON_DIR / "tests" / f"{t}.py").read_text()
                          or "wasm_lif_engine" in (PYTHON_DIR / "tests" / f"{t}.py").read_text()) for t in tests}

    selection: dict[str, list[str]] = {}
    for f in SOURCE_FILES:
        pkg = f.split("/")[0]
        chosen = [t for t in tests if any(pkg in closure(d) for d in test_pkgs[t])]
        selection[f] = sorted(chosen, key=lambda t: TEST_COST.get(t, 5.0))
    selection[C_FILE] = sorted([t for t in tests if test_uses_wasm[t]], key=lambda t: TEST_COST.get(t, 5.0))
    return selection


# ---------------------------------------------------------------------------
# Python mutant generation
# ---------------------------------------------------------------------------

_AOR = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Div, ast.Div: ast.Mult}
_ROR = {ast.GtE: ast.Gt, ast.Gt: ast.GtE, ast.LtE: ast.Lt, ast.Lt: ast.LtE, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq}
_LCR = {ast.And: ast.Or, ast.Or: ast.And}


def _protected_spans(tree: ast.AST) -> list[tuple[int, int, int, int]]:
    """Source spans never mutated: default values in signatures, dataclass
    field defaults (assignments with annotations at class level), and
    string constants (docstrings, messages)."""
    spans = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for d in list(node.args.defaults) + [d for d in node.args.kw_defaults if d is not None]:
                spans.append((d.lineno, d.col_offset, d.end_lineno, d.end_col_offset))
        if isinstance(node, ast.ClassDef):
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
                    spans.append((stmt.value.lineno, stmt.value.col_offset, stmt.value.end_lineno, stmt.value.end_col_offset))
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            spans.append((node.lineno, node.col_offset, node.end_lineno, node.end_col_offset))
        if isinstance(node, ast.JoinedStr):
            spans.append((node.lineno, node.col_offset, node.end_lineno, node.end_col_offset))
    return spans


def _inside(node: ast.AST, spans) -> bool:
    for (l0, c0, l1, c1) in spans:
        if (node.lineno, node.col_offset) >= (l0, c0) and (node.end_lineno, node.end_col_offset) <= (l1, c1):
            return True
    return False


def _in_function(node_parents: dict, node: ast.AST) -> bool:
    p = node_parents.get(node)
    while p is not None:
        if isinstance(p, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            return True
        p = node_parents.get(p)
    return False


def generate_python_mutants(rel: str, src: str) -> list[Mutant]:
    tree = ast.parse(src)
    parents = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    protected = _protected_spans(tree)
    lines = src.splitlines(keepends=True)
    out: list[Mutant] = []
    counts: dict[str, int] = {}

    def segment(node):
        return ast.get_source_segment(src, node)

    def add(op: str, node: ast.AST, new_node: ast.AST | None, replacement_text: str | None = None):
        orig = segment(node)
        if orig is None:
            return
        new = replacement_text if replacement_text is not None else ast.unparse(new_node)
        if new == orig:
            return
        key = (rel, op, orig, new)
        counts[key] = counts.get(key, 0) + 1
        occ = counts[key]
        h = hashlib.sha1(f"{rel}|{op}|{orig}|{new}|{occ}".encode()).hexdigest()[:10]
        out.append(Mutant(id=f"{op}-{h}", file=rel, operator=op, lineno=node.lineno,
                          original=orig, replacement=new, occurrence=occ,
                          span=(node.lineno, node.col_offset, node.end_lineno, node.end_col_offset)))

    for node in ast.walk(tree):
        if not hasattr(node, "lineno") or _inside(node, protected):
            continue
        # AOR
        if isinstance(node, ast.BinOp) and type(node.op) in _AOR:
            new = ast.BinOp(left=node.left, op=_AOR[type(node.op)](), right=node.right)
            add("AOR", node, new)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and _in_function(parents, node):
            add("AOR", node, node.operand)
        # ROR
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in _ROR:
            new = ast.Compare(left=node.left, ops=[_ROR[type(node.ops[0])]()], comparators=node.comparators)
            add("ROR", node, new)
        # LCR
        if isinstance(node, ast.BoolOp) and type(node.op) in _LCR:
            new = ast.BoolOp(op=_LCR[type(node.op)](), values=node.values)
            add("LCR", node, new)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            add("LCR", node, node.operand)
        # CRP (numeric constants inside function bodies only)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool) \
                and _in_function(parents, node):
            v = node.value
            if v == 0:
                nv = 1 if isinstance(v, int) else 1.0
            elif v == 1:
                nv = 0 if isinstance(v, int) else 0.0
            else:
                nv = v + 1
            add("CRP", node, ast.Constant(value=nv))
        # SDL (simple statements inside functions, not docstrings / returns / raises)
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.Expr)) and _in_function(parents, node):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                continue  # docstring
            parent = parents.get(node)
            if isinstance(parent, (ast.FunctionDef, ast.For, ast.While, ast.If, ast.With, ast.Try)):
                add("SDL", node, None, replacement_text="pass")
        # AGS
        if isinstance(node, ast.Call) and len(node.args) >= 2 and _in_function(parents, node):
            a, b = node.args[-2], node.args[-1]
            if isinstance(a, ast.Starred) or isinstance(b, ast.Starred):
                continue
            new_args = list(node.args)
            new_args[-2], new_args[-1] = b, a
            new = ast.Call(func=node.func, args=new_args, keywords=node.keywords)
            add("AGS", node, new)
    return out


def apply_python_mutant(src: str, m: Mutant) -> str | None:
    """Replaces the node's source span (ast positions are UTF-8 byte
    offsets) with the replacement. Returns None if the span no longer holds
    the original text (stale mutant)."""
    l0, c0, l1, c1 = m.span
    b = src.encode("utf-8")
    line_starts = [0]
    for i, ch in enumerate(b):
        if ch == 0x0A:
            line_starts.append(i + 1)
    if l1 > len(line_starts):
        return None
    start, end = line_starts[l0 - 1] + c0, line_starts[l1 - 1] + c1
    if b[start:end].decode("utf-8") != m.original:
        return None
    return (b[:start] + m.replacement.encode("utf-8") + b[end:]).decode("utf-8")


def generate_c_mutants() -> list[Mutant]:
    src = (PROJECT_ROOT / C_FILE).read_text()
    out = []
    for suffix, orig, new in C_MUTANTS:
        if src.count(orig) != 1:
            continue
        line = src[:src.index(orig)].count("\n") + 1
        h = hashlib.sha1(f"{C_FILE}|{suffix}|{orig}|{new}".encode()).hexdigest()[:10]
        out.append(Mutant(id=f"{suffix}-{h}", file=C_FILE, operator=suffix.split("-")[1], lineno=line,
                          original=orig, replacement=new, occurrence=1))
    return out


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def _fresh_tree(dst: Path):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(PROJECT_ROOT, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git", "method/mutation_results.json"))


def _run_tests(tree: Path, tests: list[str]) -> tuple[str, str, list[str], str]:
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    ran = []
    for t in tests:
        for pc in (tree / "python").rglob("__pycache__"):
            shutil.rmtree(pc, ignore_errors=True)
        ran.append(t)
        try:
            p = subprocess.run([sys.executable, "-B", "-m", f"tests.{t}"], cwd=tree / "python",
                               capture_output=True, text=True, timeout=PER_TEST_TIMEOUT, env=env)
        except subprocess.TimeoutExpired:
            return "timeout", t, ran, f"{t} exceeded {PER_TEST_TIMEOUT}s"
        fails = p.stdout.count("[FAIL]")
        if fails > 0:
            return "killed", t, ran, f"{fails} assertion(s) failed"
        if p.returncode != 0:
            tail = (p.stderr.strip().splitlines() or ["(no stderr)"])[-1]
            return "killed", t, ran, f"crash: {tail[:160]}"
    return "survived", "", ran, ""


def _worker(args):
    worker_id, mutants, selection, sources = args
    tree = Path(tempfile.gettempdir()) / f"hybrid-mut-w{worker_id}"
    _fresh_tree(tree)
    # The worker's copy must match the snapshot the mutants were generated
    # from, even if the live tree is edited while the campaign runs.
    for rel, text in sources.items():
        (tree / ("" if rel == C_FILE else "python") / rel).write_text(text)
    results = []
    for m in mutants:
        t0 = time.time()
        sub = "" if m.file == C_FILE else "python"
        target = tree / sub / m.file
        pristine = sources[m.file]
        if m.file == C_FILE:
            mutated = pristine.replace(m.original, m.replacement, 1)
        else:
            mutated = apply_python_mutant(pristine, m)
        if mutated is None:
            m.status, m.detail = "invalid", "site not found"
            results.append(m)
            continue
        target.write_text(mutated)
        ok = True
        if m.file == C_FILE:
            b = subprocess.run(["bash", "wasm-core/build.sh"], cwd=tree, capture_output=True, text=True)
            ok = b.returncode == 0
            if not ok:
                m.status, m.detail = "invalid", "C mutant failed to compile"
        else:
            try:
                compile(mutated, m.file, "exec")
            except SyntaxError as e:
                ok = False
                m.status, m.detail = "invalid", f"syntax: {e.msg}"
        if ok:
            m.status, m.killed_by, m.tests_run, m.detail = _run_tests(tree, selection[m.file])
        m.seconds = round(time.time() - t0, 2)
        # restore
        target.write_text(pristine)
        if m.file == C_FILE:
            subprocess.run(["bash", "wasm-core/build.sh"], cwd=tree, capture_output=True, text=True)
        results.append(m)
        print(f"  [{m.status:8}] {m.id:18} {m.file}:{m.lineno}  {m.original[:40]!r} -> {m.replacement[:40]!r}"
              + (f"  by {m.killed_by}" if m.killed_by else ""), flush=True)
    shutil.rmtree(tree, ignore_errors=True)
    return results


def main(argv: list[str]) -> int:
    workers = 2
    only = None
    for i, a in enumerate(argv):
        if a == "--workers":
            workers = int(argv[i + 1])
        if a == "--only":
            only = argv[i + 1]
    selection = build_selection()
    sources = {rel: (PYTHON_DIR / rel).read_text() for rel in SOURCE_FILES}
    sources[C_FILE] = (PROJECT_ROOT / C_FILE).read_text()
    mutants: list[Mutant] = []
    for rel in SOURCE_FILES:
        if only and rel != only:
            continue
        mutants += generate_python_mutants(rel, sources[rel])
    if not only or only == C_FILE:
        mutants += generate_c_mutants()
    print(f"generated {len(mutants)} mutants over {len(set(m.file for m in mutants))} files; "
          f"{workers} worker(s)")
    for f, ts in selection.items():
        print(f"  {f:34} -> {', '.join(ts)}")
    t0 = time.time()
    chunks = [mutants[i::workers] for i in range(workers)]
    with mp.Pool(workers) as pool:
        parts = pool.map(_worker, [(i, chunks[i], selection, sources) for i in range(workers)])
    results = [m for part in parts for m in part]
    results.sort(key=lambda m: (m.file, m.lineno, m.id))
    summary = {}
    for m in results:
        s = summary.setdefault(m.file, {"killed": 0, "survived": 0, "timeout": 0, "invalid": 0})
        s[m.status] += 1
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(time.time() - t0, 1),
        "selection": selection,
        "summary": summary,
        "mutants": [asdict(m) for m in results],
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=1))
    print(f"\nelapsed {payload['elapsed_seconds']}s -> {RESULTS_PATH}")
    total_valid = sum(v["killed"] + v["survived"] + v["timeout"] for v in summary.values())
    killed = sum(v["killed"] for v in summary.values())
    for f, s in summary.items():
        valid = s["killed"] + s["survived"] + s["timeout"]
        print(f"  {f:34} killed {s['killed']:3}/{valid:<3}  survived {s['survived']:3}  timeout {s['timeout']}  invalid {s['invalid']}")
    print(f"  {'TOTAL':34} killed {killed}/{total_valid}  ({100.0 * killed / max(total_valid, 1):.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
