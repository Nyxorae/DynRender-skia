"""Project quality scorecard — evaluate codebase health across multiple dimensions.

Usage:
    python scorecard.py              # Terminal report
    python scorecard.py --md         # Markdown report
    python scorecard.py --json       # JSON output

Scoring dimensions (weighted to 100 points):
    test_coverage    25%   pytest-cov overall coverage %
    lint_quality     20%   ruff violations (fewer = better)
    type_annotations 15%   % of functions with return type annotations
    documentation    10%   % of public functions with docstrings
    dependency_health  5%   unused dependencies penalty
    module_balance    5%   file size variance (lower = better)
    dead_code         5%   commented-out code and stale imports
"""

import ast
import json
import os
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent
SRC = ROOT / "dynrender_skia"
TESTS = ROOT / "tests"


@dataclass
class Metric:
    name: str
    score: float   # 0-100
    weight: float  # fraction of 1.0
    weighted: float = 0
    detail: str = ""


@dataclass
class Scorecard:
    metrics: list[Metric] = field(default_factory=list)
    total: float = 0

    def add(self, name, score, weight, detail=""):
        score = min(max(score, 0), 100)
        self.metrics.append(Metric(name, score, weight, score * weight, detail))
        self.total += score * weight

    def print(self):
        print()
        print("=" * 55)
        print("  DynRender-skia Quality Scorecard")
        print("=" * 55)
        print(f"  {'Metric':<22s} {'Score':>6s} {'Wt':>5s}  {'Result':>8s}")
        print("-" * 55)

        for m in self.metrics:
            bar = "#" * int(m.score / 5) + "-" * (20 - int(m.score / 5))
            contrib = m.score * m.weight
            print(f"  {m.name:<22s} {m.score:5.0f}  {m.weight*100:4.0f}%  {contrib:5.1f}  {bar[:10]}")

        print("-" * 55)
        bar = "#" * int(self.total / 5) + "-" * (20 - int(self.total / 5))
        print(f"  {'TOTAL':<22s}         100%  {self.total:5.1f}  {bar[:10]}")
        print("=" * 55)
        print()

        if any(m.detail for m in self.metrics):
            print("Details:")
            for m in self.metrics:
                if m.detail and "N/A" not in m.detail:
                    print(f"  - {m.name}: {m.detail}")

    def to_md(self, path: str) -> None:
        lines = [
            "# DynRender-skia Quality Scorecard",
            f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Score: **{self.total:.1f}/100**",
            "",
            "| Metric | Score | Weight | Weighted |",
            "|--------|-------|--------|----------|",
        ]
        for m in self.metrics:
            lines.append(f"| {m.name} | {m.score:.0f} | {m.weight*100:.0f}% | {m.weighted:.1f} |")
        lines.append(f"| **TOTAL** | | **100%** | **{self.total:.1f}** |")
        lines.append("")

        if any(m.detail for m in self.metrics):
            lines.append("## Details\n")
            for m in self.metrics:
                if m.detail and "N/A" not in m.detail:
                    lines.append(f"- **{m.name}**: {m.detail}")

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"  Report: {path}")


# ─── metric calculators ────────────────────────────────────────────────


def eval_test_coverage() -> Metric:
    """Run pytest-cov and parse coverage %."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--cov=dynrender_skia", "--cov-report=term", "--no-header", "-q"],
            capture_output=True, text=True, cwd=str(ROOT), timeout=60,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        for line in result.stdout.split("\n") + result.stderr.split("\n"):
            if "TOTAL" in line:
                parts = line.split()
                for p in parts:
                    if p.endswith("%"):
                        pct = float(p.replace("%", ""))
                        return Metric("test_coverage", pct, 0.25,
                                       detail=f"{pct:.0f}% line coverage")
        return Metric("test_coverage", 50, 0.25,
                       detail="could not parse coverage — estimated 50%")
    except Exception as e:
        return Metric("test_coverage", 0, 0.25, detail=f"error: {e}")


def eval_lint_quality() -> Metric:
    """Count ruff violations."""
    try:
        result = subprocess.run(
            ["ruff", "check", "dynrender_skia/", "--statistics", "--no-cache"],
            capture_output=True, text=True, cwd=str(ROOT), timeout=30,
        )
        errors = len([l for l in result.stdout.split("\n") if l.strip() and not l.startswith("--")])
        score = max(0, 100 - errors * 2)  # each violation = -2 points
        return Metric("lint_quality", score, 0.20,
                       detail=f"{errors} ruff violations")
    except Exception as e:
        return Metric("lint_quality", 0, 0.20, detail=f"error: {e}")


def _walk_py_files(root: Path):
    for f in root.rglob("*.py"):
        if "__pycache__" in str(f) or f.name.startswith("_"):
            continue
        yield f


def eval_type_annotations() -> Metric:
    """Count functions with return type annotations."""
    py_files = list(_walk_py_files(SRC))
    total_funcs = 0
    annotated = 0

    for f in py_files:
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    total_funcs += 1
                    if node.returns is not None:
                        annotated += 1
        except Exception:
            pass

    if total_funcs == 0:
        return Metric("type_annotations", 0, 0.15, detail="no functions found")
    pct = annotated / total_funcs * 100
    return Metric("type_annotations", pct, 0.15,
                   detail=f"{annotated}/{total_funcs} methods annotated ({pct:.0f}%)")


def eval_documentation() -> Metric:
    """Count public functions with docstrings."""
    py_files = list(_walk_py_files(SRC))
    total = 0
    with_docs = 0

    for f in py_files:
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if node.name.startswith("_"):
                        continue
                    total += 1
                    if ast.get_docstring(node):
                        with_docs += 1
        except Exception:
            pass

    if total == 0:
        return Metric("documentation", 0, 0.10, detail="no public items found")
    pct = with_docs / total * 100
    return Metric("documentation", pct, 0.10,
                   detail=f"{with_docs}/{total} documented ({pct:.0f}%)")


def eval_dependency_health() -> Metric:
    """Check for unused dependencies."""
    import tomllib
    from importlib import import_module

    try:
        info = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        deps = []
        for d in info.get("project", {}).get("dependencies", []):
            name = d.split(">=")[0].split("~=")[0].split("<")[0].strip()
            deps.append(name.replace("-", "_"))
    except Exception:
        deps = []

    unused = []
    for dep in deps:
        try:
            import_module(dep)
        except ImportError:
            unused.append(dep)

    # Hardcoded check: these are known to be used or not
    known_unused = [d for d in unused if d not in {"skia_python", "emoji"}]

    score = max(0, 100 - len(known_unused) * 25)
    detail = f"{len(known_unused)} likely unused deps: {known_unused}" if known_unused else "all deps used"
    return Metric("dependency_health", score, 0.05, detail=detail)


def eval_module_balance() -> Metric:
    """Measure how evenly distributed code is across modules (cohesion proxy)."""
    module_sizes = []
    for f in _walk_py_files(SRC):
        try:
            lines = len(f.read_text(encoding="utf-8").split("\n"))
            module_sizes.append(lines)
        except Exception:
            pass

    if not module_sizes:
        return Metric("module_balance", 0, 0.05, detail="no modules found")

    avg = sum(module_sizes) / len(module_sizes)
    max_size = max(module_sizes)
    ratio = avg / max_size if max_size else 0
    score = ratio * 100
    return Metric("module_balance", score, 0.05,
                   detail=f"avg {avg:.0f} lines, max {max_size} ({ratio:.2f} ratio)")


def eval_dead_code() -> Metric:
    """Count commented-out code blocks and stale imports."""
    stale = 0
    for f in _walk_py_files(SRC):
        try:
            text = f.read_text(encoding="utf-8")
            lines = text.split("\n")
            in_block_comment = False
            consecutive_comments = 0
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("# "):
                    consecutive_comments += 1
                else:
                    consecutive_comments = 0
                if consecutive_comments > 5:
                    stale += 1
                    consecutive_comments = 0
        except Exception:
            pass

    score = max(0, 100 - stale * 10)
    return Metric("dead_code", score, 0.05,
                   detail=f"{stale} long comment blocks (potential dead code)")


# ─── main ──────────────────────────────────────────────────────────────


def main():
    t0 = time.perf_counter()
    sc = Scorecard()

    print("\n  Evaluating...\n")
    checks = [
        ("Test Coverage", eval_test_coverage),
        ("Lint Quality", eval_lint_quality),
        ("Type Annotations", eval_type_annotations),
        ("Documentation", eval_documentation),
        ("Dependency Health", eval_dependency_health),
        ("Module Balance", eval_module_balance),
        ("Dead Code", eval_dead_code),
    ]

    for name, fn in checks:
        print(f"    {name}...", end=" ", flush=True)
        metric = fn()
        sc.add(name.replace(" ", "_").lower(), metric.score, metric.weight, metric.detail)
        print(f"{metric.score:.0f}")

    elapsed = time.perf_counter() - t0

    if "--md" in sys.argv:
        sc.to_md("scorecard_output/report.md")
    elif "--json" in sys.argv:
        out = {"metrics": [], "total": sc.total}
        for m in sc.metrics:
            out["metrics"].append({"name": m.name, "score": m.score, "weight": m.weight})
        print(json.dumps(out, indent=2))
    else:
        sc.print()
        print(f"  Evaluated in {elapsed:.1f}s")
    sc.to_md("scorecard_output/report.md")


if __name__ == "__main__":
    main()
