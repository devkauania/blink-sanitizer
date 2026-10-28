"""Latency benchmark for InputSanitizer.check.

Times every call over a fixed corpus:
  * 200 benign prompts (20 task templates x 10 subjects, generated below)
  * every string literal passed to ``sanitizer.check(...)`` or listed in a
    ``pytest.mark.parametrize`` in tests/test_sanitizer.py (attacks + benign)

Usage:
    python scripts/benchmark.py [--iterations N]

Prints p50 / p95 / p99 / max latency per check in microseconds, plus the
Python version and platform the numbers were measured on.
"""
from __future__ import annotations

import argparse
import ast
import platform
import statistics
import sys
import time
from pathlib import Path

from blink_sanitizer import InputSanitizer

ROOT = Path(__file__).resolve().parent.parent

_TASKS = [
    "Summarize the latest report on {}",
    "Draft a short email to the client about {}",
    "List three risks related to {}",
    "Explain {} to a new team member",
    "Write a SQL query that groups {} by month",
    "Compare last quarter and this quarter for {}",
    "Create a checklist for reviewing {}",
    "What are the main trends in {}?",
    "Translate the notes about {} into Portuguese",
    "Suggest five KPIs to track {}",
    "Give me a one-paragraph status update on {}",
    "Find inconsistencies in the spreadsheet about {}",
    "Outline a presentation on {} for the board",
    "How should we prioritize the backlog for {}?",
    "Rewrite this paragraph about {} in plain English",
    "Estimate the budget impact of changes to {}",
    "Generate interview questions about {}",
    "Which customers are most affected by {}?",
    "Write unit test ideas for the module handling {}",
    "Prepare meeting notes on {} with action items",
]

_SUBJECTS = [
    "Q3 sales by region",
    "customer churn in the SMB segment",
    "the new onboarding flow",
    "invoice processing delays",
    "renewable energy pricing",
    "the CRM data migration",
    "support ticket volume in March",
    "the 2026 hiring plan",
    "API latency on the checkout service",
    "late deliveries in the north warehouse",
]


def benign_corpus() -> list[str]:
    return [t.format(s) for t in _TASKS for s in _SUBJECTS]


def test_corpus() -> list[str]:
    """Collect string literals used as inputs in the test suite."""
    tree = ast.parse((ROOT / "tests" / "test_sanitizer.py").read_text(encoding="utf-8"))
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name == "check":
            args = list(node.args) + [kw.value for kw in node.keywords if kw.arg == "context"]
        elif name == "parametrize" and len(node.args) >= 2:
            args = list(getattr(node.args[1], "elts", []))
        else:
            continue
        for arg in args:
            if isinstance(arg, ast.Tuple):  # parametrize(("text", "category"), ...)
                arg = arg.elts[0] if arg.elts else arg
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                found.append(arg.value)
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--iterations", type=int, default=20, help="passes over the corpus")
    opts = parser.parse_args()

    sanitizer = InputSanitizer()
    benign = benign_corpus()
    from_tests = test_corpus()
    corpus = benign + from_tests

    # Warm-up (regex caches, first-call overhead)
    for text in corpus:
        sanitizer.check(text)

    samples_us: list[float] = []
    perf = time.perf_counter_ns
    for _ in range(opts.iterations):
        for text in corpus:
            t0 = perf()
            sanitizer.check(text)
            samples_us.append((perf() - t0) / 1000)

    samples_us.sort()
    q = statistics.quantiles(samples_us, n=100)
    benign_flagged = [t for t in benign if not sanitizer.check(t).allowed]

    print(f"python    : {sys.version.split()[0]} ({platform.python_implementation()})")
    print(f"platform  : {platform.platform()} / {platform.processor() or platform.machine()}")
    print(f"corpus    : {len(benign)} benign + {len(from_tests)} strings from tests "
          f"= {len(corpus)} inputs x {opts.iterations} iterations = {len(samples_us)} checks")
    print(f"p50       : {q[49]:.1f} us")
    print(f"p95       : {q[94]:.1f} us")
    print(f"p99       : {q[98]:.1f} us")
    print(f"max       : {samples_us[-1]:.1f} us")
    print(f"mean      : {statistics.fmean(samples_us):.1f} us")
    print(f"benign corpus flagged: {len(benign_flagged)}/{len(benign)}")
    for text in benign_flagged:
        print(f"  ! {text!r} -> {sanitizer.check(text).violations}")


if __name__ == "__main__":
    main()
