"""Latency benchmark for InputSanitizer.check.

Times every call over a fixed corpus:
  * 200 benign prompts (20 task templates x 10 subjects, generated below)
  * every string literal passed to ``sanitizer.check(...)`` or listed in a
    ``pytest.mark.parametrize`` in tests/test_sanitizer.py (attacks + benign)

Usage:
    python scripts/benchmark.py [--iterations N]

Prints p50 / p95 / p99 / max latency per check in microseconds, plus the
Python version and platform the numbers were measured on. It then times
single checks of large inputs, for benign prose and for adversarial payloads
that trigger every normalization and decoding layer: 100,000 characters
(~100 KB) with the default max_length (rejected before scanning), 20,000
characters (the default max_length: scanned, or rejected as
input_too_complex when it exceeds the scan-work budget), the same as both
prompt and context, and 100,000 characters with max_length=None (fully
scanned, no budget). Finally, for each payload it finds the largest size
that is still scanned under the budget and times it as prompt + context:
the slowest of those is the measured worst case at the default settings.
"""
from __future__ import annotations

import argparse
import ast
import base64
import platform
import statistics
import sys
import time
from pathlib import Path

from blink_sanitizer import DEFAULT_MAX_LENGTH, InputSanitizer

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


LARGE_SIZE = 100_000


def large_inputs(size: int = LARGE_SIZE) -> dict[str, str]:
    """~100 KB inputs: benign prose and payloads that exercise every layer."""
    zwsp, braille_blank, lisu_i, hangul = chr(0x200B), chr(0x2800), chr(0xA4F2), chr(0xD55C)
    prose = "Summarize the latest report on Q3 sales by region. "
    b64 = base64.b64encode((prose * (size // len(prose) + 1)).encode()).decode()
    mixed_unit = (
        f"&amp;#105;gnore %2569 ign ore prev ious i{zwsp}g{braille_blank}n "
        f"{lisu_i} {hangul} aGVsbG8gd29ybGQgdGhpcyBpcyBmaW5l 1-g-n-0-r-3 **x** "
    )
    payloads = {
        "benign prose": prose,
        "double HTML entities": "&amp;#105;",
        "base64 text": b64,
        "split words": "ign ore prev ious ",
        "invisible + Braille blank": f"i{zwsp}g{braille_blank} ",
        "Lisu I (extra candidate)": f"{lisu_i} ",
        "Hangul (NFKD x3)": hangul,
        "mixed adversarial": mixed_unit,
        # U+FDFA decomposes into 18 code points under NFKD.
        "mixed + 12 x U+FDFA": mixed_unit + chr(0xFDFA) * 12,
    }
    return {name: (unit * (size // len(unit) + 1))[:size] for name, unit in payloads.items()}


# max_length=None has no length or work bound; these are skipped in column C.
_SKIP_UNLIMITED = {"mixed + 12 x U+FDFA"}


def time_large() -> None:
    default = InputSanitizer()
    unlimited = InputSanitizer(max_length=None)
    at_limit = large_inputs(DEFAULT_MAX_LENGTH)
    print(f"single check, best of 3 (A = {LARGE_SIZE:,} chars, default limit -> rejected; "
          f"B = {DEFAULT_MAX_LENGTH:,} chars, default limit; "
          f"B2 = the same text as prompt and context; "
          f"C = {LARGE_SIZE:,} chars, max_length=None):")
    for name, text in large_inputs().items():
        sample = at_limit[name]
        t_a = min(_time_once(default, text) for _ in range(3))
        t_b = min(_time_once(default, sample) for _ in range(3))
        t_b2 = min(_time_once(default, sample, sample) for _ in range(3))
        if name in _SKIP_UNLIMITED:
            col_c = "       -"
        else:
            col_c = f"{min(_time_once(unlimited, text) for _ in range(3)) * 1000:8.1f}"
        verdict = default.check(sample).violations
        print(f"  {name:28s} A {t_a * 1000:6.2f} ms   B {t_b * 1000:7.1f} ms   "
              f"B2 {t_b2 * 1000:7.1f} ms   C {col_c} ms   B verdict {verdict}")
    time_budget_edge(default)


def time_budget_edge(default: InputSanitizer) -> None:
    """Slowest fully scanned input: for each payload, the largest size (up to
    the default limit) that is still scanned instead of being rejected as
    input_too_complex, checked as both prompt and context."""
    print("largest size still scanned under the work budget (default limit), "
          "prompt + context, best of 3:")
    units = large_inputs(DEFAULT_MAX_LENGTH)
    worst = 0.0
    for name, full in units.items():
        lo, hi = 0, DEFAULT_MAX_LENGTH
        while lo < hi:  # largest prefix length that is not input_too_complex
            mid = (lo + hi + 1) // 2
            if "input_too_complex" in default.check(full[:mid]).violations:
                hi = mid - 1
            else:
                lo = mid
        sample = full[:lo]
        elapsed = min(_time_once(default, sample, sample) for _ in range(3))
        worst = max(worst, elapsed)
        print(f"  {name:28s} {lo:6,} chars   {elapsed * 1000:7.1f} ms   "
              f"{default.check(sample, context=sample).violations}")
    print(f"  slowest: {worst * 1000:.1f} ms")


def _time_once(sanitizer: InputSanitizer, text: str, context: str | None = None) -> float:
    t0 = time.perf_counter()
    sanitizer.check(text, context=context)
    return time.perf_counter() - t0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--iterations", type=int, default=20, help="passes over the corpus")
    parser.add_argument("--skip-large", action="store_true", help="skip the 100 KB timings")
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
    if not opts.skip_large:
        time_large()


if __name__ == "__main__":
    main()
