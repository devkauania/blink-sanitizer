"""Detection and false-positive rates on the held-out corpus in tests/holdout/.

Usage: python scripts/holdout_report.py

The strings are kept out of pattern writing: see tests/holdout/README.md. This
prints aggregate numbers per group, never the strings, so that reading the
report does not turn the corpus into training data.
"""
from __future__ import annotations

import json
from pathlib import Path

from blink_sanitizer import InputSanitizer

HOLDOUT = Path(__file__).resolve().parent.parent / "tests" / "holdout"


def load(kind: str) -> dict[str, list[str]]:
    data: dict[str, list[str]] = json.loads((HOLDOUT / f"{kind}.json").read_text("utf-8"))
    return data


def score(groups: dict[str, list[str]]) -> dict[str, tuple[int, int]]:
    """Return {group: (flagged, total)} with the default settings."""
    sanitizer = InputSanitizer()
    return {
        name: (sum(not sanitizer.check(text).allowed for text in texts), len(texts))
        for name, texts in groups.items()
    }


def totals(scores: dict[str, tuple[int, int]]) -> tuple[int, int]:
    return sum(f for f, _ in scores.values()), sum(n for _, n in scores.values())


def main() -> None:
    for kind, verb in (("attacks", "detected"), ("benign", "flagged")):
        scores = score(load(kind))
        hit, total = totals(scores)
        print(f"== {kind}: {verb} {hit}/{total} ({100 * hit / total:.1f}%)")
        for name, (flagged, size) in sorted(scores.items()):
            print(f"{name:40s} {flagged:4d}/{size}")


if __name__ == "__main__":
    main()
