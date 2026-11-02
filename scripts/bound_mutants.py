"""Tighten every bounded repeat of every pattern by one and see whether the corpus notices.

Usage: python scripts/bound_mutants.py

For each ``{m,n}`` written in a pattern, two mutants: ``{m,n-1}`` and, when
``m < n``, ``{m+1,n}``. A mutant is killed when some row of
tests/pattern_corpus.json changes verdict with only that pattern replaced:
a flagged row stops reporting its category, or an allowed row gets flagged.
Survivors are printed so each one can be checked by hand; the ones known to
be unobservable are explained in the tests/test_patterns.py docstring.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from blink_sanitizer import InputSanitizer
from blink_sanitizer import sanitizer as module

CORPUS = Path(__file__).resolve().parent.parent / "tests" / "pattern_corpus.json"
BOUND = re.compile(r"(?<!\\)\{(\d+),(\d+)\}")


class _Recorder:
    def __init__(self) -> None:
        self.seen: list[str] = []

    def search(self, text: str) -> None:
        self.seen.append(text)


def main() -> None:
    original = list(module._PATTERNS)
    sanitizer = InputSanitizer()

    def candidates(text: str) -> list[str]:
        recorder = _Recorder()
        module._PATTERNS = [(recorder, "x")]  # type: ignore[list-item]
        try:
            sanitizer.check(text)
        finally:
            module._PATTERNS = original
        return recorder.seen

    rows: list[tuple[str, str, list[str], set[int]]] = []
    for label, text in json.loads(CORPUS.read_text("utf-8")):
        versions = candidates(text)
        hits = {i for i, (p, _) in enumerate(original) if any(p.search(v) for v in versions)}
        rows.append((label, text, versions, hits))

    total = killed = 0
    survivors: list[Any] = []
    for index, (pattern, category) in enumerate(original):
        source = pattern.pattern
        for bound in BOUND.finditer(source):
            low, high = int(bound.group(1)), int(bound.group(2))
            tightened = [f"{{{low},{high - 1}}}"] if high > low else []
            if low < high:
                tightened.append(f"{{{low + 1},{high}}}")
            for replacement in tightened:
                total += 1
                mutant = re.compile(
                    source[: bound.start()] + replacement + source[bound.end():], pattern.flags
                )
                # Only rows where this one pattern now answers differently can change.
                suspects = [
                    (label, text)
                    for label, text, versions, hits in rows
                    if any(mutant.search(v) for v in versions) != (index in hits)
                ]
                module._PATTERNS = list(original)
                module._PATTERNS[index] = (mutant, category)
                try:
                    dead = any(
                        not sanitizer.check(text).allowed
                        if label == "allowed"
                        else label not in sanitizer.check(text).violations
                        for label, text in suspects
                    )
                finally:
                    module._PATTERNS = original
                if dead:
                    killed += 1
                else:
                    context = source[max(0, bound.start() - 40): bound.end() + 20]
                    survivors.append((index, bound.group(0), replacement, context))
    print(f"{killed} of {total} tightened bounds fail the corpus")
    for index, was, now, context in survivors:
        print(f"  survives: pattern {index} {was} -> {now}  ...{context!a}...")


if __name__ == "__main__":
    main()
