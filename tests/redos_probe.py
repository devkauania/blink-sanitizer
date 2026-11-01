"""Pump every pattern with near-misses of strings it matches. Run as a subprocess.

Usage: python tests/redos_probe.py [limit_ms]

Generic pump strings ("a" * 20000, spaces) never reach a backtracking part that
sits behind a literal prefix, so this builds its inputs from each pattern's own
matches: the first words of a matching sentence, repeated. A catastrophic regex
cannot be interrupted once it starts, which is why tests/test_redos.py runs this
in a separate process with a timeout. Exit code 1 and one line per slow pattern
on failure. An input is timed again before it counts, because a single slow
measurement on a shared machine means nothing.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from blink_sanitizer.sanitizer import _PATTERNS

ROOT = Path(__file__).resolve().parent
SIZE = 20_000
MAX_STRINGS_PER_PATTERN = 3
SEPARATORS = (" ", "", "\n")


def strings() -> list[str]:
    corpus = json.loads((ROOT / "pattern_corpus.json").read_text("utf-8"))
    return [text for label, text in corpus if label != "allowed"] + _case_sentences()


def _case_sentences() -> list[str]:
    sys.path.insert(0, str(ROOT))
    from pattern_cases import PATTERN_CASES

    return [text for text, _ in PATTERN_CASES]


def units_for(samples: list[str]) -> set[str]:
    units: set[str] = set()
    for sample in samples:
        words = sample.split(" ")
        for count in range(1, min(len(words), 4) + 1):
            prefix = " ".join(words[:count])
            for separator in SEPARATORS:
                units.add(prefix + separator)
        for cut in (1, 3):
            if len(sample) > cut:
                units.add(sample[:-cut] + " ")
    return units


def pump(unit: str) -> str:
    return (unit * (SIZE // max(len(unit), 1) + 1))[:SIZE]


def timed(pattern: object, text: str) -> float:
    start = time.perf_counter()
    pattern.search(text)  # type: ignore[attr-defined]
    return time.perf_counter() - start


def main() -> int:
    limit = (float(sys.argv[1]) if len(sys.argv) > 1 else 250.0) / 1000
    pool = strings()
    searches = 0
    slow: list[str] = []
    for pattern, _category in _PATTERNS:
        matching = sorted((s for s in pool if pattern.search(s)), key=len)
        samples = matching[:MAX_STRINGS_PER_PATTERN]
        if not samples:
            print(f"no sample for {pattern.pattern[:80]!r}")
            return 1
        for unit in sorted(units_for(samples)):
            text = pump(unit)
            searches += 1
            elapsed = timed(pattern, text)
            if elapsed > limit:
                retry = min(timed(pattern, text) for _ in range(2))
                if retry > limit:
                    slow.append(f"{retry * 1000:.0f} ms {unit[:30]!r} {pattern.pattern[:80]!r}")
    print(f"{searches} searches, {len(slow)} slow")
    for line in slow:
        print(line)
    return 1 if slow else 0


if __name__ == "__main__":
    raise SystemExit(main())
