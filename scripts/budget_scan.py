"""How often the scan-work budget rejects long, legitimate source text.

Usage: python scripts/budget_scan.py

Checks the first 20,000 characters of every CPython stdlib module of at least
19,000 characters (site-packages excluded) and prints the ones reported as
input_too_complex, plus any other verdict.
"""
from __future__ import annotations

import sys
import sysconfig
from pathlib import Path

from blink_sanitizer import DEFAULT_MAX_LENGTH, InputSanitizer

MIN_CHARS = 19_000


def main() -> None:
    root = Path(sysconfig.get_paths()["stdlib"])
    sanitizer = InputSanitizer()
    checked = too_complex = flagged = 0
    for path in sorted(root.rglob("*.py")):
        if "site-packages" in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if len(text) < MIN_CHARS:
            continue
        checked += 1
        violations = sanitizer.check(text[:DEFAULT_MAX_LENGTH]).violations
        if "input_too_complex" in violations:
            too_complex += 1
            print("too complex:", path.relative_to(root))
        elif violations:
            flagged += 1
            print("flagged:", path.relative_to(root), violations)
    print(f"python {sys.version.split()[0]}: {checked} modules, {too_complex} too complex, "
          f"{flagged} flagged")


if __name__ == "__main__":
    main()
