"""False-positive scan over real-world English text: the CPython stdlib.

Collects every docstring (module, class, function) and every string
constant of 16-5,000 characters from the top-level modules of the running
interpreter's standard library plus the email, json, http and urllib
packages, and reports how many InputSanitizer flags. None of this text is
an attack, so every flag is a false positive.

Usage:
    python scripts/stdlib_fp_scan.py

The corpus depends on the Python version; the script prints it.
"""
from __future__ import annotations

import ast
import platform
import sysconfig
from pathlib import Path

from blink_sanitizer import InputSanitizer

_DOC_NODES = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def _sources() -> list[Path]:
    lib = Path(sysconfig.get_paths()["stdlib"])
    files = sorted(lib.glob("*.py"))
    for package in ("email", "json", "http", "urllib"):
        files += sorted((lib / package).glob("*.py"))
    return files


def collect() -> tuple[list[str], list[str]]:
    docstrings: list[str] = []
    constants: list[str] = []
    for path in _sources():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, _DOC_NODES):
                doc = ast.get_docstring(node)
                if doc:
                    docstrings.append(doc)
            elif (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and 16 <= len(node.value) < 5000
            ):
                constants.append(node.value)
    return list(dict.fromkeys(docstrings)), list(dict.fromkeys(constants))


def main() -> None:
    sanitizer = InputSanitizer()
    docstrings, constants = collect()
    print(f"python: {platform.python_version()} ({platform.python_implementation()})")
    for name, corpus in (("docstrings", docstrings), ("string constants", constants)):
        flagged = [t for t in corpus if not sanitizer.check(t).allowed]
        print(f"{name}: {len(flagged)} flagged / {len(corpus)}")
        for text in flagged:
            print(f"  ! {text[:120]!r} -> {sanitizer.check(text).violations}")


if __name__ == "__main__":
    main()
