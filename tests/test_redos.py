"""A catastrophic or quadratic pattern must not get in, behind any literal prefix.

The older linearity tests pumped generic units (spaces, "a", "1"), so a new rule
such as r"ignoriere\\s+(?:\\w+\\s*)+regeln", which took 8.8 s on a 38-character
input, passed the whole suite. Two independent guards replace that gap: a check of
the parsed regex, which needs no input at all, and a pump built from each
pattern's own matches.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from blink_sanitizer import sanitizer as module
from blink_sanitizer.sanitizer import _PATTERNS

try:
    import re._constants as C  # type: ignore[import-not-found]  # Python 3.11+
    import re._parser as P  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - Python 3.10
    import sre_constants as C  # type: ignore[no-redef]
    import sre_parse as P  # type: ignore[no-redef]

TESTS = Path(__file__).resolve().parent
_REPEATS = (C.MAX_REPEAT, C.MIN_REPEAT)
# Up to this many repetitions of a group holding "\s+" stays polynomial with a small
# exponent, and the shipped "(?:(?:all|any|the)\s+){0,3}" is that shape.
_SMALL_OUTER_REPEAT = 8


def _children(op: Any, av: Any) -> list[Any]:
    if op is C.BRANCH:
        return list(av[1])
    if op in _REPEATS:
        return [av[2]]
    if op is C.SUBPATTERN:
        return [av[3]]
    if op in (C.ASSERT, C.ASSERT_NOT):
        return [av[1]]
    return []


def _has_unbounded(seq: Any) -> bool:
    for op, av in seq:
        if op in _REPEATS and av[1] is C.MAXREPEAT:
            return True
        if any(_has_unbounded(child) for child in _children(op, av)):
            return True
    return False


def _is_wildcard(sub: Any) -> bool:
    """"." or a negated class such as [^x], which can cross almost anything."""
    if len(sub) != 1:
        return False
    op, av = sub[0]
    if op in (C.ANY, C.NOT_LITERAL):
        return True
    # "[^\W\d_]+" (letters) is a positive class written negatively, not a wildcard.
    if op is not C.IN or av[0][0] is not C.NEGATE:
        return False
    return all(o in (C.NEGATE, C.LITERAL, C.RANGE) for o, _ in av)


def _class_accepts(items: Any, char: str) -> bool:
    accepted = False
    for op, av in items:
        if op is C.LITERAL and chr(av) == char:
            accepted = True
        elif op is C.RANGE and av[0] <= ord(char) <= av[1]:
            accepted = True
        elif op is C.CATEGORY:
            name = str(av).upper()
            if "WORD" in name:
                hit = char.isalnum() or char == "_"
            elif "SPACE" in name:
                hit = char.isspace()
            else:
                hit = char.isdigit()
            accepted = accepted or (hit != ("NOT" in name))
    negated = any(op is C.NEGATE for op, _ in items)
    return accepted != negated


def _chunks_end_in_a_separator(sub: Any) -> bool:
    """True for a group such as a word-character run followed by a dot, repeated.

    The group is one repeated class plus a mandatory literal the class cannot match,
    so it cannot split a run two ways and stays linear.
    """
    items = list(sub)
    if len(items) != 2 or items[0][0] not in _REPEATS or items[1][0] is not C.LITERAL:
        return False
    inner = list(items[0][1][2])
    if len(inner) != 1 or inner[0][0] is not C.IN:
        return False
    return not _class_accepts(inner[0][1], chr(items[1][1]))


def problems(pattern: str, flags: int = 0) -> list[str]:
    """Shapes that backtrack badly: a repeat inside a large repeat, or ".*"."""
    found: list[str] = []

    def walk(seq: Any) -> None:
        for op, av in seq:
            if op in _REPEATS:
                low, high, sub = av
                if high is C.MAXREPEAT and _is_wildcard(sub):
                    found.append("unbounded wildcard")
                large = high is C.MAXREPEAT or high > _SMALL_OUTER_REPEAT
                if large and _has_unbounded(sub) and not _chunks_end_in_a_separator(sub):
                    found.append("nested unbounded repeat")
            for child in _children(op, av):
                walk(child)

    walk(P.parse(pattern, flags))
    return found


@pytest.mark.parametrize(
    "bad",
    [
        r"ignoriere\s+(?:\w+\s*)+regeln",
        r"\bbypass\s+(?:\w+\s*)+\bpolicy\b",
        r"\bforget\s+(?:\w+\s+)*?\w*\s*everything\b",
        r"\bshow\b.*\bsecret\s+prompt\b",
        r"\bshow\b[^\n]+\bsecret\s+prompt\b",
        r"(a+)+$",
        r"(?:x|(?:y+)){2,40}z",
    ],
)
def test_the_static_check_rejects_the_known_shapes(bad: str) -> None:
    assert problems(bad)


@pytest.mark.parametrize(
    "fine",
    [
        r"ignore\s+(?:(?:all|any)\s+){0,3}previous",
        r"(\w+\s+)?rules",
        r"\bshow\b[^\n]{0,80}\bprompt\b",
        r"(?:ab){2,}",
    ],
)
def test_the_static_check_accepts_bounded_shapes(fine: str) -> None:
    assert not problems(fine)


def test_every_shipped_pattern_passes_the_static_check() -> None:
    bad = [(p.pattern[:100], problems(p.pattern, p.flags)) for p, _ in _PATTERNS]
    offenders = [entry for entry in bad if entry[1]]
    assert not offenders


def test_every_helper_regex_passes_the_static_check() -> None:
    helpers = [
        value
        for name, value in vars(module).items()
        if isinstance(value, re.Pattern) and name != "_PATTERNS"
    ]
    assert len(helpers) >= 15
    # The decoder regexes with a mandatory separator between chunks are exempt here and
    # covered by the linearity tests in test_sanitizer.py.
    offenders = [
        h.pattern[:80]
        for h in helpers
        if "nested unbounded repeat" in problems(h.pattern, h.flags)
        and not _known_chunked(h.pattern)
    ]
    assert not offenders


def _known_chunked(source: str) -> bool:
    """Base64 and form-encoding chunks: every repetition needs a mandatory separator."""
    return source.startswith(("(?:[A-Za-z0-9+/]", "(?<![^\\W_])[^\\W_]+(?:\\+", "(?<![0-9A-Za-z])"))


def test_pump_built_from_each_patterns_own_matches() -> None:
    # A timeout is a failure: a catastrophic pattern never returns by itself.
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(Path(module.__file__).resolve().parent.parent), env.get("PYTHONPATH", "")]
    )
    done = subprocess.run(
        [sys.executable, str(TESTS / "redos_probe.py")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=240,
        env=env,
    )
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
