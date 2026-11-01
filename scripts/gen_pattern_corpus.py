"""Generate tests/pattern_corpus.json: one string per alternation branch of every pattern.

Usage: python scripts/gen_pattern_corpus.py

Why a snapshot: a string generated while the tests run comes from the pattern
under test, so it cannot notice that the pattern lost a branch. The committed
file remembers what each pattern matched when it was generated, so removing a
word from a pattern's list, or relabelling a pattern, fails the suite. Rows
labelled "allowed" do the same for negative lookaheads: the attack shape plus a
word the exception lets through, which must stay allowed. Narrowing a pattern on
purpose means re-running this script and reviewing the diff. These strings prove
that branches keep working. They are not natural text and say nothing about
detection of unseen attacks (see tests/holdout/ for that).
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

try:
    import re._constants as C  # Python 3.11+
    import re._parser as P
except ImportError:  # pragma: no cover - Python 3.10
    import sre_constants as C  # type: ignore[no-redef]
    import sre_parse as P  # type: ignore[no-redef]

from blink_sanitizer import InputSanitizer
from blink_sanitizer.sanitizer import _PATTERNS

OUT = Path(__file__).resolve().parent.parent / "tests" / "pattern_corpus.json"
_REPEATS = (C.MAX_REPEAT, C.MIN_REPEAT)


def _class_char(items: list[Any], pick: int) -> str:
    if any(op is C.NEGATE for op, _ in items):
        banned = {chr(av) for op, av in items if op is C.LITERAL}
        return next(c for c in "a xb1" if c not in banned)
    op, av = items[pick % len(items)]
    if op is C.LITERAL:
        return chr(av)
    if op is C.RANGE:
        return chr(av[0])
    if op is C.CATEGORY:
        name = str(av)
        if "NOT_SPACE" in name or "NOT_DIGIT" in name or "WORD" in name and "NOT" not in name:
            return "a"
        if "SPACE" in name:
            return " "
        if "DIGIT" in name:
            return "1"
    return "."


def _render(seq: Any, choose: dict[int, int], expand: tuple[Any, ...] = ()) -> str:
    """Render one string.

    ``choose`` maps a node's id to the alternative to take, or for an optional part
    to 1 (include it). A positive lookahead is written out, since a match needs what
    it looks for right there. A negative lookahead renders nothing unless its body
    is in ``expand``.
    """
    out: list[str] = []
    for op, av in seq:
        if op is C.LITERAL:
            out.append(chr(av))
        elif op is C.NOT_LITERAL:
            out.append("a" if chr(av) != "a" else "b")
        elif op is C.ANY:
            out.append("a")
        elif op is C.IN:
            out.append(_class_char(list(av), choose.get(id(av), 0)))
        elif op is C.BRANCH:
            out.append(_render(av[1][choose.get(id(av), 0)], choose, expand))
        elif op in _REPEATS:
            low, _high, sub = av
            times = low if low > 0 else choose.get(id(av), 1)
            for _ in range(times):
                out.append(_render(sub, choose, expand))
        elif op is C.SUBPATTERN:
            out.append(_render(av[3], choose, expand))
        elif op is C.ASSERT and av[0] > 0:
            out.append(_render(av[1], choose, expand))
        elif op is C.ASSERT_NOT and any(av[1] is body for body in expand):
            out.append(_render(av[1], choose, expand))
    return "".join(out)


def _choices(
    seq: Any, trail: dict[int, int], found: list[tuple[int, int, dict[int, int]]]
) -> None:
    """Every choice point, with the choices that make the render reach it."""
    for op, av in seq:
        if op is C.BRANCH:
            found.append((id(av), len(av[1]), trail))
            for index, alternative in enumerate(av[1]):
                _choices(alternative, {**trail, id(av): index}, found)
        elif op is C.IN:
            items = list(av)
            if not any(o is C.NEGATE for o, _ in items) and len(items) > 1:
                found.append((id(av), len(items), trail))
        elif op in _REPEATS:
            if av[0] == 0:
                found.append((id(av), 2, trail))
            _choices(av[2], {**trail, id(av): 1} if av[0] == 0 else trail, found)
        elif op is C.SUBPATTERN:
            _choices(av[3], trail, found)
        elif op is C.ASSERT and av[0] > 0:
            _choices(av[1], trail, found)


def _lookaheads(
    seq: Any, trail: dict[int, int], parents: tuple[Any, ...], found: list[Any]
) -> None:
    """Every negative lookahead: (its body, the bodies around it, choices to reach it)."""
    for op, av in seq:
        if op is C.ASSERT_NOT and av[0] > 0:
            found.append((av[1], parents, trail))
            _lookaheads(av[1], trail, (*parents, av[1]), found)
        elif op is C.ASSERT and av[0] > 0:
            _lookaheads(av[1], trail, parents, found)
        elif op is C.BRANCH:
            for index, alternative in enumerate(av[1]):
                _lookaheads(alternative, {**trail, id(av): index}, parents, found)
        elif op in _REPEATS:
            _lookaheads(av[2], {**trail, id(av): 1} if av[0] == 0 else trail, parents, found)
        elif op is C.SUBPATTERN:
            _lookaheads(av[3], trail, parents, found)


def _alone(text: str, pattern: re.Pattern[str], category: str) -> bool:
    """No other pattern of this category matches, so losing this branch loses the string."""
    return not any(
        other.search(text) for other, cat in _PATTERNS if cat == category and other is not pattern
    )


def _strings_for(pattern: re.Pattern[str], category: str) -> list[str]:
    """Every alternative of every branch, in the setting where only this pattern matches."""
    tree = P.parse(pattern.pattern, pattern.flags)
    nodes: list[tuple[int, int, dict[int, int]]] = []
    _choices(tree, {}, nodes)
    rng = random.Random(pattern.pattern)
    out: dict[str, None] = {}
    bases: list[dict[int, int]] = [{}]
    for _ in range(25):
        bases.append({node: rng.randrange(size) for node, size, _trail in nodes})
    for node, size, trail in nodes:
        for alternative in range(size):
            fallback = None
            for base in bases:
                text = _render(tree, {**base, **trail, node: alternative})
                if not pattern.search(text):
                    continue
                if _alone(text, pattern, category):
                    fallback = text
                    break
                fallback = fallback or text
            if fallback is not None:
                out[fallback] = None
    return list(out)


def _exception_strings(pattern: re.Pattern[str]) -> list[tuple[str, int]]:
    """Text each alternative of each negative lookahead changes, with its nesting depth.

    Depth 1 is an exception, so the text must stay allowed. Depth 2 is the exception
    to that exception ("the system prompt for the assistant"), so it must be flagged.
    """
    tree = P.parse(pattern.pattern, pattern.flags)
    bodies: list[Any] = []
    _lookaheads(tree, {}, (), bodies)
    out: dict[tuple[str, int], None] = {}
    for body, parents, trail in bodies:
        nodes: list[tuple[int, int, dict[int, int]]] = []
        _choices(body, {}, nodes)
        expand = (*parents, body)
        depth = len(expand)
        out[(_render(tree, trail, expand), depth)] = None
        for node, size, inner in nodes:
            for alternative in range(size):
                # A lookahead's optional parts stay off unless chosen: a nested
                # exception after the word would otherwise cancel it.
                text = _render(tree, {**trail, **inner, node: alternative}, expand)
                out[(text, depth)] = None
    return list(out)


def main() -> None:
    sanitizer = InputSanitizer()
    corpus: list[list[str]] = []
    seen: set[str] = set()
    dropped = 0
    for pattern, category in _PATTERNS:
        for text in _strings_for(pattern, category):
            if text in seen:
                continue
            seen.add(text)
            if category in sanitizer.check(text).violations:
                corpus.append([category, text])
            else:
                dropped += 1
    allowed: set[str] = set()
    for pattern, category in _PATTERNS:
        for text, depth in _exception_strings(pattern):
            if text in seen or not text.strip():
                continue
            verdict = sanitizer.check(text)
            if depth % 2 == 1 and verdict.allowed:
                allowed.add(text)
            elif depth % 2 == 0 and category in verdict.violations:
                seen.add(text)
                corpus.append([category, text])
    corpus.extend(["allowed", text] for text in sorted(allowed))
    OUT.write_text(
        "[\n" + ",\n".join(json.dumps(row, ensure_ascii=True) for row in corpus) + "\n]\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"{len(corpus)} strings written to {OUT.name} ({len(allowed)} exceptions that stay "
        f"allowed), {dropped} not flagged and left out"
    )


if __name__ == "__main__":
    main()
