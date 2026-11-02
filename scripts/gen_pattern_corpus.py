"""Generate tests/pattern_corpus.json: one string per alternation branch of every pattern,
and one per bound of every repeat.

Usage: python scripts/gen_pattern_corpus.py

Why a snapshot: a string generated while the tests run comes from the pattern
under test, so it cannot notice that the pattern lost a branch. The committed
file remembers what each pattern matched when it was generated, so removing a
word from a pattern's list, or relabelling a pattern, fails the suite. Every
repeat is also written at its minimum and, when bounded, at its maximum
(``{1,2}`` as one and as two), choosing where it can a string that the pattern
with that bound tightened by one no longer matches, so tightening a bound fails
the suite as well (scripts/bound_mutants.py lists the few that cannot). Rows
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
    import re._compiler as K  # Python 3.11+
    import re._constants as C
    import re._parser as P
except ImportError:  # pragma: no cover - Python 3.10
    import sre_compile as K  # type: ignore[no-redef]
    import sre_constants as C  # type: ignore[no-redef]
    import sre_parse as P  # type: ignore[no-redef]

from blink_sanitizer import InputSanitizer
from blink_sanitizer.sanitizer import _PATTERNS

OUT = Path(__file__).resolve().parent.parent / "tests" / "pattern_corpus.json"
_REPEATS = (C.MAX_REPEAT, C.MIN_REPEAT)
# Placeholder for a character that may be a letter or a space ("." or "[^.!?\n]").
# _fill writes it as "a", or as spaced filler so a following \b still holds.
_GAP = "\x00"


def _fill(text: str, spaced: bool) -> str:
    """Write the placeholders: all "a", or each run as " aaa " (a lone one as " ").

    A run of "a" glues itself to the word after it, so "[^.!?\\n]{0,80}\\bandroid"
    rendered at 80 never matches; the spaced form keeps the word boundaries.
    """
    if not spaced:
        return text.replace(_GAP, "a")
    return re.sub(
        _GAP + "+",
        lambda run: " " if len(run.group()) == 1 else " " + "a" * (len(run.group()) - 2) + " ",
        text,
    )


def _class_char(items: list[Any], pick: int) -> str:
    if any(op is C.NEGATE for op, _ in items):
        banned = {chr(av) for op, av in items if op is C.LITERAL}
        if "a" not in banned and " " not in banned:
            return _GAP
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

    ``choose`` maps a node's id to the alternative to take, or for a repeat to how
    many times to write it. An unchosen repeat is written its minimum number of
    times, an optional part once. A positive lookahead is written out, since a match
    needs what it looks for right there. A negative lookahead renders nothing unless
    its body is in ``expand``. Free characters come out as ``_GAP``; see _fill.
    """
    out: list[str] = []
    for op, av in seq:
        if op is C.LITERAL:
            out.append(chr(av))
        elif op is C.NOT_LITERAL:
            out.append(_GAP if chr(av) not in "a " else "b" if chr(av) == "a" else "a")
        elif op is C.ANY:
            out.append(_GAP)
        elif op is C.IN:
            out.append(_class_char(list(av), choose.get(id(av), 0)))
        elif op is C.BRANCH:
            out.append(_render(av[1][choose.get(id(av), 0)], choose, expand))
        elif op in _REPEATS:
            low, _high, sub = av
            times = choose.get(id(av), low if low > 0 else 1)
            for _ in range(times):
                out.append(_render(sub, choose, expand))
        elif op is C.SUBPATTERN:
            out.append(_render(av[3], choose, expand))
        elif op is C.ASSERT and av[0] > 0:
            out.append(_render(av[1], choose, expand))
        elif op is C.ASSERT_NOT and any(av[1] is body for body in expand):
            out.append(_render(av[1], choose, expand))
    return "".join(out)


# A choice point: (node id, the values it can take, the choices that reach it,
# how many of the first values the random base renders draw from, and -1 for a
# branch or class, else the repeat's bounded maximum when it is above 1, else 0).
Choice = tuple[int, tuple[int, ...], dict[int, int], int, int]


def _repeat_counts(low: int, high: int) -> tuple[int, ...]:
    """The counts a repeat is written with: its minimum, once, and a bounded maximum.

    The maximum matters as much as the minimum: narrowing ``{1,2}`` to ``{1}`` keeps
    every string written once matching, so only a string at 2 notices.
    """
    counts = [low]
    if low == 0:
        counts.append(1)
    if high != C.MAXREPEAT and high > counts[-1]:
        counts.append(high)
    return tuple(counts)


def _choices(seq: Any, trail: dict[int, int], found: list[Choice]) -> None:
    """Every choice point, with the choices that make the render reach it."""
    for op, av in seq:
        if op is C.BRANCH:
            size = len(av[1])
            found.append((id(av), tuple(range(size)), trail, size, -1))
            for index, alternative in enumerate(av[1]):
                _choices(alternative, {**trail, id(av): index}, found)
        elif op is C.IN:
            items = list(av)
            if not any(o is C.NEGATE for o, _ in items) and len(items) > 1:
                found.append((id(av), tuple(range(len(items))), trail, len(items), -1))
        elif op in _REPEATS:
            low, high, sub = av
            counts = _repeat_counts(low, high)
            if len(counts) > 1:
                # Base renders keep drawing an optional part in or out (0 or 1), as
                # before; the bounded maximum is tried once per node below.
                top = counts[-1] if counts[-1] > max(low, 1) else 0
                found.append((id(av), counts, trail, 2 if low == 0 else 0, top))
            _choices(sub, {**trail, id(av): 1} if low == 0 else trail, found)
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


def _repeat_slots(seq: Any, found: list[tuple[Any, int]]) -> None:
    """Every repeat as (the sequence holding it, its index), in a fixed order."""
    for index, (op, av) in enumerate(seq):
        if op in _REPEATS:
            found.append((seq, index))
            _repeat_slots(av[2], found)
        elif op is C.BRANCH:
            for alternative in av[1]:
                _repeat_slots(alternative, found)
        elif op is C.SUBPATTERN:
            _repeat_slots(av[3], found)
        elif op in (C.ASSERT, C.ASSERT_NOT):
            _repeat_slots(av[1], found)


def _tightened(pattern: re.Pattern[str], slot: int, low: int, high: int) -> re.Pattern[str]:
    """The pattern with one repeat (by its place in _repeat_slots) bounded to {low,high}."""
    tree = P.parse(pattern.pattern, pattern.flags)
    slots: list[tuple[Any, int]] = []
    _repeat_slots(tree, slots)
    seq, index = slots[slot]
    op, (_low, _high, sub) = seq[index]
    seq[index] = (op, (low, high, sub))
    compiled: re.Pattern[str] = K.compile(tree, pattern.flags)
    return compiled


def _pick(
    pattern: re.Pattern[str], category: str, tree: Any, choose: dict[int, int],
    bases: list[dict[int, int]], tight: re.Pattern[str] | None = None,
) -> str | None:
    """The first render over ``bases`` that only this pattern matches, else the first match.

    With ``tight`` (this pattern with a bound tightened by one), first the render
    that only this pattern matches and ``tight`` does not: the string that fails
    when the bound is tightened. Spaced filler only when plain filler matches
    nothing, so the strings that already worked stay exactly as they were.
    """
    if tight is not None:
        for spaced in (False, True):
            for base in bases:
                text = _fill(_render(tree, {**base, **choose}), spaced)
                if (
                    pattern.search(text)
                    and not tight.search(text)
                    and _alone(text, pattern, category)
                ):
                    return text
    for spaced in (False, True):
        fallback = None
        for base in bases:
            text = _fill(_render(tree, {**base, **choose}), spaced)
            if not pattern.search(text):
                continue
            if _alone(text, pattern, category):
                return text
            fallback = fallback or text
        if fallback is not None:
            return fallback
    return None


def _strings_for(pattern: re.Pattern[str], category: str) -> list[str]:
    """Every alternative of every branch and both bounds of every repeat, in the setting
    where only this pattern matches."""
    tree = P.parse(pattern.pattern, pattern.flags)
    nodes: list[Choice] = []
    _choices(tree, {}, nodes)
    rng = random.Random(pattern.pattern)
    out: dict[str, None] = {}
    bases: list[dict[int, int]] = [{}]
    for _ in range(25):
        bases.append(
            {node: values[rng.randrange(drawn)] for node, values, _t, drawn, _top in nodes if drawn}
        )
    # Every repeat at its minimum, or at its maximum: a repeat next to another one
    # ("\s*.{0,20}", ".{0,10}(X)?.{0,5}") can take up the slack of a tightened bound
    # unless the neighbours leave none.
    repeats = [(node, values) for node, values, _t, _d, top in nodes if top >= 0]
    lows = {node: values[0] for node, values in repeats}
    highs = {node: values[-1] for node, values in repeats}
    high_bases = [{**base, **highs} for base in bases]
    low_bases = [{**base, **lows} for base in bases]
    slots: list[tuple[Any, int]] = []
    _repeat_slots(tree, slots)
    where = {id(seq[index][1]): place for place, (seq, index) in enumerate(slots)}
    bounds = {id(seq[index][1]): seq[index][1][:2] for seq, index in slots}
    for node, values, trail, _drawn, top in nodes:
        for alternative in values:
            choose = {**trail, node: alternative}
            found: list[str | None] = []
            tight: re.Pattern[str] | None
            if top > 0 and alternative == top:
                # The maximum: a string that {m,n-1} no longer matches.
                tight = _tightened(pattern, where[node], values[0], top - 1)
                found.append(_pick(pattern, category, tree, choose, high_bases, tight))
                if found[0] is None or not _alone(found[0], pattern, category):
                    found.append(_pick(pattern, category, tree, choose, bases))
            else:
                found.append(_pick(pattern, category, tree, choose, bases))
                if top >= 0 and alternative == values[0]:
                    # The minimum: a string that {m+1,n} no longer matches.
                    low, high = bounds[node]
                    tight = (
                        _tightened(pattern, where[node], low + 1, high) if low < high else None
                    )
                    found.append(_pick(pattern, category, tree, choose, low_bases, tight))
            for text in found:
                if text is not None:
                    out[text] = None
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
        nodes: list[Choice] = []
        _choices(body, {}, nodes)
        expand = (*parents, body)
        depth = len(expand)
        renders = [_render(tree, trail, expand)]
        for node, values, inner, _drawn, _top in nodes:
            for alternative in values:
                # A lookahead's optional parts stay off unless chosen: a nested
                # exception after the word would otherwise cancel it.
                renders.append(_render(tree, {**trail, **inner, node: alternative}, expand))
        # Both fillers: main() keeps whichever the sanitizer treats as this depth says.
        for spaced in (False, True):
            for text in renders:
                out[(_fill(text, spaced), depth)] = None
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
