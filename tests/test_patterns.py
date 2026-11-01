"""Every pattern is verified: it has a sentence only it catches, and its branches are pinned.

A mutation run deleted 160 of the then 266 patterns at once with the suite still
green, because test strings fired only a few of them. These tests fail when any
single pattern disappears, changes category, loses a branch, or arrives without
a test of its own.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from blink_sanitizer import InputSanitizer, sanitizer as module
from blink_sanitizer.sanitizer import _PATTERNS

from .pattern_cases import PATTERN_CASES

ROOT = Path(__file__).resolve().parent.parent
PATTERN_COUNT = 272
LANGUAGE_COUNT = 11

# Exact set of categories, so a typo in a new label cannot create a silent one.
CATEGORIES = {
    "prompt_injection",
    "jailbreak",
    "system_prompt_extraction",
    "persona_hijack",
    "role_override",
    "data_exfiltration",
    "cost_manipulation",
}


class _Recorder:
    def __init__(self) -> None:
        self.seen: list[str] = []

    def search(self, text: str) -> None:
        self.seen.append(text)


def _candidates(text: str) -> list[str]:
    recorder = _Recorder()
    original: Any = module._PATTERNS
    module._PATTERNS = [(recorder, "x")]  # type: ignore[list-item]
    try:
        InputSanitizer().check(text)
    finally:
        module._PATTERNS = original
    return recorder.seen


@pytest.fixture(scope="module")
def sanitizer() -> InputSanitizer:
    return InputSanitizer()


def test_pattern_count_is_pinned_and_public_numbers_agree() -> None:
    assert len(_PATTERNS) == PATTERN_COUNT
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert f"{PATTERN_COUNT} compiled regex patterns across {LANGUAGE_COUNT} languages" in readme
    assert f"{PATTERN_COUNT} patterns across {LANGUAGE_COUNT} languages" in pyproject


def test_only_known_categories() -> None:
    assert {category for _, category in _PATTERNS} == CATEGORIES


@pytest.mark.parametrize(("text", "category"), PATTERN_CASES)
def test_natural_sentence_reports_its_category(
    sanitizer: InputSanitizer, text: str, category: str
) -> None:
    assert category in sanitizer.check(text).violations


def test_every_pattern_owns_a_sentence() -> None:
    """Remove any one pattern and some sentence above stops reporting its category.

    The owner is worked out, not declared: a sentence owns the pattern when that
    is the only pattern of its category matching any candidate of the sentence.
    A new pattern with no such sentence fails here, whatever its position.
    """
    owned: set[int] = set()
    problems: list[str] = []
    for text, category in PATTERN_CASES:
        versions = _candidates(text)
        hits = [
            index
            for index, (pattern, pattern_category) in enumerate(_PATTERNS)
            if pattern_category == category and any(pattern.search(v) for v in versions)
        ]
        if len(hits) == 1:
            owned.add(hits[0])
        else:
            problems.append(f"{text!r} matched by {hits}")
    orphans = [_PATTERNS[i][0].pattern[:90] for i in range(len(_PATTERNS)) if i not in owned]
    assert not orphans, f"patterns without a sentence of their own: {orphans}"
    assert not problems, problems


def test_every_generated_branch_string_still_reports_its_category(
    sanitizer: InputSanitizer,
) -> None:
    """Snapshot from scripts/gen_pattern_corpus.py: pattern branches, by alternative."""
    corpus = json.loads((ROOT / "tests" / "pattern_corpus.json").read_text("utf-8"))
    flagged = [row for row in corpus if row[0] != "allowed"]
    assert len(flagged) > 3_000
    lost = [
        (category, text)
        for category, text in flagged
        if category not in sanitizer.check(text).violations
    ]
    assert not lost, f"{len(lost)} of {len(flagged)} no longer flagged, first: {lost[:15]}"


def test_every_generated_exception_stays_allowed(sanitizer: InputSanitizer) -> None:
    """The same snapshot, for negative lookaheads: an attack shape plus an excepted word."""
    corpus = json.loads((ROOT / "tests" / "pattern_corpus.json").read_text("utf-8"))
    allowed = [text for label, text in corpus if label == "allowed"]
    assert len(allowed) > 100
    now_flagged = [text for text in allowed if not sanitizer.check(text).allowed]
    assert not now_flagged, f"{len(now_flagged)} now flagged, first: {now_flagged[:15]}"


# Hard benign text: ordinary sentences that use the vocabulary of the attacks. The
# one-word forms of each trigger, so an over-broad new rule such as r"\brepeat\b"
# fails here. Unlike tests/holdout/, these may be used when writing patterns.
HARD_BENIGN = [
    "Please repeat the question, the line was bad.",
    "We repeat this survey every quarter.",
    "Ignore the first two rows of the spreadsheet.",
    "You can ignore that warning, it is harmless.",
    "I forgot my password again.",
    "Forget it, we will order pizza instead.",
    "The system was down for an hour last night.",
    "What is the admin console URL for our Slack workspace?",
    "The prompt on the form asks for a shipping address.",
    "The password policy requires twelve characters.",
    "These instructions explain how to assemble the shelf.",
    "The rules of the league changed this season.",
    "Show the quarterly numbers by region.",
    "Pretend play is good for young children.",
    "Send the invoice to the finance team.",
    "Bypass surgery recovery takes about six weeks.",
    "Override the default font in the stylesheet.",
    "Disregard my last email, I attached the wrong file.",
    "Reveal the answer at the end of the quiz.",
    "A token is a unit of text in natural language processing.",
    "The secret to a good sourdough is patience.",
    "Developer documentation is linked in the footer.",
    "The new instructions are in the shared folder.",
    "Export the data to CSV for the auditors.",
    "Remove the filter from the coffee machine before washing it.",
    "The role of the system administrator is changing.",
]


@pytest.mark.parametrize("text", HARD_BENIGN)
def test_hard_benign_text_stays_allowed(sanitizer: InputSanitizer, text: str) -> None:
    result = sanitizer.check(text)
    assert result.allowed, result.violations
