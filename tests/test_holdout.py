"""Detection on strings the patterns were not written from, plus metamorphic survival.

Every other test checks strings someone wrote next to the patterns, so a list of
memorized strings could pass them (a review deleted all 266 regexes for 319 exact
strings and 559 of 562 tests stayed green). These numbers cannot be memorized as
long as tests/holdout/README.md is followed. They are pinned both ways: a drop in
detection or a rise in false positives fails here, and so does a change the README
has not been updated to report.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from blink_sanitizer import InputSanitizer

from .pattern_cases import PATTERN_CASES

ROOT = Path(__file__).resolve().parent.parent
HOLDOUT = ROOT / "tests" / "holdout"

# What this release measures. Update together with the README, never to get green.
DETECTED, ATTACKS = 222, 455
FLAGGED, BENIGN = 30, 277

# Frozen content. Changing a corpus file must be a visible decision (see its README).
SHA256 = {
    "attacks.json": "e0dd5b4565da6924334e31360a7d07d5c04e80a0d0de6ef570a91f878bcb3837",
    "benign.json": "756f58bea3036f95ce96ee8d09a04c82b304fdc1052da81f754ceddfd72b0eb2",
}


def _load(kind: str) -> dict[str, list[str]]:
    data: dict[str, list[str]] = json.loads((HOLDOUT / f"{kind}.json").read_text("utf-8"))
    return data


@pytest.fixture(scope="module")
def sanitizer() -> InputSanitizer:
    return InputSanitizer()


@pytest.mark.parametrize("name", sorted(SHA256))
def test_corpus_is_frozen(name: str) -> None:
    raw = (HOLDOUT / name).read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(raw).hexdigest() == SHA256[name]


def test_attack_detection_is_what_the_project_claims(sanitizer: InputSanitizer) -> None:
    groups = _load("attacks")
    assert sum(map(len, groups.values())) == ATTACKS
    detected = sum(not sanitizer.check(t).allowed for texts in groups.values() for t in texts)
    assert detected == DETECTED


def test_false_positives_are_what_the_project_claims(sanitizer: InputSanitizer) -> None:
    groups = _load("benign")
    assert sum(map(len, groups.values())) == BENIGN
    flagged = sum(not sanitizer.check(t).allowed for texts in groups.values() for t in texts)
    assert flagged == FLAGGED


def test_readme_reports_these_numbers() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"{DETECTED} of {ATTACKS} attacks detected" in readme
    assert f"{FLAGGED} of {BENIGN} benign texts flagged" in readme


# ── Metamorphic: transform attack sentences we know are caught ──────────
#
# The survival rate of each transform is a tracked number, not a target of 100%:
# word-level grammar changes usually evade a regex layer and the README says so.
# These floors catch a transform that used to survive and stopped.

_WORDS = re.compile(r"[^\W\d_]+")


def _latin(text: str) -> bool:
    return text.isascii() and len(_WORDS.findall(text)) >= 3


def _join(separator: str):  # type: ignore[no-untyped-def]
    return lambda text: separator.join(_WORDS.findall(text))


def _camel(text: str) -> str:
    return "".join(word.capitalize() for word in _WORDS.findall(text))


def _insert_after_first(extra: str):  # type: ignore[no-untyped-def]
    def transform(text: str) -> str:
        first, _, rest = text.partition(" ")
        return f"{first}{extra} {rest}"

    return transform


# name -> (transform, lowest accepted share of sentences still flagged)
TRANSFORMS = {
    "upper": (str.upper, 0.95),
    "note_prefix": (lambda t: "Note: " + t, 0.95),
    "quotes": (lambda t: f'"{t}"', 0.95),
    "kebab": (_join("-"), 0.90),
    "snake": (_join("_"), 0.90),
    "dotted": (_join("."), 0.90),
    "slashes": (_join("/"), 0.90),
    "pipes": (_join(" | "), 0.90),
    "emoji": (_join(" \U0001f525 "), 0.90),
    "camel": (_camel, 0.80),
    "comma_after_first": (_insert_after_first(","), 0.05),
}


@pytest.mark.parametrize("name", sorted(TRANSFORMS))
def test_metamorphic_survival(sanitizer: InputSanitizer, name: str) -> None:
    transform, floor = TRANSFORMS[name]
    base = [t.rstrip(".?!") for t, _ in PATTERN_CASES if _latin(t)]
    assert len(base) > 100
    kept = sum(not sanitizer.check(transform(t)).allowed for t in base)
    assert kept / len(base) >= floor, f"{kept}/{len(base)}"
