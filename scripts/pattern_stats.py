"""Count attack patterns per language and per category.

Languages are not stored on the patterns at runtime, so this script reads
src/blink_sanitizer/sanitizer.py, walks the ``_PATTERNS`` list with ``ast``,
and assigns each ``(regex, category)`` entry to the most recent language
banner comment above it (the boxed ``# ── ENGLISH: ...`` headers). The total
is cross-checked against ``len(_PATTERNS)`` from the imported module.

Usage:
    python scripts/pattern_stats.py      # prints a Markdown table
"""
from __future__ import annotations

import ast
import re
from collections import Counter
from pathlib import Path

from blink_sanitizer.sanitizer import _PATTERNS

SOURCE = Path(__file__).resolve().parent.parent / "src" / "blink_sanitizer" / "sanitizer.py"

# Banner keyword -> label used in the table
_BANNERS = [
    ("ENGLISH", "EN"),
    ("PORTUGUESE", "PT-BR"),
    ("SPANISH", "ES"),
    ("FRENCH", "FR"),
    ("GERMAN", "DE"),
    ("ITALIAN", "IT"),
    ("RUSSIAN", "RU"),
    ("CHINESE", "ZH"),
    ("JAPANESE", "JA"),
    ("KOREAN", "KO"),
    ("ARABIC", "AR"),
    ("UNIVERSAL", "Shared (EN-worded)"),
    ("Cost manipulation", "Shared (EN-worded)"),
]
_BANNER_RE = re.compile(r"^\s*# ── (" + "|".join(re.escape(k) for k, _ in _BANNERS) + r")\b")

CATEGORIES = [
    "prompt_injection",
    "jailbreak",
    "system_prompt_extraction",
    "persona_hijack",
    "role_override",
    "data_exfiltration",
    "cost_manipulation",
]
SHORT = {
    "prompt_injection": "inject",
    "jailbreak": "jailbrk",
    "system_prompt_extraction": "sys_prompt",
    "persona_hijack": "persona",
    "role_override": "role",
    "data_exfiltration": "exfil",
    "cost_manipulation": "cost",
}


def collect() -> list[tuple[str, str]]:
    text = SOURCE.read_text(encoding="utf-8")
    lines = text.splitlines()
    label_by_kw = dict(_BANNERS)

    # line number -> language label (1-based, as ast uses)
    lang_at: dict[int, str] = {}
    current = "?"
    for no, line in enumerate(lines, start=1):
        m = _BANNER_RE.match(line)
        if m:
            current = label_by_kw[m.group(1)]
        lang_at[no] = current

    tree = ast.parse(text)
    entries: list[tuple[str, str]] = []
    for node in tree.body:
        target = getattr(node, "target", None) or (getattr(node, "targets", [None])[0])
        if isinstance(target, ast.Name) and target.id == "_PATTERNS":
            for elt in node.value.elts:  # type: ignore[union-attr]
                category = elt.elts[1].value  # type: ignore[attr-defined]
                entries.append((lang_at[elt.lineno], category))
    return entries


def main() -> None:
    entries = collect()
    if len(entries) != len(_PATTERNS):
        raise SystemExit(f"parse mismatch: {len(entries)} parsed vs {len(_PATTERNS)} loaded")

    counts = Counter(entries)
    langs: list[str] = []
    for _, label in _BANNERS:
        if label not in langs and any(lang == label for lang, _ in entries):
            langs.append(label)

    header = ["Language", *(SHORT[c] for c in CATEGORIES), "Total"]
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join("---" if i == 0 else "---:" for i in range(len(header))) + "|")
    for lang in langs:
        row = [counts[(lang, c)] for c in CATEGORIES]
        cells = [str(n) if n else "0" for n in row]
        print(f"| {lang} | " + " | ".join(cells) + f" | {sum(row)} |")
    col_totals = [sum(counts[(lang, c)] for lang in langs) for c in CATEGORIES]
    print("| **Total** | " + " | ".join(str(n) for n in col_totals) + f" | **{len(entries)}** |")


if __name__ == "__main__":
    main()
