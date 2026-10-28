"""Generate src/blink_sanitizer/_confusables.py from Unicode confusables.txt.

Usage:
    curl -o confusables.txt https://www.unicode.org/Public/security/latest/confusables.txt
    python scripts/gen_confusables.py confusables.txt

Selection rules (each is a guard against rewriting text that is already fine):

* the source is ONE code point and it is not ASCII — plain ASCII text is
  never rewritten (so "rn", "1", "I", "|" keep their meaning);
* the target (the UTS #39 prototype) is 1-3 ASCII letters/digits;
* the source is skipped when NFKD plus combining-mark removal already turns
  it into ASCII letters/digits. NFKD runs in the pipeline anyway and is the
  better mapping there: confusables.txt maps fullwidth "I" (U+FF29) and the
  Roman numeral one (U+2160) to "l", while NFKD gives "I".

The hand-written ``_CONFUSABLES`` map in sanitizer.py is applied on top of
this table and wins on conflicts, so its long-standing entries (for example
Cyrillic "I" -> "I", where confusables.txt says "l") keep their behaviour.

The data is derived from a Unicode data file and carries its notice; see
LICENSES/Unicode-3.0.txt.
"""
from __future__ import annotations

import hashlib
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "src" / "blink_sanitizer" / "_confusables.py"

_ASCII_ALNUM = re.compile(r"[A-Za-z0-9]{1,3}")


def _fold(ch: str) -> str:
    decomposed = unicodedata.normalize("NFKD", ch)
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")


def _header_field(text: str, name: str) -> str:
    m = re.search(rf"^#\s*{name}:\s*(.+)$", text, re.M)
    if not m:
        raise SystemExit(f"confusables.txt header has no {name!r} line")
    return m.group(1).strip()


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    src = Path(sys.argv[1])
    raw = src.read_bytes()
    text = raw.decode("utf-8-sig")
    version = _header_field(text, "Version")
    date = _header_field(text, "Date")

    entries: dict[str, tuple[str, str]] = {}
    for line in text.splitlines():
        data, _, comment = line.partition("#")
        if not data.strip():
            continue
        fields = [f.strip() for f in data.split(";")]
        source = "".join(chr(int(cp, 16)) for cp in fields[0].split())
        target = "".join(chr(int(cp, 16)) for cp in fields[1].split())
        if len(source) != 1 or ord(source) < 0x80:
            continue
        if not _ASCII_ALNUM.fullmatch(target):
            continue
        if _ASCII_ALNUM.fullmatch(_fold(source)):
            continue
        # "( x → y ) SOURCE NAME → TARGET NAME" -> keep the source name.
        name = comment.split(")", 1)[-1].split("→", 1)[0].strip()
        entries[source] = (target, name)

    lines = [
        '"""Unicode confusables folded to ASCII. GENERATED FILE, DO NOT EDIT.',
        "",
        "Regenerate with: python scripts/gen_confusables.py confusables.txt",
        "",
        "Source: https://www.unicode.org/Public/security/latest/confusables.txt",
        f"        Unicode Security Mechanisms for UTS #39, Version: {version}",
        f"        Date: {date}",
        f"        sha256: {hashlib.sha256(raw).hexdigest()}",
        "",
        "Derived from Unicode data files. Copyright (c) 1991-2026 Unicode, Inc.",
        "Used under the Unicode License v3; the full notice is in",
        "LICENSES/Unicode-3.0.txt.",
        "",
        "Only single non-ASCII source code points whose prototype is 1-3 ASCII",
        "letters/digits, and that NFKD does not already fold to ASCII, are kept",
        "(see scripts/gen_confusables.py for the rules).",
        '"""',
        "from __future__ import annotations",
        "",
        f'UNICODE_CONFUSABLES_VERSION = "{version}"',
        "",
        "CONFUSABLES_TO_ASCII: dict[str, str] = {",
    ]
    for source in sorted(entries, key=ord):
        target, name = entries[source]
        cp = ord(source)
        escaped = f"\\u{cp:04X}" if cp <= 0xFFFF else f"\\U{cp:08X}"
        lines.append(f'    "{escaped}": "{target}",  # {name}')
    lines.append("}")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(entries)} entries (Unicode {version})")


if __name__ == "__main__":
    main()
