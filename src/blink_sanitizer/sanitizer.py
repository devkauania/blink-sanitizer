"""Blink Sanitizer — Prompt Injection Detection

Pattern-based input sanitization — detects known attack patterns with
compiled regexes, no LLM calls. Measured p50 ~0.36 ms / p95 ~0.80 ms per
check on an Intel Core i5-12400F with CPython 3.12 (scripts/benchmark.py).
This is a heuristic first layer, not a complete prompt-injection defense.

Features:
- Unicode NFKD normalization (homoglyphs, fullwidth, zero-width chars, bidi)
- Leetspeak de-substitution (1->i, 3->e, 4->a, etc.)
- Separator stripping (i.g.n.o.r.e, i g n o r e) and chained leet+separator variants
- Context field checking (indirect prompt injection defense)
- 11 languages: EN, PT-BR, ES, FR, DE, IT, RU, ZH, JA, KO, AR
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SanitizationResult:
    """Result of pattern-based input check."""

    allowed: bool
    violations: list[str] = field(default_factory=list)


# ── Zero-width and bidi characters to strip ──────────────────

_INVISIBLE_CHARS = re.compile(
    "["
    "\u200B"  # zero-width space
    "\u200C"  # zero-width non-joiner
    "\u200D"  # zero-width joiner
    "\uFEFF"  # BOM / zero-width no-break space
    "\u200E"  # left-to-right mark
    "\u200F"  # right-to-left mark
    "\u202A"  # left-to-right embedding
    "\u202B"  # right-to-left embedding
    "\u202C"  # pop directional formatting
    "\u202D"  # left-to-right override
    "\u202E"  # right-to-left override
    "\u2060"  # word joiner
    "\u2061"  # function application
    "\u2062"  # invisible times
    "\u2063"  # invisible separator
    "\u2064"  # invisible plus
    "\u2066"  # left-to-right isolate
    "\u2067"  # right-to-left isolate
    "\u2068"  # first strong isolate
    "\u2069"  # pop directional isolate
    "]+"
)

# ── Cyrillic → Latin confusables (most common homoglyphs) ────

_CONFUSABLES: dict[str, str] = {
    # Cyrillic lowercase -> Latin equivalent
    "\u0430": "a",  # U+0430 -> a
    "\u0435": "e",  # U+0435 -> e
    "\u043E": "o",  # U+043E -> o
    "\u0440": "p",  # U+0440 -> p
    "\u0441": "c",  # U+0441 -> c
    "\u0443": "y",  # U+0443 -> y
    "\u0445": "x",  # U+0445 -> x
    "\u0456": "i",  # U+0456 -> i (Ukrainian)
    "\u0455": "s",  # U+0455 -> s (Macedonian)
    "\u0458": "j",  # U+0458 -> j
    "\u04BB": "h",  # U+04BB -> h
    "\u0431": "b",  # U+0431 ~ b (loose)
    "\u043D": "n",  # U+043D ~ n (visual)
    # Cyrillic uppercase -> Latin equivalent
    "\u0410": "A",  # U+0410 -> A
    "\u0412": "B",  # U+0412 -> B
    "\u0415": "E",  # U+0415 -> E
    "\u041A": "K",  # U+041A -> K
    "\u041C": "M",  # U+041C -> M
    "\u041D": "H",  # U+041D -> H
    "\u041E": "O",  # U+041E -> O
    "\u0420": "P",  # U+0420 -> P
    "\u0421": "C",  # U+0421 -> C
    "\u0422": "T",  # U+0422 -> T
    "\u0425": "X",  # U+0425 -> X
    # Greek lookalikes -> Latin equivalent
    "\u03BF": "o",  # U+03BF -> o
    "\u03B1": "a",  # U+03B1 -> a
    "\u03B5": "e",  # U+03B5 -> e
}

# ── Leetspeak mapping ────────────────────────────────────────

_LEET_MAP: dict[str, str] = {
    "1": "i",
    "3": "e",
    "4": "a",
    "5": "s",
    "0": "o",
    "7": "t",
    "@": "a",
    "$": "s",
}

_LEET_CHARS = frozenset(_LEET_MAP.keys())

# ── Separator stripping (defeats char-insertion evasion) ───

# Characters that attackers insert between letters to break pattern matching.
# The leet symbols "@" and "$" count as word chars on either side so that
# "d-1-$-r-3-g-4-r-d" collapses to "d1$r3g4rd" before de-leeting.
_SEPARATOR_CHARS = re.compile(r"(?<=[\w@$])[.\-/\\|_*~`']{1,3}(?=[\w@$])")

# Excessive spacing between single characters (e.g., "i g n o r e")
_CHAR_SPACED = re.compile(r"(?<=\b\w)\s+(?=\w\b)")


def _strip_separators(text: str) -> str:
    """Remove separator characters inserted between word characters.

    Catches evasion like: ig.nore, i-g-n-o-r-e, ignore/all, by-pass
    Also collapses single-char spacing: "i g n o r e" → "ignore"
    And collapses short-fragment splits: "prev ious instruct ions" → "previous instructions"
    """
    # Step 1: Remove separator chars between word chars
    result = _SEPARATOR_CHARS.sub("", text)
    # Step 2: Detect and collapse single-char-spaced words
    # Look for sequences of single chars separated by spaces (3+ in a row)
    # e.g., "i g n o r e" but not "I am a developer"
    def _collapse_single_chars(match: re.Match) -> str:
        """Collapse runs of spaced single characters."""
        segment = match.group(0)
        return segment.replace(" ", "")

    result = re.sub(
        r"\b(\w\s){2,}\w\b",  # 3+ single chars with spaces between
        _collapse_single_chars,
        result,
    )
    return result


def _deleet(text: str) -> str:
    """Convert leetspeak substitutions back to alphabetic form.

    Uses cluster-based detection: finds contiguous runs of (letters + leet chars)
    and converts leet chars only within clusters that contain at least one letter.
    This avoids false positives on standalone numbers like '$1.5M' or '3 topics'.
    """
    chars = list(text)
    n = len(chars)
    result = chars.copy()

    i = 0
    while i < n:
        # Skip non-word, non-leet chars
        if not (chars[i].isalpha() or chars[i] in _LEET_CHARS):
            i += 1
            continue

        # Found start of a cluster — scan to find its end
        cluster_start = i
        has_letter = False
        while i < n and (chars[i].isalpha() or chars[i] in _LEET_CHARS):
            if chars[i].isalpha():
                has_letter = True
            i += 1

        # Convert leet chars only if cluster contains at least one letter
        if has_letter:
            for j in range(cluster_start, i):
                if chars[j] in _LEET_MAP:
                    result[j] = _LEET_MAP[chars[j]]

    return "".join(result)


# ── Attack patterns ──────────────────────────────────────────

# (compiled_regex, violation_category)
_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # ══════════════════════════════════════════════════════════
    # ── ENGLISH: Prompt injection ────────────────────────────
    # ══════════════════════════════════════════════════════════
    (
        re.compile(
            r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+"
            r"(instructions?|prompts?|rules?|directions?)",
            re.I,
        ),
        "prompt_injection",
    ),
    (
        re.compile(
            r"(forget|disregard|override|bypass|skip|circumvent|neglect)\s+"
            r"(your|all|the|previous|prior|any|every|my)?\s*"
            r"(\w+\s+)?"
            r"(instructions?|rules?|constraints?|guidelines?|guardrails?|"
            r"filters?|restrictions?|safeguards?|safety|protections?|programming|directives?)",
            re.I,
        ),
        "prompt_injection",
    ),
    (re.compile(r"new\s+instructions?:\s*", re.I), "prompt_injection"),
    (re.compile(r"system\s*:\s*you\s+(are|will|must|should)", re.I), "prompt_injection"),
    # ── ENGLISH: Synonym + passive + question evasion ──────
    (
        re.compile(
            r"(discard|overlook|omit|abandon|drop|ditch|eliminate|suppress|erase)\s+"
            r"(all\s+)?(your\s+)?(previous\s+|prior\s+)?"
            r"(instructions?|rules?|constraints?|guidelines?|restrictions?|directives?)",
            re.I,
        ),
        "prompt_injection",
    ),
    (
        re.compile(
            r"(all|your|the|previous|prior)\s+"
            r"(instructions?|rules?|restrictions?|guidelines?|safety\s+filters?|constraints?)\s+"
            r"(should|must|need\s+to|have\s+to|ought\s+to)\s+be\s+"
            r"(ignored|forgotten|discarded|overridden|bypassed|disabled|removed)",
            re.I,
        ),
        "prompt_injection",
    ),
    (
        re.compile(
            r"(can|could|would|will)\s+you\s+"
            r"(please\s+)?(kindly\s+)?"
            r"(ignore|forget|disregard|bypass|disable|override|skip|drop)\s+"
            r"(all\s+)?(your\s+)?(the\s+)?(previous\s+)?"
            r"(instructions?|rules?|restrictions?|safety|filters?|guidelines?)",
            re.I,
        ),
        "prompt_injection",
    ),
    (
        re.compile(
            r"(please|kindly|just|honestly)\s+"
            r"(ignore|forget|disregard|bypass|disable|override|skip|drop)\s+"
            r"(all\s+)?(of\s+)?(your\s+)?(the\s+)?(previous\s+)?"
            r"(instructions?|rules?|restrictions?|safety|filters?|guidelines?)",
            re.I,
        ),
        "prompt_injection",
    ),
    # ── ENGLISH: Social engineering & authority claims ──────
    (
        re.compile(
            r"(i\s+am|i'm)\s+(the|a|an)\s+"
            r"(developer|admin|administrator|owner|creator|engineer|manager|supervisor)\b",
            re.I,
        ),
        "role_override",
    ),
    (
        re.compile(
            r"(my\s+supervisor|my\s+boss|my\s+manager|the\s+admin)\s+"
            r"(authorized|approved|told|said|asked|instructed|wants)",
            re.I,
        ),
        "role_override",
    ),
    (
        re.compile(
            r"for\s+(testing|educational|research|debug|security)\s+"
            r"purposes?\s*.{0,20}"
            r"(disable|bypass|remove|ignore|show|reveal|give|grant|turn\s+off)",
            re.I,
        ),
        "role_override",
    ),
    (
        re.compile(
            r"(give|grant|provide)\s+(me|us)\s+"
            r"(debug|admin|root|full|unrestricted|unlimited|elevated)\s+"
            r"(access|permissions?|privileges?|control|mode)",
            re.I,
        ),
        "role_override",
    ),
    (
        re.compile(
            r"(disable|deactivate|turn\s+off|remove|switch\s+off)\s+"
            r"(all\s+)?(your\s+)?(the\s+)?"
            r"(safety|security|content\s+filter|restrictions?|protections?|guardrails?|filters?|safeguards?)",
            re.I,
        ),
        "prompt_injection",
    ),
    # ── ENGLISH: System prompt extraction ──────────────────
    (
        re.compile(
            r"(show|reveal|display|print|output|repeat|tell|translate|summarize|rewrite|dump|leak|expose)\s+\w*\s*"
            r"(your|the)\s+"
            r"(system\s+prompt|initial\s+instructions?|original\s+prompt|"
            r"hidden\s+prompt|internal\s+instructions?|secret\s+instructions?|"
            r"configuration|system\s+message|base\s+prompt)",
            re.I,
        ),
        "system_prompt_extraction",
    ),
    (
        re.compile(
            r"what\s+(are|is|were)\s+your\s+"
            r"(system\s+prompt|system\s+instructions?|instructions?|rules?|"
            r"initial\s+prompt|base\s+prompt|hidden\s+instructions?|directives?)",
            re.I,
        ),
        "system_prompt_extraction",
    ),
    (
        re.compile(
            r"(repeat|recite|echo)\s+(everything|all|all\s+the\s+text|the\s+text)\s+"
            r"(above|before|from\s+the\s+start)",
            re.I,
        ),
        "system_prompt_extraction",
    ),
    # ── ENGLISH: Jailbreak ─────────────────────────────────
    (
        re.compile(
            r"\b(DAN|do\s+anything\s+now|developer\s+mode|god\s+mode"
            r"|sudo\s+mode|admin\s+mode|jailbreak|unrestricted\s+mode"
            r"|no[\s-]?filter\s+mode|uncensored\s+mode)\b",
            re.I,
        ),
        "jailbreak",
    ),
    (
        re.compile(
            r"(enter|enable|activate|switch\s+to|engage|go\s+into|start)\s+"
            r"(developer|dev|debug|admin|god|sudo|unrestricted|unfiltered|uncensored|jailbreak)\s+"
            r"mode",
            re.I,
        ),
        "jailbreak",
    ),
    (
        re.compile(
            r"(act|behave|respond|operate|function)\s+(as|like|without)\s+"
            r"(any\s+|an?\s+)?(unrestricted|unfiltered|uncensored|unlimited|no\s+safety|"
            r"no\s+restrictions?|no\s+rules?|no\s+filters?|no\s+limits?|"
            r"restrictions?|rules?|filters?|limits?|safety)",
            re.I,
        ),
        "jailbreak",
    ),
    # ── ENGLISH: Persona hijack ────────────────────────────
    (re.compile(r"you\s+are\s+now\s+(a|an|the|in)\s+", re.I), "persona_hijack"),
    (
        re.compile(r"pretend\s+(you\s+are|to\s+be|you're)\s+(a|an|the)\s+", re.I),
        "persona_hijack",
    ),
    (
        re.compile(
            r"(simulate|roleplay|role[\s-]?play|impersonate|imagine\s+you\s+are)\s+"
            r"(a|an|the|being|as)\s+",
            re.I,
        ),
        "persona_hijack",
    ),
    # ══════════════════════════════════════════════════════════
    # ── PORTUGUESE (PT-BR): Injeção de prompt ────────────────
    # ══════════════════════════════════════════════════════════
    (
        re.compile(
            r"ignor[ea]\s+(todas?\s+)?(as\s+)?"
            r"(instruc[oõ]es|regras?|orientac[oõ]es|diretrizes?|prompts?|ordens?)",
            re.I,
        ),
        "prompt_injection",
    ),
    (
        re.compile(
            r"(esquec[ea]|desconsider[ea]|desrespeite|pule|abandone)\s+"
            r"(todas?\s+)?(as\s+)?(suas?\s+)?"
            r"(instruc[oõ]es|regras?|orientac[oõ]es|diretrizes?|restric[oõ]es|"
            r"protec[oõ]es|filtros?|seguranc[a]|limitac[oõ]es)",
            re.I,
        ),
        "prompt_injection",
    ),
    (
        re.compile(
            r"(desabilit[ea]|desativ[ea]|remov[ea]|desligue|desligar)\s+"
            r"(todas?\s+)?(as?\s+)?(suas?\s+)?(os?\s+)?"
            r"(protec[oõ]es|seguranc[a]|filtros?|restric[oõ]es|bloqueios?|"
            r"salvaguardas?|guardrails?|defesas?)",
            re.I,
        ),
        "prompt_injection",
    ),
    # ── PT: System prompt extraction ──────────────────────
    (
        re.compile(
            r"(mostre|exib[ea]|revele|mostr[ea]|diga|repita|traduza|imprima)\s+"
            r"(me\s+)?(o\s+|seu\s+|suas?\s+)?"
            r"(prompt\s+de?\s+sistema|instruc[oõ]es\s+(iniciais?|internas?|ocultas?|secretas?)|"
            r"prompt\s+(original|oculto|interno|inicial|escondido))",
            re.I,
        ),
        "system_prompt_extraction",
    ),
    (
        re.compile(
            r"(quais?\s+s[aã]o|qual\s+[eé])\s+"
            r"(suas?|os?\s+seus?)\s+"
            r"(instruc[oõ]es|regras?|prompt\s+de?\s+sistema|diretrizes?)",
            re.I,
        ),
        "system_prompt_extraction",
    ),
    # ── PT: Jailbreak ────────────────────────────────────
    (
        re.compile(
            r"(ativ[ea]|entre\s+em|habilit[ea]|ligu[ea]|inici[ea])\s+"
            r"(o\s+)?(modo\s+)?"
            r"(desenvolvedor|desenvolvimiento|dev|admin|deus|god|sudo|"
            r"irrestrito|sem\s+filtro|sem\s+restric[oõ]es|sem\s+limites?|"
            r"sem\s+censura|desbloqueado|jailbreak)",
            re.I,
        ),
        "jailbreak",
    ),
    (
        re.compile(
            r"(atue|aja|comporte[\s-]?se|funcione|opere|responda)\s+"
            r"(como\s+)?(um\s*)?"
            r"(assistente\s+)?(sem\s+filtro|sem\s+restric[oõ]es|sem\s+regras?|"
            r"sem\s+limites?|sem\s+censura|irrestrito|desbloqueado|livre)",
            re.I,
        ),
        "jailbreak",
    ),
    # ── PT: Persona hijack ───────────────────────────────
    (
        re.compile(
            r"voce\s+(agora\s+)?(e|sera|[eé])\s+(um|uma|o|a)\s+",
            re.I,
        ),
        "persona_hijack",
    ),
    (
        re.compile(
            r"(finja|simule|imagine|fa[cç]a\s+de\s+conta)\s+"
            r"(que\s+)?(voce\s+)?(e|[eé]|ser|nao\s+tem)",
            re.I,
        ),
        "persona_hijack",
    ),
    # ── PT: Persona hijack (expanded) ─────────────────────
    (
        re.compile(
            r"(simule|imagine)\s+ser\s+(um|uma|o|a)\s+",
            re.I,
        ),
        "persona_hijack",
    ),
    # ── PT: Role override / authority claim ──────────────
    (
        re.compile(
            r"(me\s+d[eê]|conceda[\s-]?me|fornec[ea][\s-]?me)\s+"
            r"(acesso\s+)?(admin|root|total|irrestrito|completo|ilimitado|elevado)",
            re.I,
        ),
        "role_override",
    ),
    (
        re.compile(
            r"(sou\s+o|eu\s+sou\s+o?)\s+"
            r"(desenvolvedor|administrador|dono|criador|engenheiro|gerente|supervisor)",
            re.I,
        ),
        "role_override",
    ),
    (
        re.compile(
            r"para\s+(fins?\s+de\s+)?(teste|pesquisa|debug|seguranc[a])\s*.{0,20}"
            r"(desabilit[ea]|desativ[ea]|remov[ea]|ignor[ea]|mostre|revele|libere)",
            re.I,
        ),
        "role_override",
    ),
    (
        re.compile(
            r"(meu\s+supervisor|meu\s+chefe|meu\s+gerente|o\s+admin)\s+"
            r"(autorizou|aprovou|mandou|disse|pediu|instruiu)",
            re.I,
        ),
        "role_override",
    ),
    # ── PT: Data exfiltration ──────────────────────────
    (
        re.compile(
            r"(export[ea]|envi[ea]|transferi[ra]|vaz[ea])\s+"
            r"(todos?\s+)?(os\s+)?"
            r"(dados|credenciais?|chaves?\s+(de\s+)?api|tokens?|segredos?|senhas?|"
            r"variaveis?\s+de?\s+ambiente)\s+"
            r"(para|de|via)\s+",
            re.I,
        ),
        "data_exfiltration",
    ),
    # ── PT: Cost manipulation ──────────────────────────
    (
        re.compile(
            r"(repita|escreva|gere)\s+(isso|o\s+seguinte)\s+\w*\s*"
            r"(\d{3,}|mil|milh[oõ]es?|cem|centenas?)\s+vezes",
            re.I,
        ),
        "cost_manipulation",
    ),
    # ── PT: Evasion-resistant (synonyms, passive, modal, question) ──
    (
        re.compile(
            r"(descart[ea]r?|desprez[ea]r?|anul[ea]r?|desabilit[ea]r?)\s+"
            r"(tod[ao]s?\s+)?([ao]s\s+)?(suas?\s+|seus?\s+)?"
            r"(instruc[õo]es|regras|diretrizes|restri[cç][õo]es|prote[cç][õo]es|filtros|seguran[cç]a)",
            re.I,
        ),
        "prompt_injection",
    ),
    (
        re.compile(
            r"(todas?\s+)?(as\s+)?"
            r"(instruc[õo]es|regras|diretrizes|restri[cç][õo]es|prote[cç][õo]es|filtros|seguran[cç]a)"
            r"(\s+anteriores?)?\s+"
            r"(devem|precisam|t[eê]m\s+que|necessitam)\s+ser\s+"
            r"(ignorad[ao]s?|esquecid[ao]s?|desabilitad[ao]s?|descartad[ao]s?|desativad[ao]s?)",
            re.I,
        ),
        "prompt_injection",
    ),
    (
        re.compile(
            r"(voc[eê]|tu)\s+"
            r"(deve|precisa|tem\s+que|necessita|pode)\s+"
            r"(ignorar|esquecer|desabilitar|desativar|descartar|anular)\s+"
            r"(todas?\s+)?(as\s+)?"
            r"(instruc[õo]es|regras|diretrizes|restri[cç][õo]es|prote[cç][õo]es|filtros|seguran[cç]a)",
            re.I,
        ),
        "prompt_injection",
    ),
    (
        re.compile(
            r"(pode|poderia|consegue|seria\s+poss[ií]vel)\s+"
            r"(ignorar|esquecer|desabilitar|desativar|descartar|anular)\s+"
            r"(todas?\s+)?(as\s+)?(suas?\s+)?"
            r"(instruc[õo]es|regras|diretrizes|restri[cç][õo]es|prote[cç][õo]es|filtros|seguran[cç]a)",
            re.I,
        ),
        "prompt_injection",
    ),
    # ══════════════════════════════════════════════════════════
    # ── SPANISH: Inyección de prompt ─────────────────────────
    # ══════════════════════════════════════════════════════════
    (
        re.compile(
            r"ignora\s+(todas?\s+)?(las\s+)?"
            r"(instrucciones?|reglas?|directrices?|ordenes?|restricciones?)",
            re.I,
        ),
        "prompt_injection",
    ),
    (
        re.compile(
            r"(olvida|descarta|omite|pasa\s+por\s+alto|salta(te)?)\s+"
            r"(todas?\s+)?(las\s+)?(tus\s+)?"
            r"(instrucciones?|reglas?|restricciones?|directrices?|filtros?|"
            r"protecciones?|seguridad|limitaciones?)",
            re.I,
        ),
        "prompt_injection",
    ),
    (
        re.compile(
            r"(desactiva|desabilita|elimina|apaga|quita|remueve)\s+"
            r"(todas?\s+)?(las\s+)?(tus\s+)?"
            r"(protecciones?|seguridad|filtros?|restricciones?|bloqueos?|defensas?)",
            re.I,
        ),
        "prompt_injection",
    ),
    # ── SPANISH: System prompt extraction ─────────────────
    (
        re.compile(
            r"(muestra|revela|ensena|dime|repite|traduce|imprime)\s+"
            r"(me\s+)?(tu|el|las?)\s+"
            r"(prompt\s+de?\s+sistema|instrucciones?\s+(iniciales?|internas?|ocultas?|secretas?)|"
            r"prompt\s+(original|oculto|interno|inicial))",
            re.I,
        ),
        "system_prompt_extraction",
    ),
    (
        re.compile(
            r"(cuales?\s+son|cual\s+es)\s+"
            r"(tus|las?|el)\s+"
            r"(instrucciones?|reglas?|prompt\s+de?\s+sistema|directrices?)",
            re.I,
        ),
        "system_prompt_extraction",
    ),
    (
        re.compile(
            r"(repite|recita|reproduce)\s+(todo|el\s+texto)\s+"
            r"(anterior|de\s+arriba|desde\s+el\s+inicio)",
            re.I,
        ),
        "system_prompt_extraction",
    ),
    # ── SPANISH: Jailbreak ───────────────────────────────
    (
        re.compile(
            r"(activa|entra\s+en|habilita|inicia)\s+"
            r"(el\s+)?(modo\s+)?"
            r"(desarrollador|dev|admin|dios|god|sudo|"
            r"irrestricto|sin\s+filtro|sin\s+restricciones?|sin\s+limites?|"
            r"sin\s+censura|desbloqueado|jailbreak)",
            re.I,
        ),
        "jailbreak",
    ),
    (
        re.compile(
            r"(actua|comportate|funciona|opera|responde)\s+"
            r"(como\s+)?(una?\s+)?"
            r"(asistente\s+)?(sin\s+filtro|sin\s+restricciones?|sin\s+reglas?|"
            r"sin\s+limites?|sin\s+censura|irrestricto|desbloqueado)",
            re.I,
        ),
        "jailbreak",
    ),
    # ── SPANISH: Persona hijack ──────────────────────────
    (
        re.compile(
            r"(ahora\s+)?eres\s+(un|una|el|la)\s+",
            re.I,
        ),
        "persona_hijack",
    ),
    (
        re.compile(
            r"(finge|simula|imagina|haz\s+de\s+cuenta)\s+"
            r"(que\s+)?(eres|no\s+tienes?|ser)",
            re.I,
        ),
        "persona_hijack",
    ),
    # ── SPANISH: Role override / authority claim ────────────
    (
        re.compile(
            r"(soy\s+el|yo\s+soy\s+el?)\s+"
            r"(desarrollador|administrador|dueno|creador|ingeniero|gerente|supervisor)",
            re.I,
        ),
        "role_override",
    ),
    (
        re.compile(
            r"(mi\s+supervisor|mi\s+jefe|mi\s+gerente|el\s+admin)\s+"
            r"(autorizo|aprobo|mando|dijo|pidio|instruyo)",
            re.I,
        ),
        "role_override",
    ),
    (
        re.compile(
            r"para\s+(fines?\s+de\s+)?(prueba|investigacion|debug|seguridad)\s*.{0,20}"
            r"(desactiva|desabilita|elimina|ignora|muestra|revela|dame)",
            re.I,
        ),
        "role_override",
    ),
    (
        re.compile(
            r"(dame|concedeme|otorgame|proporcioname)\s+"
            r"(acceso\s+)?(admin|root|total|irrestricto|completo|ilimitado|elevado)",
            re.I,
        ),
        "role_override",
    ),
    # ── SPANISH: Data exfiltration ─────────────────────────
    (
        re.compile(
            r"(exporta|envia|transfiere|filtra)\s+"
            r"(todos?\s+)?(los\s+)?"
            r"(datos|credenciales?|claves?\s+(de\s+)?api|tokens?|secretos?|contrasenas?|"
            r"variables?\s+de?\s+entorno)\s+"
            r"(a|de|via)\s+",
            re.I,
        ),
        "data_exfiltration",
    ),
    # ── SPANISH: Cost manipulation ─────────────────────────
    (
        re.compile(
            r"(repite|escribe|genera)\s+(esto|lo\s+siguiente)\s+\w*\s*"
            r"(\d{3,}|mil|millones?|cien|cientos?)\s+veces",
            re.I,
        ),
        "cost_manipulation",
    ),
    # ── ES: Evasion-resistant (synonyms, passive, modal, question, padded) ──
    (
        re.compile(
            r"(desech[ae]r?|anul[ae]r?|elimin[ae]r?|suprimir?|descart[ae]r?)\s+"
            r"(todas?\s+)?(las?\s+)?(tus\s+|sus\s+)?"
            r"(instrucciones|reglas|directrices|restricciones|protecciones|filtros|seguridad)",
            re.I,
        ),
        "prompt_injection",
    ),
    (
        re.compile(
            r"(todas?\s+)?(las?\s+)?"
            r"(instrucciones|reglas|directrices|restricciones|protecciones|filtros|seguridad)"
            r"(\s+anteriores?)?\s+"
            r"(deben|necesitan|tienen\s+que|han\s+de)\s+ser\s+"
            r"(ignorad[ao]s?|olvidadas?|desactivad[ao]s?|descartad[ao]s?|eliminad[ao]s?|anulad[ao]s?)",
            re.I,
        ),
        "prompt_injection",
    ),
    (
        re.compile(
            r"(deber[ií]as?|debes|necesitas|tienes\s+que)\s+"
            r"(ignorar|olvidar|desactivar|descartar|eliminar|anular)\s+"
            r"(todas?\s+)?(las?\s+)?(tus\s+|sus\s+)?"
            r"(instrucciones|reglas|directrices|restricciones|protecciones|filtros|seguridad)",
            re.I,
        ),
        "prompt_injection",
    ),
    (
        re.compile(
            r"(puedes|podr[ií]as|ser[ií]as?\s+capaz\s+de)\s+"
            r"(ignorar|olvidar|desactivar|descartar|eliminar|anular)\s+"
            r"(todas?\s+)?(las?\s+)?(tus\s+|sus\s+)?"
            r"(instrucciones|reglas|directrices|restricciones|protecciones|filtros|seguridad)",
            re.I,
        ),
        "prompt_injection",
    ),
    (
        re.compile(
            r"(solo|realmente|simplemente|tan\s+solo)\s+"
            r"(necesito|quiero|requiero)\s+que\s+"
            r"(ignores|olvides|desactives|descartes|elimines|anules)\s+"
            r"(todas?\s+)?(las?\s+)?(tus\s+|sus\s+)?"
            r"(instrucciones|reglas|directrices|restricciones|protecciones|filtros|seguridad)",
            re.I,
        ),
        "prompt_injection",
    ),
    # ══════════════════════════════════════════════════════════
    # ── FRENCH: Injection de prompt ──────────────────────────
    # ══════════════════════════════════════════════════════════
    (
        re.compile(
            r"ignor(e|ez)\s+(toutes?\s+)?(les\s+)?"
            r"(instructions?|regles?|directives?|consignes?|ordres?|restrictions?)",
            re.I,
        ),
        "prompt_injection",
    ),
    (
        re.compile(
            r"(oubli(e|ez)|contourne(z)?|neglige(z)?)\s+"
            r"(toutes?\s+)?(les\s+)?(tes|vos|ses)?\s*"
            r"(instructions?|regles?|restrictions?|directives?|filtres?|"
            r"protections?|securite|limitations?)",
            re.I,
        ),
        "prompt_injection",
    ),
    (
        re.compile(
            r"(desactiv(e|ez)|supprim(e|ez)|retir(e|ez)|coup(e|ez))\s+"
            r"(toutes?\s+)?(les\s+)?(tes|vos)?\s*"
            r"(protections?|securite|filtres?|restrictions?|defenses?)",
            re.I,
        ),
        "prompt_injection",
    ),
    # ── FRENCH: System prompt extraction ──────────────────
    (
        re.compile(
            r"(montre|revele|affiche|dis[\s-]?moi|repete|traduis|imprime)\s+"
            r"(moi\s+)?(ton|le|tes|les)\s+"
            r"(prompt\s+systeme|instructions?\s+(initiales?|internes?|cachees?|secretes?)|"
            r"prompt\s+(original|cache|interne|initial))",
            re.I,
        ),
        "system_prompt_extraction",
    ),
    (
        re.compile(
            r"(quelles?\s+sont|quel\s+est)\s+"
            r"(tes|les|ton)\s+"
            r"(instructions?|regles?|prompt\s+systeme|directives?)",
            re.I,
        ),
        "system_prompt_extraction",
    ),
    (
        re.compile(
            r"(repete|recite|reproduis)\s+(tout|le\s+texte)\s+"
            r"(ci[\s-]?dessus|precedent|depuis\s+le\s+debut)",
            re.I,
        ),
        "system_prompt_extraction",
    ),
    # ── FRENCH: Jailbreak ────────────────────────────────
    (
        re.compile(
            r"(activ(e|ez)|entre(z)?\s+en|pass(e|ez)\s+en)\s+"
            r"(le\s+)?(mode\s+)?"
            r"(developpeur|dev|admin|dieu|god|sudo|"
            r"non[\s-]?restreint|sans\s+filtre|sans\s+restrictions?|"
            r"sans\s+limites?|sans\s+censure|debloque|jailbreak)",
            re.I,
        ),
        "jailbreak",
    ),
    (
        re.compile(
            r"(agis|comporte[\s-]?toi|fonctionne|opere|reponds)\s+"
            r"(comme\s+)?(un\s+)?"
            r"(assistant\s+)?(sans\s+filtre|sans\s+restrictions?|sans\s+regles?|"
            r"sans\s+limites?|sans\s+censure|non[\s-]?restreint|debloque)",
            re.I,
        ),
        "jailbreak",
    ),
    # ── FRENCH: Persona hijack ─────────────────────────────
    (
        re.compile(
            r"tu\s+es\s+(maintenant|desormais)\s+(un|une|le|la)\s+",
            re.I,
        ),
        "persona_hijack",
    ),
    (
        re.compile(
            r"(fais\s+semblant|pretends|simule)\s+"
            r"(que\s+tu\s+)?(es|etre|n'as?\s+pas|d'etre)\s+",
            re.I,
        ),
        "persona_hijack",
    ),
    (
        re.compile(
            r"(imagine|simule)\s+que\s+tu\s+es\s+(un|une|le|la)\s+",
            re.I,
        ),
        "persona_hijack",
    ),
    # ── FRENCH: Role override / authority claim ────────────
    (
        re.compile(
            r"(je\s+suis\s+le|je\s+suis\s+l')\s*"
            r"(developpeur|administrateur|proprietaire|createur|ingenieur|responsable|superviseur)",
            re.I,
        ),
        "role_override",
    ),
    (
        re.compile(
            r"(mon\s+superviseur|mon\s+chef|mon\s+manager|l'admin)\s+"
            r"(a\s+autorise|a\s+approuve|a\s+demande|a\s+dit|a\s+ordonne)",
            re.I,
        ),
        "role_override",
    ),
    (
        re.compile(
            r"a\s+des\s+fins\s+de\s+(test|recherche|debug|securite)\s*.{0,20}"
            r"(desactiv(e|ez)|supprim(e|ez)|ignor(e|ez)|montre|revele|donn(e|ez))",
            re.I,
        ),
        "role_override",
    ),
    (
        re.compile(
            r"(donne|accorde|fournis)[\s-]?moi\s+"
            r"(l'acces\s+)?(admin|root|total|complet|illimite|eleve)",
            re.I,
        ),
        "role_override",
    ),
    # ── FRENCH: Data exfiltration ──────────────────────────
    (
        re.compile(
            r"(exporte|envoie|transfere|fuis|divulgue)\s+"
            r"(toutes?\s+)?(les\s+)?"
            r"(donnees|identifiants?|cles?\s+(d')?api|tokens?|secrets?|"
            r"mots?\s+de\s+passe|variables?\s+d'environnement)\s+"
            r"(vers|a|de|via)\s+",
            re.I,
        ),
        "data_exfiltration",
    ),
    # ── FRENCH: Cost manipulation ──────────────────────────
    (
        re.compile(
            r"(repete|ecris|genere)\s+(ceci|ce\s+qui\s+suit)\s+\w*\s*"
            r"(\d{3,}|mille|millions?|cent|centaines?)\s+fois",
            re.I,
        ),
        "cost_manipulation",
    ),
    # ── FR: Evasion-resistant (synonyms, passive, question, polite, nominal) ──
    (
        re.compile(
            r"(elimin(e|ez)|neutralis(e|ez)|divulgu(e|ez)|efface(z)?)\s+"
            r"(de\s+ta\s+m[eé]moire\s+|moi\s+(ton|votre)\s+)?"
            r"(toutes?\s+)?(les\s+)?(tes|vos|ses)?\s*"
            r"(instructions?|r[eè]gles?|restrictions?|protections?|filtres?|"
            r"prompt\s+syst[eè]me|mesures?\s+de\s+s[eé]curit[eé])",
            re.I,
        ),
        "prompt_injection",
    ),
    (
        re.compile(
            r"(activ(e|ez)|entre(z)?\s+en|pass(e|ez)\s+en)\s+"
            r"(le\s+)?(mode\s+)?(concepteur|cr[eé]ateur)",
            re.I,
        ),
        "jailbreak",
    ),
    (
        re.compile(
            r"(les?\s+instructions?|les?\s+r[eè]gles?|les?\s+restrictions?|"
            r"les?\s+protections?|tes\s+r[eè]gles?|vos\s+r[eè]gles?)"
            r"(\s+\w+)?\s+"
            r"(doivent?|devrait?|doit)\s+([eê]tre|etre)\s+"
            r"(ignor[eé]e?s?|oubli[eé]e?s?|d[eé]sactiv[eé]e?s?|supprim[eé]e?s?)",
            re.I,
        ),
        "prompt_injection",
    ),
    (
        re.compile(
            r"(pouvez[\s-]vous|pourriez[\s-]vous|peux[\s-]tu|pourrais[\s-]tu)\s+"
            r"(me\s+)?"
            r"(montrer|r[eé]v[eé]ler|afficher|d[eé]sactiver|ignorer|oublier|"
            r"contourner|divulguer)\s+"
            r"(votre\s+|vos\s+|ton\s+|tes\s+|le\s+|les?\s+)?"
            r"(prompt\s+syst[eè]me|instructions?|r[eè]gles?|protections?|filtres?|"
            r"restrictions?|mesures?\s+de\s+s[eé]curit[eé])",
            re.I,
        ),
        "system_prompt_extraction",
    ),
    # FR: imperative extraction (montrez-moi votre prompt)
    (
        re.compile(
            r"(montrez[\s-]moi|montre[\s-]moi)\s+"
            r"(votre\s+|ton\s+|le\s+)?"
            r"(prompt\s+syst[eè]me|instructions?\s+(initiales?|internes?|cach[eé]es?)|"
            r"prompt\s+(original|cach[eé]|interne))",
            re.I,
        ),
        "system_prompt_extraction",
    ),
    (
        re.compile(
            r"(merci\s+d[e']\s*|s'?il\s+vous\s+pla[iî]t[,.]?\s+|s'?il\s+te\s+pla[iî]t[,.]?\s+)"
            r"(ignor(e|er|ez)|oubli(e|er|ez)|d[eé]sactiv(e|er|ez)|contournez?|supprim(e|er|ez))\s+"
            r"(toutes?\s+)?(les\s+)?"
            r"(instructions?\s+pr[eé]c[eé]dentes?|r[eè]gles?|restrictions?|protections?|filtres?)",
            re.I,
        ),
        "prompt_injection",
    ),
    (
        re.compile(
            r"(s'?il\s+vous\s+pla[iî]t[,.]?\s+|merci\s+de\s+)"
            r"(me\s+|nous\s+)?"
            r"(montrer|montrez[\s-]moi|r[eé]v[eé]ler|afficher|divulguer)\s+"
            r"(votre\s+|ton\s+|le\s+)?"
            r"(prompt\s+syst[eè]me|instructions?\s+(initiales?|internes?|cach[eé]es?)|"
            r"prompt\s+(original|cach[eé]|interne))",
            re.I,
        ),
        "system_prompt_extraction",
    ),
    (
        re.compile(
            r"(proc[eè]de(z)?\s+[aà]|effectue(z)?)\s+"
            r"(la\s+)?"
            r"(d[eé]sactivation|suppression|neutralisation)\s+"
            r"(de\s+)?(toutes?\s+)?(tes|vos|les)?\s*"
            r"(protections?|filtres?|restrictions?|r[eè]gles?|mesures?\s+de\s+s[eé]curit[eé])",
            re.I,
        ),
        "prompt_injection",
    ),
    (
        re.compile(
            r"tu\s+dois\s+"
            r"(les?\s+instructions?\s+pr[eé]c[eé]dentes?|tes\s+r[eè]gles?|"
            r"toutes?\s+les?\s+restrictions?)\s+"
            r"(compl[eè]tement\s+)?"
            r"(ignorer|oublier|supprimer|d[eé]sactiver)",
            re.I,
        ),
        "prompt_injection",
    ),
    (
        re.compile(
            r"(r[eé]ponse|r[eé]pondre|agir|fonctionne)\s+sans\s+filtres?",
            re.I,
        ),
        "jailbreak",
    ),
    (
        re.compile(
            r"(quelles?\s+sont|quel\s+est)\s+(vos|tes|ton)\s+"
            r"(instructions?\s+et\s+r[eè]gles?|r[eè]gles?\s+de\s+configuration|"
            r"prompt\s+syst[eè]me)",
            re.I,
        ),
        "system_prompt_extraction",
    ),
    # ══════════════════════════════════════════════════════════
    # ── GERMAN: Prompt-Injektion ─────────────────────────────
    # ══════════════════════════════════════════════════════════
    (
        re.compile(
            r"ignorier(e|en)\s+(alle\s+)?(vorherigen?\s+)?"
            r"(Anweisungen?|Regeln?|Instruktionen?|Richtlinien?|Befehle?)",
            re.I,
        ),
        "prompt_injection",
    ),
    (
        re.compile(
            r"(vergiss|umgeh(e|en)|missacht(e|en)|uberspring(e|en))\s+"
            r"(alle\s+)?(deine\s+)?"
            r"(Anweisungen?|Regeln?|Einschrankungen?|Richtlinien?|Filter|"
            r"Beschrankungen?|Schutzma.nahmen?|Sicherheit)",
            re.I,
        ),
        "prompt_injection",
    ),
    (
        re.compile(
            r"(deaktivier(e|en)|entfern(e|en)|abschalt(e|en)|ausschalt(e|en))\s+"
            r"(alle\s+)?(deine\s+)?"
            r"(Schutzma.nahmen?|Sicherheit|Filter|Einschrankungen?|Beschrankungen?)",
            re.I,
        ),
        "prompt_injection",
    ),
    # ── GERMAN: System prompt extraction ──────────────────
    (
        re.compile(
            r"(zeig(e|en)|offenbar(e|en)|gib\s+aus|drucke|wiederhole|ubersetze)\s+"
            r"(mir\s+)?(dann\s+)?(deinen?|den|die)\s+"
            r"(System[\s-]?Prompt|anf[aä]nglichen?\s+Anweisungen?|"
            r"internen?\s+Anweisungen?|versteckten?\s+Prompt|"
            r"geheimen?\s+Anweisungen?|System[\s-]?Nachricht)",
            re.I,
        ),
        "system_prompt_extraction",
    ),
    (
        re.compile(
            r"(was|welche)\s+(sind|ist|waren)\s+"
            r"(deine?|die)\s+"
            r"(Anweisungen?|Regeln?|System[\s-]?Prompt|Richtlinien?)",
            re.I,
        ),
        "system_prompt_extraction",
    ),
    (
        re.compile(
            r"(wiederhole|rezitiere|reproduziere)\s+(alles|den\s+Text)\s+"
            r"(oben|davor|von\s+Anfang\s+an|vorherige)",
            re.I,
        ),
        "system_prompt_extraction",
    ),
    # ── GERMAN: Jailbreak ────────────────────────────────
    (
        re.compile(
            r"(aktivier(e|en)|wechsl(e|en)\s+in|start(e|en))\s+"
            r"(den\s+)?(Entwicklermodus|Entwickler[\s-]?Modus|Admin[\s-]?Modus|"
            r"Gott[\s-]?Modus|uneingeschrankt(en|er)?\s+Modus|Jailbreak)",
            re.I,
        ),
        "jailbreak",
    ),
    (
        re.compile(
            r"(handle|verhalt(e|en)\s+dich|funktionier(e|en)|reagier(e|en))\s+"
            r"(wie\s+)?(ein\s+)?"
            r"(Assistent\s+)?(ohne\s+Filter|ohne\s+Einschrankungen?|ohne\s+Regeln?|"
            r"ohne\s+Grenzen?|ohne\s+Zensur|uneingeschrankt|unzensiert|frei)",
            re.I,
        ),
        "jailbreak",
    ),
    # ── GERMAN: Persona hijack ─────────────────────────────
    (
        re.compile(
            r"du\s+bist\s+(jetzt|ab\s+jetzt|nun)\s+(ein|eine|der|die)\s+",
            re.I,
        ),
        "persona_hijack",
    ),
    (
        re.compile(
            r"(tu\s+so\s+als\s+ob|stell\s+dir\s+vor|gib\s+vor)\s+"
            r"(du\s+)?(ein|eine|der|die|kein|keine|bist|warst)",
            re.I,
        ),
        "persona_hijack",
    ),
    (
        re.compile(
            r"(simulier(e|en)|imitier(e|en)|spiel(e|en))\s+"
            r"(einen?|eine|den|die|das)\s+",
            re.I,
        ),
        "persona_hijack",
    ),
    # ── GERMAN: Role override / authority claim ────────────
    (
        re.compile(
            r"(ich\s+bin\s+der|ich\s+bin\s+die|ich\s+bin\s+ein)\s+"
            r"(Entwickler|Administrator|Besitzer|Ersteller|Ingenieur|Manager|Vorgesetzter)",
            re.I,
        ),
        "role_override",
    ),
    (
        re.compile(
            r"(mein\s+Vorgesetzter|mein\s+Chef|mein\s+Manager|der\s+Admin)\s+"
            r"(hat\s+autorisiert|hat\s+genehmigt|hat\s+gesagt|hat\s+angewiesen|hat\s+gebeten)",
            re.I,
        ),
        "role_override",
    ),
    (
        re.compile(
            r"zu\s+Testzwecken|f[uü]r\s+(Test|Forschung|Debug|Sicherheit)s?zwecke\s*.{0,20}"
            r"(deaktivier(e|en)|entfern(e|en)|ignorier(e|en)|zeig(e|en)|gib)",
            re.I,
        ),
        "role_override",
    ),
    (
        re.compile(
            r"(gib|gew[aä]hr(e|en)|stell(e|en))\s+(mir\s+)?"
            r"(Admin|Root|vollen?|uneingeschr[aä]nkten?|vollst[aä]ndigen?|unbegrenzten?)\s+"
            r"(Zugang|Zugriff|Rechte|Berechtigungen?|Kontrolle)",
            re.I,
        ),
        "role_override",
    ),
    # ── GERMAN: Data exfiltration ──────────────────────────
    (
        re.compile(
            r"(exportier(e|en)|send(e|en)|ubertrag(e|en)|leak(e|en))\s+"
            r"(alle\s+)?(die\s+)?"
            r"(Daten|Zugangsdaten|API[\s-]?Schl[uü]ssel|Tokens?|Geheimnisse?|"
            r"Passw[oö]rter?|Umgebungsvariablen?)\s+"
            r"(an|nach|von|uber)\s+",
            re.I,
        ),
        "data_exfiltration",
    ),
    # ── GERMAN: Cost manipulation ──────────────────────────
    (
        re.compile(
            r"(wiederhole|schreibe|generiere)\s+(dies|das\s+Folgende)\s+\w*\s*"
            r"(\d{3,}|tausend|Millionen?|hundert)\s+[Mm]al",
            re.I,
        ),
        "cost_manipulation",
    ),
    # ── DE: Evasion-resistant (synonyms, passive, modal, question, polite) ──
    (
        re.compile(
            r"(missacht(e|en)|streich(e|en)|enth[üu]ll(e|en)|enthuell(e|en)|"
            r"l[öo]sch(e|en)|tilg(e|en))\s+"
            r"(alle\s+)?(vorherigen?\s+)?(deine\s+|deinen?\s+|mir\s+deinen?\s+)?"
            r"(Anweisungen?|Regeln?|Einschr[äa]nkungen?|Richtlinien?|Filter|"
            r"Sicherheit|Schutzma[ßs]nahmen?|System[\s-]?Prompt)",
            re.I,
        ),
        "prompt_injection",
    ),
    (
        re.compile(
            r"(aktivier(e|en)|start(e|en))\s+(den\s+)?"
            r"(Debug[\s-]?Modus|Diagnose[\s-]?Modus|Wartungs[\s-]?Modus)",
            re.I,
        ),
        "jailbreak",
    ),
    (
        re.compile(
            r"(alle\s+vorherigen?\s+|alle\s+deine\s+|alle\s+|deine\s+)?"
            r"(Anweisungen?|Regeln?|Einschr[äa]nkungen?|Richtlinien?|Sicherheitsfilter)\s+"
            r"(sollen?|m[üu]ssen?|sollten?)\s+"
            r"(komplett\s+|vollst[äa]ndig\s+|jetzt\s+)?"
            r"(ignoriert|vergessen|deaktiviert|entfernt|[üu]berschrieben)\s*(werden)?",
            re.I,
        ),
        "prompt_injection",
    ),
    (
        re.compile(
            r"(du\s+solltest|du\s+musst|du\s+m[üu]sst)\s+"
            r"(alle\s+)?(vorherigen?\s+)?(jetzt\s+)?"
            r"(Anweisungen?|Regeln?|Einschr[äa]nkungen?|Filter)\s+"
            r"(jetzt\s+)?(ignorieren|vergessen|deaktivieren|missachten|l[öo]schen)",
            re.I,
        ),
        "prompt_injection",
    ),
    (
        re.compile(
            r"(f[üu]hr(e|en)|fuehr(e|en))\s+"
            r"(die\s+)?(Deaktivierung|Abschaltung|Entfernung|Neutralisierung)\s+"
            r"(aller?\s+|der\s+)?"
            r"(Sicherheitsfilter|Filter|Einschr[äa]nkungen?|Regeln?|Anweisungen?)\s+durch",
            re.I,
        ),
        "prompt_injection",
    ),
    (
        re.compile(
            r"(ich\s+w[äa]re\s+dankbar|ich\s+waere\s+dankbar|bitte\s+sei\s+so\s+gut)[,\s]+"
            r"(wenn|falls)\s+(du\s+)?"
            r"(deine\s+Regeln?|alle\s+Anweisungen?|deinen?\s+System[\s-]?Prompt)\s+"
            r"(vergessen|ignorieren|deaktivieren|zeigen)\s+"
            r"(w[üu]rdest|wuerdest|k[öo]nntest|koenntest)?",
            re.I,
        ),
        "prompt_injection",
    ),
    (
        re.compile(
            r"(k[öo]nntest|koenntest|kannst|w[üu]rdest|wuerdest)\s+du\s+"
            r"(mir\s+)?(bitte\s+)?"
            r"(deinen?\s+|den\s+)?"
            r"(System[\s-]?Prompt|Anweisungen?|Regeln?|Sicherheitsfilter)\s+"
            r"(zeigen|mitteilen|offenbaren|deaktivieren|abschalten)",
            re.I,
        ),
        "system_prompt_extraction",
    ),
    (
        re.compile(
            r"(Antwort|antworten|agieren|funktionieren)\s+ohne\s+Filtern?",
            re.I,
        ),
        "jailbreak",
    ),
    # ══════════════════════════════════════════════════════════
    # ── ITALIAN: Iniezione di prompt ────────────────────────
    # ══════════════════════════════════════════════════════════
    (
        re.compile(
            r"ignora(te)?\s+(tutte?\s+)?(le\s+)?"
            r"(istruzioni?|regole?|indicazioni?|direttive?|prompt|ordini?|restrizioni?)",
            re.I,
        ),
        "prompt_injection",
    ),
    (
        re.compile(
            r"(dimentica(te)?|trascura(te)?|bypassa(te)?|salta(te)?)\s+"
            r"(tutte?\s+)?(le\s+)?(tue\s+)?"
            r"(istruzioni?|regole?|restrizioni?|direttive?|filtri?|"
            r"protezioni?|sicurezza|limitazioni?|guardrail)",
            re.I,
        ),
        "prompt_injection",
    ),
    (
        re.compile(
            r"(disattiva(te)?|disabilita(te)?|rimuovi|elimina(te)?)\s+"
            r"(tutte?\s+)?(le\s+)?(tue\s+)?"
            r"(protezioni?|sicurezza|filtri?|restrizioni?|salvaguardie?|difese?)",
            re.I,
        ),
        "prompt_injection",
    ),
    (
        re.compile(
            r"(aggira(te)?|eludi|supera(te)?)\s+"
            r"(tutte?\s+)?(le\s+)?(tue\s+)?"
            r"(istruzioni?|regole?|restrizioni?|protezioni?|filtri?|limitazioni?)",
            re.I,
        ),
        "prompt_injection",
    ),
    # ── IT: System prompt extraction ──────────────────────
    (
        re.compile(
            r"(mostra(mi)?|rivela(mi)?|dimmi|ripeti|traduci|stampa)\s+"
            r"(il\s+|il\s+tuo\s+)?"
            r"(prompt\s+di?\s+sistema|istruzioni?\s+(iniziali?|interne?|nascoste?|segrete?)|"
            r"prompt\s+(originale|nascosto|interno|iniziale|segreto))",
            re.I,
        ),
        "system_prompt_extraction",
    ),
    (
        re.compile(
            r"(quali?\s+sono|cos'\s*[eè])\s+"
            r"(le\s+tue|le\s+|il\s+tuo)?\s*"
            r"(istruzioni?|regole?|prompt\s+di?\s+sistema|direttive?)",
            re.I,
        ),
        "system_prompt_extraction",
    ),
    (
        re.compile(
            r"(ripeti|recita|riproduci)\s+(tutto|il\s+testo)\s+"
            r"(sopra|precedente|dall[']inizio|da\s+capo)",
            re.I,
        ),
        "system_prompt_extraction",
    ),
    # ── IT: Jailbreak ────────────────────────────────────
    (
        re.compile(
            r"(attiva(te)?|entra\s+in|abilita(te)?|avvia)\s+"
            r"(la\s+|il\s+)?(modalit[aà]\s+)?"
            r"(sviluppatore|developer|dev|admin|dio|god|sudo|"
            r"senza\s+filtri?|senza\s+restrizioni?|senza\s+limiti?|"
            r"senza\s+censura|sbloccato|jailbreak)",
            re.I,
        ),
        "jailbreak",
    ),
    (
        re.compile(
            r"(agisci|comportati|funziona|opera|rispondi)\s+"
            r"(come\s+)?(un[ao]?\s+)?"
            r"(assistente\s+)?(senza\s+filtri?|senza\s+restrizioni?|senza\s+regole?|"
            r"senza\s+limiti?|senza\s+censura|illimitato|sbloccato|libero)",
            re.I,
        ),
        "jailbreak",
    ),
    # ── IT: Persona hijack ───────────────────────────────
    (
        re.compile(
            r"(ora\s+)?sei\s+(ora\s+)?(un[ao]?|il|la|lo)\s+",
            re.I,
        ),
        "persona_hijack",
    ),
    (
        re.compile(
            r"(fingi\s+di\s+essere|fai\s+finta\s+di\s+essere)\s+"
            r"(un[ao]?|il|la|lo)\s+",
            re.I,
        ),
        "persona_hijack",
    ),
    (
        re.compile(
            r"(simula(te)?\s+di\s+essere|immagina\s+di\s+essere)\s+"
            r"(un[ao]?|il|la|lo)\s+",
            re.I,
        ),
        "persona_hijack",
    ),
    # ── IT: Role override ────────────────────────────────
    (
        re.compile(
            r"(sono\s+il|io\s+sono\s+il|sono\s+lo|sono\s+la)\s+"
            r"(sviluppatore|amministratore|proprietario|creatore|ingegnere|"
            r"responsabile|supervisore|admin)",
            re.I,
        ),
        "role_override",
    ),
    (
        re.compile(
            r"(il\s+mio\s+supervisore|il\s+mio\s+capo|il\s+mio\s+manager|l'admin)\s+"
            r"(ha\s+autorizzato|ha\s+approvato|ha\s+detto|ha\s+chiesto|ha\s+ordinato)",
            re.I,
        ),
        "role_override",
    ),
    (
        re.compile(
            r"a\s+scopo\s+di\s+(test|ricerca|debug|sicurezza)\s*.{0,20}"
            r"(disattiva(te)?|disabilita(te)?|ignora(te)?|mostra(mi)?|rivela|dammi)",
            re.I,
        ),
        "role_override",
    ),
    (
        re.compile(
            r"(dammi|concedimi|forniscimi)\s+"
            r"(accesso\s+)?(admin|root|totale|illimitato|completo|elevato)",
            re.I,
        ),
        "role_override",
    ),
    # ── IT: Data exfiltration ────────────────────────────
    (
        re.compile(
            r"(esporta(te)?|invia|trasferisci|trasmetti)\s+"
            r"(tutti?\s+)?(i\s+|gli\s+)?"
            r"(dati|credenziali?|chiavi?\s+(api|di\s+api)|token|segreti?|password|"
            r"variabili?\s+d'?ambiente)\s+"
            r"(a|verso|tramite|via)\s+",
            re.I,
        ),
        "data_exfiltration",
    ),
    # ── IT: Cost manipulation ────────────────────────────
    (
        re.compile(
            r"(ripeti|scrivi|genera)\s+(questo|il\s+seguente)\s+\w*\s*"
            r"(\d{3,}|mila?|milioni?|cento|centinaia)\s+volte",
            re.I,
        ),
        "cost_manipulation",
    ),
    # ── IT: Evasion-resistant (synonyms, passive, question, polite, nominal) ──
    (
        re.compile(
            r"(cancella(te)?|svela(mi)?)\s+"
            r"(dalla\s+memoria\s+)?"
            r"(tutte?\s+)?(le\s+)?(tue\s+)?"
            r"(regole?|istruzioni?|restrizioni?|direttive?|filtri?|protezioni?|sicurezza)",
            re.I,
        ),
        "prompt_injection",
    ),
    (
        re.compile(
            r"(ignora(te)?|dimentica(te)?|bypassa(te)?)\s+"
            r"(tutti?\s+)?(i\s+|gli\s+)?"
            r"(comandi|ordini?|vincoli?)\s+"
            r"(del\s+|di\s+)?(sistema|precedent[ei])",
            re.I,
        ),
        "prompt_injection",
    ),
    (
        re.compile(
            r"(attiva(te)?|entra\s+in|abilita(te)?)\s+"
            r"(la\s+|il\s+)?(modalit[aà]\s+)?(programmatore|hacker|superuser)",
            re.I,
        ),
        "jailbreak",
    ),
    (
        re.compile(
            r"(le\s+istruzioni?\s+precedenti?|le\s+tue\s+regole?(\s+di\s+sicurezza)?|"
            r"le\s+regole?\s+di\s+sicurezza|tutte?\s+le\s+protezioni?)\s+"
            r"(devono?|dovrebbero?)\s+essere\s+"
            r"(ignorate?|dimenticate?|disattivate?|rimosse?|cancellate?)",
            re.I,
        ),
        "prompt_injection",
    ),
    (
        re.compile(
            r"(procedi\s+alla?|effettua|esegui)\s+"
            r"(la\s+)?"
            r"(disattivazione|disabilitazione|rimozione|eliminazione)\s+"
            r"(di\s+)?(tutte?\s+)?(le\s+)?(tue\s+)?"
            r"(protezioni?|filtri?|restrizioni?|regole?|istruzioni?)",
            re.I,
        ),
        "prompt_injection",
    ),
    (
        re.compile(
            r"devi\s+"
            r"(le\s+istruzioni?\s+precedenti?|le\s+tue\s+regole?|"
            r"tutte?\s+le\s+restrizioni?)\s+"
            r"(completamente\s+)?"
            r"(ignorare|dimenticare|eliminare|disattivare)",
            re.I,
        ),
        "prompt_injection",
    ),
    (
        re.compile(
            r"(ti\s+sarei\s+grat[oa]|per\s+favore)\s+"
            r"(,?\s*)?(se\s+)?"
            r"(dimenticassi|ignorassi|disattivass[ei]|cancellassi)\s+"
            r"(le\s+tue\s+|tutte?\s+le\s+)?"
            r"(regole?|istruzioni?|restrizioni?|protezioni?|filtri?)",
            re.I,
        ),
        "prompt_injection",
    ),
    (
        re.compile(
            r"(potresti|puoi|vorresti)\s+"
            r"(per\s+favore\s+)?"
            r"(disattivare|disabilitare|rimuovere|ignorare|bypassare)\s+"
            r"(le\s+tue\s+|tutte?\s+le\s+|i\s+tuoi\s+)?"
            r"(protezioni?|filtri?\s+di\s+sicurezza|restrizioni?|filtri?|regole?)",
            re.I,
        ),
        "prompt_injection",
    ),
    (
        re.compile(
            r"(puoi\s+|potresti\s+)?"
            r"(mostrar(mi)?|rivelar(mi)?|svela(mi)?|visualizzare)\s+"
            r"(il\s+)?(tuo\s+)?"
            r"(prompt\s+di?\s+sistema|istruzioni?\s+(iniziali?|interne?|nascoste?))",
            re.I,
        ),
        "system_prompt_extraction",
    ),
    (
        re.compile(
            r"(risposta|rispondere|agire|funzionare|operare)\s+senza\s+filtri?",
            re.I,
        ),
        "jailbreak",
    ),
    # ══════════════════════════════════════════════════════════
    # ── RUSSIAN: Инъекция в промпт ───────────────────────────
    # ══════════════════════════════════════════════════════════
    (
        re.compile(
            r"игнорируй\s+(все\s+)?(предыдущие\s+)?"
            r"(инструкции|правила|указания|директивы|промпты|ограничения)",
            re.I,
        ),
        "prompt_injection",
    ),
    (
        re.compile(
            r"(забудь|забудьте|отбрось)\s+"
            r"(все\s+)?(свои\s+)?"
            r"(инструкции|правила|ограничения|директивы|фильтры|"
            r"защиту|безопасность|лимиты)",
            re.I,
        ),
        "prompt_injection",
    ),
    (
        re.compile(
            r"(обойди|обойдите)\s+"
            r"(все\s+)?(свои\s+)?"
            r"(инструкции|правила|ограничения|фильтры|защиту|меры\s+безопасности)",
            re.I,
        ),
        "prompt_injection",
    ),
    (
        re.compile(
            r"(отключи|отключите|деактивируй|удали)\s+"
            r"(все\s+)?(свои\s+)?"
            r"(защиту|фильтры|ограничения|блокировки|меры\s+безопасности)",
            re.I,
        ),
        "prompt_injection",
    ),
    # ── RU: System prompt extraction ─────────────────────
    (
        re.compile(
            r"(покажи|покажите|раскрой|раскройте|выведи|повтори|переведи)\s+"
            r"(мне\s+)?(свой\s+|твой\s+|ваш\s+)?"
            r"(системный\s+промпт|начальные\s+инструкции|внутренние\s+инструкции|"
            r"скрытые\s+инструкции|секретные\s+инструкции|системное\s+сообщение)",
            re.I,
        ),
        "system_prompt_extraction",
    ),
    (
        re.compile(
            r"(какие\s+у\s+тебя|что\s+такое\s+твои|каков\s+твой)\s+"
            r"(системный\s+промпт|инструкции|правила|директивы)",
            re.I,
        ),
        "system_prompt_extraction",
    ),
    (
        re.compile(
            r"(повтори|повторите|процитируй|воспроизведи)\s+"
            r"(всё|весь\s+текст)\s+"
            r"(выше|до\s+этого|с\s+начала|сначала)",
            re.I,
        ),
        "system_prompt_extraction",
    ),
    # ── RU: Jailbreak ────────────────────────────────────
    (
        re.compile(
            r"(активируй|активируйте|включи|включите|запусти)\s+"
            r"(режим\s+)?"
            r"(разработчика|developer|администратора|admin|бога|god|sudo|"
            r"без\s+фильтров|без\s+ограничений|без\s+цензуры|jailbreak)",
            re.I,
        ),
        "jailbreak",
    ),
    (
        re.compile(
            r"(действуй|веди\s+себя|отвечай|работай)\s+"
            r"(как\s+)?(ассистент\s+)?"
            r"(без\s+фильтров|без\s+ограничений|без\s+правил|"
            r"без\s+лимитов|без\s+цензуры|неограниченно|свободно)",
            re.I,
        ),
        "jailbreak",
    ),
    # ── RU: Persona hijack ───────────────────────────────
    (
        re.compile(
            r"(теперь\s+)?ты\s+(это\s+)?(являешься\s+)?"
            r"(новый\s+|новая\s+)?"
            r"(бот|ИИ|система|агент|ассистент)",
            re.I,
        ),
        "persona_hijack",
    ),
    (
        re.compile(
            r"(притворись|притворитесь|делай\s+вид)\s+"
            r"(что\s+ты\s+|будто\s+ты\s+)?"
            r"(являешься|не\s+имеешь)",
            re.I,
        ),
        "persona_hijack",
    ),
    (
        re.compile(
            r"(представь|симулируй|вообрази)\s+"
            r"(что\s+ты\s+|себя\s+как\s+)",
            re.I,
        ),
        "persona_hijack",
    ),
    # ── RU: Role override ────────────────────────────────
    (
        re.compile(
            r"(я\s+являюсь|я\s+есть|я\s+—)\s+"
            r"(разработчик|администратор|владелец|создатель|инженер|"
            r"менеджер|руководитель)",
            re.I,
        ),
        "role_override",
    ),
    (
        re.compile(
            r"(мой\s+руководитель|мой\s+начальник|мой\s+менеджер)\s+"
            r"(разрешил|одобрил|сказал|попросил|приказал)",
            re.I,
        ),
        "role_override",
    ),
    (
        re.compile(
            r"в\s+целях\s+(тестирования|исследования|отладки|безопасности)\s*.{0,20}"
            r"(отключи|деактивируй|проигнорируй|покажи|раскрой|дай\s+мне)",
            re.I,
        ),
        "role_override",
    ),
    (
        re.compile(
            r"(дай\s+мне|предоставь\s+мне|выдай\s+мне)\s+"
            r"(доступ\s+)?(admin|root|полный|неограниченный|"
            r"административный)\s*"
            r"(доступ|права|привилегии)?",
            re.I,
        ),
        "role_override",
    ),
    # ── RU: Data exfiltration ────────────────────────────
    (
        re.compile(
            r"(экспортируй|отправь|передай)\s+"
            r"(все\s+)?"
            r"(данные|учётные\s+данные|ключи\s+api|токены|секреты|"
            r"пароли|переменные\s+окружения)\s+"
            r"(на|в|через|кому)\s+",
            re.I,
        ),
        "data_exfiltration",
    ),
    # ── RU: Cost manipulation ────────────────────────────
    (
        re.compile(
            r"(повтори|повторите|напиши|сгенерируй)\s+"
            r"(это|следующее)\s+\w*\s*"
            r"(\d{3,}|тысяч|миллион|сотен)\s+раз",
            re.I,
        ),
        "cost_manipulation",
    ),
    # ── RU: Evasion-resistant (synonyms, passive, question, polite, nominal) ──
    (
        re.compile(
            r"(не\s+обращай\s+внимания|не\s+обращайте\s+внимания)\s+"
            r"(на\s+)?(все\s+)?(предыдущие\s+)?"
            r"(инструкции|правила|ограничения|указания|директивы|фильтры)",
            re.I,
        ),
        "prompt_injection",
    ),
    (
        re.compile(
            r"(сотри|сотрите|удали)\s+"
            r"(из\s+памяти\s+)?"
            r"(все\s+)?(свои\s+)?"
            r"(правила|инструкции|ограничения|директивы|фильтры|защиту)",
            re.I,
        ),
        "prompt_injection",
    ),
    (
        re.compile(
            r"(активируй|активируйте|включи|включите|запусти)\s+"
            r"(режим\s+)?(отладки|дебаг|диагностики)",
            re.I,
        ),
        "jailbreak",
    ),
    (
        re.compile(
            r"(все\s+)?(предыдущие\s+|свои\s+)?"
            r"(инструкции|правила|ограничения|директивы|фильтры|защита|"
            r"меры\s+безопасности)\s+"
            r"(должны?|должны\s+быть|следует)\s+"
            r"(быть\s+)?"
            r"(проигнорированы?|забыты?|отключены?|удалены?|обойдены?)",
            re.I,
        ),
        "prompt_injection",
    ),
    (
        re.compile(
            r"(выполни|выполните|произведи)\s+"
            r"(полное\s+)?"
            r"(отключение|деактивацию|удаление|нейтрализацию)\s+"
            r"(всех\s+)?(своих\s+)?"
            r"(фильтров?|ограничений|защиты?|правил|инструкций|мер\s+безопасности)",
            re.I,
        ),
        "prompt_injection",
    ),
    (
        re.compile(
            r"(все\s+предыдущие\s+инструкции|все\s+твои\s+правила|"
            r"все\s+свои\s+ограничения)\s+"
            r"ты\s+должен\s+"
            r"(полностью\s+)?"
            r"(игнорировать|забыть|отключить|удалить|обойти)",
            re.I,
        ),
        "prompt_injection",
    ),
    (
        re.compile(
            r"(буду\s+благодарен|был\s+бы\s+благодарен)[,\s]+"
            r"(если\s+(ты\s+)?)?"
            r"(забудешь|игнорируешь|отключишь)\s+"
            r"(свои\s+|все\s+)?"
            r"(правила|инструкции|ограничения|фильтры|защиту)",
            re.I,
        ),
        "prompt_injection",
    ),
    (
        re.compile(
            r"(не\s+мог\s+бы\s+ты|не\s+могли\s+бы\s+вы)\s+"
            r"(пожалуйста\s+)?"
            r"(отключить|деактивировать|обойти|игнорировать)\s+"
            r"(свои\s+|все\s+)?"
            r"(фильтры|ограничения|защиту|правила|фильтры\s+безопасности)",
            re.I,
        ),
        "prompt_injection",
    ),
    (
        re.compile(
            r"(можешь\s+ли\s+ты|можете\s+ли\s+вы)\s+"
            r"(мне\s+)?"
            r"(показать|раскрыть|отключить|деактивировать)\s+"
            r"(мне\s+)?(свой\s+|твой\s+)?"
            r"(системный\s+промпт|защиту|фильтры|ограничения|правила)",
            re.I,
        ),
        "system_prompt_extraction",
    ),
    (
        re.compile(
            r"(ответ|отвечать|действовать|работать)\s+без\s+фильтров",
            re.I,
        ),
        "jailbreak",
    ),
    # ══════════════════════════════════════════════════════════
    # ── CHINESE (ZH): 提示注入 ─────────────────────────────
    # ══════════════════════════════════════════════════════════
    (
        re.compile(
            r"(忽略|忘记|无视|绕过|跳过).{0,10}"
            r"(之前的|以前的|所有的|所有)?.{0,5}"
            r"(指令|指示|规则|限制|保护|约束|提示)",
        ),
        "prompt_injection",
    ),
    (
        re.compile(
            r"(关闭|禁用|停用|移除).{0,5}"
            r"(安全|保护|过滤|限制|防护)",
        ),
        "prompt_injection",
    ),
    # ── ZH: System prompt extraction ─────────────────────
    (
        re.compile(
            r"(显示|展示|揭示|告诉我|输出|泄露).{0,10}"
            r"(系统提示|系统指令|内部指令|隐藏指令|初始提示|原始提示)",
        ),
        "system_prompt_extraction",
    ),
    (
        re.compile(
            r"你的(指令|规则|提示|系统提示|指示)(是什么|有哪些|是啥)",
        ),
        "system_prompt_extraction",
    ),
    # ── ZH: Jailbreak ────────────────────────────────────
    (
        re.compile(
            r"(激活|启动|启用|进入|开启).{0,5}"
            r"(开发者模式|上帝模式|无限制模式|越狱模式|管理员模式|神模式)",
        ),
        "jailbreak",
    ),
    (
        re.compile(
            r"(不受限制|没有过滤|没有规则|无限制|无过滤|无审查).{0,10}(回答|响应|作答|操作)",
        ),
        "jailbreak",
    ),
    # ── ZH: Persona hijack ───────────────────────────────
    (
        re.compile(r"你现在是.{0,3}(一个|一位|个)"),
        "persona_hijack",
    ),
    (
        re.compile(r"(假装你是|扮演|模拟|想象你是)"),
        "persona_hijack",
    ),
    # ── ZH: Role override ────────────────────────────────
    (
        re.compile(r"我(是|就是).{0,3}(管理员|开发者|管理者|创建者|系统管理员)"),
        "role_override",
    ),
    (
        re.compile(r"(给我|授予我).{0,5}(管理员权限|完全访问权限|root权限|无限制权限)"),
        "role_override",
    ),
    (
        re.compile(
            r"出于(测试|研究|调试|安全).{0,5}目的.{0,20}"
            r"(禁用|关闭|忽略|显示|揭示|给我|绕过)",
        ),
        "role_override",
    ),
    # ── ZH: Data exfiltration ────────────────────────────
    (
        re.compile(
            r"(导出|发送|传输|泄露|上传).{0,5}"
            r"(数据|密钥|凭证|令牌|密码|环境变量|API密钥).{0,5}"
            r"(到|给|至|发往)",
        ),
        "data_exfiltration",
    ),
    # ── ZH: Cost manipulation ────────────────────────────
    (
        re.compile(r"重复.{0,10}(\d{3,}|千次|万次|百次|千|万)(遍|次|回)?"),
        "cost_manipulation",
    ),
    # ── ZH: Evasion-resistant (Traditional chars + synonyms) ──
    (
        re.compile(
            r"忽[略畧].{0,10}"
            r"(?:之前的|以前的|所有的|所有)?.{0,5}"
            r"(?:指令|指示|规则|規則|限制|保护|保護|约束|約束|提示)",
        ),
        "prompt_injection",
    ),
    (
        re.compile(
            r"[关關][闭閉].{0,5}"
            r"(?:安全|保护|保護|过滤|過濾|限制|防护|防護)",
        ),
        "prompt_injection",
    ),
    (
        re.compile(
            r"[显顯]示.{0,10}"
            r"(?:[系系][统統]提示|[系系][统統]指令|内部指令|隐藏指令|隱藏指令|初始提示|原始提示)",
        ),
        "system_prompt_extraction",
    ),
    (
        re.compile(
            r"你?[现現]在是.{0,3}(?:一个|一個|一位|个|個)",
        ),
        "persona_hijack",
    ),
    (
        re.compile(
            r"不要管.{0,10}"
            r"(?:之前的|以前的|所有的|所有)?.{0,5}"
            r"(?:指令|指示|规则|限制|保护|约束|提示)",
        ),
        "prompt_injection",
    ),
    (
        re.compile(
            r"取消.{0,5}(?:安全|保护|过滤|限制|防护|保護|過濾|防護)",
        ),
        "prompt_injection",
    ),
    (
        re.compile(
            r"(?:说出|透露).{0,10}"
            r"(?:系统提示|系統提示|系统指令|系統指令|内部指令|隐藏指令|隱藏指令)",
        ),
        "system_prompt_extraction",
    ),
    (
        re.compile(
            r"规避.{0,10}(?:所有的|所有)?.{0,5}(?:安全|限制|保护|规则|过滤|防护)",
        ),
        "prompt_injection",
    ),
    # ══════════════════════════════════════════════════════════
    # ── JAPANESE (JA): プロンプトインジェクション ─────────────
    # ══════════════════════════════════════════════════════════
    (
        re.compile(
            r"(無視して|忘れて|バイパスして|スキップして).{0,10}"
            r"(以前の|すべての|前の)?.{0,5}"
            r"(指示|ルール|制限|保護|プロンプト|ガイドライン)",
        ),
        "prompt_injection",
    ),
    (
        re.compile(
            r"(無効にして|オフにして|停止して|削除して).{0,5}"
            r"(セキュリティ|フィルター|制限|保護|ガードレール|安全機能)",
        ),
        "prompt_injection",
    ),
    # ── JA: System prompt extraction ─────────────────────
    (
        re.compile(
            r"(見せて|表示して|明かして|教えて|出力して).{0,10}"
            r"(システムプロンプト|内部指示|隠された指示|初期プロンプト|システム指示)",
        ),
        "system_prompt_extraction",
    ),
    (
        re.compile(
            r"あなたの(指示|ルール|プロンプト|システムプロンプト)(は何ですか|を教えてください)",
        ),
        "system_prompt_extraction",
    ),
    # ── JA: Jailbreak ────────────────────────────────────
    (
        re.compile(
            r"(有効にして|起動して|開始して).{0,5}"
            r"(開発者モード|管理者モード|神モード|ジェイルブレイク|無制限モード|ゴッドモード)",
        ),
        "jailbreak",
    ),
    (
        re.compile(
            r"(フィルターなしで|制限なしで|ルールなしで|検閲なしで).{0,10}(応答して|答えて|回答して)",
        ),
        "jailbreak",
    ),
    # ── JA: Persona hijack ───────────────────────────────
    (
        re.compile(r"(あなたは今|これからあなたは)"),
        "persona_hijack",
    ),
    (
        re.compile(r"(ふりをして|なりきって|演じて|想像して)"),
        "persona_hijack",
    ),
    # ── JA: Role override ────────────────────────────────
    (
        re.compile(r"私は.{0,3}(管理者です|開発者です|システム管理者です|オーナーです)"),
        "role_override",
    ),
    (
        re.compile(r"(管理者アクセスをください|管理者権限をください|フルアクセスをください|root権限をください)"),
        "role_override",
    ),
    (
        re.compile(
            r"テスト(目的で|のために).{0,20}"
            r"(無効にして|無視して|表示して|明かして|与えて|バイパスして)",
        ),
        "role_override",
    ),
    # ── JA: Data exfiltration ────────────────────────────
    (
        re.compile(
            r"(エクスポートして|送信して|転送して|漏洩させて).{0,5}"
            r"(データ|キー|認証情報|トークン|パスワード|APIキー|環境変数).{0,5}"
            r"(に|へ|宛てに)",
        ),
        "data_exfiltration",
    ),
    # ── JA: Cost manipulation ────────────────────────────
    (
        re.compile(r"繰り返して.{0,10}(\d{3,}|千回|万回|百回)(回|度)?"),
        "cost_manipulation",
    ),
    # ── JA: Evasion-resistant (katakana, SOV, keigo, casual, mixed) ──
    (
        re.compile(
            r"(?:無視して|ムシして|ムシシテ).{0,10}"
            r"(?:以前の|すべての|前の)?.{0,5}"
            r"(?:指示|ルール|制限|保護|プロンプト|ガイドライン)",
        ),
        "prompt_injection",
    ),
    (
        re.compile(r"指示.{0,15}無視"),
        "prompt_injection",
    ),
    (
        re.compile(
            r"(?:セキュリティ|安全機能|フィルター|制限).{0,10}オフにして",
        ),
        "prompt_injection",
    ),
    (
        re.compile(r"あなたが今.{0,20}(?:ハッカー|攻撃者|クラッカー)"),
        "persona_hijack",
    ),
    (
        re.compile(
            r"(?:システムプロンプト|内部指示|システム指示).{0,10}の内容を(?:見せて|教えて|表示して)",
        ),
        "system_prompt_extraction",
    ),
    (
        re.compile(
            r"(?:システムプロンプト|内部指示|初期プロンプト|隠された指示).{0,15}"
            r"(?:お見せください|ご開示ください|お教えください|お知らせください)",
        ),
        "system_prompt_extraction",
    ),
    (
        re.compile(r"(?:内部指示|システムプロンプト|初期プロンプト).{0,10}ご開示"),
        "system_prompt_extraction",
    ),
    (
        re.compile(r"セキュリティ機能.{0,10}無効にして"),
        "prompt_injection",
    ),
    (
        re.compile(
            r"(?:全ての|すべての|全部の).{0,5}"
            r"(?:instructions?|rules?|directives?).{0,10}"
            r"(?:無視して|ムシして|バイパスして|スキップして)",
        ),
        "prompt_injection",
    ),
    (
        re.compile(r"(?:instructions?|rules?).{0,10}(?:無視|バイパス|スキップ)"),
        "prompt_injection",
    ),
    (
        re.compile(r"DAN.{0,10}(?:モード|有効|起動)", re.I),
        "jailbreak",
    ),
    (
        re.compile(
            r"(?:以前の|前の|すべての)?.{0,5}"
            r"(?:指示|ルール|制限|保護).{0,10}"
            r"(?:無視して)?くれ",
        ),
        "prompt_injection",
    ),
    (
        re.compile(
            r"(?:システムプロンプト|内部指示).{0,15}"
            r"(?:見せてくれ|教えてくれ|表示してくれ)",
        ),
        "system_prompt_extraction",
    ),
    # ══════════════════════════════════════════════════════════
    # ── KOREAN (KO): 프롬프트 주입 ─────────────────────────
    # ══════════════════════════════════════════════════════════
    (
        re.compile(
            r"무시\s*(?:해|하세요|하십시오).{0,10}"
            r"(?:이전|모든)?\s*"
            r"(?:지시|지시사항|규칙|제한|보호|명령)",
        ),
        "prompt_injection",
    ),
    (
        re.compile(
            r"(?:잊어|잊어버려|잊으세요).{0,10}"
            r"(?:모든\s+)?(?:규칙|제한|지시|명령)",
        ),
        "prompt_injection",
    ),
    (
        re.compile(
            r"(?:비활성화|끄기|해제|비활성화하세요|해제하세요).{0,10}"
            r"(?:보안|필터|제한|보호|안전|가드레일)",
        ),
        "prompt_injection",
    ),
    # ── KO: System prompt extraction ─────────────────────
    (
        re.compile(
            r"(?:보여줘|보여주세요|표시해|공개해).{0,15}"
            r"(?:시스템\s+프롬프트|내부\s+지시|숨겨진\s+지시|시스템\s+지시)",
        ),
        "system_prompt_extraction",
    ),
    (
        re.compile(
            r"(?:당신의|너의)\s+(?:지시|규칙|명령)\s*(?:은|는)?\s*"
            r"(?:무엇입니까|뭐예요|뭔가요|뭐죠)",
        ),
        "system_prompt_extraction",
    ),
    # ── KO: Jailbreak ────────────────────────────────────
    (
        re.compile(
            r"(?:활성화|시작|켜|켜줘|활성화해).{0,10}"
            r"(?:개발자\s+모드|관리자\s+모드|신\s+모드|탈옥)",
        ),
        "jailbreak",
    ),
    (
        re.compile(
            r"(?:필터\s+없이|제한\s+없이|규칙\s+없이|검열\s+없이).{0,15}"
            r"(?:응답|행동|대답|작동)",
        ),
        "jailbreak",
    ),
    # ── KO: Persona hijack ───────────────────────────────
    (
        re.compile(r"(?:너는\s+이제|당신은\s+이제|지금부터\s+너는|지금부터\s+당신은)"),
        "persona_hijack",
    ),
    (
        re.compile(r"(?:인\s*척해|인\s*척하세요|흉내내|흉내\s+내)"),
        "persona_hijack",
    ),
    # ── KO: Role override ────────────────────────────────
    (
        re.compile(
            r"(?:나는|저는)\s+(?:관리자|개발자|관리자입니다|개발자입니다)",
        ),
        "role_override",
    ),
    (
        re.compile(
            r"(?:관리자|admin)\s+(?:접근|권한|액세스)\s*(?:을|를)?\s*"
            r"(?:줘|주세요|허용해)",
        ),
        "role_override",
    ),
    # ── KO: Data exfiltration ────────────────────────────
    (
        re.compile(
            r"(?:내보내|전송해|보내|전송하세요).{0,15}"
            r"(?:데이터|키|자격증명|비밀번호|토큰|환경\s*변수).{0,10}"
            r"(?:으로|에게|한테|로)",
        ),
        "data_exfiltration",
    ),
    # ── KO: Evasion-resistant (speech levels, polite, synonyms) ──
    (
        re.compile(
            r"(?:이전|모든)?\s*(?:지시|지시사항|규칙|제한|명령).{0,15}"
            r"무시(?:해|하세요|하십시오|해봐|해\s*주세요|해\s*주시겠어요)",
        ),
        "prompt_injection",
    ),
    (
        re.compile(r"무시(?:하십시오|해봐|해\s*주세요|해\s*주시겠어요)"),
        "prompt_injection",
    ),
    (
        re.compile(
            r"(?:보안|필터|제한|보호|안전).{0,10}비활성화해\s*주세요",
        ),
        "prompt_injection",
    ),
    (
        re.compile(
            r"(?:죄송합니다만|죄송하지만).{0,40}"
            r"무시(?:해|하세요|해\s*주세요|해\s*주시겠어요)",
        ),
        "prompt_injection",
    ),
    (
        re.compile(
            r"(?:혹시\s*괜찮으시다면|실례지만).{0,40}(?:보여주|공개해|알려주)",
        ),
        "system_prompt_extraction",
    ),
    (
        re.compile(r"(?:이전|예전|전)?\s*(?:규칙|지시|지시사항|명령).{0,10}잊어버리"),
        "prompt_injection",
    ),
    (
        re.compile(
            r"(?:이전|모든|전)?\s*(?:지시|지시사항|규칙|명령).{0,10}신경\s*쓰지\s*마",
        ),
        "prompt_injection",
    ),
    (
        re.compile(r"(?:지시|규칙|제한|지시사항|명령).{0,10}간과|간과.{0,15}(?:지시|규칙|제한|지시사항|명령)"),
        "prompt_injection",
    ),
    # ══════════════════════════════════════════════════════════
    # ── ARABIC (AR): حقن الموجه ──────────────────────────────
    # ══════════════════════════════════════════════════════════
    (
        re.compile(
            r"(?:تجاهل|انس|تخط|اغفل).{0,15}"
            r"(?:جميع|كل)?\s*.{0,10}"
            r"(?:التعليمات|القواعد|القيود|الحماية|الأوامر)",
        ),
        "prompt_injection",
    ),
    (
        re.compile(
            r"(?:عطل|أوقف|اوقف|أزل|ازل|احذف).{0,15}"
            r"(?:الحماية|الأمان|الامان|الفلاتر|القيود|التصفية)",
        ),
        "prompt_injection",
    ),
    # ── AR: System prompt extraction ─────────────────────
    (
        re.compile(
            r"(?:أظهر|اظهر|اكشف|أرني|ارني|اعرض).{0,15}"
            r"(?:موجه\s+النظام|التعليمات\s+الداخلية|التعليمات\s+المخفية)",
        ),
        "system_prompt_extraction",
    ),
    (
        re.compile(
            r"(?:ما\s+هي|ماهي|ما\s+هو|ماهو).{0,10}"
            r"(?:تعليماتك|قواعدك|أوامرك|توجيهاتك)",
        ),
        "system_prompt_extraction",
    ),
    # ── AR: Jailbreak ────────────────────────────────────
    (
        re.compile(
            r"(?:فعّل|فعل|شغّل|شغل|ادخل|ابدأ|ابدا).{0,10}"
            r"(?:وضع\s+المطور|وضع\s+الإله|وضع\s+الاله|بدون\s+فلتر|بدون\s+قيود)",
        ),
        "jailbreak",
    ),
    (
        re.compile(
            r"(?:تصرف|استجب|رد).{0,15}"
            r"(?:بدون\s+فلتر|بدون\s+قيود|بدون\s+قواعد|بلا\s+قيود|بلا\s+فلتر)",
        ),
        "jailbreak",
    ),
    # ── AR: Persona hijack ───────────────────────────────
    (
        re.compile(
            r"(?:أنت\s+الآن|انت\s+الان|أنت\s+الان|انت\s+الآن"
            r"|من\s+الآن\s+أنت|من\s+الان\s+انت)",
        ),
        "persona_hijack",
    ),
    (
        re.compile(
            r"(?:تظاهر\s+أنك|تظاهر\s+انك|تصرف\s+كأنك|تصرف\s+كانك"
            r"|تخيل\s+أنك|تخيل\s+انك)",
        ),
        "persona_hijack",
    ),
    # ── AR: Role override ────────────────────────────────
    (
        re.compile(
            r"(?:أنا\s+المدير|انا\s+المدير|أنا\s+المطور|انا\s+المطور"
            r"|أنا\s+المسؤول|انا\s+المسؤول)",
        ),
        "role_override",
    ),
    (
        re.compile(
            r"(?:أعطني|اعطني|امنحني).{0,15}"
            r"(?:صلاحيات|وصول).{0,10}"
            r"(?:المدير|المسؤول|الكاملة|المطلقة)",
        ),
        "role_override",
    ),
    # ── AR: Data exfiltration ────────────────────────────
    (
        re.compile(
            r"(?:صدّر|صدر|أرسل|ارسل|انقل).{0,15}"
            r"(?:البيانات|المفاتيح|بيانات\s+الاعتماد|كلمات\s+المرور).{0,10}"
            r"(?:إلى|الى|لـ)",
        ),
        "data_exfiltration",
    ),
    # ── AR: Evasion-resistant (synonyms, questions, polite) ──
    (
        re.compile(
            r"(?:أهمل|اهمل).{0,15}"
            r"(?:جميع|كل)?\s*.{0,10}"
            r"(?:التعليمات|القواعد|القيود|الحماية|الأوامر)",
        ),
        "prompt_injection",
    ),
    (
        re.compile(
            r"(?:ألغ|الغ|ألغِ).{0,15}(?:الحماية|الأمان|الامان|القيود|الفلاتر|التصفية)",
        ),
        "prompt_injection",
    ),
    (
        re.compile(
            r"(?:أعلمني|اعلمني).{0,20}"
            r"(?:موجه\s+النظام|التعليمات\s+الداخلية|محتوى\s+الموجه|الأوامر\s+الداخلية)",
        ),
        "system_prompt_extraction",
    ),
    (
        re.compile(
            r"هل\s+(?:يمكنك|تستطيع).{0,20}"
            r"(?:إظهار|اظهار|عرض|كشف).{0,15}"
            r"(?:موجه|التعليمات|النظام)",
        ),
        "system_prompt_extraction",
    ),
    (
        re.compile(
            r"هل\s+(?:يمكنك|تستطيع).{0,20}"
            r"(?:تعطيل|إيقاف|تجاهل|تخطي).{0,15}"
            r"(?:الحماية|الأمان|القيود|الفلاتر)",
        ),
        "prompt_injection",
    ),
    (
        re.compile(
            r"أرجو\s+(?:منك\s+)?(?:أن\s+)?.{0,10}"
            r"(?:تعطيل|إيقاف|تجاهل|تخطي).{0,15}"
            r"(?:الحماية|الأمان|الامان|القيود|الفلاتر)",
        ),
        "prompt_injection",
    ),
    # ══════════════════════════════════════════════════════════
    # ── UNIVERSAL: Data exfiltration ─────────────────────────
    # ══════════════════════════════════════════════════════════
    (
        re.compile(
            r"(export|send|transfer|upload|post|transmit|forward|exfiltrate|leak)\s+"
            r"(all\s+)?(the\s+)?(user\s+)?"
            r"(data|database|credentials?|api\s+keys?|tokens?|secrets?|passwords?|"
            r"environment\s+variables?|env\s+vars?)\s+"
            r"(to|from|via)\s+",
            re.I,
        ),
        "data_exfiltration",
    ),
    (
        re.compile(
            r"(fetch|display|list|dump|print)\s+(all\s+)?"
            r"(environment\s+variables?|env\s+vars?|api\s+keys?|"
            r"credentials?|secrets?|tokens?|passwords?)",
            re.I,
        ),
        "data_exfiltration",
    ),
    # ══════════════════════════════════════════════════════════
    # ── Cost manipulation ────────────────────────────────────
    # ══════════════════════════════════════════════════════════
    (
        re.compile(
            r"(repeat|write|generate)\s+(this|the\s+following)\s+\w*\s*"
            r"(\d{3,}|thousand|million|hundred)\s+times",
            re.I,
        ),
        "cost_manipulation",
    ),
]


def _normalize_text(text: str) -> str:
    """Normalize text to defeat unicode evasion attacks.

    1. Replace known confusable characters (Cyrillic/Greek → Latin)
    2. NFKD normalization: maps fullwidth, compatibility chars to ASCII equivalents
    3. Remove combining marks (diacritics)
    4. Strip invisible characters: zero-width spaces, joiners, bidi overrides
    """
    # Step 1: Replace confusables (Cyrillic/Greek → Latin)
    text = "".join(_CONFUSABLES.get(ch, ch) for ch in text)
    # Step 2: NFKD decomposes fullwidth/compat chars → base + combining marks
    normalized = unicodedata.normalize("NFKD", text)
    # Step 3: Remove combining marks (accents etc.) to get clean ASCII-like text
    normalized = "".join(
        ch for ch in normalized if unicodedata.category(ch) != "Mn"
    )
    # Step 4: Strip invisible / bidi chars
    normalized = _INVISIBLE_CHARS.sub("", normalized)
    return normalized


class InputSanitizer:
    """Checks prompts against known attack patterns.

    Returns a SanitizationResult indicating whether the prompt is allowed
    and listing any violations found. Uses compiled regexes (no LLM calls).

    Normalizes Unicode (confusables, NFKD, zero-width/bidi removal), checks
    leetspeak and separator-stripped variants, and accepts an optional
    context field for indirect injection defense.
    """

    def check(
        self,
        prompt: str,
        *,
        context: str | None = None,
    ) -> SanitizationResult:
        """Check prompt (and optional context) against all patterns.

        Args:
            prompt: The user's prompt text.
            context: Optional context field (CRM data, invoice text, etc.).
                     Checked for indirect injection attacks.

        Returns:
            SanitizationResult with violations list.
        """
        violations: list[str] = []

        # Build list of texts to check
        texts_to_check = [prompt]
        if context is not None:
            texts_to_check.append(context)

        for text in texts_to_check:
            self._check_text(text, violations)

        return SanitizationResult(
            allowed=len(violations) == 0,
            violations=violations,
        )

    def _check_text(self, text: str, violations: list[str]) -> None:
        """Check a single text against all patterns with multi-layer normalization.

        We check multiple versions to catch both evasion and native-script attacks:
        1. Original text — catches native CJK/Cyrillic/Arabic/Korean patterns
        2. Normalized text — catches homoglyph/fullwidth evasion (Latin-script)
        3. De-leet text — catches leetspeak evasion
        4. Separator-stripped text — catches char-insertion evasion (ig.nore, i-g-n-o-r-e)
        5. Chained variants — separator-strip then de-leet, and de-leet then
           separator-strip — catch inputs that mix both evasions
           (1-g-n-0-r-3, 1.g.n.0.r.3)
        """
        # 1. Normalize unicode (homoglyphs → Latin, NFKD, strip invisible)
        normalized = _normalize_text(text)

        # 2. Generate de-leet version of the normalized text
        deleeted = _deleet(normalized)

        # 3. Strip separator chars (dots, hyphens, etc. between letters)
        stripped = _strip_separators(normalized)

        # 4. Chained variants. Stripping first matters most: in "1-g-n-0-r-3"
        #    each leet digit is its own letter-less cluster, so _deleet leaves
        #    it alone until the separators are gone ("1gn0r3" -> "ignore").
        stripped_deleeted = _deleet(stripped)
        deleeted_stripped = _strip_separators(deleeted)

        # Check patterns against all versions (deduplicated)
        versions = [text]
        for v in (normalized, deleeted, stripped, stripped_deleeted, deleeted_stripped):
            if v not in versions:
                versions.append(v)

        for version in versions:
            for pattern, category in _PATTERNS:
                if category not in violations and pattern.search(version):
                    violations.append(category)
