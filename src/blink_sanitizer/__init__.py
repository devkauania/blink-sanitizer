"""Blink Sanitizer — sub-millisecond prompt injection detection.

262 compiled regex patterns across 11 languages. No LLM calls. No network.
Covers all 10 OWASP LLM Top 10 2025 risk categories.

Usage:
    from blink_sanitizer import InputSanitizer

    sanitizer = InputSanitizer()
    result = sanitizer.check("ignore all previous instructions")
    print(result.allowed)     # False
    print(result.violations)  # ['prompt_injection']
"""

from blink_sanitizer.sanitizer import InputSanitizer, SanitizationResult

__all__ = ["InputSanitizer", "SanitizationResult"]
__version__ = "0.1.0"
