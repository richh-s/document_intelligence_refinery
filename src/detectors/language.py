"""Language detector wrapper around langdetect.

Provides deterministic language detection for the Triage Agent.
"""

from __future__ import annotations

from langdetect import DetectorFactory, detect_langs

# Seed set ONCE at module import for deterministic behavior.
# WARNING: If tests or workloads run in parallel via multiple threads
# in the SAME process, this global state may cause issues.
# For single-threaded or multi-process (e.g., pytest-xdist) execution,
# setting it here is safe and standard practice.
DetectorFactory.seed = 0


class LanguageDetector:
    """Detects text language using a fallback confidence threshold."""

    def __init__(self, min_tokens: int = 20) -> None:
        self._min_tokens = min_tokens

    def detect(self, text: str) -> tuple[str, float]:
        """Classify language and return ``(language_code, confidence)``.

        If text length is below the ``min_tokens`` threshold,
        returns ``("unknown", 0.0)``.
        """
        tokens = text.split()
        if len(tokens) < self._min_tokens:
            return "unknown", 0.0

        try:
            results = detect_langs(text)
            if not results:
                return "unknown", 0.0
            
            top_result = results[0]
            # langdetect returns codes like 'en', 'fr', etc.
            return top_result.lang, round(top_result.prob, 6)
        except Exception:
            # langdetect throws LangDetectException if no features can be extracted
            return "unknown", 0.0
