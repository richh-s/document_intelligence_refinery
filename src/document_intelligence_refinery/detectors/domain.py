"""Domain-hint classifier with pluggable strategy pattern.

Default strategy uses case-insensitive, stemmed keyword matching
with word-boundary regex for resilience to OCR errors and
morphological variation.
"""

from __future__ import annotations

import re
from typing import Any, Protocol, runtime_checkable

import structlog

from document_intelligence_refinery.config import TriageConfig
from document_intelligence_refinery.models import DomainHint

logger = structlog.get_logger()


# ── Strategy Protocol ─────────────────────────────────────────────────


@runtime_checkable
class DomainClassificationStrategy(Protocol):
    """Interface for domain classification backends."""

    def classify(
        self, text_sample: str, metadata: dict[str, Any]
    ) -> tuple[DomainHint, float]:
        """Return ``(domain_hint, confidence)``."""
        ...


# ── Keyword Strategy (Default) ────────────────────────────────────────


class KeywordDomainStrategy:
    """Case-insensitive, stemmed keyword matching.

    Lexicon entries are *stems* matched via ``r'\\b{stem}\\w*\\b'``
    to catch morphological variants (e.g. "fiscal" → "fiscally").
    Text is case-folded before matching.

    Scoring:
        ``score = keyword_hits / total_tokens``  (length-normalised)
        ``confidence = top_score - second_score``  (margin-based)

    Deterministic fallback:
        1. ``total_tokens < DOMAIN_MIN_TOKENS`` → ``GENERAL``, confidence 0.0
        2. No domain exceeds 0 hits → ``GENERAL``, confidence 0.0
    """

    LEXICONS: dict[DomainHint, list[str]] = {
        DomainHint.FINANCIAL: [
            "revenue",
            "audit",
            "balance sheet",
            "dividend",
            "profit",
            "loss statement",
            "asset",
            "liabilit",
            "financ",
            "capital",
            "equity",
            "interest rate",
            "income statement",
            "shareholder",
            "deposit",
            "loan",
            "bank",
            "insurance",
            "portfolio",
        ],
        DomainHint.LEGAL: [
            "pursuant",
            "statute",
            "regulat",
            "complian",
            "jurisdict",
            "legislat",
            "ordinance",
            "contractual",
            "litigat",
            "arbitrat",
            "enforc",
            "penal",
            "constitut",
        ],
        DomainHint.SCIENTIFIC: [
            "methodolog",
            "hypothes",
            "experiment",
            "empiric",
            "variable",
            "dataset",
            "peer review",
            "abstract",
            "observat",
        ],
        DomainHint.GOVERNMENTAL: [
            "proclamat",
            "directiv",
            "ministr",
            "government",
            "public sector",
            "public service",
            "polic",
            "administrat",
            "bureau",
            "commission",
            "parliament",
            "gazette",
            "decree",
            "sovereign",
            "survey",
            "reform",
            "public",
            "federal",
            "nation",
            "expenditure",
            "budget",
            "tax",
            "fiscal",
        ],
    }

    def __init__(self, config: TriageConfig) -> None:
        self._cfg = config
        # Pre-compile regex patterns for each domain
        self._patterns: dict[DomainHint, list[re.Pattern[str]]] = {}
        for domain, stems in self.LEXICONS.items():
            self._patterns[domain] = [
                re.compile(rf"\b{re.escape(stem)}\w*\b", re.IGNORECASE)
                for stem in stems
            ]

    def classify(
        self, text_sample: str, metadata: dict[str, Any]
    ) -> tuple[DomainHint, float]:
        text_lower = text_sample.lower()
        tokens = text_lower.split()
        total_tokens = len(tokens)

        # Fallback rule 1: insufficient text
        if total_tokens < self._cfg.DOMAIN_MIN_TOKENS:
            logger.info(
                "domain_fallback",
                reason="below_min_tokens",
                total_tokens=total_tokens,
                threshold=self._cfg.DOMAIN_MIN_TOKENS,
            )
            return DomainHint.GENERAL, 0.0

        # Score each domain
        scores: dict[DomainHint, float] = {}
        raw_hits: dict[DomainHint, int] = {}

        for domain, patterns in self._patterns.items():
            hits = sum(
                len(pattern.findall(text_lower)) for pattern in patterns
            )
            raw_hits[domain] = hits
            scores[domain] = hits / total_tokens

        logger.info(
            "domain_scores",
            total_tokens=total_tokens,
            raw_hits=raw_hits,
            normalised_scores={
                k.value: round(v, 6) for k, v in scores.items()
            },
        )

        # Fallback rule 2: no domain has any hits
        if all(v == 0 for v in scores.values()):
            return DomainHint.GENERAL, 0.0

        # Rank and compute margin-based confidence
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_domain, top_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        confidence = top_score - second_score

        return top_domain, round(confidence, 6)


# ── VLM Placeholder ──────────────────────────────────────────────────


class VLMDomainStrategy:
    """Placeholder for Phase 2+ VLM-based domain classification."""

    def classify(
        self, text_sample: str, metadata: dict[str, Any]
    ) -> tuple[DomainHint, float]:
        raise NotImplementedError(
            "VLM domain classification requires Phase 2 infrastructure."
        )


# ── Classifier Wrapper ────────────────────────────────────────────────


class DomainHintClassifier:
    """Facade that delegates to a pluggable strategy.

    Defaults to ``KeywordDomainStrategy`` if none is provided.
    """

    def __init__(
        self,
        strategy: DomainClassificationStrategy | None = None,
        config: TriageConfig | None = None,
    ) -> None:
        cfg = config or TriageConfig()
        self._strategy = strategy or KeywordDomainStrategy(config=cfg)

    def classify(
        self, text_sample: str, metadata: dict[str, Any]
    ) -> tuple[DomainHint, float]:
        return self._strategy.classify(text_sample, metadata)
