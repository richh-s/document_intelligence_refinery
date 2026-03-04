"""Unit tests for all detectors — origin, layout, domain.

Uses synthetic data — no file I/O — for fast, deterministic tests.
Includes adversarial edge cases.
"""

import pytest

from config import TriageConfig
from detectors.domain import (
    DomainClassificationStrategy,
    DomainHintClassifier,
    KeywordDomainStrategy,
)
from detectors.layout import LayoutComplexityDetector
from detectors.origin import OriginTypeDetector
from models.document_profile import (
    DomainHint,
    LayoutType,
    OriginType,
)


# ═══════════════════════════════════════════════════════════════════════
# Origin Detector
# ═══════════════════════════════════════════════════════════════════════


class TestOriginTypeDetector:
    def setup_method(self) -> None:
        self.cfg = TriageConfig()
        self.det = OriginTypeDetector(self.cfg)

    def _digital_page(self) -> dict:
        return {
            "ink_density": 0.08,
            "whitespace_ratio": 0.70,
            "image_ratio": 0.0,
            "font_count": 5,
            "vector_count": 20,
        }

    def _scanned_page(self) -> dict:
        return {
            "ink_density": 0.0,
            "whitespace_ratio": 0.99,
            "image_ratio": 0.95,
            "font_count": 0,
            "vector_count": 0,
        }

    def test_digital_native(self) -> None:
        origin, conf, meta = self.det.detect([self._digital_page()] * 3)
        assert origin == OriginType.DIGITAL_NATIVE
        assert conf > 0.0

    def test_scanned(self) -> None:
        origin, conf, meta = self.det.detect([self._scanned_page()] * 3)
        assert origin == OriginType.SCANNED
        assert conf > 0.0

    def test_mixed(self) -> None:
        pages = [self._digital_page(), self._scanned_page()]
        origin, conf, meta = self.det.detect(pages)
        assert origin == OriginType.MIXED

    def test_empty_pages(self) -> None:
        origin, conf, meta = self.det.detect([])
        assert origin == OriginType.MIXED
        assert conf == 0.0

    def test_confidence_bounded(self) -> None:
        _, conf, _ = self.det.detect([self._digital_page()])
        assert 0.0 <= conf <= 1.0

    # ── Adversarial ───────────────────────────────────────────────────

    def test_adversarial_ambiguous_page(self) -> None:
        """High chars + high whitespace + few vectors → low confidence."""
        ambiguous = {
            "ink_density": 0.03,
            "whitespace_ratio": 0.96,
            "image_ratio": 0.3,
            "font_count": 1,
            "vector_count": 3,
        }
        _, conf, _ = self.det.detect([ambiguous])
        # Confidence should be low (close to boundary)
        assert conf < 0.7


# ═══════════════════════════════════════════════════════════════════════
# Layout Detector
# ═══════════════════════════════════════════════════════════════════════


class TestLayoutComplexityDetector:
    def setup_method(self) -> None:
        self.cfg = TriageConfig()
        self.det = LayoutComplexityDetector(self.cfg)

    def _simple_page(self) -> dict:
        return {
            "unique_fonts": 1,
            "vectors": [],
            "line_height_variance": 1.0,
            "words": [{"x0": 50 + i * 5} for i in range(20)],
            "tables_detected": 0,
        }

    def _complex_page(self) -> dict:
        # Many fonts, many clustered vectors, high variance, tables
        vectors = [
            {"x0": 10 + i * 3, "top": 10, "x1": 12 + i * 3, "bottom": 200}
            for i in range(60)
        ]
        return {
            "unique_fonts": 12,
            "vectors": vectors,
            "line_height_variance": 50.0,
            "words": (
                [{"x0": 50 + i * 2} for i in range(10)]
                + [{"x0": 300 + i * 2} for i in range(10)]
                + [{"x0": 550 + i * 2} for i in range(10)]
            ),
            "tables_detected": 6,
        }

    def test_single_column_layout(self) -> None:
        layout, conf, meta = self.det.detect([self._simple_page()])
        assert layout == LayoutType.SINGLE_COLUMN
        # margin should be roughly 1.0(single) - 0.33(column) = 0.33 based on mock column gap extraction
        assert conf > 0.3

    def test_table_heavy_layout(self) -> None:
        table_heavy_page = {
            "tables_detected": 4,
            "vectors": [{"x0": 10, "top": 10, "x1": 100, "bottom": 100} for _ in range(50)], # lots of graphics
            "words": [{"x0": 50}]
        }
        layout, conf, meta = self.det.detect([table_heavy_page])
        assert layout == LayoutType.TABLE_HEAVY

    def test_multi_column_layout(self) -> None:
        multi_col = {
            "unique_fonts": 2,
            "line_height_variance": 5.0,
            "words": (
                [{"x0": 50 + i * 2} for i in range(10)]
                + [{"x0": 300 + i * 2} for i in range(10)]
                + [{"x0": 550 + i * 2} for i in range(10)]
            ),
            "tables_detected": 0,
        }
        layout, conf, meta = self.det.detect([multi_col])
        assert layout == LayoutType.MULTI_COLUMN

    def test_figure_heavy_layout(self) -> None:
        # pass image ratio directly via the kwarg
        layout, conf, meta = self.det.detect([self._simple_page()], image_ratio=0.85)
        assert layout == LayoutType.FIGURE_HEAVY

    def test_mixed_layout(self) -> None:
        layout, conf, meta = self.det.detect([self._complex_page()])
        # Complex page hits font, graphic, line_var, and tables. 
        # With active >= 3, margin should shrink, triggering MIXED.
        assert layout == LayoutType.MIXED

    def test_empty_pages(self) -> None:
        layout, conf, meta = self.det.detect([])
        assert layout == LayoutType.SINGLE_COLUMN
        assert conf == 0.0

    def test_confidence_bounded(self) -> None:
        _, conf, _ = self.det.detect([self._simple_page()])
        assert 0.0 <= conf <= 1.0

    # ── Vector clustering ─────────────────────────────────────────────

    def test_vector_clustering_merges_adjacent(self) -> None:
        """Adjacent vectors should cluster into fewer groups."""
        # 10 touching vectors → should form fewer groups
        vectors = [
            {"x0": i * 4, "top": 0, "x1": i * 4 + 5, "bottom": 10}
            for i in range(10)
        ]
        groups = self.det._cluster_vectors(vectors)
        assert groups < 10  # clustered, not raw

    def test_vector_clustering_distant_vectors(self) -> None:
        """Distant vectors should remain separate groups."""
        vectors = [
            {"x0": 0, "top": 0, "x1": 10, "bottom": 10},
            {"x0": 500, "top": 500, "x1": 510, "bottom": 510},
        ]
        groups = self.det._cluster_vectors(vectors)
        assert groups == 2


# ═══════════════════════════════════════════════════════════════════════
# Domain Classifier
# ═══════════════════════════════════════════════════════════════════════


class TestDomainHintClassifier:
    def setup_method(self) -> None:
        self.cfg = TriageConfig()
        self.strategy = KeywordDomainStrategy(config=self.cfg)
        self.classifier = DomainHintClassifier(strategy=self.strategy, config=self.cfg)

    def test_financial_text(self) -> None:
        text = (
            "The annual audit of revenue shows a significant increase "
            "in capital assets and dividend payments to shareholders. "
            "The balance sheet indicates strong financial performance "
            "across all subsidiaries. Profit and loss statements confirm "
            "that bank deposits and loan portfolios exceeded projections. "
            "Insurance and equity returns are at record highs. "
        ) * 3
        hint, conf = self.classifier.classify(text, {})
        assert hint == DomainHint.FINANCIAL
        assert conf > 0.0

    def test_governmental_text(self) -> None:
        text = (
            "The Ministry of Finance issued a directive in the government "
            "gazette establishing a new commission for public sector reform. "
            "The parliamentary proclamation outlines administrative policy. "
            "The bureau of administration is responsible for decree enforcement. "
            "Budget revenue and tax capital asset analysis."
        ) * 3
        hint, conf = self.classifier.classify(text, {})
        assert hint == DomainHint.GOVERNMENTAL

    def test_case_insensitive(self) -> None:
        text = (
            "REVENUE AUDIT FINANCIAL CAPITAL DIVIDEND SHAREHOLDER "
            "BALANCE SHEET PROFIT ASSET DEPOSIT LOAN BANK INSURANCE "
        ) * 5
        hint, conf = self.classifier.classify(text, {})
        assert hint == DomainHint.FINANCIAL

    def test_stemmed_matching(self) -> None:
        """Morphological variants should still match."""
        text = (
            "The regulatory compliance framework ensures legislative "
            "oversight. Contractual obligations are enforceable under "
            "the constitutional statutes of this jurisdiction. "
            "The litigants seek arbitration. "
        ) * 4
        hint, conf = self.classifier.classify(text, {})
        assert hint == DomainHint.LEGAL

    def test_below_min_tokens(self) -> None:
        text = "short text"
        hint, conf = self.classifier.classify(text, {})
        assert hint == DomainHint.GENERAL
        assert conf == 0.0

    def test_no_keywords_fallback(self) -> None:
        text = " ".join(["lorem ipsum dolor sit amet"] * 20)
        hint, conf = self.classifier.classify(text, {})
        assert hint == DomainHint.GENERAL
        assert conf == 0.0

    def test_strategy_swap(self) -> None:
        """Swapping strategy should change classification behavior."""

        class AlwaysLegalStrategy:
            def classify(self, text_sample: str, metadata: dict) -> tuple:
                return DomainHint.LEGAL, 1.0

        custom = DomainHintClassifier(strategy=AlwaysLegalStrategy(), config=self.cfg)
        hint, conf = custom.classify("anything", {})
        assert hint == DomainHint.LEGAL
        assert conf == 1.0

    # ── Adversarial ───────────────────────────────────────────────────

    def test_adversarial_mixed_domain(self) -> None:
        """Financial + Legal keywords mixed → low confidence."""
        text = (
            "The financial audit of revenue shows dividend growth. "
            "However, the statutory compliance framework requires "
            "legislative oversight and contractual arbitration for "
            "any bank deposit disputes."
        ) * 5
        hint, conf = self.classifier.classify(text, {})
        # Confidence should be meaningfully lower than pure-domain text
        assert conf < 0.4


# ═══════════════════════════════════════════════════════════════════════
# Config Validation
# ═══════════════════════════════════════════════════════════════════════


class TestTriageConfig:
    def test_default_weights_valid(self) -> None:
        cfg = TriageConfig()
        assert abs(sum(cfg.ORIGIN_WEIGHTS.values()) - 1.0) < 1e-6
        assert abs(sum(cfg.LAYOUT_WEIGHTS.values()) - 1.0) < 1e-6

    def test_invalid_origin_weights(self) -> None:
        with pytest.raises(ValueError, match="ORIGIN_WEIGHTS"):
            TriageConfig(ORIGIN_WEIGHTS={"ink_density": 0.5, "whitespace_ratio": 0.1})

    def test_invalid_layout_weights(self) -> None:
        with pytest.raises(ValueError, match="LAYOUT_WEIGHTS"):
            TriageConfig(LAYOUT_WEIGHTS={"font": 0.1})
