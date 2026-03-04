"""TriageAgent — orchestrates all detectors to produce a DocumentProfile.

This is the single entry-point for profiling a PDF.  It handles:
  • file hashing (SHA-256)
  • encryption detection
  • smart-sampled page analysis via pdfplumber
  • detector dispatch (origin → layout → domain)
  • warning collection
  • structured metadata assembly
"""

from __future__ import annotations

import hashlib
import statistics
from pathlib import Path
from typing import Any

import pypdf
import pdfplumber
import structlog

from config import TriageConfig
from detectors.domain import (
    DomainClassificationStrategy,
    DomainHintClassifier,
    KeywordDomainStrategy,
)
from detectors.language import LanguageDetector
from detectors.layout import LayoutComplexityDetector
from detectors.origin import OriginTypeDetector
from models.document_profile import (
    ConfidenceScores,
    DomainHint,
    DocumentProfile,
    ExtractionCostEstimate,
    LayoutType,
    OriginType,
)
from persistence import ProfileStore
from sampling import SmartSampler

logger = structlog.get_logger()

_BUF_SIZE = 65_536  # 64 KiB read buffer for hashing


class TriageAgent:
    """Profile a PDF document by running all classification detectors."""

    def __init__(
        self,
        config: TriageConfig | None = None,
        domain_strategy: DomainClassificationStrategy | None = None,
    ) -> None:
        self._cfg = config or TriageConfig()
        self._sampler = SmartSampler()
        self._origin_detector = OriginTypeDetector(self._cfg)
        self._layout_detector = LayoutComplexityDetector(self._cfg)
        self._language_detector = LanguageDetector(min_tokens=self._cfg.LANGUAGE_MIN_TOKENS)
        self._profile_store = ProfileStore()

        strategy = domain_strategy or KeywordDomainStrategy(config=self._cfg)
        self._domain_classifier = DomainHintClassifier(
            strategy=strategy, config=self._cfg
        )

    # ── Public API ────────────────────────────────────────────────────

    def profile(self, pdf_path: Path) -> DocumentProfile:
        """Run all detectors and assemble a complete profile."""
        pdf_path = Path(pdf_path)

        logger.info("triage_start", file=pdf_path.name)

        file_hash = self._compute_hash(pdf_path)
        warnings: list[str] = []
        metadata: dict[str, dict[str, Any]] = {}

        # ── Encryption check ──────────────────────────────────────────
        is_encrypted = self._check_encrypted(pdf_path)
        if is_encrypted:
            warnings.append("Encrypted PDF — all detection skipped")
            logger.warning("encrypted_pdf", file=pdf_path.name)
            return DocumentProfile(
                file_name=pdf_path.name,
                doc_id=file_hash,
                file_hash=file_hash,
                origin_type=OriginType.MIXED,
                layout_type=LayoutType.SINGLE_COLUMN,
                domain_hint=DomainHint.GENERAL,
                language="unknown",
                extraction_cost=ExtractionCostEstimate.NEEDS_VISION_MODEL,
                confidence=ConfidenceScores(origin=0.0, layout=0.0, domain=0.0, language=0.0),
                page_count=0,
                is_encrypted=True,
                warnings=warnings,
                processing_metadata=self._processing_meta(),
                metadata={},
            )

        # ── Open PDF and extract page-level stats ─────────────────────
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            sample_indices = self._sampler.sample_indices(total_pages)

            logger.info(
                "sampling",
                total_pages=total_pages,
                sampled=sample_indices,
            )

            # ── Form-Fillable Check ───────────────────────────────────────
            has_acroform, has_text = self._check_form_fillable(pdf_path, pdf)
            form_detected = False
            if has_acroform:
                if has_text:
                    form_detected = True
                else:
                    warnings.append("Form fields detected but no text layer — likely scanned form")

            origin_stats, layout_stats, text_chunks = self._extract_page_stats(
                pdf, sample_indices, warnings
            )

        # ── Origin detection ──────────────────────────────────────────
        if form_detected:
            origin_type = OriginType.FORM_FILLABLE
            origin_conf = 1.0
            metadata["origin"] = {"form_fillable": True}
        else:
            origin_type, origin_conf, origin_meta = self._origin_detector.detect(
                origin_stats
            )
            metadata["origin"] = origin_meta

        # ── Layout detection ──────────────────────────────────────────
        avg_image_ratio = float(statistics.mean(s.get("image_ratio", 0.0) for s in origin_stats)) if origin_stats else 0.0
        layout_type, layout_conf, layout_meta = self._layout_detector.detect(
            layout_stats, image_ratio=avg_image_ratio
        )
        metadata["layout"] = layout_meta

        # ── Domain classification ─────────────────────────────────────
        combined_text = " ".join(text_chunks)
        domain_hint, domain_conf = self._domain_classifier.classify(
            combined_text, {}
        )
        metadata["domain"] = {
            "text_sample_length": len(combined_text),
            "token_count": len(combined_text.split()),
        }

        # ── Language detection ────────────────────────────────────────
        language, lang_conf = self._language_detector.detect(combined_text)

        metadata["sampling"] = {
            "strategy": "stratified",
            "sampled_pages": [idx + 1 for idx in sample_indices],
        }

        # ── Cost Estimation ───────────────────────────────────────────
        cost_estimate = self._estimate_cost(origin_type, layout_type)

        profile = DocumentProfile(
            file_name=pdf_path.name,
            doc_id=file_hash,
            file_hash=file_hash,
            origin_type=origin_type,
            layout_type=layout_type,
            domain_hint=domain_hint,
            language=language,
            language_confidence=round(lang_conf, 6),
            extraction_cost=cost_estimate,
            confidence=ConfidenceScores(
                origin=origin_conf,
                layout=layout_conf,
                domain=domain_conf,
                language=lang_conf,
            ),
            page_count=total_pages,
            is_encrypted=False,
            warnings=warnings,
            processing_metadata=self._processing_meta(),
            metadata=metadata,
        )

        # ── Persist to disk ───────────────────────────────────────────
        self._profile_store.save(profile)

        logger.info(
            "triage_complete",
            file=pdf_path.name,
            origin=profile.origin_type.value,
            layout=profile.layout_type.value,
            domain=profile.domain_hint.value,
            language=profile.language,
            cost=profile.extraction_cost.value,
            confidence_origin=profile.confidence.origin,
            confidence_layout=profile.confidence.layout,
            confidence_domain=profile.confidence.domain,
            confidence_language=profile.confidence.language,
        )

        return profile

    # ── Page-level extraction ─────────────────────────────────────────

    def _extract_page_stats(
        self,
        pdf: pdfplumber.PDF,
        sample_indices: list[int],
        warnings: list[str],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
        """Extract origin, layout, and text data from sampled pages."""
        origin_stats: list[dict[str, Any]] = []
        layout_stats: list[dict[str, Any]] = []
        text_chunks: list[str] = []

        for idx in sample_indices:
            page = pdf.pages[idx]
            chars = page.chars
            words = page.extract_words()
            page_area = float(page.width * page.height) if page.width and page.height else 1.0

            # ── Origin signals ────────────────────────────────────────
            char_area = sum(
                (float(c["x1"]) - float(c["x0"]))
                * (float(c["bottom"]) - float(c["top"]))
                for c in chars
            )
            ink_density = char_area / page_area if page_area > 0 else 0.0
            
            # Explicit Character Density
            char_density = len(chars) / page_area if page_area > 0 else 0.0

            image_area = sum(
                float(img.get("width", 0)) * float(img.get("height", 0))
                for img in page.images
            )
            image_ratio = image_area / page_area if page_area > 0 else 0.0

            fonts = {c.get("fontname") for c in chars if c.get("fontname")}
            font_count = len(fonts)

            vectors_raw = list(page.rects) + list(page.lines) + list(page.curves)
            vector_count = len(vectors_raw) + len(page.images)

            # Explicit Whitespace Analysis
            whitespace_area = max(0.0, page_area - char_area - image_area)
            ws_ratio = whitespace_area / page_area if page_area > 0 else 1.0

            origin_stats.append({
                "ink_density": ink_density,
                "char_density": char_density,
                "whitespace_ratio": ws_ratio,
                "image_ratio": image_ratio,
                "font_count": font_count,
                "vector_count": vector_count,
            })

            # ── Layout signals ────────────────────────────────────────
            # line-height variance
            sorted_words = sorted(words, key=lambda w: w["top"])
            line_heights: list[float] = []
            last_top: float | None = None
            for w in sorted_words:
                if last_top is not None:
                    diff = float(w["top"]) - last_top
                    if diff > 0:
                        line_heights.append(diff)
                last_top = float(w["top"])
            lh_variance = (
                statistics.variance(line_heights)
                if len(line_heights) >= 2
                else 0.0
            )

            # Table bbox detection (from intersecting rects)
            tables_found = len(page.find_tables()) if hasattr(page, "find_tables") else 0

            layout_stats.append({
                "unique_fonts": font_count,
                "vectors": [
                    {
                        "x0": v.get("x0", 0),
                        "top": v.get("top", 0),
                        "x1": v.get("x1", 0),
                        "bottom": v.get("bottom", 0),
                    }
                    for v in vectors_raw
                ],
                "line_height_variance": lh_variance,
                "words": [{"x0": w["x0"]} for w in words],
                "tables_detected": tables_found,
            })

            # ── Text extraction ───────────────────────────────────────
            text = page.extract_text() or ""
            text_chunks.append(text)

            # ── Warnings ──────────────────────────────────────────────
            if len(chars) == 0 and len(page.images) == 0:
                warnings.append(f"Page {idx}: empty (no chars, no images)")
            elif ink_density < 0.001 and len(chars) > 0:
                warnings.append(f"Page {idx}: extremely low text density")

        return origin_stats, layout_stats, text_chunks

    # ── Utilities ─────────────────────────────────────────────────────

    @staticmethod
    def _compute_hash(path: Path) -> str:
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                data = f.read(_BUF_SIZE)
                if not data:
                    break
                sha.update(data)
        return sha.hexdigest()

    @staticmethod
    def _check_encrypted(path: Path) -> bool:
        try:
            with pdfplumber.open(path) as pdf:
                # If we can open and access pages, it's not encrypted
                _ = len(pdf.pages)
            return False
        except Exception:
            return True

    @staticmethod
    def _check_form_fillable(pdf_path: Path, pdfplumber_doc: pdfplumber.PDF) -> tuple[bool, bool]:
        """Returns (has_acroform, has_text_layer)."""
        has_acroform = False
        try:
            reader = pypdf.PdfReader(pdf_path)
            if "/AcroForm" in reader.trailer["/Root"]:
                has_acroform = True
        except Exception:
            pass

        # check first up to 3 sampled pages for text
        has_text = any(len(page.chars) > 0 for page in pdfplumber_doc.pages[:3])
        return has_acroform, has_text

    @staticmethod
    def _estimate_cost(origin: OriginType, layout: LayoutType) -> ExtractionCostEstimate:
        if origin in (OriginType.SCANNED, OriginType.MIXED):
            return ExtractionCostEstimate.NEEDS_VISION_MODEL
        if origin == OriginType.FORM_FILLABLE:
            if layout == LayoutType.TABLE_HEAVY:
                return ExtractionCostEstimate.NEEDS_LAYOUT_MODEL
            return ExtractionCostEstimate.FAST_TEXT_SUFFICIENT
        
        # DIGITAL_NATIVE
        if layout in (LayoutType.MULTI_COLUMN, LayoutType.TABLE_HEAVY):
            return ExtractionCostEstimate.NEEDS_LAYOUT_MODEL
        return ExtractionCostEstimate.FAST_TEXT_SUFFICIENT

    def _processing_meta(self) -> dict[str, Any]:
        import pdfplumber as _pp

        return {
            "pdfplumber_version": getattr(_pp, "__version__", "unknown"),
            "triage_config_version": "1.0.0",
            "detector_version": "1.0.0",
        }
