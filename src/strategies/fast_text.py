"""Strategy A: Fast Text Extraction (pdfplumber)."""

import pdfplumber
import time
from pathlib import Path
from typing import Any

from models.document_profile import DocumentProfile
from models.extracted_document import (
    ExtractedDocument,
    ExtractedPage,
    TextBlock,
    normalize_coordinates,
)
from strategies.base import BaseExtractor, ExtractionResult


class FastTextExtractor(BaseExtractor):
    """Cost: Low. Triggers on DIGITAL_NATIVE/FORM_FILLABLE + SINGLE_COLUMN."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.nonsense_ratio_max = float(self.config.get("NONSENSE_RATIO_MAX", 0.30))
        self.min_chars_per_page = 100
        
    def _is_nonsense(self, word: str) -> bool:
        """Heuristic N-gram / junk token check for hidden OCR layers."""
        if not word:
            return True
            
        # Fast junk detection without heavy ML deps
        # 1. Very long words without spaces
        if len(word) > 40:
            return True
        # 2. Words with no alphabetic characters (e.g., "$$123-///")
        # unless it's a pure number which is fine in financial docs.
        # Check if doing purely symbols/letters and no alphabet.
        has_alpha = any(c.isalpha() for c in word)
        has_digit = any(c.isdigit() for c in word)
        if not has_alpha and not has_digit:
            return True
        # If it has digits but is overwhelmed by symbols (e.g "$$$$-11"):
        if has_digit and not has_alpha:
            alnum_count = sum(c.isalnum() for c in word)
            if alnum_count / len(word) < 0.5:
                return True

        # 3. Repeated characters "sssssss"
        for i in range(len(word) - 3):
            if word[i] == word[i+1] == word[i+2] == word[i+3]:
                if word[i].isalpha():
                    return True
                    
        return False

    def extract(self, pdf_path: Path, profile: DocumentProfile) -> ExtractionResult:
        start_time = self._timer_start()
        pages = []
        page_confidences = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, raw_page in enumerate(pdf.pages, start=1):
                page_width = raw_page.width
                page_height = raw_page.height
                
                # Check for heavily images/graphics-driven pages that might mask as digital
                image_area = sum(img["width"] * img["height"] for img in raw_page.images)
                page_area = page_width * page_height
                image_ratio = image_area / page_area if page_area > 0 else 0
                
                # Extract text blocks
                # pdfplumber extract_words() groups by spatial clustering
                words = raw_page.extract_words(
                    keep_blank_chars=False,
                    use_text_flow=True,
                    extra_attrs=["fontname", "size"]
                )
                
                text_blocks = []
                junk_tokens = 0
                total_tokens = len(words)
                total_chars = 0
                
                for word_data in words:
                    text = word_data["text"]
                    total_chars += len(text)
                    
                    if self._is_nonsense(text):
                        junk_tokens += 1
                        
                    # pdfplumber origin is Top-Left (x0, top, x1, bottom)
                    bbox = normalize_coordinates(
                        (word_data["x0"], word_data["top"], word_data["x1"], word_data["bottom"]),
                        float(page_width),
                        float(page_height),
                        source_origin="top_left"
                    )
                    
                    text_blocks.append(TextBlock(
                        text=text,
                        bbox=bbox,
                        page_number=page_num,
                        source_strategy="FastTextExtractor",
                        reading_order=len(text_blocks) + 1
                    ))
                
                # Gather signal metrics
                char_density = total_chars / page_area if page_area > 0 else 0.0
                has_fonts = any(w.get("fontname") is not None for w in words) if words else False
                
                # Compute Confidence Signals for this page
                nonsense_ratio = (junk_tokens / total_tokens) if total_tokens > 0 else 1.0
                quality_score = max(0.0, 1.0 - nonsense_ratio)
                
                # Base structural confidence
                structural_conf = 1.0
                if total_chars < self.min_chars_per_page:
                    structural_conf *= 0.1 # Severely penalize sparse pages
                    
                if image_ratio > 0.5 and total_chars < 50:
                    # A page with high image area but near-zero character count is likely scanned
                    structural_conf = 0.0 
                elif image_ratio > 0.5:
                    structural_conf *= 0.5 # Penalize image-dominated pages
                    
                if not has_fonts and total_chars > 0:
                    # Lack of programmatic font metadata in a "digital" pdf hints at raw paths or OCR
                    structural_conf *= 0.5
                    
                if char_density < 0.0005 and total_chars > 0:
                    # Very sparse text spread across a large page area (often just noise or watermarks)
                    structural_conf *= 0.8
                    
                # If nonsense ratio blows past MAX, force deep penalty
                if nonsense_ratio > self.nonsense_ratio_max:
                    quality_score = 0.0 # Force gate failure
                    
                page_confidence = min(structural_conf, quality_score)
                page_confidences.append(page_confidence)
                
                pages.append(ExtractedPage(
                    page_number=page_num,
                    source_strategy="FastTextExtractor",
                    text_blocks=text_blocks,
                    tables=[], # Strategy A does not extract structured tables natively
                    figures=[],
                    metadata={
                        "nonsense_ratio": round(nonsense_ratio, 4),
                        "char_count": total_chars,
                        "image_ratio": round(image_ratio, 4),
                        "char_density": round(char_density, 6),
                        "has_fonts": has_fonts
                    }
                ))

        # Overall document confidence is the minimum page confidence
        final_confidence = min(page_confidences) if page_confidences else 0.0
        
        doc = ExtractedDocument(
            file_hash=profile.file_hash,
            pages=pages
        )

        return ExtractionResult(
            document=doc,
            confidence=round(final_confidence, 6),
            cost=0.0,
            time_ms=self._timer_end_ms(start_time),
            model_name="pdfplumber",
            pages_sent=len(pages),
            signals={
                "ocr_quality": round(sum(page_confidences)/len(page_confidences), 4) if page_confidences else 0.0,
                "layout_consistency": 1.0, # Strategy A is only used for single-column
                "structural_fidelity": 0.8 if any(p.metadata.get("has_fonts") for p in pages) else 0.5,
                "completeness_ratio": 1.0 # Scans all pages by default
            }
        )
