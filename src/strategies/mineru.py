"""Strategy B: MinerU-based Local Extraction with Docling fallback."""

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

from models.document_profile import DocumentProfile
from models.extracted_document import (
    ExtractedDocument,
    ExtractedPage,
    TextBlock,
    StructuredTable,
    Figure,
    normalize_coordinates,
)
from strategies.base import BaseExtractor, ExtractionResult

logger = logging.getLogger(__name__)


class MinerUExtractor(BaseExtractor):
    """Cost: Zero (Local). Uses MinerU for high-fidelity OCR + layout extraction.
    
    Falls back to Docling (LayoutAwareExtractor) if MinerU fails.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self._mineru_available = self._check_mineru()
        self._docling_fallback = None  # lazy-initialized

    @staticmethod
    def _check_mineru() -> bool:
        """Check if MinerU's magic-pdf library is importable."""
        try:
            from magic_pdf.data.read_api import read_local_pdfs  # noqa: F401
            return True
        except ImportError:
            logger.warning("magic-pdf not installed. MinerU strategy unavailable; will use Docling fallback.")
            return False

    def _get_docling_fallback(self):
        """Lazy-load Docling extractor as fallback."""
        if self._docling_fallback is None:
            from strategies.layout import LayoutAwareExtractor
            self._docling_fallback = LayoutAwareExtractor(self.config)
        return self._docling_fallback

    def extract(self, pdf_path: Path, profile: DocumentProfile) -> ExtractionResult:
        """Extract using MinerU Python API, falling back to Docling on failure."""
        
        if not self._mineru_available:
            logger.info("MinerU unavailable. Falling back to Docling.")
            return self._get_docling_fallback().extract(pdf_path, profile)
        
        try:
            return self._extract_with_mineru(pdf_path, profile)
        except Exception as e:
            logger.error(f"MinerU extraction failed: {e}. Falling back to Docling.")
            return self._get_docling_fallback().extract(pdf_path, profile)

    def _extract_with_mineru(self, pdf_path: Path, profile: DocumentProfile) -> ExtractionResult:
        """Core MinerU extraction using the direct Python API."""
        start_time = self._timer_start()

        from magic_pdf.data.data_reader_writer import FileBasedDataReader, FileBasedDataWriter
        from magic_pdf.data.dataset import PymuDocDataset
        from magic_pdf.config.enums import SupportedPdfParseMethod
        from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze

        logger.info(f"Running MinerU on {pdf_path.name} (Python API)")
        
        # 1. Read PDF bytes and create dataset
        reader = FileBasedDataReader()
        pdf_bytes = reader.read(str(pdf_path))
        ds = PymuDocDataset(pdf_bytes)

        # 2. Classify (txt vs ocr) and run model inference
        parse_method = ds.classify()
        is_ocr = (parse_method == SupportedPdfParseMethod.OCR)
        
        logger.info(f"MinerU classified document as: {'OCR' if is_ocr else 'TXT'} mode")
        
        infer_result = ds.apply(doc_analyze, ocr=is_ocr)

        # 3. Run the appropriate pipeline (we need a temp dir for images)
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_writer = FileBasedDataWriter(tmp_dir)
            
            if is_ocr:
                pipe_result = infer_result.pipe_ocr_mode(image_writer, debug_mode=False)
            else:
                pipe_result = infer_result.pipe_txt_mode(image_writer, debug_mode=False)

        # 4. Extract the raw pipeline result dict
        pipe_res = pipe_result._pipe_res
        pdf_info = pipe_res.get("pdf_info", [])
        
        logger.info(f"MinerU returned {len(pdf_info)} pages of data")

        # 5. Convert to our normalized ExtractedDocument
        normalized_doc = self._parse_pipe_result(pdf_info, profile, len(ds))

        # 6. Calculate confidence signals
        total_blocks = sum(len(p.text_blocks) for p in normalized_doc.pages)
        total_tables = sum(len(p.tables) for p in normalized_doc.pages)
        total_figures = sum(len(p.figures) for p in normalized_doc.pages)
        pages_with_content = sum(
            1 for p in normalized_doc.pages 
            if len(p.text_blocks) > 0 or len(p.tables) > 0
        )

        logger.info(
            f"MinerU extracted: {total_blocks} text blocks, "
            f"{total_tables} tables, {total_figures} figures "
            f"across {pages_with_content}/{profile.page_count} pages"
        )

        comp_ratio = pages_with_content / profile.page_count if profile.page_count > 0 else 0.0
        ocr_sig = 0.92 if is_ocr else 0.98
        layout_sig = min(1.0, 0.8 + (total_tables * 0.02)) if total_blocks > 0 else 0.5
        struct_sig = min(1.0, total_blocks / (profile.page_count * 10)) if total_blocks > 0 else 0.0

        confidence = (0.4 * comp_ratio) + (0.3 * layout_sig) + (0.2 * struct_sig) + (0.1 * ocr_sig)
        confidence = max(0.1, min(1.0, confidence)) if total_blocks > 0 else 0.0

        return ExtractionResult(
            document=normalized_doc,
            confidence=round(confidence, 4),
            cost=0.0,
            time_ms=self._timer_end_ms(start_time),
            model_name="mineru-local",
            pages_sent=profile.page_count,
            signals={
                "ocr_quality": ocr_sig,
                "layout_consistency": round(layout_sig, 4),
                "structural_fidelity": round(struct_sig, 4),
                "completeness_ratio": round(comp_ratio, 4)
            }
        )

    def _parse_pipe_result(
        self, pdf_info: list, profile: DocumentProfile, total_pages: int
    ) -> ExtractedDocument:
        """Parse MinerU's pdf_info list into our normalized ExtractedDocument format.
        
        MinerU's pdf_info is a list of page dicts. Each page contains:
        - 'preproc_blocks' or 'para_blocks': list of block dicts
        - 'page_size': [width, height]
        - Each block has 'type', 'bbox' [x0, y0, x1, y1], and content fields
        """
        pages_dict = {
            i: ExtractedPage(page_number=i, source_strategy="MinerUExtractor")
            for i in range(1, total_pages + 1)
        }

        for p_idx, p_data in enumerate(pdf_info):
            page_num = p_idx + 1
            if page_num not in pages_dict:
                continue

            # Page dimensions (MinerU uses top-left origin, units in points)
            page_size = p_data.get("page_size", [595.0, 842.0])
            page_width = float(page_size[0]) if len(page_size) > 0 else 595.0
            page_height = float(page_size[1]) if len(page_size) > 1 else 842.0

            # MinerU stores blocks in 'preproc_blocks' or 'para_blocks'
            blocks = p_data.get("preproc_blocks", p_data.get("para_blocks", []))
            
            # Also check for direct 'blocks' key used in some versions
            if not blocks:
                blocks = p_data.get("blocks", [])

            reading_order = 1
            for block in blocks:
                bbox_raw = block.get("bbox", [0, 0, page_width, page_height])
                if len(bbox_raw) != 4:
                    continue

                normalized_bbox = normalize_coordinates(
                    tuple(bbox_raw),
                    page_width,
                    page_height,
                    source_origin="top_left"
                )
                
                # Ensure valid geometry (skip degenerate boxes)
                if normalized_bbox[0] >= normalized_bbox[2] or normalized_bbox[1] >= normalized_bbox[3]:
                    continue

                b_type = block.get("type", "text")

                # Extract text from various MinerU block structures
                text = self._extract_text_from_block(block)

                if b_type in ("text", "title", "paragraph", "plain_text", 0, 1):
                    if text and text.strip():
                        pages_dict[page_num].text_blocks.append(
                            TextBlock(
                                text=text.strip(),
                                bbox=normalized_bbox,
                                page_number=page_num,
                                source_strategy="MinerUExtractor",
                                reading_order=reading_order
                            )
                        )
                        reading_order += 1

                elif b_type in ("table", 5):
                    markdown = block.get("markdown", block.get("html", text or ""))
                    pages_dict[page_num].tables.append(
                        StructuredTable(
                            bbox=normalized_bbox,
                            page_number=page_num,
                            source_strategy="MinerUExtractor",
                            markdown=markdown,
                            has_headers=True
                        )
                    )

                elif b_type in ("image", "figure", 3):
                    caption = block.get("caption", None)
                    pages_dict[page_num].figures.append(
                        Figure(
                            bbox=normalized_bbox,
                            page_number=page_num,
                            source_strategy="MinerUExtractor",
                            caption=caption
                        )
                    )

        return ExtractedDocument(
            file_hash=profile.file_hash,
            pages=[pages_dict[i] for i in sorted(pages_dict.keys())]
        )

    @staticmethod
    def _extract_text_from_block(block: dict) -> str:
        """Extract text content from a MinerU block, handling nested structures.
        
        MinerU blocks can have text in various places:
        - block['text'] (simple)
        - block['lines'][*]['spans'][*]['content'] (detailed)
        - block['content'] (alternative)
        """
        # Direct text field
        if "text" in block and block["text"]:
            return block["text"]
        
        # Content field
        if "content" in block and block["content"]:
            return block["content"]

        # Nested lines/spans structure (common in MinerU middle.json)
        lines = block.get("lines", [])
        if lines:
            text_parts = []
            for line in lines:
                spans = line.get("spans", [])
                for span in spans:
                    content = span.get("content", span.get("text", ""))
                    if content:
                        text_parts.append(content)
            if text_parts:
                return " ".join(text_parts)

        return ""
