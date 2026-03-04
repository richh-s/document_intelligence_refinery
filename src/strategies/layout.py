"""Strategy B: Layout-Aware Extraction (Docling)."""

import logging
from pathlib import Path
from typing import Any

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel

from models.document_profile import DocumentProfile
from models.extracted_document import (
    ExtractedDocument,
    ExtractedPage,
    TextBlock,
    StructuredTable,
    normalize_coordinates,
)
from strategies.base import BaseExtractor, ExtractionResult


logger = logging.getLogger(__name__)


class DoclingDocumentAdapter:
    """Class to convert Docling's internal schema to our normalized internal schema."""
    
    @staticmethod
    def adapt(docling_doc: DoclingDocument, profile: DocumentProfile) -> ExtractedDocument:
        pages_dict = {i: ExtractedPage(page_number=i, source_strategy="LayoutAwareExtractor") 
                      for i in range(1, profile.page_count + 1)}
        
        # Docling stores nodes in a flat list mapped to pages
        for item, level in docling_doc.iterate_items():
            if not hasattr(item, "label"):
                continue
                
            if not getattr(item, "prov", None):
                page_no = 1
                normalized_bbox = (0.0, 0.0, 1.0, 1.0)
            else:
                prov = item.prov[0]
                page_no = prov.page_no
                if page_no not in pages_dict:
                    continue

                # Docling origin is Bottom-Left. We must normalize and apply Y-Inversion
                page_dimensions = docling_doc.pages[page_no].size
                page_width = page_dimensions.width
                page_height = page_dimensions.height
                bbox_raw = (prov.bbox.l, prov.bbox.b, prov.bbox.r, prov.bbox.t)

                normalized_bbox = normalize_coordinates(
                    bbox_raw, 
                    page_width, 
                    page_height, 
                    source_origin="bottom_left"
                )

            if item.label in (DocItemLabel.TEXT, DocItemLabel.TITLE, DocItemLabel.SECTION_HEADER, DocItemLabel.LIST_ITEM):
                pages_dict[page_no].text_blocks.append(
                    TextBlock(
                        text=item.text,
                        bbox=normalized_bbox,
                        page_number=page_no,
                        source_strategy="LayoutAwareExtractor",
                        reading_order=len(pages_dict[page_no].text_blocks) + 1
                    )
                )
            elif item.label == DocItemLabel.TABLE:
                has_headers = False
                try:
                    if item.data:
                        cells = getattr(item.data, 'table_cells', getattr(item.data, 'grid_cells', []))
                        has_headers = any(getattr(cell, 'column_header', False) for cell in cells)
                except Exception as e:
                    logger.warning(f"Failed to parse table headers on page {page_no}: {e}")
                    
                pages_dict[page_no].tables.append(
                    StructuredTable(
                        bbox=normalized_bbox,
                        page_number=page_no,
                        source_strategy="LayoutAwareExtractor",
                        markdown=item.export_to_markdown(),
                        has_headers=has_headers
                    )
                )
                
        # Fill in empty arrays for unparsed pages
        return ExtractedDocument(
            file_hash=profile.file_hash,
            pages=[pages_dict[i] for i in sorted(pages_dict.keys())]
        )


class LayoutAwareExtractor(BaseExtractor):
    """Cost: Medium. Triggers on Complex Layouts (Tables, Multi-column)."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.converter = DocumentConverter(allowed_formats=[InputFormat.PDF])

    def extract(self, pdf_path: Path, profile: DocumentProfile) -> ExtractionResult:
        start_time = self._timer_start()
        
        # Memory-Aware Circuit Breaker handled in router. 
        # But we capture and re-raise MemoryError explicitly just in case.
        try:
            conversion_result = self.converter.convert(pdf_path)
            docling_doc = conversion_result.document
        except MemoryError as e:
            logger.error(f"Docling MemoryError on {pdf_path}: {e}")
            raise
            
        normalized_doc = DoclingDocumentAdapter.adapt(docling_doc, profile)
        
        # Docling Confidence Heuristic
        # 1. Did it find text?
        # 2. Did it find tables but fail to identify headers?
        total_tables = 0
        tables_with_headers = 0
        total_blocks = 0
        
        for page in normalized_doc.pages:
            total_blocks += len(page.text_blocks)
            for table in page.tables:
                total_tables += 1
                if table.has_headers:
                    tables_with_headers += 1
                    
        confidence = 0.0
        if total_blocks > 0:
            # Baseline structural confidence scales by volume (cap at +0.8)
            confidence = min(1.0, 0.5 + (total_blocks / (profile.page_count * 50)) * 0.4)
            
        if total_tables > 0:
            header_ratio = tables_with_headers / total_tables
            if header_ratio > 0.5:
                confidence = min(1.0, confidence + 0.1)
            else:
                confidence = max(0.0, confidence - 0.1)
                
        if total_blocks == 0 and total_tables > 0:
            confidence = 0.8  # Edge case: Valid page of pure tables
            
        confidence = max(0.0, min(1.0, confidence))
                
        return ExtractionResult(
            document=normalized_doc,
            confidence=round(confidence, 6),
            cost=0.0,
            time_ms=self._timer_end_ms(start_time),
            model_name="docling (cpu)",
            pages_sent=profile.page_count
        )
