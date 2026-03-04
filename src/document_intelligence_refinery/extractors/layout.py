"""Strategy B: Layout-Aware Extraction (Docling)."""

import logging
from pathlib import Path
from typing import Any

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel

from document_intelligence_refinery.models import DocumentProfile
from document_intelligence_refinery.schema import (
    ExtractedDocument,
    ExtractedPage,
    TextBlock,
    StructuredTable,
    normalize_coordinates,
)
from document_intelligence_refinery.extractors.base import BaseExtractor, ExtractionResult


logger = logging.getLogger(__name__)


class DoclingDocumentAdapter:
    """Class to convert Docling's internal schema to our normalized internal schema."""
    
    @staticmethod
    def adapt(docling_doc: DoclingDocument, profile: DocumentProfile) -> ExtractedDocument:
        pages_dict = {i: ExtractedPage(page_number=i, source_strategy="LayoutAwareExtractor") 
                      for i in range(1, profile.page_count + 1)}
        
        # Docling stores nodes in a flat list mapped to pages
        for node, data in docling_doc.iterate_items():
            if not getattr(data, "prov", None):
                continue
                
            prov = data.prov[0]
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

            if data.label in (DocItemLabel.TEXT, DocItemLabel.TITLE, DocItemLabel.SECTION_HEADER, DocItemLabel.LIST_ITEM):
                pages_dict[page_no].text_blocks.append(
                    TextBlock(
                        text=data.text,
                        bbox=normalized_bbox,
                        page_number=page_no,
                        source_strategy="LayoutAwareExtractor",
                        reading_order=len(pages_dict[page_no].text_blocks) + 1
                    )
                )
            elif data.label == DocItemLabel.TABLE:
                has_headers = any(cell.column_header for cell in data.data.grid_cells) if data.data else False
                pages_dict[page_no].tables.append(
                    StructuredTable(
                        bbox=normalized_bbox,
                        page_number=page_no,
                        source_strategy="LayoutAwareExtractor",
                        markdown=data.export_to_markdown(),
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
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = True
        
        self.converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            format_options={
                InputFormat.PDF: pipeline_options
            }
        )

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
                    
        confidence = 1.0
        if total_blocks == 0:
            confidence = 0.0 # Total failure to parse
        elif total_tables > 0:
            header_ratio = tables_with_headers / total_tables
            if header_ratio < 0.5:
                # If Docling found tables but failed to parse grid headers for most of them,
                # it's likely hallucinating or failing layout boundaries.
                confidence *= 0.7 
                
        return ExtractionResult(
            document=normalized_doc,
            confidence=round(confidence, 6),
            cost=0.0,
            time_ms=self._timer_end_ms(start_time),
            model_name="docling (cpu)",
            pages_sent=profile.page_count
        )
