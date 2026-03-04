"""Strategy C: Vision-Augmented Extraction (VLM via OpenRouter)."""

import httpx
import json
import logging
import base64
from pathlib import Path
from typing import Any
import pypdfium2 as pdfium

from document_intelligence_refinery.models import DocumentProfile
from document_intelligence_refinery.schema import (
    ExtractedDocument,
    ExtractedPage,
    TextBlock,
    StructuredTable,
    Figure,
)
from document_intelligence_refinery.extractors.base import BaseExtractor, ExtractionResult, PartialExtractionResult

logger = logging.getLogger(__name__)

class BudgetExceededError(Exception):
    """Raised when document extraction costs exceed the global budget cap."""
    pass


def strip_markdown_json(raw_text: str) -> str:
    """Robust JSON Sanitization: Extracts JSON content from Markdown code blocks."""
    text = raw_text.strip()
    if text.startswith("```"):
        # Find the first { or [ after the first line
        lines = text.split("\n")
        if len(lines) > 1:
            try:
                first_brace = next(i for i, line in enumerate(lines) if "{" in line or "[" in line)
                last_brace = next(i for i in range(len(lines)-1, -1, -1) if "}" in lines[i] or "]" in lines[i])
                return "\n".join(lines[first_brace:last_brace+1])
            except StopIteration:
                pass
    
    # Fallback brute-force search
    start_idx = text.find("{")
    end_idx = text.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        return text[start_idx:end_idx+1]
        
    start_idx_list = text.find("[")
    end_idx_list = text.rfind("]")
    if start_idx_list != -1 and end_idx_list != -1 and end_idx_list > start_idx_list:
        return text[start_idx_list:end_idx_list+1]
        
    return text


class VisionExtractor(BaseExtractor):
    """Cost: High. Triggers on Scanned, Handwritten, or Fallbacks."""
    
    # Standard System Prompt forcing JSON output
    SYSTEM_PROMPT = """You are a precision document extraction engine.
You will be provided with images of document pages.
Extract all text, tables, and figures precisely as they appear.
Do NOT attempt to complete sentences that span across pages. Focus only on what is optically visible.

You MUST respond strictly with a valid JSON object matching this schema:
{
  "pages": [
    {
      "page_number": 1,
      "text_blocks": [
        {"text": "Extracted string", "bbox": [0.0, 0.0, 1.0, 1.0]}
      ],
      "tables": [
        {"markdown_table": "| Header |\\n|---|", "has_headers": true, "bbox": [0.0, 0.0, 1.0, 1.0]}
      ],
      "figures": [
        {"caption": "Figure 1", "bbox": [0.0, 0.0, 1.0, 1.0]}
      ]
    }
  ]
}
Return ONLY valid JSON. No conversational filler. Bounding boxes must be [x0, y0, x1, y1] normalized from 0.0 to 1.0 relative to the page dimensions.
"""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.max_pages_per_call = int(self.config.get("MAX_PAGES_PER_VLM_CALL", 5))
        self.budget_cap_usd = float(self.config.get("GLOBAL_DOCUMENT_BUDGET_USD", 0.05))
        self.avg_tokens_per_page = int(self.config.get("AVG_TOKENS_PER_PAGE_IMAGE", 2000))
        self.prompt_tokens = int(self.config.get("PROMPT_TOKENS", 500))
        
        # Default pricing: GPT-4o-mini equivalents via OpenRouter
        self.price_input_1m = float(self.config.get("VLM_PRICE_INPUT_1M", 0.15))
        self.price_output_1m = float(self.config.get("VLM_PRICE_OUTPUT_1M", 0.60))
        self.model_name = self.config.get("VLM_MODEL_NAME", "openai/gpt-4o-mini")
        self.api_key = self.config.get("OPENROUTER_API_KEY", "")

    def _estimate_cost(self, num_pages: int) -> float:
        """Pre-Flight cost estimation."""
        est_input_tokens = (num_pages * self.avg_tokens_per_page) + self.prompt_tokens
        # Assume output tokens are roughly 50% of input text density, roughly 500 per page
        est_output_tokens = num_pages * 500
        
        cost_input = (est_input_tokens / 1_000_000) * self.price_input_1m
        cost_output = (est_output_tokens / 1_000_000) * self.price_output_1m
        
        return cost_input + cost_output

    def _calculate_actual_cost(self, tokens_in: int, tokens_out: int) -> float:
        cost_input = (tokens_in / 1_000_000) * self.price_input_1m
        cost_output = (tokens_out / 1_000_000) * self.price_output_1m
        return cost_input + cost_output

    def extract(self, pdf_path: Path, profile: DocumentProfile) -> ExtractionResult:
        start_time = self._timer_start()
        
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is not configured.")
            
        total_pages = profile.page_count
        
        # Pre-flight Budget Guard
        estimated_total_cost = self._estimate_cost(total_pages)
        if estimated_total_cost > self.budget_cap_usd:
            logger.error(f"BudgetExceededError: Est limit ${estimated_total_cost:.4f} > Cap ${self.budget_cap_usd:.4f}")
            raise BudgetExceededError(f"Estimated cost ${estimated_total_cost:.4f} exceeds budget cap of ${self.budget_cap_usd:.2f}")

        # Render PDF to Images using pypdfium2 (fastest)
        pdf_images = []
        pdf = pdfium.PdfDocument(pdf_path)
        for i in range(total_pages):
            page = pdf[i]
            # 72 DPI rendering is usually enough for VLM, or maybe 150 for dense tables
            bitmap = page.render(scale=2.0) 
            pil_image = bitmap.to_pil()
            pdf_images.append(pil_image)
        pdf.close()
        
        all_extracted_pages = []
        accumulated_cost = 0.0
        total_tokens_in = 0
        total_tokens_out = 0
        
        pages_processed = 0
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Context Window Batching loop
        for batch_start in range(0, total_pages, self.max_pages_per_call):
            batch_end = min(batch_start + self.max_pages_per_call, total_pages)
            batch_images = pdf_images[batch_start:batch_end]
            
            # Encode images to base64 payload
            content_payload = [{"type": "text", "text": "Extract these pages."}]
            import io
            for idx, img in enumerate(batch_images):
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=85)
                b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
                content_payload.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64_str}"
                    }
                })

            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": content_payload}
                ],
                "response_format": {"type": "json_object"}
            }

            try:
                with httpx.Client(timeout=120.0) as client:
                    resp = client.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    
                    usage = data.get("usage", {})
                    batch_in = usage.get("prompt_tokens", 0)
                    batch_out = usage.get("completion_tokens", 0)
                    total_tokens_in += batch_in
                    total_tokens_out += batch_out
                    
                    batch_cost = self._calculate_actual_cost(batch_in, batch_out)
                    accumulated_cost += batch_cost
                    
                    # Robust JSON Sanitization
                    raw_content = data["choices"][0]["message"]["content"]
                    clean_json = strip_markdown_json(raw_content)
                    
                    try:
                        parsed_data = json.loads(clean_json)
                        
                        for p_data in parsed_data.get("pages", []):
                            page_num = p_data.get("page_number", batch_start + 1)
                            
                            t_blocks = [
                                TextBlock(
                                    text=b.get("text", ""),
                                    bbox=tuple(b.get("bbox", [0.0, 0.0, 1.0, 1.0])),
                                    page_number=page_num,
                                    source_strategy="VisionExtractor"
                                ) for b in p_data.get("text_blocks", [])
                            ]
                            
                            tables = [
                                StructuredTable(
                                    bbox=tuple(t.get("bbox", [0.0, 0.0, 1.0, 1.0])),
                                    page_number=page_num,
                                    source_strategy="VisionExtractor",
                                    markdown=t.get("markdown_table", ""),
                                    has_headers=t.get("has_headers", False)
                                ) for t in p_data.get("tables", [])
                            ]
                            
                            figures = [
                                Figure(
                                    bbox=tuple(f.get("bbox", [0.0, 0.0, 1.0, 1.0])),
                                    page_number=page_num,
                                    source_strategy="VisionExtractor",
                                    caption=f.get("caption", None)
                                ) for f in p_data.get("figures", [])
                            ]
                            
                            all_extracted_pages.append(ExtractedPage(
                                page_number=page_num,
                                source_strategy="VisionExtractor",
                                text_blocks=t_blocks,
                                tables=tables,
                                figures=figures,
                                metadata={"batch_cost": batch_cost}
                            ))
                            pages_processed += 1
                            
                    except json.JSONDecodeError as e:
                        logger.error(f"VLM JSON parsing failed for batch {batch_start}-{batch_end}: {e}")
                        # On hard JSON failure, we break and trigger Partial Success
                        break
                        
            except httpx.HTTPError as e:
                logger.error(f"HTTP Error calling OpenRouter: {e}")
                break

            # Check if mid-flight budget cap is breached
            if accumulated_cost >= self.budget_cap_usd:
                logger.warning(f"Budget capped hit during processing. Paid: ${accumulated_cost:.4f}. Preserving parsed data.")
                break
                
        # Synthesize final document 
        doc = ExtractedDocument(
            file_hash=profile.file_hash,
            pages=all_extracted_pages
        )
        
        # Calculate Vision Confidence (JSON Structural validation penalty)
        confidence = 1.0
        # If we didn't parse all pages, apply a heavy continuity penalty 
        if pages_processed < total_pages:
            penalty = pages_processed / max(1, total_pages)
            confidence *= penalty
            
        kwargs = {
            "document": doc,
            "confidence": round(confidence, 6),
            "cost": round(accumulated_cost, 6),
            "time_ms": self._timer_end_ms(start_time),
            "model_name": self.model_name,
            "tokens_in": total_tokens_in,
            "tokens_out": total_tokens_out,
            "pages_sent": pages_processed
        }
        
        # Return Partial if we bailed out early to preserve the pipeline
        if pages_processed < total_pages and len(all_extracted_pages) > 0:
            return PartialExtractionResult(**kwargs)

        return ExtractionResult(**kwargs)
