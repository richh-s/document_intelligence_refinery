"""Normalized extraction schemas for the Document Intelligence Refinery.

Provides standard minimum contracts for all strategy engines.
"""

from typing import Any
from pydantic import BaseModel, Field, model_validator


class TextBlock(BaseModel):
    """A snippet of text extracted from a page."""

    text: str
    bbox: tuple[float, float, float, float]
    page_number: int  # 1-indexed
    source_strategy: str
    reading_order: int
    column_id: int | None = None

    @model_validator(mode="after")
    def validate_bounds(self) -> "TextBlock":
        x0, y0, x1, y1 = self.bbox
        if not (0.0 <= x0 <= 1.0 and 0.0 <= y0 <= 1.0 and 0.0 <= x1 <= 1.0 and 0.0 <= y1 <= 1.0):
            raise ValueError(f"Bounding box coords must be normalized [0, 1]. Got: {self.bbox}")
        if x0 >= x1 or y0 >= y1:
            raise ValueError(f"Invalid bounding box geometry: {self.bbox}")
        return self


class StructuredTable(BaseModel):
    """A tabular data structure extracted from a page."""

    bbox: tuple[float, float, float, float]
    page_number: int
    source_strategy: str
    markdown: str
    has_headers: bool = False


class Figure(BaseModel):
    """An image or figure representation extracted from a page."""

    bbox: tuple[float, float, float, float]
    page_number: int
    source_strategy: str
    caption: str | None = None


class ExtractedPage(BaseModel):
    """Normalized representation of a single page of extraction data."""

    page_number: int
    source_strategy: str
    text_blocks: list[TextBlock] = Field(default_factory=list)
    tables: list[StructuredTable] = Field(default_factory=list)
    figures: list[Figure] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def reconstruct_reading_order(self, gutter_threshold: float = 0.05) -> None:
        """
        Reconstruct reading order using an XY-Cut algorithm derivative.
        Sorts blocks into Column Containers based on horizontal gutters,
        then orders them top-to-bottom within each column.
        """
        if not self.text_blocks:
            return

        # Sort by x0 to sweep left-to-right and find column gutters
        sorted_by_x = sorted(self.text_blocks, key=lambda b: (b.bbox[0], b.bbox[1]))
        
        columns = []
        current_col = [sorted_by_x[0]]
        
        for block in sorted_by_x[1:]:
            # If the gap between current block's x0 and the column's max x1 > gutter_threshold, it's a new column
            max_x1 = max(b.bbox[2] for b in current_col)
            if block.bbox[0] - max_x1 > gutter_threshold:
                columns.append(current_col)
                current_col = [block]
            else:
                current_col.append(block)
        columns.append(current_col)
        
        # Assign reading order and column_id top-to-bottom per column
        global_order = 1
        for col_idx, col_blocks in enumerate(columns):
            # Sort top-to-bottom
            col_blocks.sort(key=lambda b: b.bbox[1])
            for block in col_blocks:
                block.column_id = col_idx
                block.reading_order = global_order
                global_order += 1


class ExtractedDocument(BaseModel):
    """The unified standard payload produced by all Extraction Engines."""

    file_hash: str
    pages: list[ExtractedPage] = Field(default_factory=list)


def normalize_coordinates(
    bbox: tuple[float, float, float, float],
    page_width: float,
    page_height: float,
    source_origin: str = "top_left",
) -> tuple[float, float, float, float]:
    """Convert engine-specific coordinates to a [0.0, 1.0] normalized space.

    Args:
        bbox: (x0, y0, x1, y1) in raw engine points/pixels.
        page_width: Raw page width.
        page_height: Raw page height.
        source_origin: "top_left" (e.g., pdfplumber) or "bottom_left" (e.g., Docling).

    Returns:
        (x0_norm, y0_norm, x1_norm, y1_norm) bounded [0.0, 1.0] with top-left origin.
    """
    if page_width <= 0 or page_height <= 0:
        return (0.0, 0.0, 0.0, 0.0)

    x0_raw, y0_raw, x1_raw, y1_raw = bbox

    # Normalize roughly to [0, 1]
    x0_norm = max(0.0, min(1.0, x0_raw / page_width))
    x1_norm = max(0.0, min(1.0, x1_raw / page_width))
    y0_norm = max(0.0, min(1.0, y0_raw / page_height))
    y1_norm = max(0.0, min(1.0, y1_raw / page_height))

    # Handle Y-Axis Inversion based on origin Protocol
    if source_origin == "bottom_left":
        # In bottom_left, higher Y is further UP the page.
        # Top-left norm requires higher Y to be further DOWN the page.
        y0_inverted = 1.0 - y1_norm  # Old top becomes new top
        y1_inverted = 1.0 - y0_norm  # Old bottom becomes new bottom
        y0_norm, y1_norm = y0_inverted, y1_inverted

    # Ensure well-formed geometry
    return (
        min(x0_norm, x1_norm),
        min(y0_norm, y1_norm),
        max(x0_norm, x1_norm),
        max(y0_norm, y1_norm),
    )
