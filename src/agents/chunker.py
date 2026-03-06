"""Semantic Chunking Engine for Logical Document Units."""

import re
import logging
from typing import Callable, List, Optional, Tuple, Dict, Literal
from models.extracted_document import ExtractedDocument
from models.ldu import LogicalDocumentUnit, LDUMetadata
from chunking.hasher import generate_ldu_hash

logger = logging.getLogger(__name__)


class ChunkingEngine:
    """
    Transforms raw extracted content into RAG-safe Logical Document Units (LDUs).
    Enforces precise contextual, structural, and relational rules.
    """
    
    def __init__(self, tokenizer_fn: Callable[[str], int], max_tokens: int = 600, overlap_tokens: int = 50):
        self.tokenizer = tokenizer_fn
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        
        # Cross-reference regex patterns
        self.ref_pattern = re.compile(r"(?i)(?:see\s+|as\s+shown\s+in\s+)(table|figure|section)\s+([a-zA-Z0-9.\-]+)")
        
        # Table explanation regex pattern
        self.table_rel_pattern = re.compile(r"(?i)(?:as\s+shown\s+in\s+table|table)\s+([a-zA-Z0-9.\-]+)\s+(?:shows|illustrates|demonstrates|details)")

    def process_document(self, doc: ExtractedDocument) -> List[LogicalDocumentUnit]:
        """Convert an ExtractedDocument into an ordered list of LDUs."""
        ldus: List[LogicalDocumentUnit] = []
        
        # Context tracking
        current_section_id: Optional[str] = None
        global_context_prefix: Optional[str] = None
        
        for page in doc.pages:
            # Reconstruct reading order before processing
            page.reconstruct_reading_order()
            
            # 1. Process Figures First
            for fig in page.figures:
                content = fig.caption if fig.caption else "[Figure Content]"
                ldus.append(self._create_ldu(
                    content=content,
                    chunk_type="figure",
                    page_refs=[page.page_number],
                    bbox=list(fig.bbox),
                    parent_section=current_section_id
                ))
                
            # 2. Process Tables with Header-Injected Row Chunking
            for table in page.tables:
                content = table.markdown
                token_count = self.tokenizer(content)
                if token_count > self.max_tokens:
                    table_chunks = self._table_header_injected_split(content, self.max_tokens)
                    for tc in table_chunks:
                        ldus.append(self._create_ldu(
                            content=tc,
                            chunk_type="table",
                            page_refs=[page.page_number],
                            bbox=list(table.bbox),
                            parent_section=current_section_id
                        ))
                else:
                    ldus.append(self._create_ldu(
                        content=content,
                        chunk_type="table",
                        page_refs=[page.page_number],
                        bbox=list(table.bbox),
                        parent_section=current_section_id
                    ))

            # 3. Process Text Blocks (now sorted by reading_order)
            # Ensure text blocks are sorted by reading order
            sorted_blocks = sorted(page.text_blocks, key=lambda b: b.reading_order)
            
            for block in sorted_blocks:
                content = block.text.strip()
                if not content:
                    continue
                    
                chunk_type = "text"
                
                # Naive Header Detection (could be improved by looking at font size/boldness if available)
                if len(content) < 100 and content.istitle() and not content.endswith("."):
                    chunk_type = "header"
                    current_section_id = f"section_{len(ldus)}"
                
                # Naive List Detection
                is_list = re.match(r"^(\d+\.|-|\*)\s+", content)
                if is_list:
                    chunk_type = "list"
                    if global_context_prefix:
                        content = f"[Context: {global_context_prefix} (Continued)]\n{content}"
                        # Reset for next item unless it overflows
                        global_context_prefix = None
                
                # Check for table relations and cross-references
                metadata = LDUMetadata()
                
                # Table-to-Text Relationship
                table_rel_match = self.table_rel_pattern.search(content)
                if table_rel_match:
                    metadata.related_table_id = f"table_{table_rel_match.group(1).lower()}"
                    
                # Cross-Reference Detection
                ref_match = self.ref_pattern.search(content)
                if ref_match:
                    ref_type = ref_match.group(1).lower()
                    ref_val = ref_match.group(2).lower()
                    metadata.cross_reference = f"{ref_type}_{ref_val}"
                    metadata.cross_reference_type = ref_type
                    # Mark as dangling until resolved later
                    metadata.dangling_reference = metadata.cross_reference

                # Handle Token Overflows (Contextual Prepending for Lists/Text)
                token_count = self.tokenizer(content)
                if token_count > self.max_tokens:
                    if chunk_type == "list":
                        # Emergency list splitter
                        first_line = content.split("\n")[0]
                        global_context_prefix = first_line[:50]
                        sub_chunks = self._emergency_list_split(content, self.max_tokens)
                    else:
                        sub_chunks = self._semantic_split(content, self.max_tokens, self.overlap_tokens)
                        
                    for i, sc in enumerate(sub_chunks):
                        # Prepend context to split overflow chunks
                        if i > 0 and chunk_type == "list" and global_context_prefix:
                            sc = f"[Context: {global_context_prefix} (Continued)]\n{sc}"
                            
                        ldus.append(self._create_ldu(
                            content=sc,
                            chunk_type=chunk_type,
                            page_refs=[page.page_number],
                            bbox=list(block.bbox),
                            parent_section=current_section_id,
                            metadata=metadata.model_copy()
                        ))
                else:
                    ldus.append(self._create_ldu(
                        content=content,
                        chunk_type=chunk_type,
                        page_refs=[page.page_number],
                        bbox=list(block.bbox),
                        parent_section=current_section_id,
                        metadata=metadata
                    ))

        # 4. Cross-Reference Resolution Pass
        for ldu in ldus:
            if ldu.metadata.cross_reference:
                ref_type = ldu.metadata.cross_reference_type
                ref_val = ldu.metadata.cross_reference.split('_')[-1]
                
                # Find matching target LDU
                target_hash = None
                for target_ldu in ldus:
                    # Match by chunk type and a textual presence of the ref
                    # e.g., if looking for 'table 3', check if 'table 3' is in the table LDU content.
                    if target_ldu.chunk_type == ref_type:
                        search_str = f"{ref_type} {ref_val}".lower()
                        if search_str in target_ldu.content.lower():
                            target_hash = target_ldu.content_hash
                            break
                            
                if target_hash:
                    ldu.metadata.chunk_relationships.append({
                        "type": "refers_to",
                        "target_content_hash": target_hash
                    })
                    ldu.metadata.dangling_reference = None

        return ldus

    def _create_ldu(
        self, 
        content: str, 
        chunk_type: Literal["text", "table", "list", "figure", "header"],
        page_refs: List[int],
        bbox: List[float],
        parent_section: Optional[str],
        metadata: Optional[LDUMetadata] = None
    ) -> LogicalDocumentUnit:
        
        md = metadata if metadata else LDUMetadata()
        
        # Enforce exact token count mapping
        token_count = self.tokenizer(content)
        
        # Generate spatial content hash
        c_hash = generate_ldu_hash(content, bbox, page_refs, chunk_type)
        
        # Use a bounding box kwargs dict to map correctly
        bbox_obj = {"x0": bbox[0], "y0": bbox[1], "x1": bbox[2], "y1": bbox[3]}
        
        return LogicalDocumentUnit(
            content=content,
            chunk_type=chunk_type,
            page_refs=page_refs,
            bounding_box=bbox_obj,
            parent_section_id=parent_section,
            token_count=token_count,
            content_hash=c_hash,
            metadata=md
        )

    def _semantic_split(self, text: str, max_tokens: int, overlap_tokens: int) -> List[str]:
        """Splits sentences with a 'soft' overlap buffer."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            s_tokens = self.tokenizer(sentence)
            if current_tokens + s_tokens > max_tokens and current_chunk:
                chunks.append(" ".join(current_chunk))
                # Create overlap buffer from the last sentence(s)
                current_chunk = [current_chunk[-1]] if current_chunk else []
                current_tokens = self.tokenizer(" ".join(current_chunk))
                
                # Add current sentence to new chunk
                current_chunk.append(sentence)
                current_tokens += s_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += s_tokens
                
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks

    def _table_header_injected_split(self, markdown_table: str, max_tokens: int) -> List[str]:
        """Splits large markdown tables, forcing header repetition on each chunk."""
        lines = markdown_table.strip().split("\n")
        if len(lines) < 3:
            return [markdown_table]
            
        headers = lines[:2] # header text + separator line
        rows = lines[2:]
        
        chunks = []
        current_chunk = list(headers)
        current_tokens = self.tokenizer("\n".join(current_chunk))
        
        for row in rows:
            r_tokens = self.tokenizer(row)
            if current_tokens + r_tokens > max_tokens and len(current_chunk) > 2:
                chunks.append("\n".join(current_chunk))
                current_chunk = list(headers)
                current_tokens = self.tokenizer("\n".join(current_chunk))
            
            current_chunk.append(row)
            current_tokens += r_tokens
            
        if len(current_chunk) > 2:
            chunks.append("\n".join(current_chunk))
            
        return chunks

    def _emergency_list_split(self, text: str, max_tokens: int) -> List[str]:
        """Splits lists precisely by list item boundaries."""
        # Find all list item lines using bullet or number regex
        lines = text.split("\n")
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for line in lines:
            l_tokens = self.tokenizer(line)
            # If line is a new list item and chunk exceeds limit, split here
            is_item_boundary = bool(re.match(r"^(\d+\.|-|\*)\s+", line.strip()))
            
            if is_item_boundary and current_tokens + l_tokens > max_tokens and current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = [line]
                current_tokens = l_tokens
            else:
                current_chunk.append(line)
                current_tokens += l_tokens
                
        if current_chunk:
            chunks.append("\n".join(current_chunk))
            
        return chunks
