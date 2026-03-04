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
    
    def __init__(self, tokenizer_fn: Callable[[str], int], max_tokens: int = 500):
        self.tokenizer = tokenizer_fn
        self.max_tokens = max_tokens
        
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
            # 1. Process Figures First (to capture layout)
            for fig in page.figures:
                content = fig.caption if fig.caption else "[Figure Content]"
                ldus.append(self._create_ldu(
                    content=content,
                    chunk_type="figure",
                    page_refs=[page.page_number],
                    bbox=list(fig.bbox),
                    parent_section=current_section_id
                ))
                
            # 2. Process Tables
            for idx, table in enumerate(page.tables):
                # Ensure header injection tracking if needed
                content = table.markdown
                ldus.append(self._create_ldu(
                    content=content,
                    chunk_type="table",
                    page_refs=[page.page_number],
                    bbox=list(table.bbox),
                    parent_section=current_section_id
                ))

            # 3. Process Text Blocks
            for block in page.text_blocks:
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
                    # If this is a list, save the context for the next iteration
                    if chunk_type == "list":
                        first_line = content.split("\n")[0]
                        global_context_prefix = first_line[:50]
                        
                    # Split logic (simplified naive split by period for semantic integrity)
                    sub_chunks = self._semantic_split(content, self.max_tokens)
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
                    ldu.metadata.relations.append({
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
        
        return LogicalDocumentUnit(
            content=content,
            chunk_type=chunk_type,
            page_refs=page_refs,
            bounding_box=bbox,
            parent_section_id=parent_section,
            token_count=token_count,
            content_hash=c_hash,
            metadata=md
        )

    def _semantic_split(self, text: str, max_tokens: int) -> List[str]:
        """Splits sentences without breaking them mid-way."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            s_tokens = self.tokenizer(sentence)
            if current_tokens + s_tokens > max_tokens and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_tokens = s_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += s_tokens
                
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks
