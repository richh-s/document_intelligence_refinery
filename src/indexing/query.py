"""Hybrid Retrieval Safeguard."""

import logging
from typing import List, Dict, Any
from indexing.vector_store import RefineryVectorStore

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Executes a two-stage hybrid retrieval strategy:
    1. PageIndex Traversal (Top-3 Sections) -> Vector Search within sections
    2. Global Fallback (Top-3 Global LDUs)
    """
    
    def __init__(self, vector_store: RefineryVectorStore):
        self.vector_store = vector_store
        
    def retrieve(self, query: str, top_section_k: int = 3, top_chunk_k: int = 3) -> List[Dict[str, Any]]:
        """
        Executes the hierarchical + global retrieval fallback safely.
        """
        # Step 1: PageIndex Traversal
        top_sections = self.vector_store.query_page_index(query, top_k=top_section_k)
        section_ids = [s["id"] for s in top_sections]
        
        # Step 2: Vector Search within those sections
        hierarchical_results = []
        if section_ids:
            hierarchical_results = self.vector_store.query_ldus(
                query=query, 
                section_ids=section_ids, 
                top_k=top_chunk_k
            )
            
        # Step 3: Global Fallback
        global_results = self.vector_store.query_ldus(
            query=query, 
            section_ids=None, 
            top_k=top_chunk_k
        )
        
        # Merge and deduplicate by content_hash (id)
        merged = {}
        for res in hierarchical_results + global_results:
            uid = res["id"]
            if uid not in merged:
                merged[uid] = res
                
        # Sort by distance (assuming lower is better in Chroma default cosine/l2)
        sorted_results = sorted(list(merged.values()), key=lambda x: x["distance"])
        return sorted_results

    def retrieve_global_only(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Baseline embedding search comparison."""
        return self.vector_store.query_ldus(
            query=query,
            section_ids=None,
            top_k=top_k
        )
