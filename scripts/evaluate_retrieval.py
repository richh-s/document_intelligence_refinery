"""
Evaluation script to quantify the benefit of the PageIndex layer.

Measures and compares:
- Retrieval precision
- Recall
- Average retrieval latency
With vs without PageIndex traversal.
"""

import time
import logging
import argparse
from typing import List, Dict, Any

from document_intelligence_refinery.chunking.models import LogicalDocumentUnit, LDUMetadata
from document_intelligence_refinery.indexing.page_index import PageIndexNode
from document_intelligence_refinery.indexing.vector_store import RefineryVectorStore
from document_intelligence_refinery.indexing.query import HybridRetriever

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def generate_mock_data() -> tuple[List[LogicalDocumentUnit], List[PageIndexNode]]:
    """Synthesize a massive mock document to test hierarchical scaling retrieval."""
    ldus = []
    nodes = []
    
    # Section 1: Financials
    nodes.append(PageIndexNode(
        section_id="sec_fin",
        section_title="Financial Report Q3",
        summary="Detailed breakdown of Q3 revenue, operating costs, and profit."
    ))
    ldus.append(LogicalDocumentUnit(
        content="Operating costs decreased by 4% due to cloud optimizations.",
        chunk_type="text",
        page_refs=[1],
        bounding_box=[0,0,1,1],
        parent_section_id="sec_fin",
        token_count=10,
        content_hash="hash_fin_1",
        metadata=LDUMetadata()
    ))
    ldus.append(LogicalDocumentUnit(
        content="Total revenue reached a record $52 million across all enterprise sectors.",
        chunk_type="text",
        page_refs=[1],
        bounding_box=[0,0,1,1],
        parent_section_id="sec_fin",
        token_count=15,
        content_hash="hash_fin_2",
        metadata=LDUMetadata()
    ))
    
    # Section 2: Security & Safety
    nodes.append(PageIndexNode(
        section_id="sec_sec",
        section_title="Security and Compliance",
        summary="Security policies, data encryption methodologies, and regulatory compliance standards."
    ))
    ldus.append(LogicalDocumentUnit(
        content="All data is encrypted at rest using AES-256 standards.",
        chunk_type="text",
        page_refs=[2],
        bounding_box=[0,0,1,1],
        parent_section_id="sec_sec",
        token_count=10,
        content_hash="hash_sec_1",
        metadata=LDUMetadata()
    ))
    
    # Inject noise (distractor LDUs) to test recall precision
    for i in range(100):
        ldus.append(LogicalDocumentUnit(
            content=f"Irrelevant boilerplate filler content block {i} spanning pages. Used to dilute the embedding space.",
            chunk_type="text",
            page_refs=[i+5],
            bounding_box=[0,0,1,1],
            parent_section_id=f"sec_noise_{i//10}",
            token_count=20,
            content_hash=f"hash_noise_{i}",
            metadata=LDUMetadata()
        ))
        
        if i % 10 == 0:
            nodes.append(PageIndexNode(
                section_id=f"sec_noise_{i//10}",
                section_title=f"Appendix Section {i//10}",
                summary=f"Various legal boilerplate and irrelevant distractor content block {i//10}."
            ))
            
    return ldus, nodes


def execute_evaluation(tmp_path: str):
    logger.info("Setting up ChromaDB Evaluation Environment...")
    vector_store = RefineryVectorStore(db_path=tmp_path)
    
    ldus, nodes = generate_mock_data()
    logger.info(f"Ingesting {len(ldus)} chunks and {len(nodes)} PageIndex summaries...")
    vector_store.ingest_ldus(ldus)
    vector_store.ingest_page_index(nodes)
    
    retriever = HybridRetriever(vector_store)
    
    # Define Ground Truth
    queries = [
        {"q": "How much was the total revenue?", "expected_ids": ["hash_fin_2"]},
        {"q": "What encryption standard is used for data at rest?", "expected_ids": ["hash_sec_1"]},
        {"q": "Did cloud operating costs improve?", "expected_ids": ["hash_fin_1"]},
    ]
    
    def run_tests(name: str, fn_call, top_k: int):
        total_precision = 0.0
        total_recall = 0.0
        total_time = 0.0
        
        for case in queries:
            start = time.perf_counter()
            if name == "Hybrid":
                results = fn_call(query=case["q"], top_section_k=3, top_chunk_k=top_k)
            else:
                results = fn_call(query=case["q"], top_k=top_k)
            latency = (time.perf_counter() - start) * 1000  # ms
            
            retrieved_ids = [r["id"] for r in results]
            expected = set(case["expected_ids"])
            retrieved = set(retrieved_ids)
            
            # True Positives
            true_pos = len(expected.intersection(retrieved))
            precision = true_pos / len(retrieved) if retrieved else 0.0
            recall = true_pos / len(expected) if expected else 0.0
            
            total_time += latency
            total_precision += precision
            total_recall += recall
            
        N = len(queries)
        logger.info(f"--- {name} Retrieval Performance ---")
        logger.info(f"Avg Precision: {total_precision/N:.2%}")
        logger.info(f"Avg Recall:    {total_recall/N:.2%}")
        logger.info(f"Avg Latency:   {total_time/N:.2f} ms")
        logger.info("-" * 40)
        
    run_tests("Baseline Global", retriever.retrieve_global_only, top_k=3)
    run_tests("Hybrid", retriever.retrieve, top_k=3)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", type=str, default="./eval_chroma_db", help="Path to ephemeral evaluation db")
    args = parser.parse_args()
    
    # Run evaluation
    execute_evaluation(args.db_path)
