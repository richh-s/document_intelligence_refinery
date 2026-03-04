import pytest
import chromadb
from unittest.mock import patch, MagicMock
from document_intelligence_refinery.chunking.models import LogicalDocumentUnit, LDUMetadata
from document_intelligence_refinery.indexing.page_index import PageIndexNode
from document_intelligence_refinery.indexing.vector_store import RefineryVectorStore
from document_intelligence_refinery.indexing.query import HybridRetriever

@pytest.fixture
def mock_vector_store(tmp_path):
    """Creates a temporary isolated ChromaDB store."""
    import warnings
    warnings.filterwarnings("ignore")
    
    # We create an ephemeral client for speed to avoid writing to disk
    store = RefineryVectorStore(db_path=str(tmp_path), embedding_model="all-MiniLM-L6-v2")
    # Patch the PersistentClient back to Ephemeral for this test lifecycle 
    # to avoid disk lock issues if concurrency occurs, though tmp_path should be safe
    return store

def test_ingestion_and_retrieval(mock_vector_store):
    # Setup test 
    ldu1 = LogicalDocumentUnit(
        content="The quarterly revenue reached $50M in Q2.",
        chunk_type="text",
        page_refs=[1],
        bounding_box=[0,0,1,1],
        parent_section_id="sec_1",
        token_count=10,
        content_hash="h1",
        metadata=LDUMetadata()
    )
    ldu2 = LogicalDocumentUnit(
        content="Our primary focus is safety procedures and risk management.",
        chunk_type="text",
        page_refs=[2],
        bounding_box=[0,0,1,1],
        parent_section_id="sec_2",
        token_count=10,
        content_hash="h2",
        metadata=LDUMetadata()
    )
    
    mock_vector_store.ingest_ldus([ldu1, ldu2])
    
    # Simple global retrieval
    results = mock_vector_store.query_ldus(query="financial revenue", top_k=1)
    assert len(results) == 1
    assert results[0]["id"] == "h1"
    
def test_hybrid_retriever_logic(mock_vector_store):
    # Setup LDUs
    ldu1 = LogicalDocumentUnit(
        content="Financial earnings detail goes here.",
        chunk_type="text",
        page_refs=[1],
        bounding_box=[0,0,1,1],
        parent_section_id="sec_1",
        token_count=10,
        content_hash="h1",
        metadata=LDUMetadata()
    )
    
    # Setup PageIndex Node summarizing the section
    node1 = PageIndexNode(
        section_id="sec_1",
        section_title="Financial Quarter Review",
        summary="A summary of our Q3 financial earnings and revenue streams."
    )
    
    mock_vector_store.ingest_ldus([ldu1])
    mock_vector_store.ingest_page_index([node1])
    
    retriever = HybridRetriever(mock_vector_store)
    
    # Test retrieving
    results = retriever.retrieve(query="earnings", top_section_k=1, top_chunk_k=3)
    
    assert len(results) == 1
    assert results[0]["id"] == "h1"
    
    # Assert Point 10 constraints: LDU metadata is properly attached and returned
    metadata = results[0].get("metadata", {})
    assert metadata.get("content_hash") == "h1"
    assert metadata.get("parent_section_id") == "sec_1"
    assert metadata.get("chunk_type") == "text"
