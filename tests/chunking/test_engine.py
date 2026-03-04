import pytest
from document_intelligence_refinery.schema import ExtractedDocument, ExtractedPage, TextBlock, StructuredTable
from document_intelligence_refinery.chunking.engine import ChunkingEngine

def dummy_tokenizer(text: str) -> int:
    return len(text.split())

def test_engine_cross_reference_detection():
    engine = ChunkingEngine(tokenizer_fn=dummy_tokenizer, max_tokens=100)
    
    page = ExtractedPage(
        page_number=1,
        source_strategy="test",
        text_blocks=[
            TextBlock(text="As detailed earlier, see Table 3 for the financial analysis.", bbox=(0.1, 0.1, 0.5, 0.5), page_number=1, source_strategy="test", reading_order=1)
        ]
    )
    doc = ExtractedDocument(file_hash="xyz", pages=[page])
    
    ldus = engine.process_document(doc)
    assert len(ldus) == 1
    metadata = ldus[0].metadata
    assert metadata.cross_reference == "table_3"
    assert metadata.cross_reference_type == "table"
    assert metadata.dangling_reference == "table_3"

def test_engine_table_to_text_relationship():
    engine = ChunkingEngine(tokenizer_fn=dummy_tokenizer, max_tokens=100)
    
    page = ExtractedPage(
        page_number=1,
        source_strategy="test",
        text_blocks=[
            TextBlock(text="As shown in Table 4 illustrates the drop in revenue.", bbox=(0.1, 0.1, 0.5, 0.5), page_number=1, source_strategy="test", reading_order=1)
        ]
    )
    doc = ExtractedDocument(file_hash="xyz", pages=[page])
    
    ldus = engine.process_document(doc)
    assert len(ldus) == 1
    metadata = ldus[0].metadata
    assert metadata.related_table_id == "table_4"

def test_engine_contextual_list_prepending():
    engine = ChunkingEngine(tokenizer_fn=dummy_tokenizer, max_tokens=5)
    
    # Define a list that is too long (gt 5 tokens) to fit in one chunk
    list_text = "1. Safety Procedures\nAlways wear your hard hat.\nNever run near machinery.\nReport safety issues."
    
    page = ExtractedPage(
        page_number=1,
        source_strategy="test",
        text_blocks=[
            TextBlock(text=list_text, bbox=(0.1, 0.1, 0.5, 0.5), page_number=1, source_strategy="test", reading_order=1)
        ]
    )
    doc = ExtractedDocument(file_hash="xyz", pages=[page])
    
    ldus = engine.process_document(doc)
    # The first chunk should contain the first 5 words
    # The engine splits sentences.
    # 1. Safety Procedures -> 3 words
    # Always wear your hard hat. -> 5 words. So it will be split.
    # The continuation chunk must have the context prefix.
    
    assert len(ldus) >= 2
    assert "Context:" in ldus[1].content
    assert "Safety Procedures" in ldus[1].content
