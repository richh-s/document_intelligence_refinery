"""Tests for the Semantic Chunking Engine."""

import pytest
from chunking.engine import ChunkingEngine
from chunking.validator import ChunkValidator, ChunkValidationError
from models.extracted_document import ExtractedDocument, ExtractedPage, TextBlock, StructuredTable, Figure

def dummy_tokenizer(text: str) -> int:
    return len(text.split())

def test_engine_semantic_splitting():
    engine = ChunkingEngine(tokenizer_fn=dummy_tokenizer, max_tokens=10)
    
    # Text block with 15 words
    text = "This is a simple sentence. This is another sentence that pushes us over the limit."
    
    block = TextBlock(
        text=text,
        bbox=(0.1, 0.1, 0.9, 0.2),
        page_number=1,
        source_strategy="test",
        reading_order=1
    )
    
    page = ExtractedPage(
        page_number=1,
        source_strategy="test",
        text_blocks=[block]
    )
    
    doc = ExtractedDocument(
        file_hash="testhash",
        pages=[page]
    )
    
    ldus = engine.process_document(doc)
    # Should be split into 2 LDUs since max_tokens=10 and the text has 15 words
    assert len(ldus) == 2
    assert "This is a simple sentence." in ldus[0].content
    assert "This is another sentence" in ldus[1].content


def test_validator_hallucination_check():
    validator = ChunkValidator(dummy_tokenizer)
    engine = ChunkingEngine(tokenizer_fn=dummy_tokenizer)
    
    doc = ExtractedDocument(
        file_hash="testhash",
        pages=[
            ExtractedPage(
                page_number=1,
                source_strategy="test",
                text_blocks=[
                    TextBlock(
                        text="Valid text here",
                        bbox=(0.0, 0.0, 0.5, 0.5),
                        page_number=1,
                        source_strategy="test",
                        reading_order=1
                    )
                ]
            )
        ]
    )
    
    ldus = engine.process_document(doc)
    assert len(ldus) == 1
    
    # Store original and validate
    ldu = ldus[0]
    original_hash = ldu.content_hash
    validator.verify_post_transformation_hash(original_hash, ldu)
    
    # Change the content
    ldu.content = "Invalid text here"
    with pytest.raises(ChunkValidationError, match="Post-Transformation Hash Verification failed"):
        validator.verify_post_transformation_hash(original_hash, ldu)

def test_xy_cut_reconstruction():
    # Blocks out of order but horizontally aligned into 2 columns
    b1 = TextBlock(text="Col1 Top", bbox=(0.1, 0.1, 0.4, 0.2), page_number=1, source_strategy="test", reading_order=0)
    b2 = TextBlock(text="Col2 Top", bbox=(0.6, 0.1, 0.9, 0.2), page_number=1, source_strategy="test", reading_order=0)
    b3 = TextBlock(text="Col1 Bottom", bbox=(0.1, 0.3, 0.4, 0.4), page_number=1, source_strategy="test", reading_order=0)
    
    page = ExtractedPage(
        page_number=1,
        source_strategy="test",
        text_blocks=[b2, b3, b1]  # Shuffled
    )
    
    page.reconstruct_reading_order()
    
    # Check column grouping and top-to-bottom assignment
    assert page.text_blocks[0].text == "Col2 Top"  # It mutated original list? No, reconstruct mutates the blocks inside, reading_order will be set
    
    # Re-sort to check final reading order
    ordered = sorted(page.text_blocks, key=lambda b: b.reading_order)
    assert ordered[0].text == "Col1 Top"
    assert ordered[0].column_id == 0
    assert ordered[1].text == "Col1 Bottom"
    assert ordered[1].column_id == 0
    assert ordered[2].text == "Col2 Top"
    assert ordered[2].column_id == 1
