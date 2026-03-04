import pytest
from unittest.mock import patch, MagicMock
from models.page_index import PageIndexBuilder
from models.ldu import LogicalDocumentUnit, LDUMetadata

@patch("models.page_index.httpx.Client")
def test_page_index_builder(mock_client_class):
    # Setup mock
    mock_client = mock_client_class.return_value.__enter__.return_value
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "Mocked summary for the section."}}]
    }
    mock_client.post.return_value = mock_resp
    
    # Create test LDUs
    ldu1 = LogicalDocumentUnit(
        content="Section 1 Header",
        chunk_type="header",
        page_refs=[1],
        bounding_box=[0.1, 0.1, 0.9, 0.2],
        parent_section_id="sec_1",
        token_count=3,
        content_hash="hash1",
        metadata=LDUMetadata()
    )
    ldu2 = LogicalDocumentUnit(
        content="Some text inside section 1.",
        chunk_type="text",
        page_refs=[1],
        bounding_box=[0.1, 0.2, 0.9, 0.3],
        parent_section_id="sec_1",
        token_count=5,
        content_hash="hash2",
        metadata=LDUMetadata()
    )
    
    builder = PageIndexBuilder(api_key="test_key")
    nodes = builder.build_index([ldu1, ldu2])
    
    assert len(nodes) == 1
    node = nodes[0]
    assert node.section_id == "sec_1"
    assert node.section_title == "Section 1 Header"
    assert node.summary == "Mocked summary for the section."
