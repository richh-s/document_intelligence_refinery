import pytest
import os
import json
from unittest.mock import MagicMock
from agents.indexer import PageIndexBuilder, PageIndexNode
from models.ldu import LogicalDocumentUnit, LDUMetadata

@pytest.fixture
def mock_builder():
    return PageIndexBuilder(api_key="test_key")

def create_ldu(content: str, chunk_type: str, parent_section: str = None) -> LogicalDocumentUnit:
    return LogicalDocumentUnit(
        content=content,
        chunk_type=chunk_type,
        page_refs=[1],
        bounding_box={"x0": 0, "y0": 0, "x1": 1, "y1": 1},
        parent_section=parent_section,
        token_count=len(content.split()),
        content_hash=f"hash_{content}",
        metadata=LDUMetadata()
    )

def test_hierarchical_build(mock_builder):
    # Mock LLM response
    mock_builder._generate_batched_summaries = MagicMock(return_value={
        "section_1": {"summary": "Main section summary", "key_entities": ["Alpha"], "data_types_present": ["Table"]},
        "section_1.1": {"summary": "Sub section summary", "key_entities": ["Beta"], "data_types_present": ["Figure"]}
    })
    
    ldus = [
        create_ldu("Main Title", "header", parent_section="section_1"),
        create_ldu("Sub Title", "header", parent_section="section_1.1")
    ]
    
    nodes = mock_builder.build_index(ldus)
    
    assert len(nodes) == 1 # Only section_1 is root
    root = nodes[0]
    assert root.section_id == "section_1"
    assert len(root.child_sections) == 1
    assert root.child_sections[0].section_id == "section_1.1"
    assert "Alpha" in root.key_entities
    assert "Beta" in root.child_sections[0].key_entities

def test_serialization(mock_builder, tmp_path):
    nodes = [PageIndexNode(
        section_id="sec_1",
        title="Title",
        summary="Summary",
        key_entities=["E1"],
        data_types_present=["T1"],
        child_sections=[]
    )]
    
    path = os.path.join(tmp_path, "index.json")
    mock_builder.save_index(nodes, path)
    
    loaded_nodes = mock_builder.load_index(path)
    assert len(loaded_nodes) == 1
    assert loaded_nodes[0].section_id == "sec_1"
    assert loaded_nodes[0].key_entities == ["E1"]

def test_navigate_api(mock_builder):
    node1 = PageIndexNode(section_id="1", title="Financial Report", summary="Numbers", key_entities=["CBE"], data_types_present=[], child_sections=[])
    node2 = PageIndexNode(section_id="2", title="Audit Findings", summary="Logic", key_entities=["Entity"], data_types_present=[], child_sections=[])
    
    # Test top-k navigation
    top = mock_builder.navigate([node1, node2], "Financial", k=1)
    assert len(top) == 1
    assert top[0].section_id == "1"

def test_recursive_search(mock_builder):
    child = PageIndexNode(section_id="1.1", title="Details", summary="Deep", key_entities=["Secret"], data_types_present=["Internal"], child_sections=[])
    parent = PageIndexNode(section_id="1", title="Top", summary="High", key_entities=[], data_types_present=[], child_sections=[child])
    
    # Search for entity in child
    results = mock_builder.search([parent], "Secret")
    assert len(results) == 1
    assert results[0].section_id == "1.1"
    
    # Search for data type in child
    results = mock_builder.search([parent], "Internal")
    assert len(results) == 1
    assert results[0].section_id == "1.1"
