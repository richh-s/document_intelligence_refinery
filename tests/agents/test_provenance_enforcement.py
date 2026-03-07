import pytest
from agents.query_agent import AgentState, synthesize_answer, ProvenanceMissingError

def test_provenance_enforcement():
    """
    Rubric Requirement:
    Agent cannot return text without provenance. If not provenance, raise ProvenanceMissingError.
    """
    
    # 1. Test missing provenance
    state_empty = AgentState(
        query="What is the revenue?",
        original_query="What is the revenue?",
        provenance_links=[]
    )
    
    with pytest.raises(ProvenanceMissingError):
        synthesize_answer(state_empty)
        
    # 2. Test valid provenance
    state_valid = AgentState(
        query="What is the revenue?",
        original_query="What is the revenue?",
        provenance_links=[{"content_hash": "hash123", "document_name": "Doc1"}]
    )
    
    result = synthesize_answer(state_valid)
    assert result.final_answer != "", "Did not synthesize despite strict provenance."
