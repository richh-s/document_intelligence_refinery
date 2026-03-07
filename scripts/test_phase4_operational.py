import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from agents.query_agent import workflow, AgentState
from agents.audit_mode import ClaimAuditor
from indexing.vector_store import RefineryVectorStore
from indexing.fact_table import FactTableStore, Fact

def test_operational_agent():
    print("--- Testing Phase 4: Operational Agent ---")
    
    # Initialize backends
    vs = RefineryVectorStore()
    ft = FactTableStore(":memory:") # Use memory for test
    # ft.ingest_facts expects doc_name, section, hash, facts_list
    ft.ingest_facts("test_doc.pdf", "Summary", "h1", [Fact(entity="revenue", value="4.2B")])
    
    # 1. Test Quantitative Route (SQL)
    state_quant = AgentState(query="What is the revenue?", original_query="What is the revenue?")
    state_quant.classification = "quantitative"
    
    # Manually invoke nodes to simulate graph flow with injected dependencies
    from agents.query_agent import structured_query_node, synthesize_answer
    state_quant = structured_query_node(state_quant, ft)
    state_quant = synthesize_answer(state_quant)
    
    print(f"Quant Answer: {state_quant.final_answer}")
    assert "4.2B" in state_quant.final_answer
    assert "Fact_Table.sql" in state_quant.final_answer

    # 2. Test Conceptual Route (Vector)
    # Note: This assumes some data is in the local .chromadb from previous runs
    from agents.query_agent import semantic_search_node
    state_concept = AgentState(query="finding", original_query="What are the findings?")
    state_concept.target_sections = ["sec_01"]
    
    try:
        state_concept = semantic_search_node(state_concept, vs)
        state_concept = synthesize_answer(state_concept)
        print(f"Concept Answer: {state_concept.final_answer}")
    except Exception as e:
        print(f"Vector search skipped (no local data): {e}")

    print("\n--- Testing Phase 4: Audit System ---")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        auditor = ClaimAuditor(api_key=api_key)
        context = [{"hash": "h1", "text": "Total revenue was $4.2B in Q3."}]
        result = auditor.verify_claim("Revenue was $4.2B", context)
        print(f"Audit Status: {result.status}")
        print(f"Reasoning: {result.reasoning}")
    else:
        print("Skipping LLM Audit test (no API key).")

if __name__ == "__main__":
    test_operational_agent()
