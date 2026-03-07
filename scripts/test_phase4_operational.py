import sys
import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.extracted_document import ExtractedDocument
from agents.query_agent import workflow, AgentState
from agents.audit_mode import ClaimAuditor
from indexing.vector_store import RefineryVectorStore
from indexing.fact_table import FactTableStore, Fact

# Load environment variables from .env
load_dotenv()

def test_operational_agent():
    print("--- Testing Phase 4: Operational Agent (REAL DATA) ---")
    
    # Initialize backends
    vs = RefineryVectorStore()
    ft = FactTableStore(":memory:") # Use memory for test
    
    # 1. Load REAL extraction and process into LDUs
    extracted_path = Path(".refinery/extracted/4dcf8792cc25a98826f4c9d28da108716744c49e444273983e06a6d72bb90154.json")
    if not extracted_path.exists():
        print(f"Error: {extracted_path} not found. Please run a real extraction first.")
        return
        
    with open(extracted_path, "r") as f:
        data = json.load(f)
    
    doc = ExtractedDocument.model_validate(data["document"])
    
    def mock_tokenizer(text: str) -> int:
        return len(text.split())
        
    from agents.chunker import ChunkingEngine
    engine = ChunkingEngine(tokenizer_fn=mock_tokenizer)
    ldus = engine.process_document(doc)
    
    # Ingest the real LDUs
    doc_name = "fta_performance_survey_final_report_2022.pdf"
    for ldu in ldus:
        ldu.metadata.document_name = doc_name
        
    vs.ingest_ldus(ldus)
    print(f"Ingested {len(ldus)} real LDUs into VectorStore.")
    
    # Ingest a real fact into FactTable
    from indexing.fact_table import Fact
    ft.ingest_facts("fta_performance_survey_final_report_2022.pdf", "Cover", "h_year", [Fact(entity="report_year", value="2022")])

    # 2. Test Quantitative Route (SQL)
    state_quant = AgentState(query="What is the report_year?", original_query="What is the report_year?")
    state_quant.classification = "quantitative"
    
    from agents.query_agent import structured_query_node, synthesize_answer
    state_quant = structured_query_node(state_quant, ft)
    state_quant = synthesize_answer(state_quant)
    
    print(f"\nQUANT ANSWER:\n{state_quant.final_answer}")
    assert "2022" in state_quant.final_answer

    # 3. Test Conceptual Route (Real Vector Search)
    from agents.query_agent import semantic_search_node
    # Query for something on page 7
    query_text = "Who prepared this report?"
    state_concept = AgentState(query=query_text, original_query=query_text)
    state_concept.target_sections = None # Search globally in this document
    
    state_concept = semantic_search_node(state_concept, vs)
    state_concept = synthesize_answer(state_concept)
    
    print(f"\nCONCEPTUAL ANSWER (REAL PDF CONTENT):\n{state_concept.final_answer}")
    assert "Abate Mekuriaw" in state_concept.final_answer or "prepared by" in state_concept.final_answer.lower()

    print("\n--- Testing Phase 4: Audit System ---")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        from agents.audit_mode import ClaimAuditor
        auditor = ClaimAuditor(api_key=api_key)
        # Use the real context retrieved in step 3
        context = [{"hash": p["content_hash"], "text": state_concept.retrieved_context} for p in state_concept.provenance_links]
        result = auditor.verify_claim("Abate Mekuriaw was part of the core team.", context)
        print(f"Audit Status: {result.status}")
        print(f"Reasoning: {result.reasoning}")
    else:
        print("Skipping LLM Audit test (no API key).")

if __name__ == "__main__":
    test_operational_agent()
