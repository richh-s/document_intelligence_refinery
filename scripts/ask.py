#!/usr/bin/env python3
"""
Refinery Query CLI: Ask questions to your processed document corpus.
Usage: PYTHONPATH=src python scripts/ask.py "Who prepared the report?"
"""

import sys
import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from agents.query_agent import workflow, AgentState
from agents.audit_mode import ClaimAuditor
from indexing.vector_store import RefineryVectorStore
from indexing.fact_table import FactTableStore

# Load environment variables (OPENROUTER_API_KEY)
load_dotenv()

def main():
    if len(sys.argv) < 2:
        print("Usage: PYTHONPATH=src python scripts/ask.py \"Your question here\"")
        sys.exit(1)

    query = sys.argv[1]
    
    # Initialize backends
    vs = RefineryVectorStore()
    ft = FactTableStore(".refinery/facts.db")
    
    print(f"\n🧠 Thinking about: '{query}'...")
    
    # Initialize state
    state = AgentState(query=query, original_query=query)
    
    # 1. Routing (Simple implementation for CLI)
    classification = "conceptual"
    quant_triggers = ["year", "revenue", "cost", "total", "count", "number", "how many"]
    if any(word in query.lower() for word in quant_triggers):
        classification = "quantitative"
    
    state.classification = classification
    
    from agents.query_agent import structured_query_node, semantic_search_node, synthesize_answer
    
    # Execute Path
    if state.classification == "quantitative":
        state = structured_query_node(state, ft)
    else:
        state = semantic_search_node(state, vs)
        
    state = synthesize_answer(state)
    
    # Output result
    print("\n" + "="*50)
    print(f"🤖 ANSWER:\n{state.final_answer}")
    print("="*50)
    
    # Run Audit if API key exists
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key and state.retrieved_context:
        print("\n🔍 AUDIT CHECK...")
        auditor = ClaimAuditor(api_key=api_key)
        # Format context for auditor
        context = []
        if state.provenance_links:
            context = [{"hash": p["content_hash"], "text": state.retrieved_context} for p in state.provenance_links]
        else:
            context = [{"hash": "unknown", "text": state.retrieved_context}]
            
        audit_result = auditor.verify_claim(state.final_answer, context)
        print(f"Status: {audit_result.status}")
        print(f"Reasoning: {audit_result.reasoning}")
    
    print("\n")

if __name__ == "__main__":
    main()
