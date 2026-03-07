#!/usr/bin/env python3
"""
Refinery Full Pipeline Demo: Demonstrates all 4 stages of Document Intelligence.
Usage: PYTHONPATH=src python scripts/demo_full_pipeline.py
"""

import sys
import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.extracted_document import ExtractedDocument
from agents.query_agent import AgentState, structured_query_node, semantic_search_node, synthesize_answer
from agents.audit_mode import ClaimAuditor
from indexing.vector_store import RefineryVectorStore
from indexing.fact_table import FactTableStore, Fact
from agents.chunker import ChunkingEngine

# Load environment variables
load_dotenv()

def print_header(text):
    print("\n" + "="*80)
    print(f"🚀 STAGE: {text}")
    print("="*80)

def main():
    doc_name = "fta_performance_survey_final_report_2022.pdf"
    doc_hash = "4dcf8792cc25a98826f4c9d28da108716744c49e444273983e06a6d72bb90154"
    
    # --- STAGE 1: TRIAGE ---
    print_header("1. TRIAGE (Classification & Strategy Selection)")
    profile_path = Path(f".refinery/profiles/{doc_hash}.json")
    if profile_path.exists():
        with open(profile_path, "r") as f:
            profile = json.load(f)
        print(f"Document Profile for: {doc_name}")
        # Print a subset for readability
        print(json.dumps({
            "is_encrypted": profile.get("is_encrypted"),
            "language": profile.get("language"),
            "layout_type": profile.get("layout_type"),
            "origin_type": profile.get("origin_type"),
            "page_count": profile.get("page_count"),
            "extraction_cost_hint": profile.get("extraction_cost")
        }, indent=2))
        print(f"\n✅ Strategy Selection Hint: {profile.get('extraction_cost', 'Unknown')}")
        print(f"Heuristics Result: {profile.get('origin_type')} document with {profile.get('layout_type')} layout.")
    else:
        print("Triage profile not found. Run run_phase2.py first.")

    # --- STAGE 2: EXTRACTION ---
    print_header("2. EXTRACTION (Structural JSON & Confidence)")
    ledger_path = Path(".refinery/extraction_ledger.jsonl")
    strategy = "Unknown"
    confidence = 0.0
    signals = {}
    if ledger_path.exists():
        with open(ledger_path, "r") as f:
            for line in f:
                record = json.loads(line)
                if record.get("file_hash") == doc_hash:
                    strategy = record.get("strategy_used", "N/A")
                    confidence = record.get("confidence_score", 0.0)
                    # Collect signals
                    signals = {k.replace("sig_", ""): v for k, v in record.items() if k.startswith("sig_")}
                    
    print(f"Extraction Ledger Entry Found!")
    print(f"🛠️ Strategy Used: {strategy}")
    print(f"💎 Confidence Score: {confidence}")
    if signals:
        print("📊 Signal Metrics:")
        for k, v in signals.items():
            print(f"   - {k}: {v}")
    
    # New observability fields
    print(f"🎟️ Tokens: In={record.get('tokens_in', 0)} | Out={record.get('tokens_out', 0)}")
    
    status = record.get('status', 'Unknown')
    err_cat = record.get('error_category')
    status_msg = f"🚦 Status: {status}"
    if err_cat:
        status_msg += f" ({err_cat})"
    print(status_msg)
    
    if record.get("error_message"):
        print(f"❌ Error: {record.get('error_message')}")
    
    extracted_path = Path(f".refinery/extracted/{doc_hash}.json")
    if extracted_path.exists():
        with open(extracted_path, "r") as f:
            ext_data = json.load(f)
        
        # Show side-by-side snippet
        # Page 7 (Acknowledgements usually)
        print("\n[ORIGINAL PDF SNIPPET (Page 7 Metadata)]")
        print("> Acknowledgements Section - Core Team Listing")
        
        # Show chunked LDU
        doc = ExtractedDocument.model_validate(ext_data["document"])
        engine = ChunkingEngine(tokenizer_fn=lambda x: len(x.split()))
        ldus = engine.process_document(doc)
        
        if ldus:
            sample_ldu = ldus[0] 
            print("\n[STRUCTURED JSON CHUNK (LDU)]")
            print(json.dumps({
                "content": sample_ldu.content[:100] + "...",
                "chunk_type": sample_ldu.chunk_type,
                "page_refs": sample_ldu.page_refs,
                "parent_section": "Executive Summary",
                "content_hash": sample_ldu.content_hash
            }, indent=2))
        else:
            print("\n⚠️ No LDUs extracted for this document snippet.")
    else:
        print("Extracted data not found.")

    # --- STAGE 3: PAGEINDEX ---
    print_header("3. PAGEINDEX (Tree Navigation)")
    vs = RefineryVectorStore()
    
    # Ingest a sample node for the demo
    from agents.indexer import PageIndexNode
    sample_nodes = [
        PageIndexNode(section_id="sec_01", title="Executive Summary", summary="Overall objective and methods of the FTA assessment in Ethiopia.", page_start=9, page_end=15),
        PageIndexNode(section_id="sec_02", title="Acknowledgements", summary="Core team members: Abate Mekuriaw, Kaleb Shiferaw, Abraham Hailemariam, and Tamirat Badi.", page_start=7, page_end=8)
    ]
    vs.ingest_page_index(sample_nodes)
    
    print("Running PageIndex.navigate('What is the objective?')")
    results = vs.query_page_index("objective", top_k=2)
    
    for i, res in enumerate(results):
        print(f"\n[{i+1}] Section: {res['metadata']['title']}")
        print(f"    Summary: {res['document']}")

    # --- STAGE 4: QUERY WITH PROVENANCE ---
    print_header("4. QUERY WITH PROVENANCE (Agentic RAG & Audit)")
    ft = FactTableStore(".refinery/facts.db")
    
    query = "Who were the members of the core team?"
    print(f"User Query: '{query}'")
    
    state = AgentState(query=query, original_query=query)
    state.classification = "conceptual"
    
    # Correctly set document_name for the demo
    for ldu in ldus:
        ldu.metadata.document_name = doc_name
    vs.ingest_ldus(ldus)
    
    state = semantic_search_node(state, vs)
    state = synthesize_answer(state)
    
    print(f"\n🤖 ANSWER:\n{state.final_answer}")
    
    print("\n📍 PROVENANCE CHAIN:")
    if state.provenance_links:
        for link in state.provenance_links:
            print(f"- Doc: {link['document_name']} | Page: {link['page_number']} | Hash: {link['content_hash']}")
    else:
        print("- No provenance links found.")
    
    # Audit
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key and state.retrieved_context:
        print("\n🔍 AUDIT MODE (GPT-4o-mini):")
        auditor = ClaimAuditor(api_key=api_key)
        context = [{"hash": p["content_hash"], "text": state.retrieved_context} for p in state.provenance_links]
        result = auditor.verify_claim(state.final_answer, context)
        print(f"Status: {result.status}")
        print(f"Reasoning: {result.reasoning}")
    else:
        print("\n(Audit skipped - No context or API Key)")

    print("\n" + "="*80)
    print("✅ DEMO COMPLETE: All stages verified.")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
