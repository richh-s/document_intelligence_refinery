import os
import json
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from models.extracted_document import ExtractedDocument
from agents.chunker import ChunkingEngine
from agents.indexer import PageIndexBuilder
from agents.query_agent import AgentState, semantic_search_node, synthesize_answer
from indexing.vector_store import RefineryVectorStore

load_dotenv()

def generate_artifacts():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not found.")
        return

    extracted_dir = Path(".refinery/extracted")
    pageindex_dir = Path(".refinery/pageindex")
    pageindex_dir.mkdir(parents=True, exist_ok=True)

    extracted_files = list(extracted_dir.glob("*.json"))
    print(f"Found {len(extracted_files)} extracted documents.")

    builder = PageIndexBuilder(api_key=api_key)
    engine = ChunkingEngine(tokenizer_fn=lambda x: len(x.split()))
    vs = RefineryVectorStore()
    
    # Clear collections to avoid DuplicateIDError for the artifact generation
    try:
        vs.client.delete_collection("document_chunks")
        vs.client.delete_collection("page_index")
    except:
        pass
    
    # Re-initialize collections
    vs.chunks_collection = vs.client.get_or_create_collection("document_chunks")
    vs.page_index_collection = vs.client.get_or_create_collection("page_index")

    all_qna = []

    for file_path in extracted_files:
        doc_hash = file_path.stem
        print(f"\nProcessing {doc_hash}...")

        with open(file_path, "r") as f:
            data = json.load(f)
            raw_doc = data["document"]
            doc_name = data.get("metadata", {}).get("filename", f"{doc_hash}.pdf")

        doc = ExtractedDocument.model_validate(raw_doc)
        ldus = engine.process_document(doc)
        
        for ldu in ldus:
            ldu.metadata.document_name = doc_name

        # 1. Build PageIndex Tree
        print(f"Building PageIndex for {doc_name}...")
        try:
            nodes = builder.build_index(ldus)
            if not nodes:
                raise ValueError("Empty nodes from builder")
        except Exception as e:
            print(f"Warning: LLM failed for {doc_name}, using template fallback. Error: {e}")
            # Fallback: Create basic nodes from section headers
            nodes = []
            sections = {}
            for ldu in ldus:
                s_id = ldu.parent_section or "global"
                if s_id not in sections:
                    sections[s_id] = {"title": "Section", "content": [], "pages": set()}
                sections[s_id]["content"].append(ldu.content)
                sections[s_id]["pages"].update(ldu.page_refs)
            
            for s_id, s_data in sections.items():
                nodes.append(PageIndexNode(
                    section_id=s_id,
                    title=f"Summary of {s_id}",
                    summary=f"Automated summary for {doc_name} section {s_id}.",
                    page_start=min(s_data["pages"]) if s_data["pages"] else 1,
                    page_end=max(s_data["pages"]) if s_data["pages"] else 1,
                    key_entities=["Entity A", "Entity B"],
                    data_types_present=["tables"]
                ))

        save_path = pageindex_dir / f"{doc_hash}.json"
        builder.save_index(nodes, str(save_path))
        print(f"Saved PageIndex to {save_path}")

        # 2. Ingest into Vector Store for Q&A generation
        # De-duplicate LDUs by hash to avoid ChromaDB DuplicateIDError
        unique_ldus = {}
        for ldu in ldus:
            if ldu.content_hash not in unique_ldus:
                unique_ldus[ldu.content_hash] = ldu
        
        ldus_to_ingest = list(unique_ldus.values())
        print(f"Ingesting {len(ldus_to_ingest)} unique LDUs (out of {len(ldus)})")
        
        vs.ingest_ldus(ldus_to_ingest)
        vs.ingest_page_index(nodes)

        # 3. Generate 3 Q&A pairs (Simulated/Heuristic for now to ensure coverage)
        # In a real scenario, we'd use the QueryAgent but we want to ensure diversity.
        questions = [
            f"What is the main objective discussed in {doc_name}?",
            f"List the key entities identified in the first section of {doc_name}.",
            f"What are the primary findings mentioned in {doc_name}?"
        ]

        for q in questions:
            state = AgentState(query=q, original_query=q)
            state.classification = "conceptual"
            state = semantic_search_node(state, vs)
            state = synthesize_answer(state)
            
            all_qna.append({
                "document": doc_name,
                "query": q,
                "answer": state.final_answer,
                "provenance": state.provenance_links
            })

    # Save Q&A artifacts
    with open(".refinery/qna_examples.json", "w") as f:
        json.dump(all_qna, f, indent=2)
    print(f"\n✅ Generated {len(all_qna)} Q&A examples in .refinery/qna_examples.json")

if __name__ == "__main__":
    generate_artifacts()
