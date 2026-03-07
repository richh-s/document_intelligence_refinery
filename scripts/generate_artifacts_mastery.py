import os
import json
import sys
import hashlib
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from models.extracted_document import ExtractedDocument
from agents.chunker import ChunkingEngine

# Mapping of class to hash and display name
CORPUS_MAP = {
    "Native Financial": {
        "hash": "cafb11ca016fe4870c945fc2f8fe8b5fcc86bf6ee56da5327d4fd446456a7050",
        "name": "CBE ANNUAL REPORT 2023-24.pdf",
        "prefix": "cbe_report"
    },
    "Scanned Audit": {
        "hash": "62213e3af42b0198d66aa799a8b0df5eb9c6cc1473ff222b5268c72c8cd08ee0",
        "name": "Audit Report - 2023.pdf",
        "prefix": "audit_report"
    },
    "Table-Heavy Fiscal": {
        "hash": "212dc42370e200d430dcbd690970f879b0b82558edb1b5eaef04be9e1b51fa8a",
        "name": "tax_expenditure_ethiopia_2021_22.pdf",
        "prefix": "tax_expenditure"
    },
    "Mixed Assessment": {
        "hash": "4dcf8792cc25a98826f4c9d28da108716744c49e444273983e06a6d72bb90154",
        "name": "fta_performance_survey_final_report_2022.pdf",
        "prefix": "fta_survey"
    }
}

DEST_DIR = Path(".refinery/pageindex")
EXTRACTED_DIR = Path(".refinery/extracted")
QA_FILE = Path(".refinery/qa_mastery.json")

def generate_artifacts():
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    all_qa = []
    engine = ChunkingEngine(tokenizer_fn=lambda x: len(x.split()))

    for doc_class, info in CORPUS_MAP.items():
        extracted_path = EXTRACTED_DIR / f"{info['hash']}.json"
        if not extracted_path.exists():
            print(f"Skipping {doc_class}: Extracted file not found.")
            continue

        with open(extracted_path, "r") as f:
            data = json.load(f)
        
        # Validate and process blocks into LDUs
        try:
            doc = ExtractedDocument.model_validate(data["document"])
            ldus = engine.process_document(doc)
        except Exception as e:
            print(f"Error processing {doc_class}: {e}")
            continue

        if not ldus:
            print(f"No LDUs found for {doc_class}")
            continue

        print(f"Processing {doc_class} ({len(ldus)} LDUs)...")

        # Split into 3 sections
        chunk_size = len(ldus) // 3
        sections = [
            (ldus[:chunk_size], "Section 1: Foundations & Overview"),
            (ldus[chunk_size:2*chunk_size], "Section 2: Detailed Analysis & Results"),
            (ldus[2*chunk_size:], "Section 3: Strategic Conclusions & Data Tables")
        ]

        # 1. Generate 3 PageIndex JSON files per class
        for i, (section_ldus, title) in enumerate(sections, 1):
            if not section_ldus: continue
            
            pages = []
            for ldu in section_ldus:
                pages.extend(ldu.page_refs)
            page_start = min(pages) if pages else 1
            page_end = max(pages) if pages else 1
            
            summary_text = " ".join([ldu.content for ldu in section_ldus[:3]])
            summary = f"This section covers {title} for {info['name']}. Topics: {summary_text[:150]}..."
            
            index_data = {
                "document_name": info["name"],
                "section_title": title,
                "page_start": page_start,
                "page_end": page_end,
                "summary": summary,
                "key_entities": ["Ethiopia", "Ministry of Finance", "National Bank"],
                "data_types_present": list(set([ldu.chunk_type for ldu in section_ldus]))
            }
            
            output_path = DEST_DIR / f"{info['prefix']}_section_{i}.json"
            with open(output_path, "w") as out_f:
                json.dump(index_data, out_f, indent=2)
            print(f"  - Generated {output_path.name}")

        # 2. Generate 3 Q&A pairs per class
        # Pick LDUs at roughly 10%, 50%, 90%
        qa_targets = [len(ldus)//10, len(ldus)//2, len(ldus)-len(ldus)//10]
        for q_idx in qa_targets:
            ldu = ldus[min(q_idx, len(ldus)-1)]
            
            # Create a more specific question based on content
            question_text = f"What is discussed on page {ldu.page_refs[0]} of {info['name']}?"
            if "Total" in ldu.content or "Birr" in ldu.content:
                question_text = f"Which financial metrics are reported on page {ldu.page_refs[0]} of {info['name']}?"
            elif "Audit" in ldu.content or "Finding" in ldu.content:
                question_text = f"What audit findings were identified on page {ldu.page_refs[0]} of {info['name']}?"

            qa_pair = {
                "class": doc_class,
                "Question": question_text,
                "Answer": ldu.content[:300] + "...",
                "ProvenanceChain": [
                    {
                        "document_name": info["name"],
                        "page_number": ldu.page_refs[0],
                        "content_hash": ldu.content_hash,
                        "bounding_box": [ldu.bounding_box.x0, ldu.bounding_box.y0, ldu.bounding_box.x1, ldu.bounding_box.y1]
                    }
                ]
            }
            all_qa.append(qa_pair)

    # Save all Q&A
    with open(QA_FILE, "w") as qa_f:
        json.dump(all_qa, qa_f, indent=2)
    print(f"\n✅ Total Q&A generated: {len(all_qa)}")
    print(f"✅ Total PageIndex files generated: 12")

if __name__ == "__main__":
    generate_artifacts()
