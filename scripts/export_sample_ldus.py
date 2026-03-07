import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from agents.chunker import ChunkingEngine
from models.extracted_document import ExtractedDocument
from models.ldu import LogicalDocumentUnit

def mock_tokenizer(text: str) -> int:
    return len(text.split())

def export_ldus():
    # Use one of the already extracted documents
    extracted_dir = Path(".refinery/extracted")
    if not extracted_dir.exists():
        print("Error: .refinery/extracted does not exist. Run the pipeline first.")
        return

    # Pick the first JSON file
    json_files = list(extracted_dir.glob("*.json"))
    if not json_files:
        print("Error: No extracted JSON files found in .refinery/extracted")
        return

    target_file = json_files[0]
    print(f"Processing {target_file.name} to generate LDUs...")

    with open(target_file, "r") as f:
        data = json.load(f)
    
    # Map to ExtractedDocument
    doc = ExtractedDocument.model_validate(data["document"])
    
    # Process through ChunkingEngine
    engine = ChunkingEngine(tokenizer_fn=mock_tokenizer)
    ldus = engine.process_document(doc)
    
    # Save to file
    output_path = Path(".refinery/ldus_sample.json")
    with open(output_path, "w") as f:
        json.dump([ldu.model_dump() for ldu in ldus], f, indent=2)
    
    print(f"Successfully exported {len(ldus)} LDUs to {output_path}")

if __name__ == "__main__":
    export_ldus()
