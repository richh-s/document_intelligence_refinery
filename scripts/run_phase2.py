import json
import logging
from pathlib import Path
import sys

# Allow importing from src
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.triage import TriageAgent
from agents.extractor import ExtractionRouter
from extractors.ledger import ExtractionLedger
from config_loader import load_extraction_rules
import structlog

# Setup basic logging
logging.basicConfig(level=logging.INFO)
structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
)
logger = logging.getLogger(__name__)

def main():
    root_dir = Path(__file__).parent.parent
    data_dir = root_dir / "data"
    out_dir = root_dir / ".refinery" / "extracted"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Initialize Triage Layer
    triage_agent = TriageAgent()
    
    # 2. Initialize Extraction Layer
    rules = load_extraction_rules()
    ledger_path = root_dir / ".refinery" / "extraction_ledger.jsonl"
    ledger = ExtractionLedger(ledger_path)
    router = ExtractionRouter(rules, ledger)
    
    # 3. Find target PDFs
    pdfs = list(data_dir.glob("*.pdf"))
    if not pdfs:
        logger.error(f"No PDFs found in {data_dir}")
        return
        
    logger.info(f"Found {len(pdfs)} PDFs to process.")
    
    for pdf_path in pdfs:
        logger.info(f"--- Processing {pdf_path.name} ---")
        
        # Phase 1: Triage
        logger.info("Executing TriageAgent...")
        profile = triage_agent.profile(pdf_path)
        logger.info(f"Triage Profile Complete: Origin={profile.origin_type.value}, Layout={profile.layout_type.value}, Format={profile.domain_hint.value}")
        
        # Phase 2: Extraction
        logger.info("Executing ExtractionRouter...")
        result = router.route(pdf_path, profile)
        
        if result and result.document:
            logger.info(f"Extraction Completed via {result.model_name}. Final Confidence: {result.confidence:.3f}")
            out_file = out_dir / f"{profile.file_hash}.json"
            
            # Serialize the entire ExtractionResult to preserve routing/escalation flags
            doc_data = result.model_dump()
            with open(out_file, "w") as f:
                json.dump(doc_data, f, indent=2)
            logger.info(f"Saved extracted output to: {out_file.name}")
        else:
            logger.error(f"Extraction failed for {pdf_path.name}")
            
    logger.info("--- All Processing Complete! Check .refinery/extraction_ledger.jsonl ---")

if __name__ == "__main__":
    main()
