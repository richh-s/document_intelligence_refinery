import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any

import structlog

# Set up simple import resolution for the other scripts
import sys
sys.path.append(str(Path(__file__).parent))

from density_analysis import analyze_pdf as analyze_density
from docling_analysis import analyze_with_docling as analyze_docling

logger = structlog.get_logger()

def compare_extractions(pdf_path: Path, sample_mode: str, every_n: int) -> Dict[str, Any]:
    if not pdf_path.exists():
        logger.error("file_not_found", path=str(pdf_path))
        return {}

    logger.info("starting_comparison", file=pdf_path.name)
    
    # 1. Run Density Analysis (pdfplumber)
    density_start = time.time()
    density_stats = analyze_density(pdf_path)
    density_time = time.time() - density_start
    
    if not density_stats:
        return {}

    # 2. Run Docling Analysis
    docling_start = time.time()
    docling_stats = analyze_docling(pdf_path, sample_mode, every_n)
    docling_time = time.time() - docling_start
    
    if not docling_stats:
        return {}

    # 3. Compute Comparative Metrics
    pdfplumber_chars = density_stats.get("total_characters", 0)
    docling_chars = docling_stats.get("markdown_characters", 0)
    
    char_difference = docling_chars - pdfplumber_chars
    char_ratio = docling_chars / pdfplumber_chars if pdfplumber_chars > 0 else 0
    
    comparison_results = {
        "file": pdf_path.name,
        "pdfplumber_total_characters": pdfplumber_chars,
        "docling_markdown_characters": docling_chars,
        "character_difference": char_difference,
        "character_extraction_ratio": round(char_ratio, 2),
        "density_classification_hint": density_stats.get("classification_hint", "unknown"),
        "docling_failure_modes": docling_stats.get("failure_modes", []),
        "potential_issues": []
    }
    
    # Analyze discrepancies
    if char_ratio < 0.5 and pdfplumber_chars > 1000:
        comparison_results["potential_issues"].append("severe_text_loss")
        logger.error("comparison_alert", file=pdf_path.name, issue="severe_text_loss", ratio=char_ratio)
    elif char_ratio > 1.5 and pdfplumber_chars > 1000:
        comparison_results["potential_issues"].append("potential_hallucination_or_over_segmentation")
        logger.warning("comparison_alert", file=pdf_path.name, issue="potential_hallucination_or_over_segmentation", ratio=char_ratio)
        
    if density_stats.get("classification_hint") == "likely_scanned" and char_ratio < 0.2:
        comparison_results["potential_issues"].append("scanned_pdf_failure")
        logger.warning("comparison_alert", file=pdf_path.name, issue="scanned_pdf_failure")

    logger.info("comparison_complete", 
                file=pdf_path.name, 
                ratio=round(char_ratio, 2), 
                issues=comparison_results["potential_issues"])
                
    return comparison_results

def main() -> None:
    parser = argparse.ArgumentParser(description="Run cross-tool comparative analysis (pdfplumber vs docling)")
    parser.add_argument("files", nargs="+", type=Path, help="Paths to PDF files to analyze")
    parser.add_argument("--sample-mode", choices=["first", "middle", "every-n", "full"], default="full", help="Sampling strategy for Docling")
    parser.add_argument("--every-n", type=int, default=10, help="N parameter for every-n sampling in Docling")
    parser.add_argument("--json", action="store_true", help="Output raw JSON summary")
    args = parser.parse_args()

    results = []
    for f in args.files:
        res = compare_extractions(f, args.sample_mode, args.every_n)
        if res:
            results.append(res)
            
    if args.json:
        print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
