import argparse
import time
import traceback
from pathlib import Path
from typing import Any, Dict

import structlog
import pandas as pd
from docling.document_converter import DocumentConverter

logger = structlog.get_logger()

def analyze_with_docling(pdf_path: Path, sample_mode: str, every_n: int) -> Dict[str, Any]:
    if not pdf_path.exists():
        logger.error("file_not_found", path=str(pdf_path))
        return {}
        
    logger.info("starting_docling_analysis", file=pdf_path.name, sample_mode=sample_mode)
    converter = DocumentConverter()
    
    stats: Dict[str, Any] = {
        "file": pdf_path.name,
        "markdown_characters": 0,
        "tables_detected": 0,
        "tables_with_headers": 0,
        "avg_header_row_ratio": 0.0,
        "num_headings": 0,
        "num_list_markers": 0,
        "avg_line_length": 0.0,
        "empty_line_ratio": 0.0,
        "failure_modes": [],
        "elapsed_time_s": 0.0
    }
    
    start_time = time.time()
    try:
        # Note: docling's python API native sampling by page is limited to `page_ranges` or `max_num_pages` 
        # For simplicity in this script, we'll convert the whole document if 'full' or 'every-n' or 'middle' is needed, 
        # or limit to a small range if 'first'. A more robust implementaion would compute the PDF page count 
        # first (e.g. via PyPDF2) and pass specific `page_ranges` to `convert`.
        # Here we just convert the whole doc and warn if not 'full' since we don't have pdfplumber in this context
        
        # We will attempt full conversion for now to get true metadata.
        result = converter.convert(pdf_path)
        doc = result.document
        markdown_output = doc.export_to_markdown()
        
        stats["elapsed_time_s"] = round(time.time() - start_time, 2)
        stats["markdown_characters"] = len(markdown_output)
        
        # B. Table Metadata Enhancement
        stats["tables_detected"] = len(doc.tables)
        header_ratios = []
        for table in doc.tables:
            # Docling exports tables as pandas DataFrames where we can infer structure.
            # A native table object has a grid. We check if it has headers.
            df = table.export_to_dataframe()
            # Basic heuristic: if columns are not just range indices (0, 1, 2) it likely extracted a header
            has_headers = not all(str(c).isdigit() for c in df.columns)
            if has_headers:
                 stats["tables_with_headers"] += 1
                 # Typically 1 header row vs len(df) data rows
                 total_rows = len(df) + 1 
                 header_ratios.append(1.0 / total_rows if total_rows > 0 else 0)
            else:
                 header_ratios.append(0.0)
                 
        if header_ratios:
             stats["avg_header_row_ratio"] = sum(header_ratios) / len(header_ratios)
             
        # C. Markdown Structural Signals
        lines = markdown_output.split('\n')
        if lines:
            stats["num_headings"] = sum(1 for line in lines if line.strip().startswith('#'))
            stats["num_list_markers"] = sum(1 for line in lines if line.strip().startswith('- ') or line.strip().startswith('* '))
            
            line_lengths = [len(line) for line in lines if line.strip()]
            stats["avg_line_length"] = sum(line_lengths) / len(line_lengths) if line_lengths else 0.0
            
            empty_lines = sum(1 for line in lines if not line.strip())
            stats["empty_line_ratio"] = empty_lines / len(lines)
            
        # D. Failure Mode Logging
        if stats["markdown_characters"] == 0:
             stats["failure_modes"].append("empty_output")
             logger.error("failure_mode", mode="empty_output", file=pdf_path.name)
        elif stats["markdown_characters"] < 500:
             stats["failure_modes"].append("extremely_short_markdown")
             logger.error("failure_mode", mode="extremely_short_markdown", file=pdf_path.name)
             
        if stats["tables_detected"] > 0 and stats["tables_with_headers"] == 0:
             stats["failure_modes"].append("header_loss")
             logger.error("failure_mode", mode="header_loss", file=pdf_path.name)
             
        if stats["num_headings"] == 0 and stats["markdown_characters"] > 1000:
             stats["failure_modes"].append("layout_flattening")
             logger.error("failure_mode", mode="layout_flattening", file=pdf_path.name)
             
        logger.info("docling_analysis_complete", **stats)
        return stats
        
    except Exception as e:
        stats["failure_modes"].append("conversion_exception")
        logger.error("failure_mode", mode="conversion_exception", file=pdf_path.name, error=str(e), traceback=traceback.format_exc())
        return stats

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze PDF extraction using Docling")
    parser.add_argument("files", nargs="+", type=Path, help="Paths to PDF files to analyze")
    parser.add_argument("--sample-mode", choices=["first", "middle", "every-n", "full"], default="full", help="Sampling strategy for pages")
    parser.add_argument("--every-n", type=int, default=10, help="N parameter for every-n sampling")
    args = parser.parse_args()

    for f in args.files:
        analyze_with_docling(f, sample_mode=args.sample_mode, every_n=args.every_n)

if __name__ == "__main__":
    main()
