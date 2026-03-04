import argparse
import json
import traceback
from pathlib import Path
from typing import Any, Dict, List
import statistics

import pdfplumber
import structlog

logger = structlog.get_logger()

def compute_variance(data: List[float]) -> float:
    if len(data) < 2:
        return 0.0
    return statistics.variance(data)

def analyze_pdf(pdf_path: Path) -> Dict[str, Any]:
    if not pdf_path.exists():
        logger.error("file_not_found", path=str(pdf_path))
        return {}

    logger.info("analyzing_pdf", file=pdf_path.name)
    
    stats: Dict[str, Any] = {
        "file": pdf_path.name,
        "total_pages": 0,
        "total_characters": 0,
        "chars_per_page": [],
        "unique_fonts_per_page": [],
        "avg_unique_fonts": 0.0,
        "max_unique_fonts": 0,
        "line_height_variance_per_page": [],
        "avg_line_height_variance": 0.0,
        "vector_count_per_page": [],
        "avg_vector_count": 0.0,
        "whitespace_ratio_per_page": [],
        "bbox_height_mean": 0.0,
        "bbox_height_std": 0.0,
        "bbox_width_mean": 0.0,
        "bbox_width_std": 0.0,
        "page_density_variance": 0.0,
        "classification_hint": "unknown"
    }

    all_heights = []
    all_widths = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            stats["total_pages"] = len(pdf.pages)
            if stats["total_pages"] == 0:
                 return stats

            for i, page in enumerate(pdf.pages):
                chars = page.chars
                words = page.extract_words()
                
                num_chars = len(chars)
                stats["total_characters"] += num_chars
                stats["chars_per_page"].append(num_chars)
                
                # A. Font Diversity Count
                fonts = {c.get("fontname") for c in chars if c.get("fontname")}
                stats["unique_fonts_per_page"].append(len(fonts))
                
                # B. Average Line Height Variance
                # Sort words by top y-coordinate to roughly approximate lines
                sorted_words = sorted(words, key=lambda w: w['top'])
                line_heights = []
                last_top = None
                for w in sorted_words:
                    if last_top is not None:
                        diff = w['top'] - last_top
                        if diff > 0: # Only count vertical moves down
                            line_heights.append(diff)
                    last_top = w['top']
                    
                stats["line_height_variance_per_page"].append(compute_variance(line_heights))

                # C. Image / Vector Count
                vector_count = len(page.images) + len(page.rects) + len(page.lines) + len(page.curves)
                stats["vector_count_per_page"].append(vector_count)
                
                # D. Structural Metrics (Whitespace, Bbox)
                char_area = 0
                for c in chars:
                    width = c['x1'] - c['x0']
                    height = c['bottom'] - c['top']
                    char_area += width * height
                    all_widths.append(width)
                    all_heights.append(height)

                page_area = float(page.width * page.height) if page.width and page.height else 1.0
                whitespace_ratio = 1.0 - (char_area / page_area) if page_area > 0 else 1.0
                stats["whitespace_ratio_per_page"].append(round(whitespace_ratio, 4))
                
        # Aggregations
        stats["avg_unique_fonts"] = statistics.mean(stats["unique_fonts_per_page"]) if stats["unique_fonts_per_page"] else 0
        stats["max_unique_fonts"] = max(stats["unique_fonts_per_page"]) if stats["unique_fonts_per_page"] else 0
        
        stats["avg_line_height_variance"] = statistics.mean(stats["line_height_variance_per_page"]) if stats["line_height_variance_per_page"] else 0
        stats["avg_vector_count"] = statistics.mean(stats["vector_count_per_page"]) if stats["vector_count_per_page"] else 0
        
        if all_heights:
             stats["bbox_height_mean"] = statistics.mean(all_heights)
             stats["bbox_height_std"] = statistics.stdev(all_heights) if len(all_heights) > 1 else 0.0
        if all_widths:
             stats["bbox_width_mean"] = statistics.mean(all_widths)
             stats["bbox_width_std"] = statistics.stdev(all_widths) if len(all_widths) > 1 else 0.0
             
        stats["page_density_variance"] = compute_variance(stats["chars_per_page"])

        # E. Classification Hint
        avg_ws = statistics.mean(stats["whitespace_ratio_per_page"]) if stats["whitespace_ratio_per_page"] else 1.0
        avg_chars = stats["total_characters"] / stats["total_pages"] if stats["total_pages"] > 0 else 0
        
        if avg_ws > 0.95 and avg_chars < 500 and stats["avg_vector_count"] < 5:
            stats["classification_hint"] = "likely_scanned"
        elif stats["avg_unique_fonts"] >= 4 or stats["avg_vector_count"] > 20:
            stats["classification_hint"] = "layout_complex"
        else:
             stats["classification_hint"] = "likely_native"
             
        logger.info("analysis_complete", 
                    file=pdf_path.name, 
                    total_pages=stats["total_pages"], 
                    total_characters=stats["total_characters"],
                    hint=stats["classification_hint"])
        return stats
    except Exception as e:
        logger.error("analysis_error", file=pdf_path.name, error=str(e), traceback=traceback.format_exc())
        return stats

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze PDF character density and structure using pdfplumber")
    parser.add_argument("files", nargs="+", type=Path, help="Paths to PDF files to analyze")
    parser.add_argument("--json", action="store_true", help="Output raw JSON instead of log lines")
    args = parser.parse_args()

    results = []
    for f in args.files:
        res = analyze_pdf(f)
        if res:
            results.append(res)
            
    if args.json:
        print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
