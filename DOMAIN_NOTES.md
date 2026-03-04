# Document Intelligence Extraction Notes

This document synthesizes findings from a structural and cross-tool comparative analysis of a sample of heavy Ethiopian regulatory and financial PDF documents. Tools evaluated were `pdfplumber` (for low-level spatial/structural layout heuristics) and `Docling` (for high-level layout extraction and markdown conversion).

## 1. Document Characteristics & Classification
The analyzed documents demonstrated significant layout complexity and structural variance.
- **Scanned vs Native**: `Audit Report - 2023.pdf` was strongly identified as a scanned document (`pdfplumber` detected only 116 readable characters vs Docling's OCR extraction of 539,681 characters).
- **Layout Complexity**: All other reports (`CBE ANNUAL REPORT`, `fta_performance`, `tax_expenditure`) were correctly flagged as having highly complex native layouts due to significant font diversity, variance in line heights, and high vector counts per page.

### Cross-Tool Comparative Metrics
| Document Name | pdfplumber (Chars) | Docling (Chars) | Extraction Ratio | Classification Hint | Potential Issues |
|---|---|---|---|---|---|
| `Audit Report - 2023.pdf` | 116 | 539,681 | 4652.42x | `likely_scanned` | Clean OCR Recovery |
| `tax_expenditure_ethiopia_2021_22.pdf` | 105,205 | 150,944 | 1.43x | `layout_complex` | Clean Extraction |
| `CBE ANNUAL REPORT 2023-24.pdf` | 319,492 | 520,263 | 1.63x | `layout_complex` | `potential_hallucination_or_over_segmentation` |
| `fta_performance_survey_final_report_2022.pdf` | 263,370 | 523,519 | 1.99x | `layout_complex` | `potential_hallucination_or_over_segmentation` |

## 2. Failure Modes Observed

A rigorous cross-tool comparison revealed the following major failure modes in the current extraction pipeline:

### A. Extreme Over-Segmentation / Potential Hallucincanation
In highly complex, natively digital PDFs with many tables and sidebars (e.g., `CBE ANNUAL REPORT`, `fta_performance_survey`), `Docling` returned between **1.6x and 2.0x** the character count extracted by `pdfplumber`. 
- **Cause**: Docling often aggressively flattens surrounding metadata, duplicate headers, or hallucinates tabular grid structures as text characters in complex financial reports.

### B. Header Ratio Imbalance in Tables
While Docling successfully detects tables, its ability to confidently detect the semantic *headers* of those tables drops significantly in complex documents:
- *Tax Expenditure Ethiopia*: 29 tables detected, 24 with inferred headers.
- *FTA Performance Survey*: 57 tables detected, only 25 with inferred headers.
- **Impact**: Without robust header row identification, semantic extraction (RAG) on financial tables degrades heavily.

## 3. Extraction Strategy Decision Tree

Based on these heuristics, a static "one size fits all" parser will fail. We need a dynamic routing strategy.

```mermaid
graph TD
    A[Ingest PDF Document] --> B(Run pdfplumber Structural Analysis\nMax 5 Pages)
    B --> C{Calculate Density & Variance}
    
    C -- Whitespace > 95% AND Vectors < 5<br/>AND Chars < 500 --> D[Classifier: Scanned / Image PDF]
    C -- High Font Diversity > 4<br/>OR High Vector Count > 20 --> E[Classifier: Complex Native Layout]
    C -- High Char Density<br/>AND Low Line Height Variance --> F[Classifier: Simple Native Layout]

    D --> G[Route to Heavy OCR Pipeline<br/>e.g., Docling with OCR enabled / Tesseract]
    E --> H[Route to Hybrid Parser<br/>Docling for Markdown + Custom BBox Tabular Logic]
    F --> I[Route to Lightweight Parser<br/>e.g., PyMuPDF or pure Docling via CLI]
```

## 4. Proposed Parsing Pipeline Architecture

To execute on the Decision Tree above, the ingestion flow must incorporate an initial "Classification Node" that governs downstream processing.

```mermaid
sequenceDiagram
    participant S as Storage Bucket
    participant R as Refinery Router (Main Agent)
    participant DA as Density Analyzer (pdfplumber)
    participant EX as Extraction Engines (Docling/OCR)
    participant V as Validation Filter

    S->>R: New PDF Uploaded
    R->>DA: Request Structural Heuristics (Pages 1-5)
    DA-->>R: Return (WhitespaceRatio, CharDensity, Vectors, Fonts)
    
    Note over R, DA: Router assigns Document Type based on Decision Tree.
    
    alt is Scanned
        R->>EX: Dispatch to OCR-Heavy Engine
    else is Layout Complex
        R->>EX: Dispatch to Layout-Aware Engine (Docling)
    else is Simple
        R->>EX: Dispatch to Fast Text Extractor
    end
    
    EX-->>V: Raw Markdown / Structured JSON
    
    Note over V: Validator checks for: Empty Output, Missing Table Headers
    
    alt Table Headers Missing
        V-->>R: Flag for Human Review / Advanced Table Rescue
    else Passed
        V->>S: Store Cleaned Artifacts
    end
```

## 5. MinerU Processing Pipeline

Below is a simplified flowchart of the MinerU pipeline for document processing:

```mermaid
flowchart TD
    A[Input: PDF or Images] --> B{Choose Backend}
    
    B -->|Default: hybrid-auto-engine| C[<b>Hybrid-Auto-Engine</b><br>Best accuracy, auto platform detection]
    B -->|CPU-only / Compatibility| D[<b>Pipeline Backend</b><br>Traditional processing]
    B -->|Pure VLM| E[<b>VLM-Auto-Engine</b><br>End-to-end model]
    
    C --> F[Smart Processing]
    F --> G{PDF Type}
    G -->|Text PDF| H[Direct Text Extraction]
    G -->|Scanned PDF| I[OCR - 109 languages]
    
    H & I --> J[Parse Formulas & Tables]
    J --> K[Merge Cross-Page Elements]
    K --> L[Generate Output]
    
    D --> M[Multi-Model Chain]
    M --> L
    
    E --> N[Two-Stage VLM]
    N --> L
    
    L --> O[<b>Output Files</b>]
    O --> P[Markdown]
    O --> Q[Structured JSON]
    O --> R[Visual PDFs]
    
    %% Simple styling with good contrast
    style A fill:#e1f5fe,color:#000000
    style B fill:#ffffff,color:#000000,stroke:#333
    style C fill:#2e7d32,color:#ffffff
    style D fill:#b71c1c,color:#ffffff
    style E fill:#b85e00,color:#ffffff
    style F fill:#4a6da7,color:#ffffff
    style G fill:#ffffff,color:#000000
    style H fill:#2e7d32,color:#ffffff
    style I fill:#2e7d32,color:#ffffff
    style J fill:#4a6da7,color:#ffffff
    style K fill:#4a6da7,color:#ffffff
    style L fill:#4a6da7,color:#ffffff
    style M fill:#b71c1c,color:#ffffff
    style N fill:#b85e00,color:#ffffff
    style O fill:#01579b,color:#ffffff
    style P fill:#01579b,color:#ffffff
    style Q fill:#01579b,color:#ffffff
    style R fill:#01579b,color:#ffffff
```

## 6. Conclusion 

Based on the empirical evidence gathered from our internal extraction tests on complex regulatory documents:

1.  **pdfplumber excels at structural heuristics:** It is highly performant and accurate for gathering metadata bounding boxes, vector counts, and font diversity. It provides the essential, low-level metrics required to reliably classify a PDF's complexity *before* heavy processing begins.
2.  **Docling struggles as a generalized fallback for complex layouts:** While Docling produces visually clean Markdown, it is computationally expensive and prone to severe over-segmentation and hallucination on highly complex, natively digital PDFs (e.g., repeating grid lines or duplicate headers as text). Furthermore, its header-detection for embedded tables degrades significantly in non-standard layouts.
3.  **A dynamic pipeline is required:** A hybrid routing approach (as outlined in Sections 3 and 4) is the only viable path forward. By explicitly calculating document density and variance via pdfplumber first, the system can selectively dispatch documents to the appropriate extraction engine (e.g., lightweight parsing for clean text, or heavy OCR/VLM backends like MinerU for complex unstructured data), avoiding complete failure on edge cases while optimizing for cost and speed.

---

## Phase 2: Extraction Rules Rationale

The file `extraction_rules.yaml` establishes the exact physical constants governing the Multi-Strategy router constraints and the fallback triggers.

### Escalation & Circuit Breaker Logic
* `MIN_EXTRACTION_CONFIDENCE` (0.85): High standard for pass-through. If the validator evaluates a document's layout fidelity, reading-order coherency, and parsed metadata below an 85% confidence score, we assume data degradation and force an escalation tier retry.
* `PAGE_CONTINUITY_PENALTY` (0.4): Missing pages during extraction structurally break reading order flows downstream. A multiplier penalty of 0.4 almost mathematically guarantees the document will fail the `0.85` gate (e.g., `0.95 * 0.4 = 0.38 < 0.85`) forcing an escalation.
* `MAX_STRATEGY_RETRIES` (1): Prevents infinite loops. Each backend engine gets one structural attempt to parse a document per tier.

### Strategy A: Text & Garbage Avoidance
* `NONSENSE_RATIO_MAX` (0.30): Digital "native" documents frequently carry hidden, corrupted OCR text layers overlaid on scans. Setting the tolerance to 30% allows minor header/footer artifact parsing while aggressively aborting unreadable layers back to Strategy C (Vision).
* `MIN_CHARS_PER_PAGE` (100): Determines valid digital data flow. Prevents blank or heavily damaged sparse pages from passing through the pipeline, punishing page confidence to force an escalation.
* `MAX_IMAGE_RATIO` (0.50): If more than 50% of the page area is consumed by raw graphics or scans, it is overwhelmingly likely to hold un-extractable embedded data. Structural confidence is slashed by half to route it toward heavier layout-aware or vision engines.

### Strategy C: Vision Budgets
Vision language models are cost-prohibitive on massive document batches. Budget limits must be calculated _pre-flight_.
* `GLOBAL_DOCUMENT_BUDGET_USD` ($0.05): Hard cap per individual document to prevent cost overruns.
* `AVG_TOKENS_PER_PAGE_IMAGE` (2000): An empirical average token slice for standard high-resolution A4 scans in Gemini/OpenAI multimodal ingestion.
* `MAX_PAGES_PER_VLM_CALL` (5): Prevents context dilution and hallucination in massive token streams, and allows the pre-flight routine to halt parsing gracefully with a `PartialExtractionResult` before burning through the entire document budget.
* `PROMPT_TOKENS` (500): Average size of the strict parsing JSON schema prompt instructions.

### Universal Constraints
* `COORDINATE_SYSTEM` (`normalized_float_0_1`): Eradicates PDF point versus image pixel mismatches by enforcing a uniform `[0.0, 1.0]` grid relative to page bounds, standardizing output for downstream spatial RAG chunking.
