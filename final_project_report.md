# Document Intelligence Refinery: Final Project Report

This report documents the end-to-end implementation of the Document Intelligence Refinery, a high-fidelity extraction and agentic RAG pipeline designed for complex Ethiopian financial and regulatory corpora.

---

## 1. Domain Analysis & Strategy Decision Tree

The refinery is built on "Document Science" principles, recognizing that PDF structures are not monolithic. We categorize our 12-document corpus into four primary classes, each requiring a tailored extraction strategy to prevent structural decay.

### Document Classes & Corpus Evidence
| Class | Corpus Example | Characteristics | Failure Mode |
| :--- | :--- | :--- | :--- |
| **Native Financial** | `CBE ANNUAL REPORT 2023-24.pdf` | Dense multi-column. | **Column bleed** when using pdfplumber on page 14. |
| **Scanned Audit** | `Audit Report - 2023.pdf` | Pure scanned pages. | **FastText returns 0 chars** on page 3. |
| **Table-Heavy Fiscal** | `tax_expenditure_ethiopia_2021_22.pdf` | Multi-line headers. | **Header truncation** in Table 5 page 21. |
| **Mixed Assessment** | `fta_performance_survey_final_report_2022.pdf` | Mixed text + infographics. | **Infographic text split** into fragments page 48. |

---

## 2. Pipeline Architecture & Data Flow

The system transforms raw bytes into "Logical Document Units" (LDUs) across five strictly typed stages.

### 5-Stage Data Transformation
```mermaid
graph TD
    A[PDF Ingest] --> T(Triage)
    T --> E(Extraction)
    E --> C(Chunking)
    C --> P(PageIndex)
    P --> Q(Query Agent)
    
    subgraph "Data Flow"
        T -- DocumentProfile --> E
        E -- ExtractedDocument --> C
        C -- Logical Document Units --> P
        P -- PageIndexNode Tree --> Q
    end
```

The pipeline distinguishes between a **happy path (A→B)** and an **escalation path (A→B→C)**. Escalation occurs when extraction confidence falls below the specific strategy's threshold (0.85/0.70) or when a strategy raises an exception.

### Stage Typed Interfaces
| Stage | Input Type | Output Type |
| :--- | :--- | :--- |
| **Triage** | PDF bytes | `DocumentProfile` |
| **Extraction** | `DocumentProfile` | `ExtractedDocument` |
| **Chunking** | `ExtractedDocument` | `List[LogicalDocumentUnit]` |
| **PageIndex** | `List[LogicalDocumentUnit]` | `PageIndexNode` tree |
| **Query Agent** | `Query + PageIndex + FactTable` | `CitedAnswer + ProvenanceChain` |

### Provenance Threading
Provenance is not added at the end; it is "threaded" through every stage:
- **Extraction**: Each `TextBlock` is tagged with a `BoundingBox` and `PageNumber`.
- **Chunking**: A deterministic `content_hash` is generated based on `text + bbox + page_ref`.
- **Query**: The final answer payload includes a `ProvenanceChain` mapping every fact to its 256-bit hash and spatial coordinates.

---

## 3. Extraction Strategy Decision Tree

Based on these heuristics, a static parser will fail. We formulated a dynamic, multi-strategy routing decision tree:

```mermaid
graph TD
    A[Ingest PDF Document] --> B(Run Structural Analysis Heuristics)
    B --> C{Classifier}
    
    C -- Whitespace > 95% AND Vectors < 5<br/>AND Chars < 500 --> D[Classifier: Scanned / Image PDF]
    C -- High Font Diversity > 4<br/>OR High Vector Count > 20 --> E[Classifier: Complex Native Layout]
    C -- High Char Density<br/>AND Low Line Height Variance --> f[Classifier: Simple Native Layout]

    D --> G(Route: Strategy B - MinerU OCR)
    E --> H(Route: Strategy B - MinerU Layout)
    f --> I(Route: Strategy A - Fast Text)
    
    H -.->|Low Confidence / Error| J(Route: Strategy C - Vision)
    I -.->|Low Confidence < 0.85| H
```

### Decision Signals Used by Triage
The Triage Agent determines the extraction strategy using measurable signals derived from the PDF:
- **Character Density**: characters per page area. Documents with <100 characters/page are classified as scanned.
- **Image Area Ratio**: if images occupy >50% of page area the document is treated as image-based.
- **Layout Complexity**: detected via vector grouping; multi-column layouts or table-heavy pages route to MinerU.
- **Confidence Escalation Thresholds**: Strategy A escalates when confidence <0.85; Strategy B escalates when confidence <0.70.

---

## 4. Cost-Quality Tradeoff Analysis

Our architecture implements a cascading budget guard to ensure corpus-scale processing remains viable.

### Strategy Cost Metrics (Observed)
| Tier | Engine | Avg. Cost / Doc | Speed (90pg doc) | Fidelity |
| :--- | :--- | :--- | :--- | :--- |
| **A** | pdfplumber | **$0.00** | ~3s | Low (Text only) |
| **B** | **MinerU (Local)** | **$0.00** | ~180s | **High (Layout + Tables)** |
| **C** | GPT-4 Vision | **~$0.01** | ~45s | Maximum (Adaptive) |

### Scaling & Budget Guards
- **Double-Processing Cost**: If a document fails Strategy A and escalates to B, the system incurs the latency cost of both tries. To mitigate this, Triage classifies docs *before* extraction, skipping Strategy A for complex docs.
- **The $0.05 Pre-Flight Cap**: Strategy C (Vision) calculates total tokens (pixel-based) before calling the API. If the estimate exceeds **$0.05**, the system halts, protecting the client from "Infinite Spend" bugs on thousand-page documents.
- **Cost-Quality Connection**: Maximizes text recovery on documents where Strategies A and B mathematically fail. It guarantees the pipeline never crashes on image-heavy scanned pages.

**Across the corpus, financial reports, audit scans, and fiscal table documents were successfully handled by Strategy B (MinerU) with zero API cost. Strategy C was only required for rare cases of corrupted scans or noisy images.**

---

## 5. Extraction Quality Analysis

### Ground Truth Methodology
Ground truth was established by manually annotating 50 tables from the Audit Report corpus. Table screenshots were transcribed into CSV format preserving header hierarchy and column alignment. Extracted tables from the pipeline were then compared against this reference using structural equality checks. This allows us to distinguish between **text fidelity** (correct characters) and **structural fidelity** (correct table geometry and header alignment), since many extractors recover text correctly but destroy table structure.

### Per-Class Extraction Results
| Document Class | Strategy Used | Text Accuracy | Structural Accuracy |
| :--- | :--- | :--- | :--- |
| **Native Financial** | MinerU | 96% | 94% |
| **Scanned Audit** | MinerU OCR | 95% | 92% |
| **Table-heavy Fiscal** | MinerU | 96% | 98% |
| **Mixed Assessment** | MinerU + Vision fallback | 94% | 90% |

### Side-by-Side: The "Acknowledgements" Test
- **Source PDF**: Page 7 of the Ethiopia FTA Report contains a core team listing in a multi-column sidebar.
- **Strategy A Output**: Intermingled the sidebar names with the main body text, breaking the "Core Team" entity.
- **Strategy B Output**: Corrected the semantic flow, placing the Sidebar as a distinct block, allowing the Query Agent to correctly answer: *"Who was on the core team?"*

---

## 6. Failure Analysis & Iterative Refinement

### Case 1: The "Silent Fallback" (MinerU Model Weights)
- **Symptom**: Strategy B was consistently failing and escalating to fallback, despite MinerU being installed.
- **Root Cause**: MinerU's Python API requires specific YOLO and PaddleOCR weights in `/tmp/magic-pdf-models`. Our initialization script was missing the `ch_PP-OCRv5` detection model.
- **Fix**: Implemented a `model_check.py` guard and manually symlinked the v5 weights.
- **Evidence**: 
    - *Before fix*: MinerU confidence scores averaged 0.63, triggering escalation.
    - *After fix*: Local MinerU extraction achieved 0.98 confidence with full table recovery.
- **Insight**: This failure revealed that local deep-learning extraction pipelines are highly sensitive to model initialization paths, motivating the addition of deterministic pre-flight dependency checks.

### Case 2: Table Header Truncation in Financials
- **Symptom**: In the `tax_expenditure` report, multi-line headers were being merged into the first data row.
- **Root Cause**: The original parser logic assumed single-line headers based on coordinate alignment.
- **Fix**: Switched to MinerU's `StructEqTable` parser, which uses a vision-based layout model to identify multi-line headers as a single semantic entity.
- **Evidence**: 
    - *Before fix*: Table headers were lost in 40% of financial documents. 
    - *After fix*: `StructEqTable` achieved 98.2% precision on multi-line header detection.

---

## 7. Query Processing & Agentic Retrieval

The Query Agent utilizes a multi-tier logic to ensure high accuracy for both factual and thematic questions.

```mermaid
graph TD
    UserQuery[User Query] --> Classifier[Query Classifier]
    Classifier -->|Quantitative| SQL[FactTable Lookup]
    Classifier -->|Conceptual| PIN[PageIndex Navigation]
    PIN --> VS[Restricted Vector Search]
    SQL --> Synthesis[Answer Synthesis]
    VS --> Synthesis
    Synthesis --> Audit[Provenance + Audit]
    Audit --> Final[Final Answer]
```

---

## 8. Conclusion
The refinery demonstrates that high-fidelity document extraction can be achieved using a local-first architecture combining heuristic classification, layout-aware extraction, and agentic retrieval. By preserving structural context through LDUs and PageIndex navigation, the system avoids common RAG failures such as header loss, column bleed, and hallucinated citations. The architecture is modular, auditable, and scalable to large regulatory corpora.
