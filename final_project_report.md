# Document Intelligence Refinery - Final Report

This report consolidates Phase 0 deliverables (Domain Notes), extraction strategies, architecture definitions, and cost analyses for the Document Intelligence Refinery pipeline.

---

## 1. Domain Notes (Phase 0 Deliverable)

An empirical structural analysis of Ethiopian regulatory and financial PDF documents was conducted to determine optimal extraction pathways.

### Document Characteristics & Classification (4 Corpus Classes)
The analyzed documents were empirically evaluated across 4 distinct document classes from the corpus, demonstrating significant layout complexity and structural variance:
- **1. Native Financial**: 
  - *Corpus Example*: `CBE ANNUAL REPORT 2023-24.pdf`
  - *Characteristics & Failure Modes*: Highly complex native layouts with intense font diversity and repeating vector grid lines. Causes pure-text parsers to fail on column reading order, and occasionally causes Docling to hallucinate structure (returning up to 2.0x raw characters).
- **2. Scanned Legal / Administrative**:
  - *Corpus Example*: `Audit Report - 2023.pdf`
  - *Characteristics & Failure Modes*: Visually dense but structurally empty (high whitespace, ~116 raw chars via `pdfplumber` vs ~539k chars via OCR). Fails natively on all standard text extraction approaches without Vision/OCR pipelines.
- **3. Table-Heavy Fiscal**:
  - *Corpus Example*: `tax_expenditure_ethiopia_2021_22.pdf`
  - *Characteristics & Failure Modes*: High vector density delineating massive financial tables. Frequently causes header-inferencing to drop confidence in layout-aware models (e.g. only 24/29 headers detected natively).
- **4. Mixed Assessment**:
  - *Corpus Example*: `fta_performance_survey_final_report_2022.pdf`
  - *Characteristics & Failure Modes*: Blends standard digital layouts with massive vector infographics. Extremely prone to severe over-segmentation and garbage capture without precise layout-aware parsing logic.

### Structural Signals Used by the Triage Agent
The system uses several metrics to classify documents accurately prior to extraction:

- **Character Density** (`chars / page_area`): Helps determine if text is sparse or dense.
- **Whitespace Ratio**: Percentage of blank page area.
- **Vector Density**: Number of graphical vector objects detected.
- **Font Diversity**: Count of unique font families detected.

These signals determine the `origin_type`, `layout_complexity`, and `domain_hint`. For example:
High whitespace (>95%) combined with very low character density indicates scanned PDFs. High vector density and high font diversity typically indicate complex financial reports with tables and diagrams.

---

## 2. Failure Modes Observed Across Document Types

Rigorous cross-tool comparisons (`pdfplumber` vs `Docling`) revealed primary extraction failure modes:

### A. Extreme Over-Segmentation & Hallucination
In highly complex, natively digital PDFs with dense tables and sidebars, layout-aware models (like Docling) occasionally returned **1.6x to 2.0x** the character count extracted by pure text parsers.
- **Cause**: Heavy extractors often flatten surrounding metadata, duplicate headers, or hallucinate tabular grid lines as actual text characters when processing non-standard financial reports.

### B. Header Ratio Imbalance in Tables
While layout-aware models successfully detect tables, their ability to confidently detect the semantic *headers* drops significantly in complex documents:
- *Tax Expenditure Ethiopia*: 29 tables detected, 24 with inferred headers.
- *FTA Performance Survey*: 57 tables detected, only 25 with inferred headers.
- **Impact**: Without robust header row identification, downstream semantic extraction (RAG) degrades.

### C. The "Hidden OCR" Layer
Some digital "native" documents carry hidden, corrupted OCR text layers overlaid on scans. If ingested blindly, the parser returns garbage characters ("nonsense ratio"), destroying the index. 

---

## 3. Extraction Strategy Decision Tree

Based on these heuristics, a static parser will fail. We formulated a dynamic, multi-strategy routing decision tree:

```mermaid
graph TD
    A[Ingest PDF Document] --> B(Run Structural Analysis Heuristics)
    B --> C{Calculate Density & Vectors}
    
    C -- Whitespace > 95% AND Vectors < 5<br/>AND Chars < 500 --> D[Classifier: Scanned / Image PDF]
    C -- High Font Diversity > 4<br/>OR High Vector Count > 20 --> E[Classifier: Complex Native Layout]
    C -- High Char Density<br/>AND Low Line Height Variance --> F[Classifier: Simple Native Layout]

    D --> G(Route: Strategy C - Vision VLM)
    E --> H(Route: Strategy B - Layout-Aware)
    F --> I(Route: Strategy A - Fast Text)
    
    H -.->|Low Confidence / Grid Failure| G
    I -.->|Garbage OCR / High Nonsense| H
```

### Strategy Comparison Table

| Strategy | Tool | Trigger | Cost |
| :--- | :--- | :--- | :--- |
| Strategy A | pdfplumber | simple native PDFs | $0 |
| Strategy B | Docling | complex layout / tables | $0 |
| Strategy C | GPT-4o-mini | scanned or fallback | ~$0.01 |

### Extraction Confidence Scoring

Confidence scoring evaluates extraction quality for Strategy A and Strategy B. 

FastTextExtractor scores documents based on multiple signals:
- `character_count`
- `character_density`
- `image_to_page_area_ratio`
- `font_metadata_presence`

Additionally, a strict scanning heuristic is applied:
If `image_ratio > 0.5` AND `character_count < 50` → confidence is forced to `0.0`.
Low confidence immediately triggers router escalation.

### Confidence-Gated Escalation Guard

The `ExtractionRouter` enforces a tiered extraction pipeline relying on specific thresholds:
- Strategy A threshold: `0.85`
- Strategy B threshold: `0.70`
- Strategy C threshold: `0.60`

The escalation sequence proceeds as follows:
`FastTextExtractor`
↓ if confidence < threshold
`LayoutExtractor` (Docling)
↓ if confidence < threshold
`VisionExtractor` (VLM)

This prevents garbage extraction, hallucinated tables, and broken OCR text from corrupting the index.

---

## 4. Full 5-Stage Document Intelligence Pipeline

To execute the intelligent routing and chunking, the complete 5-Stage Pipeline architecture is defined below:

```mermaid
sequenceDiagram
    participant S as Storage (Provenance Ledger)
    participant T as Stage 1: Triage
    participant EX as Stage 2: Structure Extraction (Router & A, B, C)
    participant C as Stage 3: Semantic Chunking
    participant P as Stage 4: PageIndex Builder
    participant Q as Stage 5: Query Interface

    S->>T: Ingest Document
    T->>T: Analyze Origin, Domain, Layout
    T-->>EX: DocumentProfile (Heuristics)
    
    Note over EX: Route to optimal Strategy Tier
    
    alt is Simple Digital
        EX->>EX: Strategy A (Fast Text)
    else is Complex Layout
        EX->>EX: Strategy B (Docling)
    else is Scanned / Fallback
        EX->>EX: Strategy C (Vision)
    end
    
    Note over EX: If Confidence < threshold, escalate sequence A -> B -> C
    
    EX->>S: Write Audit/Provenance Ledger
    EX->>C: Clean Normalized ExtractedDocument
    
    Note over C: Enforce Chunking Constitution rules (context prepending, table integrity)
    
    C->>P: Validated Logical Document Units (LDUs)
    
    Note over P: LLM Summarize Sections (PageIndex Builder)
    P->>Q: Hierarchical Tree + LDU Vectors
    
    Note over Q: Query Interface retrieves via Hybrid Search
    Q->>S: Store to ChromaDB / Disk
```

### Extraction Observability Ledger

All routing and extraction actions are logged in `.refinery/extraction_ledger.jsonl`.
This ledger records the following fields for every extraction:
- `strategy_used`
- `confidence_score`
- `processing_time`
- `tokens_in`
- `tokens_out`
- `page_count`
- `cost_estimate`
- `error_category`

This atomic ledger provides crucial benefits including auditability, cost monitoring, failure analysis, and simple debugging of extraction quality over time.

---

## 5. Cost Analysis & Estimated Cost Per Document Tier

The extraction engine operates on a cascading cost-efficiency model, ensuring expensive operations are only used when absolutely necessary.

### **Strategy A: Fast Text (pdfplumber)**
- **Target**: Clean, simple, digital-native text.
- **Cost Aspect**: Pure standard local CPU compute. Extremely fast (under 2 seconds avg processing time), minimal RAM overhead.
- **Estimated API / Usage Cost**: **$0.00** per document.
- **Cost-Quality Connection**: Provides the cheapest and fastest extraction natively, but sacrifices visual and table fidelity which completely breaks down on fiscal reports.

### **Strategy B: Layout-Aware (Docling)**
- **Target**: Complex native layouts, heavy multi-column text, multi-page spanning tables.
- **Cost Aspect**: High CPU/GPU utilization and high RAM footprint. Can take several minutes (e.g., ~172 seconds for 95 pages) per document locally. Infrastructure overhead is required to scale, but no per-token API fees are triggered.
- **Estimated API / Usage Cost**: **$0.00** per document (excluding local cloud compute runtime).
- **Cost-Quality Connection**: Provides extremely high structural and tabular fidelity compared to Strategy A (restoring grid lines and read-order), but demands massive operational runtime and compute processing time.

### **Strategy C: Vision-Augmented (VLM via OpenRouter)**
- **Target**: Heavily scanned documents, corrupted PDFs, unreadable images, or structural router fallbacks.
- **Cost Aspect**: High external API dependency. Dispatches high-res base64 images to `gpt-4o-mini` (or equivalent).
- **Hard Cap Budget Guard**: Enforced at `GLOBAL_DOCUMENT_BUDGET_USD` = **$0.05** per document.
- **Budget Guard Algorithm**: Before dispatching a Vision batch, the system estimates the exact cost:
  `estimated_cost = (tokens_in / 1M * price_input) + (tokens_out / 1M * price_output)`
  If `estimated_cost` exceeds `GLOBAL_DOCUMENT_BUDGET_USD`, the VisionExtractor stops processing, returns a `PartialExtractionResult`, and logs a `BUDGET_CAP_HIT`.
- **Estimated Average Cost**:
  - Assuming standard $0.15 / 1M Input Tokens and $0.60 / 1M Output Tokens.
  - A standard 15-page scanned document yields ~30,000 input tokens and ~8,000 output tokens.
  - Calculation: `(30,000/1M * $0.15) + (8,000/1M * $0.60)` = `$0.0045 + $0.0048`
  - **Estimated Cost**: **~$0.01** per average document. Corpus variation limits costs strictly linearily based on token volume footprint, halting cleanly at the $0.05 cap for massive documents.
- **Cost-Quality Connection**: Maximizes text recovery on documents where Strategies A and B mathematically fail (triggering 0.0 confidence). It guarantees the pipeline never crashes on image-heavy scanned pages, but trades significant monetary API cost limitations and higher processing times for that fallback coverage.

---

## 6. Preparation for Semantic Retrieval

Extracted content from the pipeline is normalized into strict `ExtractedDocument` schemas. 
These objects will later be converted into Logical Document Units (LDUs) during Phase 3 semantic chunking, paving the way for seamless ingestion into the vector store and optimal hierarchical Retrieval-Augmented Generation (RAG).
