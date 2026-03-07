# Document Intelligence Refinery: System Architecture

This document details the technical architecture, implementation logic, and data flow of the Document Intelligence Refinery (Phase 0 to Phase 4).

---

## 🗺️ Pipeline Architecture (End-to-End)

The refinery follows a tiered, agentic approach to document processing, prioritizing local extraction and fallback to VLMs only when confidence is low.

```mermaid
graph TD
    A[PDF Document] --> B[Phase 1: Triage Agent]
    B -->|Digital Native| C[Strategy A: FastText]
    B -->|Complex/Scanned| D[Strategy B: MinerU]
    C -->|Low Conf < 0.85| D
    D -->|Fail/OOM/Low Conf < 0.70| E[Strategy C: Vision]
    
    subgraph "Extraction Strategy B (MinerU)"
        D1[YOLO Layout Detection] --> D2[PaddleOCR Text Recognition]
        D2 --> D3[StructEqTable/TableMaster]
        D3 --> D4[Structured JSON Output]
    end
    
    D4 --> F[Phase 3: Chunking Engine]
    F --> G[Logical Document Units (LDUs)]
    
    subgraph "Indexing layer"
        G --> H[FactExtractor]
        H --> I[facts.db (SQLite)]
        G --> J[VectorStore (ChromaDB)]
        G --> K[PageIndex Builder]
    end
    
    K --> L[Phase 4: Query Agent]
    I --> L
    J --> L
    L --> M[Reasoner & Auditor]
    M --> N[Final Citied Answer]
```

---

## 🛠️ Implementation Details

### Phase 1: Triage Agent
- **Origin Detection**: Uses `char_density`, `ink_density`, and `whitespace_ratio` to classify documents as `DIGITAL_NATIVE` or `SCANNED`.
- **Strategy Selection**:
    - `DIGITAL_NATIVE` & `SIMPLE_LAYOUT` → Strategy A.
    - `SCANNED` or `MULTI_COLUMN` → Strategy B (MinerU).

### Phase 2: Extraction Engine & Strategy Escalation
We employ a 3-tier circuit-breaker strategy to balance cost and fidelity.

| Strategy | Engine | Trigger | Budget |
| :--- | :--- | :--- | :--- |
| **Strategy A** | FastText (pdfplumber) | Default for Digital Docs | $0.00 |
| **Strategy B** | **MinerU ( magic-pdf )** | Scanned OR A < 0.85 confidence | $0.00 |
| **Strategy C** | Vision (GPT-4) | B < 0.70 OR B failure (OOM/Error) | $0.05 cap |

#### 💎 Confidence Scoring
**Formula**: `0.4 * completeness + 0.3 * layout + 0.2 * structural + 0.1 * ocr`
- **Signals**: `completeness_ratio`, `layout_consistency`, `structural_fidelity`, `ocr_quality`.
- **Normalization**: All signals are normalized to a `[0.0 - 1.0]` range.

#### 🚦 Failure Handling
If Strategy C fails, or its pre-flight budget ($0.05) is exceeded:
- Document is marked as **FAILED** in the `extraction_ledger.jsonl`.
- Extraction halts and the document is **flagged for manual review**.

### Phase 3: Semantic Chunking & Indexing
- **LDU Formation**: Converts raw blocks into **Logical Document Units (LDUs)**.
- **Token Limit**: `MAX_LDU_TOKENS = 800`. Oversized chunks are split at sentence/row boundaries with a contextual overlap buffer.
- **FactTable (SQL)**: Every LDU is processed by the `FactExtractor` (GPT-4o-mini) to extract entities, values, units, and time periods into **SQLite (`facts.db`)**.
- **VectorStore (ChromaDB)**: Stores embeddings (`all-MiniLM-L6-v2`) for conceptual retrieval.
    - **Metadata Keys**: `document_name`, `page_refs`, `bounding_box`, `chunk_type`, `parent_section`, `content_hash`.

### Phase 4: Query Routing & Provenance
The **Query Agent** uses a tiered routing approach:
1.  **QueryClassifier**: Determines if the query is **Quantitative** (e.g., "What is the total?") or **Conceptual**.
2.  **Routing**:
    - **Quantitative** → Structured SQL lookup on `facts.db`.
    - **Conceptual** → `PageIndex.navigate()` to find the section "map", then restricted Vector Search.
3.  **Synthesis**: Strictly enforces that every answer contains a **Provenance Chain** (Doc Name, Page, Hash, BBox).
4.  **Audit**: A second pass by the `ClaimAuditor` verifies the synthesis against the original source text.

---

## 🖼️ Handling Complex Elements

### 1. Table Extraction (MinerU)
MinerU uses **YOLO** for detection and **StructEqTable** for recognition to preserve merged cells and headers, outputting clean Markdown/HTML.

### 2. Image & Figure Handling
Figures are extracted with absolute bounding boxes. The `ChunkingEngine` performs contextual linking (e.g., "See Figure 4") to create bidirectional relationships between text and images.

### 3. PageIndex Summarization
We use **Gemini-1.5-Flash** to create semantic summaries of every section. This results in a "Human-like" Table of Contents that eliminates retrieval noise.
