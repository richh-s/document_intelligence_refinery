# Document Intelligence Refinery

This document details the architectural structure and implementation details of the Document Intelligence Refinery pipeline, covering Phase 0 to Phase 3.

## Directory Structure

```text
document_intelligence_refinery/
├── pyproject.toml              # Dependency and project configuration (uv)
├── extraction_rules.yaml       # Empirical rules, thresholds, and budget caps
├── DOMAIN_NOTES.md             # Threshold justifications and experimental findings
├── README.md                   # Quickstart and overview
├── .refinery/                  # Local persistence directory
│   ├── profiles/               # DocumentProfile JSON outputs (named by doc_id)
│   └── ledgers/                # Extraction cost & confidence tracking ledgers
├── scripts/                    # CLI tools for evaluation and testing
│   ├── evaluate_retrieval.py   # RAG precision/recall evaluation scripts
│   └── compare_extractions.py  # Script for testing strategy outputs
├── src/document_intelligence_refinery/
│   ├── __init__.py
│   ├── config.py               # Global settings & environment variables
│   ├── models.py               # Core Pydantic data models (DocumentProfile, ConfidenceScores)
│   ├── persistence.py          # Profile serialization & storage logic
│   ├── sampling.py             # SmartSampler for deterministic pagination
│   ├── schema.py               # Normalized extraction schemas (ExtractedDocument, TextBlock)
│   ├── triage.py               # Phase 1: Main TriageAgent orchestrator
│   │
│   ├── detectors/              # Phase 1: Classification layers
│   │   ├── domain.py           # DomainHintClassifier (FINANCIAL, LEGAL, TECHNICAL, etc.)
│   │   ├── language.py         # langdetect wrapper for document language
│   │   ├── layout.py           # LayoutComplexityDetector (vector & font grouping)
│   │   └── origin.py           # OriginTypeDetector (DIGITAL_NATIVE, SCANNED, etc.)
│   │
│   ├── extractors/             # Phase 2: Multi-Strategy Extractors
│   │   ├── base.py             # BaseExtractor and Result models
│   │   ├── router.py           # ExtractionRouter (Circuit breakers & fallback escalation)
│   │   ├── ledger.py           # Atomic locked ExtractionLedger for observability limits
│   │   ├── validator.py        # Validates extracted schemas meet minimum quality
│   │   ├── fast_text.py        # Strategy A (pdfplumber) - Low Cost, strict nonsense rejection
│   │   ├── layout.py           # Strategy B (Docling) - Medium cost, handles complex grids
│   │   └── vision.py           # Strategy C (VLM OpenRouter) - High cost, pre-flight budget checks
│   │
│   ├── chunking/               # Phase 3: RAG Semantic Chunking Engine
│   │   ├── engine.py           # ChunkingEngine (enforces Constitution & relationship mapping)
│   │   ├── hasher.py           # Deterministic spatial origin hashing (provenance tracking)
│   │   ├── models.py           # LogicalDocumentUnit (LDU) & LDUMetadata schemas
│   │   └── validator.py        # Validates tokens and bounding box rules prior to index insertion
│   │
│   └── indexing/               # Phase 3: Semantic Storage & Retrieval
│       ├── vector_store.py     # ChromaDB wrapper for LDUs and PageIndex nodes
│       ├── page_index.py       # Hierarchical document summarization map via gemini-1.5-flash
│       └── query.py            # HybridRetriever (Tree + Vector constraints)
│
└── tests/                      # Pytest verification suites
    ├── conftest.py
    ├── test_models.py
    ├── test_sampling.py
    ├── test_detectors.py       # Triage assertions (char_density, whitespaces)
    ├── test_triage.py
    ├── chunking/               # Asserts semantic relationships & spatial hashes
    └── indexing/               # Validates metadata filtering & Chroma vectors
```

---

## Technical Implementation Details

### Phase 1: Triage Agent & Classification
**Goal:** Ingest a PDF securely and output a deterministic `DocumentProfile` that dictates all downstream rules.
*   **Determinism & Persistence**: Results are rounded to 6 decimal places. The `TriageAgent` saves JSON payloads to `.refinery/profiles/{doc_id}.json` using sorted keys and UTF-8 enforcement to ensure reproducibility.
*   **Origin Type (`detectors/origin.py`)**: Computes `char_density` (chars/page area), `whitespace_ratio`, and `ink_density` (bbox geometry sum). This is mathematically superior to raw char counts for sparse digital covers.
*   **Layout Type (`detectors/layout.py`)**: Groups graphical vectors using Union-Find algorithms. Prevents a 500-cell table from simulating 500 individual graphics, mapping to 5 discrete enum classes seamlessly.
*   **Domain Hint (`detectors/domain.py`)**: Uses case-insensitive regex word boundaries (`\b`) over static word lists, mapping into strict `FINANCIAL`, `LEGAL`, `TECHNICAL`, `MEDICAL`, and `GENERAL` classifications.

### Phase 2: Multi-Strategy Extraction Engine
**Goal:** Route documents to the cheapest, most effective parser. Execute orderly fail-overs via a circuit breaker if data fidelity drops.
*   **Unified Schema (`schema.py`)**: Every backend must resolve to `ExtractedDocument`, packing standard `TextBlock` and `StructuredTable` arrays. We enforce normalized geometries (`[0.0, 1.0]`) globally to resolve axis inversions (e.g., Docling uses Bottom-Left; pdfplumber uses Top-Left).
*   **ExtractionRouter (`router.py`)**: The central circuit breaker. Analyzes the `OriginType` and `LayoutComplexity` to dispatch the task. If confidence dips below `0.85` (e.g., due to sparse characters or missing pages), the process catches the failure and bumps the document to a heavier tier.
*   **Strategy A (pdfplumber)**: Extremely fast CPU extraction. Employs `NONSENSE_RATIO_MAX` (30%) N-gram checkers to catch and immediately fail documents containing corrupted hidden OCR layers.
*   **Strategy B (Docling)**: Extracts reading-order-aware tabular grids and complex multi-column wraps, transforming them into markdown syntax.
*   **Strategy C (Vision VLM)**: Dispatches PDF pages as base64 images to `gpt-4o-mini` (via OpenRouter). Employs **Pre-Flight Budget Checks**, throwing immediate `BudgetExceededError`s if the estimated document token slice exceeds `$0.05`. Ensures conversational markdown wrapper texts are forcefully stripped to retain strict JSON.
*   **Ledger Mechanics (`ledger.py`)**: Utilizes POSIX `fcntl` file locks ensuring multi-threading doesn't corrupt the logging of `strategy_used`, `cost_estimate`, `processing_time`, and `confidence_score` arrays.

### Phase 3: Semantic Chunking Engine & PageIndex
**Goal:** Convert disjointed extraction frames into RAG-safe Logical Document Units (LDUs) ready for high-fidelity vector searches.
*   **Semantic Data Contracts (`chunking/models.py`)**: Each chunk translates to an `LDU` strictly enforcing token counts, absolute bounding box positions, and rigorous metadata sets (`relations`, `cross_reference_type`).
*   **Constitution rules in `ChunkingEngine`**:
    *   **Context Prepending**: Long lists are cut to fit token bounds, but subsequent pieces receive a `[Context: ... (Continued)]` injection so indexers don't lose the anchor subject.
    *   **Cross-Reference Resolutions**: A dynamic 2nd-pass inspects text (e.g., `"As shown in Figure 2"`), scans the memory structure for `Figure 2`'s chunk, and links the precise 256-bit `content_hash` natively into the referencing LDU's parent arrays.
*   **Provenance Spatial Hashing**: Calculates `hash(content + rounded_bbox + pagerefs + type)`. If a PDF table structure shifts by even a pixel in a future version limit test, the hash changes, catching supply-chain document tampering.
*   **PageIndex & ChromaDB (`indexing/vector_store.py`)**:
    *   Top-level headers are bundled and dispatched to `gemini-1.5-flash` to write semantic summaries of the chapter.
    *   These form the `PageIndex`. When a query is initiated ("Show me revenue distributions"), the `HybridRetriever` searches the Tree first to grab the optimal 3 chapters, then heavily restricts the Vector Store (ChromaDB) to search **only** those chapters, terminating false-positive hallucination links.
    *   The raw results strictly return the original metadata, retaining line-by-line traceability back to the `ExtractedDocument` payload.
