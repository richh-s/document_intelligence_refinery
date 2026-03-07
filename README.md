# Document Intelligence Refinery

A robust, multi-strategy document intelligence system designed for precise triage, extraction, and semantic chunking of diverse PDF documents. 

It intelligently routes documents to the most cost-effective extraction strategy and provides an agentic RAG system with mathematically verifiable provenance.

## 🚀 Key Features

*   **Intelligent Triage**: Automatically detects document types (Financial, Legal, Survey) and Layout Complexity.
*   **Multi-Strategy Extraction**: Routes to Strategy A (pdfplumber), B (docling), or C (Vision VLMs) based on triage.
*   **PageIndex Navigation**: High-level hierarchical indexing of document sections for efficient retrieval.
*   **Agentic Query Interface**: A LangGraph agent that routes queries to either structured SQL (Quantitative) or semantic Vector (Conceptual) backends.
*   **Audit Mode**: Third-party verification of agent claims using LLM-based auditing and deterministic fallback checks.
*   **Provenance Enforcement**: Every answer is strictly cited with `document_name`, `page_number`, and `bounding_boxes`.

## 🛠️ Architecture

1.  **Triage Agent**: Determines the origin (Digital vs. Scanned) and Domain.
2.  **Extraction Logic**: Implements 5 chunking rules (Text, Table, List, Figure, Header) via `ChunkingEngine`.
3.  **Indexing Engine**: Builds the `PageIndex` tree and populates `RefineryVectorStore` (ChromaDB) and `FactTableStore` (SQLite).
4.  **Query Agent**: Orchestrates retrieval and synthesis with mandatory source verification.

## 📦 Setup Instructions

### 1. Prerequisites
- **Python 3.11+**
- **Virtual Environment** (e.g., `.venv`)

### 2. Installation
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 3. Environment Configuration
Create a `.env` file with your OpenRouter credentials:
```env
OPENROUTER_API_KEY="your-sk-or-v1-key-here"
```

## 🎮 How to Test

### 1. Extraction Pipeline (Phase 2 & 3)
Run the orchestrator to process your PDFs and create LDUs:
```bash
PYTHONPATH=src python scripts/run_phase2.py
```

### 2. Manual Query Interface (Phase 4 & 5)
Interactive manual testing via the terminal:
```bash
# Ask a conceptual question (Vector)
PYTHONPATH=src python scripts/ask.py "Who prepared the report?"

# Ask a quantitative question (SQL)
PYTHONPATH=src python scripts/ask.py "What is the report_year?"
```

### 3. Automated Verification
Run the end-to-end operational test and guardrail checks:
```bash
# Core operational test (SQL + Vector + Audit)
PYTHONPATH=src python scripts/test_phase4_operational.py

# Rubric Compliance: Anti-Hallucination Guardrail
PYTHONPATH=src pytest tests/agents/test_provenance_enforcement.py -v

# Rubric Compliance: SQL Injection Protection
PYTHONPATH=src pytest tests/indexing/test_fact_table_queries.py -v
```

## 📂 Artifacts
The application generates the following deterministic artifacts in the `.refinery/` directory:
- `.refinery/extraction_ledger.jsonl`: Full performance and confidence logs.
- `.refinery/pageindex/`: JSON representation of document hierarchies.
- `.refinery/extracted/`: Structured JSON output of raw extractions.
- `.refinery/facts.db`: Persistent SQLite store for quantitative data.
- `.refinery/ldus_sample.json`: Exported Logical Document Units for rubric validation.

