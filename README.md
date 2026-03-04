# Document Intelligence Refinery

A robust, multi-strategy document intelligence system designed for precise triage, extraction, and semantic chunking of diverse PDF documents. 

It intelligently routes documents to the most cost-effective extraction strategy based on deterministic triage heuristics.

## Architecture & Strategies

**Triage Phase:** The Triage Agent analyzes documents to determine their origin (Digital vs. Scanned), Layout Complexity, Domain, and Language.

**Extraction Phase:**
- **Strategy A (Fast Text):** Uses `pdfplumber` for structured, single-column digital-native documents.
- **Strategy B (Layout-Aware):** Uses `docling` to parse multi-column, table-heavy, and complex layouts.
- **Strategy C (Vision-Augmented):** Uses Vision Language Models (VLMs) via OpenRouter for heavily visual, scanned, or fallback documents. It is protected by strict budget guards.

## Setup Instructions

### 1. Prerequisites
- **Python 3.11+**
- **[uv](https://github.com/astral-sh/uv)** (Fast Python package manager)

### 2. Installation
Install the project dependencies and set up the virtual environment automatically using `uv`:

```bash
uv sync
```

### 3. Environment Configuration
Copy the provided environment example file to instantiate your local `.env`:

```bash
cp .env.example .env
```

Open your new `.env` file and insert your OpenRouter API key (mandatory for Strategy C Vision processing):
```env
OPENROUTER_API_KEY="sk-or-v1-..."
```

## Running the Pipeline

You can run the Phase 2 extraction orchestrator script, which evaluates test PDFs and routes them accordingly:

```bash
uv run scripts/run_phase2.py
```

### Reviewing Telemetry & Output
The application outputs deterministic artifacts for every extraction attempt:

1. **Extracted Payloads:** Stored securely as JSON records governed by a strict internal schema.
   - `Location:` `.refinery/extracted/<file_hash>.json`
2. **Extraction Ledger:** An atomic JSONL file recording extraction metrics, latency, confidence scores (`[0.0, 1.0]`), fallback triggers, and total budget consumed per document.
   - `Location:` `.refinery/extraction_ledger.jsonl`
