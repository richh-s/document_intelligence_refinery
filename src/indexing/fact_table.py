"""Structured quantitative data indexing via SQLite FactTable."""

import json
import sqlite3
import logging
from typing import List, Optional, Dict
from pathlib import Path
import httpx
from pydantic import BaseModel

from models.ldu import LogicalDocumentUnit
from models.page_index import PageIndexNode

logger = logging.getLogger(__name__)

class Fact(BaseModel):
    """Schema for a single extracted numerical fact."""
    entity: str
    value: str
    unit: Optional[str] = None
    time_period: Optional[str] = None
    fact_type: Optional[str] = None # e.g. "actual", "projected"


class FactExtractor:
    """Uses Gemini 1.5 Flash to pull structured facts from numerical chunks."""
    
    SYSTEM_PROMPT = """You are a meticulous financial analyst.
Extract key quantitative facts from the provided document chunk. Focus on metrics, financials, and major quantities.
For each fact, extract:
- entity: The name of the metric (e.g. "Revenue", "Operating Costs")
- value: The exact value (e.g. "4.2B", "150,000")
- unit: The currency or measurement (e.g. "USD", "%") - OPTIONAL
- time_period: The scope (e.g. "Q3 2024", "FY23") - OPTIONAL
- fact_type: Is this an "actual" result, a "projected" forecast, or "historical"? - OPTIONAL

Return ONLY a JSON array of these objects:
[
  {"entity": "Revenue", "value": "4.2", "unit": "Billion USD", "time_period": "Q3 2024", "fact_type": "actual"}
]
If there are no clear financial or numerical data points, return an empty array []."""

    def __init__(self, api_key: str, model: str = "google/gemini-1.5-flash", timeout: float = 60.0):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        
    def extract_from_ldu(self, ldu: LogicalDocumentUnit) -> List[Fact]:
        """Runs the LLM over the LDU and returns parsed factual rows."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": ldu.content}
            ]
        }
        
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"].strip()
                
                # Cleanup markdown and parse
                if content.startswith("```json"):
                    content = content[7:]
                if content.endswith("```"):
                    content = content[:-3]
                    
                parsed = json.loads(content)
                
                # Check if it returned a raw list or a dict containing a list
                facts_list = parsed if isinstance(parsed, list) else parsed.get("facts", [])
                
                return [Fact(**f) for f in facts_list]
        except Exception as e:
            logger.error(f"Failed to extract facts from LDU {ldu.content_hash}: {e}")
            return []


class FactTableStore:
    """Manages the SQLite `.refinery/facts.db` table."""
    
    def __init__(self, db_path: str = ".refinery/facts.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()
        
    def _init_db(self):
        """Creates the fact table schema."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_name TEXT NOT NULL,
                section_title TEXT,
                entity TEXT NOT NULL,
                value TEXT NOT NULL,
                unit TEXT,
                time_period TEXT,
                fact_type TEXT,
                source_hash TEXT NOT NULL
            )
        """)
        self.conn.commit()
        
    def ingest_facts(self, doc_name: str, section_title: Optional[str], source_hash: str, facts: List[Fact]):
        """Inserts extracted facts into SQLite."""
        if not facts:
            return
            
        cursor = self.conn.cursor()
        for f in facts:
            cursor.execute("""
                INSERT INTO facts (document_name, section_title, entity, value, unit, time_period, fact_type, source_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (doc_name, section_title, f.entity, f.value, f.unit, f.time_period, f.fact_type, source_hash))
        self.conn.commit()
        
    def query(self, sql_query: str) -> List[Dict]:
        """Executes a strictly READ-ONLY SELECT query against the facts table."""
        if not sql_query.strip().upper().startswith("SELECT"):
            raise ValueError("Only SELECT queries are permitted on the FactTable.")
            
        cursor = self.conn.cursor()
        cursor.execute(sql_query)
        
        # Convert to dictionary output
        columns = [description[0] for description in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        return results
