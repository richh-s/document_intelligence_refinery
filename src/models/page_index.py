"""PageIndex tree builder for semantic section summaries."""

import logging
import httpx
from typing import List, Dict, Optional, Any
from pydantic import BaseModel
from models.ldu import LogicalDocumentUnit

logger = logging.getLogger(__name__)


class PageIndexNode(BaseModel):
    """A hierarchical semantic index node storing section abstracts."""
    section_id: str
    section_title: str
    summary: str
    embedding: Optional[List[float]] = None
    child_sections: List[str] = []


class PageIndexBuilder:
    """
    Constructs the PageIndex tree by analyzing the LDU hierarchy and using
    gemini-1.5-flash to generate fast, localized abstracts for each section.
    """
    
    SYSTEM_PROMPT = """You are a highly capable document analyst. 
Provided with the extracted text from a specific section of a document, generate a concise but highly descriptive summary (under 100 words). 
Capture the main topics, key metrics, and core arguments to enable precise semantic search queries. Return ONLY the summary text."""
    
    def __init__(self, api_key: str, model: str = "google/gemini-1.5-flash", timeout: float = 60.0):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        
    def build_index(self, ldus: List[LogicalDocumentUnit]) -> List[PageIndexNode]:
        """Traverse LDUs, group by section, and generate PageIndexNodes."""
        if not self.api_key:
            logger.warning("No OPENROUTER_API_KEY provided; returning empty PageIndex.")
            return []
            
        # Group content by section_id
        sections: Dict[str, Dict[str, Any]] = {}
        for ldu in ldus:
            s_id = ldu.parent_section_id
            if not s_id:
                continue
                
            if s_id not in sections:
                # Naive title extraction (assume the first header chunk observed for this ID is the title)
                sections[s_id] = {"title": "Unknown Section", "content": []}
                
            if ldu.chunk_type == "header" and sections[s_id]["title"] == "Unknown Section":
                sections[s_id]["title"] = ldu.content
                
            sections[s_id]["content"].append(ldu.content)
            
        nodes = []
        for s_id, s_data in sections.items():
            full_text = "\n".join(s_data["content"])
            # Fallback title if no header was found
            if s_data["title"] == "Unknown Section":
                s_data["title"] = f"Section: {s_id}"
                
            summary = self._generate_summary(full_text)
            if summary:
                nodes.append(PageIndexNode(
                    section_id=s_id,
                    section_title=s_data["title"],
                    summary=summary
                ))
            else:
                logger.error(f"Failed to generate summary for section {s_id}")
                
        return nodes
        
    def _generate_summary(self, text: str) -> Optional[str]:
        """Invoke the OpenRouter VLM to generate a section summary."""
        # Truncate to reasonable context if absolutely massive just for cost protection 
        # (Though Flash handles 1M, let's keep it under, say, 20k chars for speed)
        truncated_text = text[:80000] 
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"Summarize this section:\n\n{truncated_text}"}
            ]
        }
        
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error(f"Error generating section summary via API: {e}")
            return None
