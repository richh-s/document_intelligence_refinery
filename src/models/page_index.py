"""PageIndex tree builder for semantic section summaries."""

import json
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
    children: List['PageIndexNode'] = []


class PageIndexBuilder:
    """
    Constructs the PageIndex tree by analyzing the LDU hierarchy and using
    gemini-1.5-flash to generate fast, localized abstracts for each section.
    """
    
    SYSTEM_PROMPT = """You are a highly capable document analyst. 
Provided with a JSON mapping of section IDs to their extracted text content, generate a concise but highly descriptive summary (under 2 sentences) for EACH section. 
Capture the main topics, key metrics, and core arguments to enable precise semantic search queries. 

You MUST return ONLY a valid JSON object mapping the section IDs to their summaries, like this:
{
  "section_0": "Summary text...",
  "section_1": "Summary text..."
}
Do not include any markdown formatting or extra text outside the JSON object."""
    
    def __init__(
        self, 
        api_key: str, 
        primary_model: str = "google/gemini-1.5-flash", 
        fallback_model: str = "openai/gpt-4o-mini",
        timeout: float = 120.0
    ):
        self.api_key = api_key
        self.primary_model = primary_model
        self.fallback_model = fallback_model
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
            
        if not sections:
            return []
            
        # Prepare the batched payload
        batch_payload = {}
        for s_id, s_data in sections.items():
            if s_data["title"] == "Unknown Section":
                s_data["title"] = f"Section: {s_id}"
            
            # Truncate section content to prevent overwhelming the context window per section, though Flash is huge
            full_text = "\n".join(s_data["content"])
            batch_payload[s_id] = full_text[:10000] 
            
        # Attempt Primary Model
        summaries = self._generate_batched_summaries(batch_payload, self.primary_model)
        
        # Fallback to GPT-4o-mini if primary fails or returns invalid JSON
        if not summaries:
            logger.warning(f"Primary model {self.primary_model} failed. Falling back to {self.fallback_model}.")
            summaries = self._generate_batched_summaries(batch_payload, self.fallback_model)
            
        if not summaries:
            logger.error("Both primary and fallback models failed to generate PageIndex summaries.")
            summaries = {}
            
        # Build Nodes
        nodes = []
        for s_id, s_data in sections.items():
            summary = summaries.get(s_id, "Summary unavailable.")
            nodes.append(PageIndexNode(
                section_id=s_id,
                section_title=s_data["title"],
                summary=summary
            ))
            
        return nodes
        
    def _generate_batched_summaries(self, section_map: Dict[str, str], model: str) -> Optional[Dict[str, str]]:
        """Invoke the OpenRouter VLM to generate a batched JSON summary response."""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # We enforce JSON response format natively where supported
        payload = {
            "model": model,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(section_map)}
            ]
        }
        
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"].strip()
                
                # Strip markdown code blocks if the model ignored the system prompt
                if content.startswith("```json"):
                    content = content[7:]
                if content.endswith("```"):
                    content = content[:-3]
                    
                parsed_json = json.loads(content)
                if not isinstance(parsed_json, dict):
                    raise ValueError("Response is not a JSON dictionary.")
                return parsed_json
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse batched JSON summaries from {model}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error generating batched summaries via API ({model}): {e}")
            return None
