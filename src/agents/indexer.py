"""PageIndex tree builder for semantic section summaries."""

import json
import logging
import httpx
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from models.ldu import LogicalDocumentUnit

logger = logging.getLogger(__name__)


class PageIndexNode(BaseModel):
    """A hierarchical semantic index node storing section abstracts."""
    section_id: str
    title: str
    summary: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    key_entities: List[str] = Field(default_factory=list, description="Canonicalized named entities")
    data_types_present: List[str] = Field(default_factory=list, description="List of 'tables', 'figures', etc.")
    embedding: Optional[List[float]] = None
    child_sections: List['PageIndexNode'] = []


class PageIndexBuilder:
    """
    Constructs the PageIndex tree by analyzing the LDU hierarchy and using
    gemini-1.5-flash to generate fast, localized abstracts for each section.
    """
    
    SYSTEM_PROMPT = """You are a highly capable document analyst. 
Provided with a JSON mapping of section IDs to their extracted text content, analyze EACH section and generate:
1. summary: A concise, highly descriptive summary (under 2 sentences) capturing main topics, key metrics, and core arguments.
2. key_entities: A normalized, canonical list of the most important entities (e.g., standardizing "Apple" and "Apple Inc." to "Apple Corporation").
3. data_types_present: A list of string indicators (e.g. "tables", "figures", "equations") if they are present or strongly referenced in the text.

You MUST return ONLY a valid JSON object mapping the section IDs to their analysis, like this:
{
  "section_0": {
      "summary": "Summary text...",
      "key_entities": ["Entity A", "Entity B"],
      "data_types_present": ["tables"]
  }
}
Do not include any markdown formatting or extra text outside the JSON object."""
    
    def __init__(
        self, 
        api_key: str, 
        primary_model: str = "openai/gpt-4o-mini", 
        fallback_model: str = "openai/gpt-4o-mini",
        timeout: float = 120.0
    ):
        self.api_key = api_key
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.timeout = timeout
        
    def build_index(self, ldus: List[LogicalDocumentUnit]) -> List[PageIndexNode]:
        """Traverse LDUs, group by section, and generate PageIndexNodes in a hierarchical tree."""
        if not self.api_key:
            logger.warning("No OPENROUTER_API_KEY provided; returning empty PageIndex.")
            return []
            
        # 1. Group content and identify relationships
        sections: Dict[str, Dict[str, Any]] = {}
        section_to_parent: Dict[str, Optional[str]] = {}
        
        for ldu in ldus:
            s_id = ldu.parent_section
            if not s_id:
                continue
                
            if s_id not in sections:
                sections[s_id] = {"title": "Unknown Section", "content": [], "ldus": []}
                # Heuristic: infer parent from ID nesting (e.g., section_1.1 -> section_1)
                parent_id = None
                if "." in s_id:
                    parent_id = "section_" + ".".join(s_id.split("_")[-1].split(".")[:-1])
                section_to_parent[s_id] = parent_id
                
            if ldu.chunk_type == "header" and sections[s_id]["title"] == "Unknown Section":
                sections[s_id]["title"] = ldu.content
                
            sections[s_id]["content"].append(ldu.content)
            sections[s_id]["ldus"].append(ldu)
            
        if not sections:
            return []
            
        # 2. LLM Summarization Pass
        batch_payload = {s_id: "\n".join(s_data["content"])[:10000] for s_id, s_data in sections.items()}
        summaries = self._generate_batched_summaries(batch_payload, self.primary_model)
        if not summaries:
            summaries = self._generate_batched_summaries(batch_payload, self.fallback_model) or {}
            
        # 3. Build Flat Node Map
        node_map: Dict[str, PageIndexNode] = {}
        for s_id, s_data in sections.items():
            analysis = summaries.get(s_id, {})
            if isinstance(analysis, str):
                analysis = {"summary": analysis, "key_entities": [], "data_types_present": []}
                
            # Calculate page spans
            pages = {p for l in s_data["ldus"] for p in l.page_refs}
            
            node_map[s_id] = PageIndexNode(
                section_id=s_id,
                title=s_data["title"],
                summary=analysis.get("summary", "Summary unavailable."),
                page_start=min(pages) if pages else None,
                page_end=max(pages) if pages else None,
                key_entities=analysis.get("key_entities", []),
                data_types_present=analysis.get("data_types_present", []),
                child_sections=[]
            )
            
        # 4. Assemble Hierarchical Tree
        root_nodes = []
        for s_id, node in node_map.items():
            parent_id = section_to_parent.get(s_id)
            if parent_id and parent_id in node_map:
                node_map[parent_id].child_sections.append(node)
            else:
                root_nodes.append(node)
                
        return root_nodes

    def save_index(self, nodes: List[PageIndexNode], path: str) -> None:
        """Serialize the PageIndex tree to a JSON file."""
        with open(path, 'w') as f:
            json.dump([n.model_dump() for n in nodes], f, indent=2)
            
    def load_index(self, path: str) -> List[PageIndexNode]:
        """Load the PageIndex tree from a JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
            return [PageIndexNode(**n) for n in data]

    def navigate(self, nodes: List[PageIndexNode], query: str, k: int = 3) -> List[PageIndexNode]:
        """
        Recursive traversal method that returns the top-k most relevant nodes
        anywhere in the hierarchy based on keyword relevance.
        """
        all_nodes = []
        def flatten(n_list):
            for n in n_list:
                all_nodes.append(n)
                flatten(n.child_sections)
        
        flatten(nodes)
        
        query_words = set(query.lower().split())
        def score_node(node: PageIndexNode) -> float:
            score = 0.0
            score += 2.0 * len(query_words.intersection(set(node.title.lower().split())))
            score += 1.0 * len(query_words.intersection(set(node.summary.lower().split())))
            score += 1.5 * len(query_words.intersection(set(" ".join(node.key_entities).lower().split())))
            return score

        return sorted(all_nodes, key=score_node, reverse=True)[:k]

    def search(self, nodes: List[PageIndexNode], entity: str) -> List[PageIndexNode]:
        """Recursive search API for specific entities or data types."""
        results = []
        for node in nodes:
            if entity.lower() in [e.lower() for e in node.key_entities] or \
               entity.lower() in [d.lower() for d in node.data_types_present]:
                results.append(node)
            results.extend(self.search(node.child_sections, entity))
        return results
        
    def _generate_batched_summaries(self, section_map: Dict[str, str], model: str) -> Optional[Dict[str, Any]]:
        """Invoke the OpenRouter VLM to generate a batched JSON summary response."""
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
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
                content = resp.json()["choices"][0]["message"]["content"].strip()
                if content.startswith("```json"): content = content[7:-3].strip()
                return json.loads(content)
        except Exception as e:
            logger.error(f"Error generating PageIndex summaries ({model}): {e}")
            return None
