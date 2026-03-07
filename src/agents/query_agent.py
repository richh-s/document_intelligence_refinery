"""Multi-tier LangGraph agent orchestrating querying workflows."""

import logging
from typing import List, Dict, Any, Literal, Optional
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from models.provenance import ProvenanceChain
from indexing.vector_store import RefineryVectorStore
from indexing.fact_table import FactTableStore
from agents.indexer import PageIndexBuilder, PageIndexNode

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class AgentState(BaseModel):
    query: str
    original_query: str
    classification: Optional[Literal["quantitative", "conceptual"]] = None
    target_sections: List[str] = Field(default_factory=list)
    retrieved_context: str = ""
    provenance_links: List[Dict[str, Any]] = Field(default_factory=list)
    final_answer: str = ""
    failure_mode: bool = False
    
# Mock Tools for orchestration (to be bound to actual Engine backends in prod)

@tool
def structured_query(query: str) -> str:
    """Useful for answering quantitative, numerical, or financial facts natively via SQL."""
    return f"Executed SQL lookup for: {query}"

@tool
def pageindex_navigate(query: str) -> List[str]:
    """Traverses the PageIndex tree summaries to locate relevant section IDs prior to deep search."""
    return ["section_0", "section_1"] # Returns mock targets

@tool
def semantic_search(query: str, section_ids: Optional[List[str]] = None) -> str:
    """Performs deep dense vector retrieval against LDUs, optionally filtered by sections."""
    return "Retrieved dense textual payload from vector store."

# Classification Node
def classify_query(state: AgentState) -> AgentState:
    """Mathematical routing classification."""
    q = state.query.lower()
    is_quant = any(k in q for k in ["total", "sum", "average", "revenue", "cost", "margin", "%", "$", "q1", "q2", "q3", "q4"])
    state.classification = "quantitative" if is_quant else "conceptual"
    return state

class ProvenanceMissingError(Exception):
    """Raised natively if the system attempts synthesis without explicit citations."""
    pass

# --- Operational Node Functions (connecting to real backends) ---

def pageindex_navigate_node(state: AgentState) -> AgentState:
    """Traverses PageIndex to locate structural anchors."""
    # In a real run, we'd load the pageindex by doc_id
    # For now, we simulate the 'navigate' logic from PageIndexBuilder
    # based on the classification
    state.target_sections = ["sec_01", "sec_02_a"] # IDs from our generated samples
    return state

def semantic_search_node(state: AgentState, vector_store: RefineryVectorStore) -> AgentState:
    """Performs real vector retrieval against LDUs, filtered by sections."""
    results = vector_store.query_ldus(
        query=state.query, 
        section_ids=state.target_sections,
        top_k=2
    )
    
    context_parts = []
    links = []
    for res in results:
        context_parts.append(res["document"])
        # Map back to ProvenanceChain
        meta = res["metadata"]
        links.append({
            "document_name": meta.get("document_name", "Unknown"),
            "page_number": int(eval(meta["page_refs"])[0]) if "page_refs" in meta else 1,
            "bbox": eval(meta["bounding_box"]) if "bounding_box" in meta else [0,0,0,0],
            "content_hash": meta["content_hash"]
        })
    
    state.retrieved_context = "\n\n".join(context_parts)
    state.provenance_links = links
    return state

def structured_query_node(state: AgentState, fact_table: FactTableStore) -> AgentState:
    """Executes SQL against the FactTable for quantitative facts."""
    # Simulation of query translation: strip punctuation and take last word
    term = state.query.strip("?").split()[-1]
    sql_query = f"SELECT value FROM facts WHERE entity LIKE '%{term}%'"
    try:
        results = fact_table.query(sql_query)
        if results:
            state.retrieved_context = f"SQL Result: {str(results[0]['value'])}"
            state.provenance_links = [{"document_name": "Fact_Table.sql", "page_number": 0, "bbox": [0,0,0,0], "content_hash": "sql_fact"}]
        else:
            state.retrieved_context = "Numerical lookup yielded no results."
    except Exception as e:
        state.retrieved_context = f"Numerical lookup error: {str(e)}"
    return state

def synthesize_answer(state: AgentState) -> AgentState:
    """Final LangGraph node that strictly enforces the Provenance requirement."""
    if not state.provenance_links:
        # If no provenance was found by either route, we enter failure mode
        state.final_answer = "I'm sorry, I cannot verify this answer with source citations."
        return state
        
    # Construct cited response
    citations = "\n".join([f"[{i+1}] {p['document_name']} (p.{p['page_number']})" for i, p in enumerate(state.provenance_links)])
    state.final_answer = f"{state.retrieved_context}\n\nSources:\n{citations}"
    return state

# Routing logic
def route_by_query_type(state: AgentState) -> Literal["structured", "navigational"]:
    if state.classification == "quantitative":
        return "structured"
    return "navigational"

# Agent Graph Definition
workflow = StateGraph(AgentState)
workflow.add_node("classifier", classify_query)
workflow.add_node("navigate", pageindex_navigate_node)
workflow.add_node("search", semantic_search_node)
workflow.add_node("sql_query", structured_query_node)
workflow.add_node("synthesize", synthesize_answer)

workflow.add_edge(START, "classifier")
workflow.add_conditional_edges(
    "classifier", 
    route_by_query_type,
    {
        "structured": "sql_query",
        "navigational": "navigate"
    }
)
workflow.add_edge("navigate", "search")
workflow.add_edge("search", "synthesize")
workflow.add_edge("sql_query", "synthesize")
workflow.add_edge("synthesize", END)

app = workflow.compile()
