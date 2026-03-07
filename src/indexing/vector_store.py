"""ChromaDB local vector store integration for LDUs and PageIndex."""

import logging
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional
from models.ldu import LogicalDocumentUnit
from agents.indexer import PageIndexNode

from config import PipelineConfig

logger = logging.getLogger(__name__)


class RefineryVectorStore:
    """Manages ingestion and retrieval of LDUs and PageIndex nodes via ChromaDB."""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        cfg = config or PipelineConfig()
        self.db_path = cfg.VECTOR_DB_PATH
        self.embedding_model = cfg.EMBEDDING_MODEL
        
        # Initialize persistent client
        self.client = chromadb.PersistentClient(path=self.db_path)
        
        # Use SentenceTransformers via Chroma's built-in function
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=self.embedding_model)
        
        # Collections
        self.chunks_collection = self.client.get_or_create_collection(
            name="document_chunks",
            embedding_function=self.ef,
            metadata={"description": "Stores RAG-ready Logical Document Units (LDUs)"}
        )
        
        self.page_index_collection = self.client.get_or_create_collection(
            name="page_index",
            embedding_function=self.ef,
            metadata={"description": "Stores high-level semantic summaries of document sections"}
        )

    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """ChromaDB metadata only accepts int, float, str, or bool."""
        clean = {}
        for k, v in metadata.items():
            if v is None:
                continue
            if isinstance(v, (int, float, str, bool)):
                clean[k] = v
            else:
                clean[k] = str(v)
        return clean

    def ingest_ldus(self, ldus: List[LogicalDocumentUnit]) -> None:
        """Embed and store LDUs in the vector database."""
        if not ldus:
            return
            
        ids = [ldu.content_hash for ldu in ldus]
        documents = [ldu.content for ldu in ldus]
        
        metadatas = []
        for ldu in ldus:
            # Flatten core LDU properties into metadata for rich filtering
            meta = {
                "content_hash": ldu.content_hash,
                "chunk_type": ldu.chunk_type,
                "parent_section": ldu.parent_section if ldu.parent_section else "global",
                "page_refs": str(ldu.page_refs),
                "token_count": ldu.token_count,
                "bounding_box": str([ldu.bounding_box.x0, ldu.bounding_box.y0, ldu.bounding_box.x1, ldu.bounding_box.y1])
            }
            # Include optional extracted metadata
            meta.update(ldu.metadata.model_dump(exclude_none=True))
            metadatas.append(self._sanitize_metadata(meta))
            
        # Chroma handles embedding automatically via the configured EF
        self.chunks_collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        logger.info(f"Ingested {len(ldus)} LDUs into vector store.")

    def ingest_page_index(self, nodes: List[PageIndexNode]) -> None:
        """Embed and store PageIndex summary nodes for hierarchical retrieval."""
        if not nodes:
            return
            
        ids = [node.section_id for node in nodes]
        documents = []
        metadatas = []
        
        # We embed a composite of the title and the summary
        for node in nodes:
            composite_text = f"Section: {node.title}\nSummary: {node.summary}"
            documents.append(composite_text)
            metadatas.append({
                "section_id": node.section_id,
                "title": node.title
            })
            
        self.page_index_collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        logger.info(f"Ingested {len(nodes)} PageIndex nodes into vector store.")

    def query_page_index(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve the most semantic relevant sections based on LLM summaries."""
        results = self.page_index_collection.query(
            query_texts=[query],
            n_results=top_k
        )
        # Format the standardized output Dict containing id, metadata, distance
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                "id": results['ids'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i] if 'distances' in results and results['distances'] else 0.0,
                "document": results['documents'][0][i]
            })
        return formatted_results

    def query_ldus(self, query: str, section_ids: Optional[List[str]] = None, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve chunks, optionally filtered to specific section_ids."""
        
        where_filter = None
        if section_ids:
            if len(section_ids) == 1:
                where_filter = {"parent_section": section_ids[0]}
            else:
                where_filter = {"parent_section": {"$in": section_ids}}
                
        results = self.chunks_collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_filter
        )
        
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                "id": results['ids'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i] if 'distances' in results and results['distances'] else 0.0,
                "document": results['documents'][0][i]
            })
        return formatted_results
