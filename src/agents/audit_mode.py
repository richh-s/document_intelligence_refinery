"""Audit Mechanism: Verifies declarative claims mathematically against Provenance."""

import logging
import httpx
from typing import Literal, List, Dict
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class AuditResult(BaseModel):
    """The strict verification output of the Audit process."""
    claim: str
    status: Literal["Verified", "Partially Supported", "Not Found / Unverifiable"]
    reasoning: str
    provenance_hashes: List[str] # Exact LDU ID linking
    
    
class ClaimAuditor:
    """Verifies declarative facts strictly against known corpus geometry."""
    
    SYSTEM_PROMPT = """You are an algorithmic verification agent. 
Given a CLAIM and a set of SOURCE TEXTS retrieved from the document corpus, evaluate if the claim is mathematically and semantically true.
Return your evaluation structured precisely:
status: Either "Verified" (completely matches), "Partially Supported" (numbers match but dates/context differ), or "Not Found / Unverifiable" (cannot be proven natively by text).
reasoning: A 1-sentence analytical defense of your status.
provenance_hashes: ONLY the content_hashes of the exact texts that proved the case."""

    def __init__(self, api_key: str, model: str = "google/gemini-1.5-flash", timeout: float = 60.0):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        
    def verify_claim(self, claim: str, retrieved_context: List[Dict[str, str]]) -> AuditResult:
        """
        Submits the retrieved vectors mathematically against the claim constraints.
        `retrieved_context` expects dicts with "hash" and "text" keys.
        """
        
        # Assemble verifiable context cleanly
        context_string = "\n\n---\n\n".join([f"HASH [{c['hash']}]:\n{c['text']}" for c in retrieved_context])
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # We explicitly force the JSON topology for predictability
        payload = {
            "model": self.model,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"CLAIM: {claim}\n\nSOURCES:\n{context_string}"}
            ]
        }
        
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"].strip()
                
                parsed_json = json.loads(content)
                return AuditResult(
                    claim=claim,
                    status=parsed_json.get("status", "Not Found / Unverifiable"),
                    reasoning=parsed_json.get("reasoning", "Failed to parse reasoning."),
                    provenance_hashes=parsed_json.get("provenance_hashes", [])
                )
                
        except Exception as e:
            logger.error(f"Audit failed natively: {e}")
            return AuditResult(
                claim=claim,
                status="Not Found / Unverifiable",
                reasoning="System timeout or evaluation failure.",
                provenance_hashes=[]
            )
