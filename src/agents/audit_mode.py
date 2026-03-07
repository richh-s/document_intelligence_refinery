"""Audit Mechanism: Verifies declarative claims mathematically against Provenance."""

import logging
import httpx
import json
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
Determine whether the CLAIM is provided below is supported by the SOURCE texts.
You must respond ONLY in valid JSON format. Do not include any conversational text or markdown blocks.

Required JSON Structure:
{
  "status": "Verified" | "Not Found / Unverifiable",
  "reasoning": "A short, 1-sentence analytical defense of your status.",
  "provenance_hashes": ["exact_hash_of_source"]
}"""

    def __init__(self, api_key: str, model: str = "openai/gpt-4o-mini", timeout: float = 60.0):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        
    def verify_claim(self, claim: str, retrieved_context: List[Dict[str, str]]) -> AuditResult:
        """
        Submits the retrieved vectors mathematically against the claim constraints.
        Includes robust JSON parsing and a deterministic fallback.
        """
        
        # Assemble verifiable context cleanly
        context_string = "\n\n---\n\n".join([f"HASH [{c['hash']}]:\n{c['text']}" for c in retrieved_context])
        all_source_text = " ".join([c["text"] for c in retrieved_context]).lower()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/richh-s/document_intelligence_refinery",
            "X-Title": "Document Intelligence Refinery"
        }
        
        payload = {
            "model": self.model,
            "max_tokens": 500,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"CLAIM: {claim}\n\nSOURCES:\n{context_string}"}
            ]
        }
        
        status = "Not Found / Unverifiable"
        reasoning = "System timeout or evaluation failure."
        hashes = []
        
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"].strip()
                    
                    # Remove markdown code blocks if present
                    if content.startswith("```"):
                        content = content.split("```")[1]
                        if content.startswith("json"):
                            content = content[4:].strip()
                            
                    try:
                        parsed = json.loads(content)
                        status = parsed.get("status", status)
                        reasoning = parsed.get("reasoning", "Parsed from LLM.")
                        hashes = parsed.get("provenance_hashes", [])
                        return AuditResult(claim=claim, status=status, reasoning=reasoning, provenance_hashes=hashes)
                    except json.JSONDecodeError:
                        logger.warning(f"Audit JSON parse failed. Raw content: {content}")
                else:
                    logger.error(f"OpenRouter Error {resp.status_code}: {resp.text}")
                    
        except Exception as e:
            logger.error(f"Audit API call failed: {e}")

        # --- DETERMINISTIC FALLBACK ---
        # If API or Parsing fails, we perform a basic keyword/claim check
        # This ensures the function NEVER crashes and provides a safety net.
        if claim.lower() in all_source_text:
            status = "Verified"
            reasoning = "Deterministic match: Claim text was found directly in the source context."
            # Tag all retrieved hashes as supporting if we do a text match
            hashes = [c["hash"] for c in retrieved_context]
        else:
            status = "Not Found / Unverifiable"
            reasoning = "Claim could not be verified via LLM and no direct text match was found."

        return AuditResult(
            claim=claim,
            status=status,
            reasoning=reasoning,
            provenance_hashes=hashes
        )
