"""Routing and Circuit Breaker logic governing the Extraction Pipeline."""

from pathlib import Path
import logging
from typing import Any

from document_intelligence_refinery.models import DocumentProfile, OriginType, LayoutType
from document_intelligence_refinery.extractors.base import ExtractionResult, PartialExtractionResult
from document_intelligence_refinery.extractors.validator import ExtractionValidator
from document_intelligence_refinery.extractors.ledger import ExtractionLedger
from document_intelligence_refinery.extractors.fast_text import FastTextExtractor
from document_intelligence_refinery.extractors.layout import LayoutAwareExtractor
from document_intelligence_refinery.extractors.vision import VisionExtractor, BudgetExceededError

logger = logging.getLogger(__name__)

class ExtractionRouter:
    """Primary traffic controller for handling Document intelligence processing."""

    def __init__(self, config: dict[str, Any], ledger: ExtractionLedger):
        self.config = config
        self.ledger = ledger
        
        self.validator = ExtractionValidator(config)
        self.strategy_a = FastTextExtractor(config)
        self.strategy_b = LayoutAwareExtractor(config)
        self.strategy_c = VisionExtractor(config)
        
        self.min_confidence = float(self.config.get("MIN_EXTRACTION_CONFIDENCE", 0.85))

    def _log_attempt(
        self, profile: DocumentProfile, strategy: str, result: ExtractionResult | None, 
        val_conf: float, status: str, error_cat: str = "", error_msg: str = ""
    ):
        """Append to the Atomic Ledger."""
        record = {
            "file_hash": profile.file_hash,
            "strategy_used": strategy,
            "extractor_confidence": result.confidence if result else 0.0,
            "validator_confidence": val_conf,
            "final_confidence": min(result.confidence, val_conf) if result else 0.0,
            "cost_estimate": result.cost if result else 0.0,
            "processing_time_ms": result.time_ms if result else 0,
            "model_name": result.model_name if result else None,
            "tokens_in": result.tokens_in if result else 0,
            "tokens_out": result.tokens_out if result else 0,
            "pages_sent": result.pages_sent if result else 0,
            "status": status,
            "error_category": error_cat,
            "error_message": error_msg
        }
        self.ledger.append(record)

    def route(self, pdf_path: Path, profile: DocumentProfile) -> ExtractionResult | None:
        """Executes the extraction workflow with deterministic circuit breakers."""
        
        origin = profile.origin_type
        layout = profile.layout_type
        
        # ==========================================
        # STRATEGY A (Fast Text)
        # ==========================================
        if origin in [OriginType.DIGITAL_NATIVE, OriginType.FORM_FILLABLE] and layout == LayoutType.SINGLE_COLUMN:
            logger.info("Routing to Strategy A (Fast Text)")
            try:
                res_a = self.strategy_a.extract(pdf_path, profile)
                val_conf_a = self.validator.validate(res_a.document, profile)
                final_a = min(res_a.confidence, val_conf_a)
                
                if final_a >= self.min_confidence:
                    self._log_attempt(profile, "StrategyA", res_a, val_conf_a, "SUCCESS")
                    return res_a
                else:
                    self._log_attempt(profile, "StrategyA", res_a, val_conf_a, "ESCALATED", "LOW_CONFIDENCE_ESCALATION", f"Confidence {final_a} < {self.min_confidence}")
                    
            except Exception as e:
                self._log_attempt(profile, "StrategyA", None, 0.0, "ESCALATED", "EXTRACTOR_EXCEPTION", str(e))
                logger.error(f"Strategy A Failed: {e}. Escalating...")

        # ==========================================
        # STRATEGY B (Layout Aware - Docling)
        # ==========================================
        # Triggers: Flow falls through from A, or it's native but complex
        if origin in [OriginType.DIGITAL_NATIVE, OriginType.MIXED] or layout in [LayoutType.MULTI_COLUMN, LayoutType.TABLE_HEAVY, LayoutType.MIXED]:
            logger.info("Routing to Strategy B (Layout Aware)")
            try:
                # Memory-Aware Circuit Breaker handled by the Extractor raising MemoryError
                res_b = self.strategy_b.extract(pdf_path, profile)
                val_conf_b = self.validator.validate(res_b.document, profile)
                final_b = min(res_b.confidence, val_conf_b)
                
                if final_b >= self.min_confidence:
                    self._log_attempt(profile, "StrategyB", res_b, val_conf_b, "SUCCESS")
                    return res_b
                else:
                    self._log_attempt(profile, "StrategyB", res_b, val_conf_b, "ESCALATED", "LOW_CONFIDENCE_ESCALATION", f"Confidence {final_b} < {self.min_confidence}")
                    
            except MemoryError as e:
                # Explicit Memory Circuit Breaker
                self._log_attempt(profile, "StrategyB", None, 0.0, "ESCALATED", "MEMORY_CIRCUIT_BREAKER", str(e))
                logger.error("Strategy B hit MemoryError (OOM). Escalating to Vision...")
            except Exception as e:
                self._log_attempt(profile, "StrategyB", None, 0.0, "ESCALATED", "EXTRACTOR_EXCEPTION", str(e))
                logger.error(f"Strategy B Failed: {e}. Escalating to Vision...")

        # ==========================================
        # STRATEGY C (Vision Augmented VLM)
        # ==========================================
        # Triggers: All native paths failed, or document is definitively Scanned
        logger.info("Routing to Strategy C (Vision Augmented)")
        try:
            res_c = self.strategy_c.extract(pdf_path, profile)
            val_conf_c = self.validator.validate(res_c.document, profile)
            
            # Did it hit the budget ceiling mid-way?
            if isinstance(res_c, PartialExtractionResult):
                self._log_attempt(profile, "StrategyC", res_c, val_conf_c, "PARTIAL_SUCCESS", "BUDGET_CAP_HIT", "Document exceeded global token budget.")
                return res_c
                
            final_c = min(res_c.confidence, val_conf_c)
            # Strategy C is final tier. Return whatever we got. 
            # If it's terrible, we note the failure in the ledger but still return the object.
            status = "SUCCESS" if final_c >= self.min_confidence else "COMPLETED_WITH_LOW_CONFIDENCE"
            self._log_attempt(profile, "StrategyC", res_c, val_conf_c, status)
            return res_c
            
        except BudgetExceededError as e:
            # Pre-flight budget rejection
            self._log_attempt(profile, "StrategyC", None, 0.0, "FAILED_UNRECOVERABLE", "BUDGET_EXCEEDED", str(e))
            logger.error(f"Strategy C Pre-flight aborted: {e}")
            return None
        except Exception as e:
            self._log_attempt(profile, "StrategyC", None, 0.0, "FAILED_UNRECOVERABLE", "EXTRACTOR_EXCEPTION", str(e))
            logger.error(f"Strategy C Unrecoverable Failure: {e}")
            return None
