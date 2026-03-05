"""Routing and Circuit Breaker logic governing the Extraction Pipeline."""

from pathlib import Path
import logging
from typing import Any

from models.document_profile import DocumentProfile, OriginType, LayoutType
from strategies.base import ExtractionResult, PartialExtractionResult
from extractors.validator import ExtractionValidator
from extractors.ledger import ExtractionLedger
from strategies.fast_text import FastTextExtractor
from strategies.layout import LayoutAwareExtractor
from strategies.vision import VisionExtractor, BudgetExceededError

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
        
        self.min_confidence_fast_text = float(self.config.get("MIN_CONFIDENCE_FAST_TEXT", 0.85))
        self.min_confidence_layout = float(self.config.get("MIN_CONFIDENCE_LAYOUT", 0.70))
        self.min_confidence_vision = float(self.config.get("MIN_CONFIDENCE_VISION", 0.60))
        self.max_retries = int(self.config.get("MAX_STRATEGY_RETRIES", 1))

    def _log_attempt(
        self, profile: DocumentProfile, strategy: str, result: ExtractionResult | None, 
        val_conf: float, status: str, error_cat: str = "", error_msg: str = ""
    ):
        """Append to the Atomic Ledger."""
        final_conf = (result.confidence * val_conf) if result else 0.0
        
        record = {
            "file_hash": profile.file_hash,
            "strategy_used": strategy,
            "confidence_score": final_conf,
            "extractor_confidence": result.confidence if result else 0.0,
            "validator_confidence": val_conf,
            "cost_estimate": result.cost if result else 0.0,
            "processing_time": result.time_ms if result else 0,
            "model_name": result.model_name if result else None,
            "tokens_in": result.tokens_in if result else 0,
            "tokens_out": result.tokens_out if result else 0,
            "page_count": result.pages_sent if result else 0,
            "status": status,
            "error_category": error_cat,
            "error_message": error_msg
        }
        self.ledger.append(record)

    def route(self, pdf_path: Path, profile: DocumentProfile) -> ExtractionResult | None:
        """Executes the extraction workflow with deterministic circuit breakers."""
        
        origin = profile.origin_type
        layout = profile.layout_type
        
        # Track the specific path this document took through the router
        strategies_attempted = []
        escalation_occurred = False
        
        # Check retries via Ledger 
        attempts_a = self.ledger.get_attempt_count(profile.file_hash, "StrategyA")
        attempts_b = self.ledger.get_attempt_count(profile.file_hash, "StrategyB")
        attempts_c = self.ledger.get_attempt_count(profile.file_hash, "StrategyC")

        # ==========================================
        # STRATEGY A (Fast Text)
        # ==========================================
        if origin in [OriginType.DIGITAL_NATIVE, OriginType.FORM_FILLABLE] and layout == LayoutType.SINGLE_COLUMN:
            if attempts_a < self.max_retries:
                logger.info("Routing to Strategy A (Fast Text)")
                try:
                    res_a = self.strategy_a.extract(pdf_path, profile)
                    val_conf_a = self.validator.validate(res_a.document, profile)
                    final_a = res_a.confidence * val_conf_a
                    
                    if final_a >= self.min_confidence_fast_text:
                        res_a.escalation_occurred = escalation_occurred
                        res_a.strategies_attempted = strategies_attempted + ["StrategyA"]
                        self._log_attempt(profile, "StrategyA", res_a, val_conf_a, "SUCCESS")
                        return res_a
                    else:
                        escalation_occurred = True
                        strategies_attempted.append(f"StrategyA({final_a:.2f})")
                        self._log_attempt(profile, "StrategyA", res_a, val_conf_a, "ESCALATED", "LOW_CONFIDENCE_ESCALATION", f"Confidence {final_a} < {self.min_confidence_fast_text}")
                        
                except Exception as e:
                    escalation_occurred = True
                    strategies_attempted.append("StrategyA(Exception)")
                    self._log_attempt(profile, "StrategyA", None, 0.0, "ESCALATED", "EXTRACTOR_EXCEPTION", str(e))
                    logger.error(f"Strategy A Failed: {e}. Escalating...")
            else:
                escalation_occurred = True
                strategies_attempted.append("StrategyA(MaxRetries)")
                logger.info("Strategy A max retries exceeded. Escalating...")

        # ==========================================
        # STRATEGY B (Layout Aware - Docling)
        # ==========================================
        # Triggers: Flow falls through from A, or it's native but complex
        if origin in [OriginType.DIGITAL_NATIVE, OriginType.MIXED] or layout in [LayoutType.MULTI_COLUMN, LayoutType.TABLE_HEAVY, LayoutType.MIXED]:
            if attempts_b < self.max_retries:
                logger.info("Routing to Strategy B (Layout Aware)")
                try:
                    # Memory-Aware Circuit Breaker handled by the Extractor raising MemoryError
                    res_b = self.strategy_b.extract(pdf_path, profile)
                    val_conf_b = self.validator.validate(res_b.document, profile)
                    final_b = res_b.confidence * val_conf_b
                    
                    if final_b >= self.min_confidence_layout:
                        res_b.escalation_occurred = escalation_occurred
                        res_b.strategies_attempted = strategies_attempted + ["StrategyB"]
                        self._log_attempt(profile, "StrategyB", res_b, val_conf_b, "SUCCESS")
                        return res_b
                    else:
                        escalation_occurred = True
                        strategies_attempted.append(f"StrategyB({final_b:.2f})")
                        self._log_attempt(profile, "StrategyB", res_b, val_conf_b, "ESCALATED", "LOW_CONFIDENCE_ESCALATION", f"Confidence {final_b} < {self.min_confidence_layout}")
                        
                except MemoryError as e:
                    # Explicit Memory Circuit Breaker
                    escalation_occurred = True
                    strategies_attempted.append("StrategyB(MemoryError)")
                    self._log_attempt(profile, "StrategyB", None, 0.0, "ESCALATED", "MEMORY_CIRCUIT_BREAKER", str(e))
                    logger.error("Strategy B hit MemoryError (OOM). Escalating to Vision...")
                except Exception as e:
                    escalation_occurred = True
                    strategies_attempted.append("StrategyB(Exception)")
                    self._log_attempt(profile, "StrategyB", None, 0.0, "ESCALATED", "EXTRACTOR_EXCEPTION", str(e))
                    logger.error(f"Strategy B Failed: {e}. Escalating to Vision...")
            else:
                escalation_occurred = True
                strategies_attempted.append("StrategyB(MaxRetries)")
                logger.info("Strategy B max retries exceeded. Escalating to Vision...")

        # ==========================================
        # STRATEGY C (Vision Augmented VLM)
        # ==========================================
        # Triggers: All native paths failed, or document is definitively Scanned
        if attempts_c < self.max_retries:
            logger.info("Routing to Strategy C (Vision Augmented)")
            try:
                res_c = self.strategy_c.extract(pdf_path, profile)
                val_conf_c = self.validator.validate(res_c.document, profile)
                
                # Did it hit the budget ceiling mid-way?
                if isinstance(res_c, PartialExtractionResult):
                    res_c.escalation_occurred = escalation_occurred
                    res_c.strategies_attempted = strategies_attempted + ["StrategyC(Partial)"]
                    res_c.requires_human_review = True # Always flag partials
                    self._log_attempt(profile, "StrategyC", res_c, val_conf_c, "PARTIAL_SUCCESS", "BUDGET_CAP_HIT", "Document exceeded global token budget.")
                    return res_c
                    
                final_c = res_c.confidence * val_conf_c
                
                # Graceful Degradation flag 
                if final_c < self.min_confidence_vision:
                    res_c.requires_human_review = True
                    
                res_c.escalation_occurred = escalation_occurred
                res_c.strategies_attempted = strategies_attempted + [f"StrategyC({final_c:.2f})"]
                
                status = "SUCCESS" if final_c >= self.min_confidence_vision else "COMPLETED_WITH_LOW_CONFIDENCE"
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
        else:
            logger.info("Strategy C max retries exceeded. Extraction blocked.")
            return None
