"""Routing and Circuit Breaker logic governing the Extraction Pipeline."""

from pathlib import Path
import logging
from typing import Any

from models.document_profile import DocumentProfile, OriginType, LayoutType
from strategies.base import ExtractionResult, PartialExtractionResult
from extractors.validator import ExtractionValidator
from extractors.ledger import ExtractionLedger
from strategies.fast_text import FastTextExtractor
from strategies.mineru import MinerUExtractor
from strategies.vision import VisionExtractor, BudgetExceededError

logger = logging.getLogger(__name__)

class ExtractionRouter:
    """Primary traffic controller for handling Document intelligence processing."""

    def __init__(self, config: dict[str, Any], ledger: ExtractionLedger):
        self.config = config
        self.ledger = ledger
        
        self.validator = ExtractionValidator(config)
        self.strategy_a = FastTextExtractor(config)
        self.strategy_b = MinerUExtractor(config)
        self.strategy_c = VisionExtractor(config)
        
        self.min_confidence_fast_text = float(self.config.get("MIN_CONFIDENCE_FAST_TEXT", 0.85))
        self.min_confidence_layout = float(self.config.get("MIN_CONFIDENCE_LAYOUT", 0.70))
        self.min_confidence_vision = float(self.config.get("MIN_CONFIDENCE_VISION", 0.60))
        self.max_retries = int(self.config.get("MAX_STRATEGY_RETRIES", 1))

    def _log_attempt(
        self, profile: DocumentProfile, strategy: str, result: ExtractionResult | None, 
        val_score: float, status: str, val_flag: str = "HEALTHY", error_cat: str = "", error_msg: str = ""
    ):
        """Append to the Atomic Ledger with flattened signals and weighted confidence."""
        
        # Weighted Confidence Calculation (v2)
        # Formula: 0.4 * completeness + 0.3 * layout + 0.2 * structural + 0.1 * ocr
        final_conf = 0.0
        signals = result.signals if result and result.signals else {}
        
        if result:
            comp = signals.get("completeness_ratio", 0.0)
            lay = signals.get("layout_consistency", 0.0)
            struct = signals.get("structural_fidelity", 0.0)
            ocr = signals.get("ocr_quality", 0.0)
            
            # Weighted combine
            weighted = (0.4 * comp) + (0.3 * lay) + (0.2 * struct) + (0.1 * ocr)
            # Apply validator score as a global multiplier for consistency
            final_conf = weighted * val_score
            
            # Minimum Baseline: If content exists, don't show 0.0
            if final_conf < 0.1 and (len(result.document.pages) > 0):
                final_conf = 0.1
        
        val_flag = val_flag
        error_cat = error_cat or (result.error_category if result and hasattr(result, 'error_category') else "")
        error_msg = error_msg or (result.error_message if result and hasattr(result, 'error_message') else "")

        record = {
            "file_hash": profile.file_hash,
            "strategy_used": strategy,
            "confidence_score": round(final_conf, 6),
            "validator_score": round(val_score, 4),
            "validator_flag": val_flag,
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
        
        # Flatten signals into the record for observability
        if signals:
            for k, v in signals.items():
                record[f"sig_{k}"] = v
                
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
                    val_score_a, val_flag_a = self.validator.validate(res_a.document, profile)
                    
                    # We use the internal confidence for routing decisions
                    if res_a.confidence >= self.min_confidence_fast_text and val_flag_a == "HEALTHY":
                        res_a.escalation_occurred = escalation_occurred
                        res_a.strategies_attempted = strategies_attempted + ["StrategyA"]
                        self._log_attempt(profile, "StrategyA", res_a, val_score_a, "SUCCESS", val_flag_a)
                        return res_a
                    else:
                        escalation_occurred = True
                        strategies_attempted.append(f"StrategyA({res_a.confidence:.2f})")
                        self._log_attempt(profile, "StrategyA", res_a, val_score_a, "ESCALATED", val_flag_a, "LOW_CONFIDENCE_ESCALATION", f"Confidence {res_a.confidence} < {self.min_confidence_fast_text}")
                        
                except Exception as e:
                    escalation_occurred = True
                    strategies_attempted.append("StrategyA(Exception)")
                    self._log_attempt(profile, "StrategyA", None, 0.0, "ESCALATED", "HEALTHY", "EXTRACTOR_EXCEPTION", str(e))
                    logger.error(f"Strategy A Failed: {e}. Escalating...")
            else:
                escalation_occurred = True
                strategies_attempted.append("StrategyA(MaxRetries)")
                logger.info("Strategy A max retries exceeded. Escalating...")

        # ==========================================
        # STRATEGY B (MinerU Local OCR/Layout)
        # ==========================================
        # Triggers: Flow falls through from A, or it's Scanned/Complex Layout
        if (origin in [OriginType.SCANNED, OriginType.MIXED]) or \
           (layout in [LayoutType.MULTI_COLUMN, LayoutType.TABLE_HEAVY, LayoutType.MIXED]) or \
           (escalation_occurred):
            
            if attempts_b < self.max_retries:
                logger.info("Routing to Strategy B (MinerU Local)")
                try:
                    res_b = self.strategy_b.extract(pdf_path, profile)
                    val_score_b, val_flag_b = self.validator.validate(res_b.document, profile)
                    
                    if res_b.confidence >= self.min_confidence_layout and val_flag_b == "HEALTHY":
                        res_b.escalation_occurred = escalation_occurred
                        res_b.strategies_attempted = strategies_attempted + ["StrategyB"]
                        self._log_attempt(profile, "StrategyB", res_b, val_score_b, "SUCCESS", val_flag_b)
                        return res_b
                    else:
                        escalation_occurred = True
                        strategies_attempted.append(f"StrategyB({res_b.confidence:.2f})")
                        self._log_attempt(profile, "StrategyB", res_b, val_score_b, "ESCALATED", val_flag_b, "LOW_CONFIDENCE_ESCALATION", f"Confidence {res_b.confidence} < {self.min_confidence_layout}")
                        
                except MemoryError as e:
                    escalation_occurred = True
                    strategies_attempted.append("StrategyB(MemoryError)")
                    self._log_attempt(profile, "StrategyB", None, 0.0, "ESCALATED", "HEALTHY", "MEMORY_CIRCUIT_BREAKER", str(e))
                    logger.error("Strategy B hit MemoryError (OOM). Escalating to Vision...")
                except Exception as e:
                    escalation_occurred = True
                    strategies_attempted.append("StrategyB(Exception)")
                    self._log_attempt(profile, "StrategyB", None, 0.0, "ESCALATED", "HEALTHY", "EXTRACTOR_EXCEPTION", str(e))
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
                val_score_c, val_flag_c = self.validator.validate(res_c.document, profile)
                
                # Did it hit the budget ceiling mid-way?
                if isinstance(res_c, PartialExtractionResult):
                    res_c.escalation_occurred = escalation_occurred
                    res_c.strategies_attempted = strategies_attempted + ["StrategyC(Partial)"]
                    res_c.requires_human_review = True 
                    self._log_attempt(profile, "StrategyC", res_c, val_score_c, "PARTIAL_SUCCESS", val_flag_c, "BUDGET_CAP_HIT", "Document exceeded global token budget.")
                    return res_c
                    
                # Graceful Degradation flag 
                if res_c.confidence < self.min_confidence_vision or val_flag_c != "HEALTHY":
                    res_c.requires_human_review = True
                    
                res_c.escalation_occurred = escalation_occurred
                res_c.strategies_attempted = strategies_attempted + [f"StrategyC({res_c.confidence:.2f})"]
                
                status = "SUCCESS" if (res_c.confidence >= self.min_confidence_vision and val_flag_c == "HEALTHY") else "COMPLETED_WITH_LOW_CONFIDENCE"
                self._log_attempt(profile, "StrategyC", res_c, val_score_c, status, val_flag_c)
                return res_c
                
            except BudgetExceededError as e:
                self._log_attempt(profile, "StrategyC", None, 0.0, "FAILED_UNRECOVERABLE", "HEALTHY", "BUDGET_EXCEEDED", str(e))
                logger.error(f"Strategy C Pre-flight aborted: {e}")
                return None
            except Exception as e:
                self._log_attempt(profile, "StrategyC", None, 0.0, "FAILED_UNRECOVERABLE", "HEALTHY", "EXTRACTOR_EXCEPTION", str(e))
                logger.error(f"Strategy C Unrecoverable Failure: {e}")
                return None
        else:
            logger.info("Strategy C max retries exceeded. Extraction blocked.")
            return None
