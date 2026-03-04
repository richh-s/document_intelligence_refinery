"""Atomic ledger operations for tracking extraction cost and quality."""

import json
import logging
import fcntl
from pathlib import Path
from typing import Any
from datetime import datetime

logger = logging.getLogger(__name__)

class ExtractionLedger:
    """Thread-safe JSONL appender for extraction observability."""

    def __init__(self, ledger_path: Path):
        self.ledger_path = ledger_path
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.ledger_path.exists():
            self.ledger_path.touch()

    def get_attempt_count(self, file_hash: str, strategy: str) -> int:
        """Read the ledger and count instances of strategy use for a specific file hash."""
        if not self.ledger_path.exists():
            return 0
        count = 0
        with open(self.ledger_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    record = json.loads(line)
                    if record.get("file_hash") == file_hash and record.get("strategy_used") == strategy:
                        count += 1
                except json.JSONDecodeError:
                    pass
        return count

    def append(self, record: dict[str, Any]) -> None:
        """Atomically append a record to the ledger using file locking."""
        required = {
            "strategy_used", "confidence_score", "cost_estimate", "processing_time",
            "tokens_in", "tokens_out", "page_count"
        }
        if not required.issubset(record.keys()):
            raise ValueError(f"Ledger record missing mandatory fields. Required: {required}")

        record["timestamp"] = datetime.utcnow().isoformat()
        
        json_line = json.dumps(record) + "\n"
        
        # Lock-Write-Unlock mechanism to prevent race condition corruption
        try:
            with open(self.ledger_path, "a", encoding="utf-8") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(json_line)
                    f.flush()
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception as e:
            logger.error(f"Failed to write to Extraction Ledger: {e}")
