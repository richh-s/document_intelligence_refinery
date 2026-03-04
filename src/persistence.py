"""Profile persistence for the Triage Agent."""

from __future__ import annotations

import json
from pathlib import Path

from models.document_profile import DocumentProfile


class ProfileStore:
    """Handles deterministic persistence of document profiles to disk."""

    @staticmethod
    def save(profile: DocumentProfile, base_dir: Path = Path(".refinery")) -> Path:
        """Serialize and save profile to `.refinery/profiles/{doc_id}.json`."""
        out = base_dir / "profiles" / f"{profile.doc_id}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        
        # Serialize with complete determinism: sorted keys, UTF-8, no exclude_none
        data = profile.model_dump(mode="json")
        out.write_text(
            json.dumps(data, sort_keys=True, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return out
