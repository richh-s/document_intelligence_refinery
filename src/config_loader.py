import os
import yaml
from pathlib import Path
from typing import Any

def load_extraction_rules() -> dict[str, Any]:
    """Loads operational constants from the configuration directory."""
    config_path = Path("rubric/extraction_rules.yaml")
    if not config_path.exists():
        config_path = Path("config/extraction_rules.yaml")
    
    if not config_path.exists():
        return {}
        
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f) or {}

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    # Force inject secrets from Environment strictly separate from YAML
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if api_key:
        config_data["OPENROUTER_API_KEY"] = api_key

    return config_data
