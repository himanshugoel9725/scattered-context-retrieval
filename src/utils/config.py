"""YAML configuration loader with environment variable support."""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_DIR = _PROJECT_ROOT / "configs"

# Load .env file from project root if it exists
load_dotenv(_PROJECT_ROOT / ".env")


def load_yaml(name: str) -> dict[str, Any]:
    """Load a YAML config file by name (without extension)."""
    path = _CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def get_datasets_config() -> dict[str, Any]:
    return load_yaml("datasets")


def get_models_config() -> dict[str, Any]:
    return load_yaml("models")


def get_experiments_config() -> dict[str, Any]:
    return load_yaml("experiments")


def get_attribute_schemas() -> dict[str, Any]:
    return load_yaml("attribute_schemas")


def get_query_schema() -> dict[str, Any]:
    return load_yaml("query_schema")


def project_root() -> Path:
    return _PROJECT_ROOT


def data_dir(subdir: str = "") -> Path:
    """Return a path under data/, creating if needed."""
    p = _PROJECT_ROOT / "data" / subdir
    p.mkdir(parents=True, exist_ok=True)
    return p


def results_dir(subdir: str = "") -> Path:
    """Return a path under results/, creating if needed."""
    p = _PROJECT_ROOT / "results" / subdir
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_api_key(provider: str) -> str:
    """Get API key from environment variables."""
    key_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
        "together": "TOGETHER_API_KEY",
        "groq": "GROQ_API_KEY",
    }
    env_var = key_map.get(provider)
    if not env_var:
        raise ValueError(f"Unknown provider: {provider}")
    key = os.getenv(env_var)
    if not key:
        raise ValueError(
            f"API key not set for {provider}. Set {env_var} in .env file."
        )
    return key
