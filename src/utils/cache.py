"""Disk-based cache for LLM calls, embeddings, and expensive computations.

Uses diskcache for persistent key-value storage. All LLM API calls must go
through this cache to avoid paying for duplicate requests during reruns.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import diskcache

from src.utils.config import project_root

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = project_root() / ".diskcache"
_cache_instance: diskcache.Cache | None = None


def get_cache(cache_dir: Path | None = None) -> diskcache.Cache:
    """Get or create the global disk cache."""
    global _cache_instance
    if _cache_instance is None:
        d = cache_dir or _DEFAULT_CACHE_DIR
        d.mkdir(parents=True, exist_ok=True)
        _cache_instance = diskcache.Cache(str(d), size_limit=10 * 2**30)  # 10GB
    return _cache_instance


def make_cache_key(prefix: str, **kwargs: Any) -> str:
    """Create a deterministic cache key from prefix + sorted kwargs."""
    serialized = json.dumps(kwargs, sort_keys=True, default=str)
    h = hashlib.sha256(serialized.encode()).hexdigest()
    return f"{prefix}:{h}"


def cached_call(prefix: str, fn, **kwargs: Any) -> Any:
    """Execute fn(**kwargs) with caching. Returns cached result if available."""
    cache = get_cache()
    key = make_cache_key(prefix, **kwargs)
    result = cache.get(key)
    if result is not None:
        logger.debug("Cache hit: %s", key)
        return result
    logger.debug("Cache miss: %s", key)
    result = fn(**kwargs)
    cache.set(key, result)
    return result


class CostTracker:
    """Track cumulative API costs across experiments."""

    def __init__(self):
        self._costs: list[dict[str, Any]] = []

    def record(self, model: str, input_tokens: int, output_tokens: int,
               cost_per_1m_input: float, cost_per_1m_output: float):
        cost = (input_tokens * cost_per_1m_input + output_tokens * cost_per_1m_output) / 1_000_000
        self._costs.append({
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost,
        })

    @property
    def total_cost(self) -> float:
        return sum(c["cost_usd"] for c in self._costs)

    @property
    def total_input_tokens(self) -> int:
        return sum(c["input_tokens"] for c in self._costs)

    @property
    def total_output_tokens(self) -> int:
        return sum(c["output_tokens"] for c in self._costs)

    def summary(self) -> dict[str, Any]:
        by_model: dict[str, float] = {}
        for c in self._costs:
            by_model[c["model"]] = by_model.get(c["model"], 0) + c["cost_usd"]
        return {
            "total_cost_usd": round(self.total_cost, 4),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "by_model": {k: round(v, 4) for k, v in by_model.items()},
            "num_calls": len(self._costs),
        }


# Global cost tracker
cost_tracker = CostTracker()
