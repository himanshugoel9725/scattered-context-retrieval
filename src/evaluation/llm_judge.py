"""LLM-as-judge evaluation with structured scoring.

Uses GPT-4o for judging (worth the cost for reliable evaluation).
Structured 1-5 scoring for completeness, accuracy, coherence.
"""

import json
import logging
import math
from typing import Any

from src.generation.llm_client import generate
from src.generation.prompts import LLM_JUDGE_PROMPT

logger = logging.getLogger(__name__)


def llm_judge_score(query: str, generated_answer: str,
                    reference_answer: str | None = None,
                    model: str = "gpt-4o",
                    temperature: float = 0.0) -> dict[str, Any]:
    """Score a generated answer using LLM-as-judge.

    Returns:
        {
            "completeness": int (1-5),
            "accuracy": int (1-5),
            "coherence": int (1-5),
            "overall": float (average),
            "reasoning": str,
        }
    """
    ref_text = reference_answer or "Not available"
    prompt = LLM_JUDGE_PROMPT.format(
        question=query,
        answer=generated_answer,
        reference=ref_text,
    )

    response = generate(prompt, model=model, temperature=temperature, max_tokens=500)

    # Parse structured output
    try:
        # Try to extract JSON from the response
        start = response.find("{")
        end = response.rfind("}") + 1
        if start != -1 and end > start:
            parsed = json.loads(response[start:end])
            completeness = int(parsed.get("completeness", 3))
            accuracy = int(parsed.get("accuracy", 3))
            coherence = int(parsed.get("coherence", 3))
            reasoning = parsed.get("reasoning", "")
        else:
            raise ValueError("No JSON found")
    except (json.JSONDecodeError, ValueError, KeyError):
        logger.warning("Failed to parse LLM judge response: %s", response[:200])
        return {
            "completeness": math.nan,
            "accuracy": math.nan,
            "coherence": math.nan,
            "overall": math.nan,
            "reasoning": response,
            "parse_failed": True,
        }

    overall = (completeness + accuracy + coherence) / 3.0
    return {
        "completeness": completeness,
        "accuracy": accuracy,
        "coherence": coherence,
        "overall": overall,
        "reasoning": reasoning,
    }


def batch_judge(items: list[dict[str, str]],
                model: str = "gpt-4o") -> list[dict[str, Any]]:
    """Score multiple items. Each item has 'query', 'generated_answer', 'reference_answer'."""
    results = []
    for item in items:
        result = llm_judge_score(
            query=item["query"],
            generated_answer=item["generated_answer"],
            reference_answer=item.get("reference_answer"),
            model=model,
        )
        results.append(result)
    return results
