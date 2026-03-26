"""Gold alignment scorer: score generated answers against gold attribute inventory.

ICS scoring pipeline (Fix #3):
1. Load gold attribute inventory
2. For each attribute in gold, use LLM to check binary presence in generated answer
3. ICS = count(present) / count(total_gold_attributes)
"""

import logging
from typing import Any

from src.evaluation.gold_schema import GoldEvidence
from src.generation.llm_client import generate
from src.generation.prompts import ICS_PRESENCE_CHECK_PROMPT

logger = logging.getLogger(__name__)


def check_attribute_presence(generated_answer: str, attribute_name: str,
                             gold_evidence: str,
                             model: str = "gpt-4o-mini") -> bool:
    """Check if a specific attribute is present in the generated answer.

    Uses LLM as a binary presence checker (not a soft judge).
    """
    prompt = ICS_PRESENCE_CHECK_PROMPT.format(
        attribute_name=attribute_name,
        gold_evidence=gold_evidence,
        generated_answer=generated_answer,
    )
    response = generate(prompt, model=model, temperature=0.0, max_tokens=10)
    return response.strip().upper().startswith("YES")


def score_ics(generated_answer: str, gold: GoldEvidence,
              model: str = "gpt-4o-mini") -> dict[str, Any]:
    """Compute ICS for a generated answer against gold attributes.

    Returns:
        {
            "ics": float,
            "present_attributes": [str],
            "missing_attributes": [str],
            "total_attributes": int,
        }
    """
    if not gold.gold_attributes:
        return {"ics": 0.0, "present_attributes": [], "missing_attributes": [], "total_attributes": 0}

    present = []
    missing = []

    for attr_name, gold_evidence in gold.gold_attributes.items():
        is_present = check_attribute_presence(
            generated_answer, attr_name, gold_evidence, model=model
        )
        if is_present:
            present.append(attr_name)
        else:
            missing.append(attr_name)

    ics = len(present) / len(gold.gold_attributes) if gold.gold_attributes else 0.0
    return {
        "ics": ics,
        "present_attributes": present,
        "missing_attributes": missing,
        "total_attributes": len(gold.gold_attributes),
    }
