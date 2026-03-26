"""Exp 3.3: Error Analysis.

Manually categorize 200 failure cases into 6 error types.
Report distribution.
"""

import json
import logging
from pathlib import Path
from collections import Counter

from src.generation.llm_client import generate
from src.utils.config import get_experiments_config, results_dir
from src.utils.plotting import create_figure, save_figure

logger = logging.getLogger(__name__)

ERROR_TYPES = [
    "entity_not_detected",       # NER missed the target entity
    "coref_failure",             # Pronoun not resolved
    "incomplete_retrieval",      # Not all relevant chunks retrieved
    "wrong_chunks_retrieved",    # Irrelevant chunks contaminate context
    "synthesis_hallucination",   # LLM generates info not in context
    "ordering_confusion",        # Correct chunks but poor synthesis due to ordering
]

ERROR_CLASSIFICATION_PROMPT = """Analyze this QA failure case and classify the primary error type.

Question: {query}
Reference Answer: {reference}
Generated Answer: {generated}
Retrieved Chunks (first 3): {chunks}

Error types:
- entity_not_detected: The target entity was not recognized by NER
- coref_failure: Pronouns referring to the entity were not resolved
- incomplete_retrieval: Not all relevant passages were retrieved
- wrong_chunks_retrieved: Irrelevant passages were included
- synthesis_hallucination: The LLM generated information not present in the context
- ordering_confusion: Correct passages but poor synthesis due to ordering/structure

Respond with ONLY the error type name, nothing else."""


def run(config: dict | None = None):
    if config is None:
        config = get_experiments_config()["phase3"]["exp3_3_error_analysis"]

    out_dir = results_dir("exp3_3")
    n_cases = config.get("sample_size", 200)

    # Load exp2_1 results to find failure cases (low ROUGE-L)
    exp2_1_path = results_dir("exp2_1") / "exp2_1_results.json"
    if not exp2_1_path.exists():
        logger.warning("Exp 2.1 results not found")
        return

    with open(exp2_1_path) as f:
        exp2_data = json.load(f)

    # Collect failure cases (HybridEntity with low ROUGE-L)
    failures = []
    for ds_name, entries in exp2_data.items():
        for e in entries:
            rouge = e.get("HybridEntity_rougeL", 1.0)
            if rouge < 0.3:
                failures.append(e)
    failures = failures[:n_cases]

    # Classify errors using LLM
    classified = []
    for f in failures:
        prompt = ERROR_CLASSIFICATION_PROMPT.format(
            query=f["query"],
            reference=f.get("reference", "N/A"),
            generated=f.get("HybridEntity_answer", "N/A"),
            chunks="(not available in summary)",
        )
        response = generate(prompt, model="gpt-4o-mini", temperature=0.0, max_tokens=30)
        error_type = response.strip().lower()
        matched = "unknown"
        for et in ERROR_TYPES:
            if et in error_type:
                matched = et
                break
        classified.append({"query": f["query"], "error_type": matched,
                           "rougeL": f.get("HybridEntity_rougeL", 0)})

    dist = Counter(c["error_type"] for c in classified)

    with open(out_dir / "exp3_3_results.json", "w") as f:
        json.dump({"distribution": dict(dist), "cases": classified}, f, indent=2)

    # Plot
    fig, ax = create_figure(figsize=(8, 5))
    labels = list(dist.keys())
    values = list(dist.values())
    ax.barh(range(len(labels)), values)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([l.replace("_", " ").title() for l in labels])
    ax.set_xlabel("Count")
    ax.set_title("Error Type Distribution in Failure Cases")
    ax.invert_yaxis()
    save_figure(fig, out_dir / "figure_error_analysis.pdf")

    logger.info("Exp 3.3: Error distribution: %s", dict(dist))
    return {"distribution": dict(dist), "cases": classified}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
