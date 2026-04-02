"""Exp 3.3: Error Analysis.

Manually categorize 200 failure cases into 6 error types.
Report distribution.
"""

import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path

from src.data.loaders import load_dataset
from src.data.processors import clean_text
from src.generation.llm_client import generate
from src.indexing.chunker import Chunker
from src.indexing.entity_index import EntityIndex
from src.indexing.vector_index import VectorIndex
from src.retrieval.hybrid_entity import HybridEntityRetriever
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


def _retrieve_chunks_for_failures(failures: list[dict], k: int = 15) -> list[str]:
    """Re-retrieve HybridEntity chunks for each failure case.

    Groups by dataset then by doc_id so each dataset is loaded once and each
    doc's index is built once.  Returns a parallel list of formatted chunk
    strings (or "(retrieval failed)" on error).
    """
    chunker = Chunker(chunk_size=512, overlap=128)
    results = ["(retrieval failed)"] * len(failures)

    # Index into the failure list keyed by (dataset, doc_id)
    by_dataset: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for i, f in enumerate(failures):
        ds = f.get("dataset", "unknown")
        by_dataset[ds].append((i, f))

    for ds_name, items in by_dataset.items():
        dataset = load_dataset(ds_name)
        if not dataset:
            logger.warning("Could not load dataset %s for chunk re-retrieval", ds_name)
            continue
        doc_map = {doc.doc_id: doc for doc in dataset}

        # Group by doc_id to build each index once
        by_doc: dict[str, list[tuple[int, dict]]] = defaultdict(list)
        for i, f in items:
            by_doc[f.get("doc_id", "")].append((i, f))

        for doc_id, doc_items in by_doc.items():
            doc = doc_map.get(doc_id)
            if doc is None:
                logger.warning("doc_id %s not found in %s", doc_id, ds_name)
                continue
            try:
                text = clean_text(doc.text)
                chunks = chunker.chunk_document(doc_id, text)
                if len(chunks) < 5:
                    continue
                chunk_map = {c.chunk_id: c for c in chunks}

                vector_idx = VectorIndex()
                vector_idx.add([c.chunk_id for c in chunks], [c.text for c in chunks])
                entity_idx = EntityIndex(doc_id)
                entity_idx.build_from_chunks(chunks)

                retriever = HybridEntityRetriever(vector_idx, entity_idx, chunk_map)

                for i, f in doc_items:
                    retrieved = retriever.retrieve(f["query"], k=k)
                    chunk_texts = []
                    for r in retrieved[:3]:
                        # Truncate each chunk to ~300 chars to keep prompt manageable
                        chunk_texts.append(f"[Chunk {r.chunk_id}]: {r.text[:300]}")
                    results[i] = "\n\n".join(chunk_texts) if chunk_texts else "(no chunks)"
            except Exception as e:
                logger.warning("Chunk re-retrieval failed for doc %s: %s", doc_id, e)

    return results


def run(config: dict | None = None):
    if config is None:
        config = get_experiments_config()["phase3"]["exp3_3_error_analysis"]

    random.seed(42)

    out_dir = results_dir("exp3_3")
    n_cases = config.get("sample_size", 50)

    # Load exp2_1 results to find failure cases (low ROUGE-L)
    exp2_1_path = results_dir("exp2_1") / "exp2_1_results.json"
    if not exp2_1_path.exists():
        logger.warning("Exp 2.1 results not found")
        return

    with open(exp2_1_path) as f:
        exp2_data = json.load(f)

    # Collect failure cases (HybridEntity with low ROUGE-L)
    all_failures = []
    for ds_name, entries in exp2_data.items():
        for e in entries:
            rouge = float(e.get("HybridEntity_rougeL", 1.0))
            if rouge < 0.3:
                all_failures.append(e)

    # Random sample (not head-cut) so we get diverse failure cases
    n_sample = min(n_cases, len(all_failures))
    failures = random.sample(all_failures, n_sample)
    logger.info("Sampled %d / %d failure cases for error analysis", n_sample, len(all_failures))

    # Re-retrieve actual chunks for each failure case
    logger.info("Re-retrieving chunks for %d failure cases...", len(failures))
    chunks_list = _retrieve_chunks_for_failures(failures)

    # Classify errors using LLM
    classified = []
    for f, chunks_text in zip(failures, chunks_list):
        prompt = ERROR_CLASSIFICATION_PROMPT.format(
            query=f["query"],
            reference=f.get("reference", "N/A"),
            generated=f.get("HybridEntity_answer", "N/A"),
            chunks=chunks_text[:1200],
        )
        response = generate(prompt, model="gpt-4.1-nano", temperature=0.0, max_tokens=30)
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
