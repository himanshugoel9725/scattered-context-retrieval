"""Exp 2.2: Ablation Study.

Full system → remove each component one at a time (6 ablations).
Run on NarrativeQA subset (200 scattered queries).
Generate Figure 9 (waterfall chart).
"""

import json
import logging
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.data.loaders import load_dataset
from src.data.processors import clean_text
from src.indexing.chunker import Chunker
from src.indexing.entity_index import EntityIndex
from src.indexing.vector_index import VectorIndex
from src.retrieval.standard_semantic import StandardSemanticRetriever
from src.retrieval.hybrid_entity import HybridEntityRetriever
from src.generation.synthesizer import synthesize, to_synthesis_chunks
from src.retrieval.base import detect_query_entities
from src.evaluation.standard_metrics import compute_rouge_l
from src.evaluation.metrics import scatter_coverage_at_k
from src.utils.config import get_experiments_config, results_dir
from src.utils.plotting import create_figure, save_figure

logger = logging.getLogger(__name__)


ABLATION_CONFIGS = [
    {"name": "Full System", "alpha": 0.5, "beta": 0.3, "gamma": 0.2},
    {"name": "No Entity (β=0)", "alpha": 0.7, "beta": 0.0, "gamma": 0.3},
    {"name": "No Diversity (γ=0)", "alpha": 0.6, "beta": 0.4, "gamma": 0.0},
    {"name": "No Semantic (α=0)", "alpha": 0.0, "beta": 0.6, "gamma": 0.4},
    {"name": "Entity Only", "alpha": 0.0, "beta": 1.0, "gamma": 0.0},
    {"name": "Semantic Only", "alpha": 1.0, "beta": 0.0, "gamma": 0.0},
]


def run(config: dict | None = None):
    """Run Exp 2.2: Ablation Study."""
    if config is None:
        config = get_experiments_config()["phase2"]["exp2_2_ablation"]

    random.seed(42)

    out_dir = results_dir("exp2_2")
    chunker = Chunker(chunk_size=512, overlap=128)
    n_queries = config.get("queries", 200)
    k = config.get("k", 15)

    ds_name = config.get("dataset", "scatterqa")
    dataset = load_dataset(ds_name)
    if not dataset:
        return

    # ------------------------------------------------------------------
    # Phase 1: Collect all valid (doc_id, qa_index) pairs — no index building
    # ------------------------------------------------------------------
    doc_store: dict[str, tuple] = {}
    all_pairs: list[tuple[str, int]] = []

    for doc in dataset:
        text = clean_text(doc.text)
        chunks = chunker.chunk_document(doc.doc_id, text)
        if len(chunks) < 5:
            continue
        doc_store[doc.doc_id] = (doc, chunks)
        for qa_idx in range(len(doc.questions)):
            all_pairs.append((doc.doc_id, qa_idx))

    # ------------------------------------------------------------------
    # Phase 2: Random sample to spread queries across documents/novels
    # ------------------------------------------------------------------
    n_sample = min(n_queries, len(all_pairs))
    sampled_pairs = random.sample(all_pairs, n_sample)
    logger.info("[%s] Sampled %d / %d available queries for ablation",
                ds_name, n_sample, len(all_pairs))

    by_doc: dict[str, list[int]] = defaultdict(list)
    for doc_id, qa_idx in sampled_pairs:
        by_doc[doc_id].append(qa_idx)

    # ------------------------------------------------------------------
    # Phase 3: Build indices per needed doc, run all ablation configs
    # ------------------------------------------------------------------
    ablation_results = {abl["name"]: [] for abl in ABLATION_CONFIGS}
    queries_done = 0

    for doc_id, qa_indices in by_doc.items():
        doc, chunks = doc_store[doc_id]
        chunk_map = {c.chunk_id: c for c in chunks}

        vector_idx = VectorIndex()
        vector_idx.add([c.chunk_id for c in chunks], [c.text for c in chunks])
        entity_idx = EntityIndex(doc_id)
        entity_idx.build_from_chunks(chunks)

        for qa_idx in qa_indices:
            qa = doc.questions[qa_idx]

            entities = detect_query_entities(qa["question"], entity_idx)
            entity = entities[0] if entities else None

            for abl in ABLATION_CONFIGS:
                retriever = HybridEntityRetriever(
                    vector_idx, entity_idx, chunk_map,
                    alpha=abl["alpha"], beta=abl["beta"], gamma=abl["gamma"],
                )
                retrieved = retriever.retrieve(qa["question"], k=k)
                context_texts = [r.text for r in retrieved]
                syn_chunks = to_synthesis_chunks(retrieved, chunk_map, doc_id)
                result = synthesize(qa["question"], syn_chunks, entity=entity, model="gpt-4.1-nano")
                answer = result["answer"]
                rouge = compute_rouge_l(answer, qa.get("answer", ""))
                sc = scatter_coverage_at_k([r.chunk_id for r in retrieved], chunk_map)

                ablation_results[abl["name"]].append({
                    "rougeL": rouge["rougeL_fmeasure"],
                    "scatter_coverage": sc,
                })

            queries_done += 1

    with open(out_dir / "exp2_2_results.json", "w") as f:
        json.dump(ablation_results, f, indent=2)

    _plot_figure9(ablation_results, out_dir)
    logger.info("Exp 2.2 complete. Results in %s", out_dir)
    return ablation_results


def _plot_figure9(results: dict, out_dir: Path):
    """Figure 9: Ablation waterfall chart."""
    fig, ax = create_figure(figsize=(10, 5))

    names = list(results.keys())
    means = [float(np.nanmean([r["rougeL"] for r in results[n]])) if results[n] else 0 for n in names]

    bars = ax.barh(range(len(names)), means, color=["#0072B2"] + ["#E69F00"] * (len(names) - 1))
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("ROUGE-L F1")
    ax.set_title("Ablation Study: Component Contributions")
    ax.invert_yaxis()
    save_figure(fig, out_dir / "figure9_ablation.pdf")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
