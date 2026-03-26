"""Exp 3.2: Cross-Domain Transfer.

Train entity detection on novels → test on contracts, and vice versa.
Report transfer degradation.
"""

import json
import logging
from pathlib import Path

import numpy as np

from src.data.loaders import load_dataset
from src.data.processors import clean_text
from src.indexing.chunker import Chunker
from src.indexing.entity_index import EntityIndex
from src.indexing.vector_index import VectorIndex
from src.retrieval.hybrid_entity import HybridEntityRetriever
from src.generation.synthesizer import synthesize, to_synthesis_chunks
from src.retrieval.base import detect_query_entities
from src.evaluation.standard_metrics import compute_rouge_l
from src.utils.config import get_experiments_config, results_dir
from src.utils.plotting import create_figure, save_figure

logger = logging.getLogger(__name__)


def run(config: dict | None = None):
    if config is None:
        config = get_experiments_config()["phase3"]["exp3_2_cross_domain"]

    out_dir = results_dir("exp3_2")
    chunker = Chunker(chunk_size=512, overlap=128)

    # Compare performance on each domain: in-domain vs out-of-domain entity types
    results = {}
    for ds_name in ["quality", "cuad", "qasper"]:
        dataset = load_dataset(ds_name)
        if not dataset:
            continue

        ds_results = {"in_domain": [], "cross_domain": []}

        for doc in dataset[:config.get("max_docs", 15)]:
            text = clean_text(doc.text)
            chunks = chunker.chunk_document(doc.doc_id, text)
            if len(chunks) < 5:
                continue
            chunk_map = {c.chunk_id: c for c in chunks}

            # In-domain: use all entity types
            entity_idx_full = EntityIndex(doc.doc_id)
            entity_idx_full.build_from_chunks(chunks)

            # Cross-domain: restrict to PERSON only (narrative entity type)
            entity_idx_person = EntityIndex(doc.doc_id)
            entity_idx_person.build_from_chunks(chunks, entity_types=("PERSON",))

            vector_idx = VectorIndex()
            vector_idx.add([c.chunk_id for c in chunks], [c.text for c in chunks])

            for qa in doc.questions[:5]:
                for label, eidx in [("in_domain", entity_idx_full), ("cross_domain", entity_idx_person)]:
                    # Detect entity using the corresponding entity index
                    entities = detect_query_entities(qa["question"], eidx)
                    entity = entities[0] if entities else None

                    retriever = HybridEntityRetriever(vector_idx, eidx, chunk_map)
                    retrieved = retriever.retrieve(qa["question"], k=15)
                    context_texts = [r.text for r in retrieved]
                    syn_chunks = to_synthesis_chunks(retrieved, chunk_map, doc.doc_id)
                    result = synthesize(qa["question"], syn_chunks, entity=entity, model="gpt-4o-mini")
                    answer = result["answer"]
                    rouge = compute_rouge_l(answer, qa.get("answer", ""))
                    ds_results[label].append(rouge["rougeL_fmeasure"])

        results[ds_name] = {
            "in_domain_mean": float(np.mean(ds_results["in_domain"])) if ds_results["in_domain"] else 0,
            "cross_domain_mean": float(np.mean(ds_results["cross_domain"])) if ds_results["cross_domain"] else 0,
            "degradation": float(np.mean(ds_results["in_domain"]) - np.mean(ds_results["cross_domain"]))
            if ds_results["in_domain"] and ds_results["cross_domain"] else 0,
        }

    with open(out_dir / "exp3_2_results.json", "w") as f:
        json.dump(results, f, indent=2)

    _generate_figures(results, out_dir)
    logger.info("Exp 3.2 complete. Results: %s", results)
    return results


def _generate_figures(results: dict, out_dir: Path):
    """Generate cross-domain transfer heatmap."""
    domains = list(results.keys())
    if not domains:
        return

    labels = ["In-Domain", "Cross-Domain"]
    data = np.array([[results[d]["in_domain_mean"], results[d]["cross_domain_mean"]]
                     for d in domains])

    fig, ax = create_figure(figsize=(6, 4))
    import seaborn as sns
    sns.heatmap(data, annot=True, fmt=".3f", xticklabels=labels,
                yticklabels=[d.upper() for d in domains], cmap="YlOrRd_r",
                vmin=0, ax=ax)
    ax.set_title("Cross-Domain Transfer: ROUGE-L Performance")
    save_figure(fig, out_dir / "figure_cross_domain_transfer.pdf")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
