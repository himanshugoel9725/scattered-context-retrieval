"""Exp 1.1: Scatter Factor Measurement.

Sample 100 entities across 3 domains (NarrativeQA characters, CUAD contract parties,
QASPER methods/models). Compute scatter factors. Generate Figure 1 (violin) and Figure 2 (completeness).
"""

import json
import logging
from pathlib import Path

import numpy as np

from src.data.loaders import load_dataset
from src.data.processors import clean_text
from src.indexing.chunker import Chunker
from src.indexing.entity_index import EntityIndex
from src.evaluation.metrics import compute_scatter_factor
from src.utils.config import get_experiments_config, results_dir
from src.utils.plotting import create_figure, save_figure

logger = logging.getLogger(__name__)


def run(config: dict | None = None):
    """Run Experiment 1.1: Scatter Factor Measurement."""
    if config is None:
        config = get_experiments_config()["phase1"]["exp1_1_scatter_factor"]

    out_dir = results_dir("exp1_1")
    chunker = Chunker(chunk_size=512, overlap=128)

    all_scatter_factors = {}  # {domain: [SF values]}

    for domain in ["narrativeqa", "qasper", "cuad"]:
        logger.info("Processing domain: %s", domain)
        dataset = load_dataset(domain)
        if not dataset:
            logger.warning("Could not load %s, skipping", domain)
            continue

        domain_sfs = []
        n_entities_target = config.get("entities_per_domain", 34)  # ~100 total / 3

        # Process subset of documents
        for doc in dataset[:config.get("docs_per_domain", 20)]:
            text = clean_text(doc.text)
            doc_id = doc.doc_id
            chunks = chunker.chunk_document(doc_id, text)
            chunk_map = {c.chunk_id: c for c in chunks}

            entity_idx = EntityIndex(doc_id)
            entity_idx.build_from_chunks(chunks)

            top_ents = entity_idx.get_top_entities(5)
            for ent in top_ents:
                sf = compute_scatter_factor(
                    ent["canonical"], ent["chunk_ids"], chunk_map
                )
                domain_sfs.append({
                    "entity": ent["canonical"],
                    "scatter_factor": sf,
                    "n_chunks": len(ent["chunk_ids"]),
                    "domain": domain,
                    "doc_id": doc_id,
                })
                if len(domain_sfs) >= n_entities_target:
                    break
            if len(domain_sfs) >= n_entities_target:
                break

        all_scatter_factors[domain] = domain_sfs
        logger.info("Domain %s: computed SF for %d entities", domain, len(domain_sfs))

    # Save raw results
    with open(out_dir / "scatter_factors.json", "w") as f:
        json.dump(all_scatter_factors, f, indent=2)

    # Generate Figure 1: Violin plot of SF distributions
    _plot_figure1(all_scatter_factors, out_dir)

    # Generate Figure 2: Completeness vs retrieved chunks
    _plot_figure2(all_scatter_factors, out_dir)

    logger.info("Exp 1.1 complete. Results in %s", out_dir)
    return all_scatter_factors


def _plot_figure1(data: dict, out_dir: Path):
    """Figure 1: Violin plot of scatter factor distributions by domain."""
    fig, ax = create_figure(figsize=(8, 5))

    domains = []
    sf_values = []
    for domain, entities in data.items():
        for ent in entities:
            domains.append(domain.upper())
            sf_values.append(ent["scatter_factor"])

    if not sf_values:
        return

    import pandas as pd
    import seaborn as sns
    df = pd.DataFrame({"Domain": domains, "Scatter Factor": sf_values})
    sns.violinplot(data=df, x="Domain", y="Scatter Factor", ax=ax, inner="box")
    ax.set_title("Distribution of Entity Scatter Factors Across Domains")
    ax.set_ylabel("Scatter Factor (SF)")
    save_figure(fig, out_dir / "figure1_scatter_factor_distribution.pdf")


def _plot_figure2(data: dict, out_dir: Path):
    """Figure 2: Number of chunks per entity vs scatter factor."""
    fig, ax = create_figure(figsize=(8, 5))
    for domain, entities in data.items():
        xs = [e["n_chunks"] for e in entities]
        ys = [e["scatter_factor"] for e in entities]
        ax.scatter(xs, ys, label=domain.upper(), alpha=0.7, s=30)

    ax.set_xlabel("Number of Chunks Containing Entity")
    ax.set_ylabel("Scatter Factor")
    ax.set_title("Entity Scatter Factor vs. Information Spread")
    ax.legend()
    save_figure(fig, out_dir / "figure2_sf_vs_chunks.pdf")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
