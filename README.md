# Scatter-RAG

**Beyond Single-Chunk Retrieval: Entity-Centric Aggregation for Distributed Information**

Information about entities in long documents is often *scattered* across many passages — a character's traits unfold over chapters, a legal clause's implications span multiple sections. Standard RAG retrieves top-k chunks by similarity but misses these distributed mentions, leading to incomplete answers.

**Scatter-RAG** addresses this with entity-centric retrieval:

1. **Entity detection** — identify entities in the query using spaCy NER + fuzzy matching against a pre-built entity index
2. **Coreference resolution** — resolve aliases and pronouns to canonical entity forms (fastcoref)
3. **Entity→chunk index** — map each entity to every chunk where it appears
4. **Hybrid retrieval** — combine entity-aware, dense semantic (FAISS), and keyword (BM25) scoring to maximize coverage of scattered information
5. **Scatter-aware synthesis** — reorder retrieved chunks (chronologically, by entity cluster) before LLM generation

## Key Metrics

| Metric | Description |
|--------|-------------|
| **Scatter Factor (SF)** | Measures how distributed entity information is across a document: `SF = (N_chunks × avg_pairwise_distance) / doc_length` |
| **Information Completeness Score (ICS)** | Measures how complete entity descriptions are in retrieved chunks |
| **Scatter Coverage@K** | Fraction of document deciles represented in top-k retrieved chunks |

## Repository Structure

```
scatter-rag/
├── configs/                  # YAML configuration files
│   ├── datasets.yaml         # Dataset paths, splits, metadata
│   ├── experiments.yaml      # All 12 experiment definitions
│   ├── models.yaml           # Embedding, NER, LLM model configs
│   ├── attribute_schemas.yaml
│   └── query_schema.yaml
├── data/
│   ├── annotations/          # Entity annotations
│   └── scatterqa/            # ScatterQA benchmark (custom)
├── experiments/
│   ├── run_experiment.py     # CLI entrypoint for all experiments
│   ├── phase1/               # Problem quantification (exp 1.1–1.3)
│   ├── phase2/               # System experiments (exp 2.1–2.5)
│   └── phase3/               # Deep analysis (exp 3.1–3.4)
├── results/
│   ├── exp1_1/ … exp3_4/     # Per-experiment JSON results + figures
│   ├── figures/              # All 16 figures (PDF + PNG)
│   └── logs/                 # Experiment logs
├── scripts/                  # Data prep, index building, utilities
├── src/
│   ├── benchmark/            # ScatterQA benchmark construction
│   ├── data/                 # Dataset loaders & processors
│   ├── evaluation/           # Metrics (SF, ICS, ROUGE, BERTScore, RAGAS)
│   ├── generation/           # LLM clients, prompts, synthesis
│   ├── indexing/             # Chunker, entity index, vector index, coref
│   ├── retrieval/            # 7 retrieval strategies
│   └── utils/                # Config, caching, plotting
├── tests/                    # 40 tests across 6 modules
├── pyproject.toml
├── requirements.txt
└── .env.example
```

## Setup

```bash
# Clone
git clone https://github.com/himanshugoel9725/scattered-context-retrieval.git
cd scattered-context-retrieval

# Virtual environment
python -m venv .venv
source .venv/bin/activate

# Dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_lg

# API keys
cp .env.example .env
# Edit .env with your OpenAI / Anthropic / Google API keys
```

## Data Preparation

```bash
# Download NarrativeQA, CUAD, QASPER, Quality, NovelHopQA
python scripts/download_datasets.py

# Build entity + vector indices
python scripts/build_indices.py

# Build ScatterQA benchmark (from Project Gutenberg novels)
python scripts/build_scatterqa.py
```

## Running Experiments

```bash
# Run a single experiment
python -m experiments.run_experiment --exp 1.1

# List all 12 experiments
python -m experiments.run_experiment --list

# Run with all 3 seeds (42, 123, 456) for variance estimation
python -m experiments.run_experiment --exp 2.1 --multi-seed

# Override seed / enable debug logging
python -m experiments.run_experiment --exp 2.1 --seed 123 --verbose
```

## Experiments

### Phase 1 — Problem Quantification

| ID | Name | Description |
|----|------|-------------|
| 1.1 | Scatter Factor | Quantify how scattered entity info is across narrative, legal, and scientific domains |
| 1.2 | RAG Failure | Demonstrate standard RAG fails on scattered queries vs. localized queries |
| 1.3 | Completeness Audit | Measure retrieval completeness at k = 5, 10, 20 |

### Phase 2 — System Experiments

| ID | Name | Description |
|----|------|-------------|
| 2.1 | Strategy Comparison | Compare 5 retrieval strategies on 4 datasets (main result) |
| 2.2 | Ablation | Remove one component at a time to measure contribution |
| 2.3 | Chunk Count | Optimal k for low / medium / high scatter factor bins |
| 2.4 | LLM Comparison | 6 LLMs (GPT-4o, Claude Sonnet, Gemini Pro, Llama 3 70B/8B) on same chunks |
| 2.5 | Ordering | Impact of chunk ordering (chronological, entity-clustered, relevance, random) |

### Phase 3 — Deep Analysis

| ID | Name | Description |
|----|------|-------------|
| 3.1 | Scatter Taxonomy | Categorize scatter types (progressive, distributed, contradictory, cross-ref, implicit) |
| 3.2 | Cross-Domain | Domain transfer of entity detection across narrative ↔ legal ↔ scientific |
| 3.3 | Error Analysis | Failure mode categorization (entity detection, coref, over/under-retrieval, synthesis) |
| 3.4 | Long Context | Scatter-aware RAG vs. full-document LLM (128K context) on quality vs. cost |

## Datasets

| Dataset | Domain | Size | Source |
|---------|--------|------|--------|
| NarrativeQA | Narrative | 1,567 docs / 46,765 QA | [deepmind/narrativeqa](https://github.com/deepmind/narrativeqa) |
| CUAD | Legal | 510 contracts / 13K+ annotations | [TheAtticusProject/cuad](https://github.com/TheAtticusProject/cuad) |
| QASPER | Scientific | 1,585 papers / 5,049 questions | [allenai/qasper](https://github.com/allenai/qasper-led-baseline) |
| ScatterQA | Narrative | 50 novels / 500 entities / 2,500 QA | Custom (built from Project Gutenberg) |
| Quality | Narrative | 265 articles / 6,737 questions | Supplementary |
| NovelHopQA | Narrative | 4,000 multi-hop QA | Supplementary |

## Figures

All 16 figures are in `results/figures/` (PDF + PNG):

| # | Figure | Experiment |
|---|--------|------------|
| 1 | Scatter factor distribution | 1.1 |
| 2 | SF vs. chunks needed | 1.1 |
| 3 | RAG failure on scattered queries | 1.2 |
| 4 | Scatter degradation curve | 1.2 |
| 5 | Completeness by scatter factor | 1.3 |
| 6 | Strategy comparison (retrieval) | 2.1 |
| 7 | Scatter coverage (main result) | 2.1 |
| 8 | Quality vs. latency | 2.1 |
| 9 | Ablation waterfall | 2.2 |
| 10 | ICS vs. chunk count | 2.3 |
| 11 | LLM comparison | 2.4 |
| 12 | Ordering comparison | 2.5 |
| 13 | Quality vs. cost | 3.4 |
| — | Scatter taxonomy | 3.1 |
| — | Cross-domain transfer | 3.2 |
| — | Error analysis | 3.3 |

## Testing

```bash
pytest                    # Run all 40 tests
pytest -v                 # Verbose output
pytest tests/test_metrics.py  # Run specific module
```

6 test modules: entity index, metrics, RAGAS metrics, retrieval strategies, robustness, scatter factor.

## License

[MIT](LICENSE)
