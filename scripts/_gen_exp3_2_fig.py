"""One-shot script to generate exp3_2 cross-domain heatmap from existing results."""
import json
from pathlib import Path
from experiments.phase3.exp3_2_cross_domain import _generate_figures

results = json.load(open("results/exp3_2/exp3_2_results.json"))
_generate_figures(results, Path("results/exp3_2"))
print("exp3_2 figure generated")
