"""Run paired bootstrap tests on the four key paper claims."""
import json
import numpy as np
from pathlib import Path
from src.evaluation.statistics import paired_bootstrap_test, significance_annotation

ROOT = Path(__file__).parent.parent
exp2_1 = json.loads((ROOT / "results/exp2_1/exp2_1_results.json").read_text())
attr   = json.loads((ROOT / "results/attr_eval/attribute_metrics.json").read_text())
exp2_2 = json.loads((ROOT / "results/exp2_2/exp2_2_results.json").read_text())

# ── 1 & 2: ScatterQA scatter_coverage from exp2_1 ─────────────────────────
sqa     = exp2_1["scatterqa"]
hy_cov  = [r["HybridEntity_scatter_coverage"]     for r in sqa if "HybridEntity_scatter_coverage"     in r]
sem_cov = [r["StandardSemantic_scatter_coverage"] for r in sqa if "StandardSemantic_scatter_coverage" in r]
bm_cov  = [r["BM25_scatter_coverage"]             for r in sqa if "BM25_scatter_coverage"             in r]
n12 = min(len(hy_cov), len(sem_cov), len(bm_cov))

r1 = paired_bootstrap_test(hy_cov[:n12], sem_cov[:n12])
print(f"1. HybridEntity vs Semantic — Scatter Coverage (ScatterQA, n={n12})")
print(f"   HE={np.mean(hy_cov[:n12]):.4f}  Sem={np.mean(sem_cov[:n12]):.4f}  diff={r1.observed_diff:+.4f}")
print(f"   p={r1.p_value:.4f}  {significance_annotation(r1.p_value)}  95%CI=[{r1.ci_lower:+.4f},{r1.ci_upper:+.4f}]")
print()

r2 = paired_bootstrap_test(hy_cov[:n12], bm_cov[:n12])
print(f"2. HybridEntity vs BM25 — Scatter Coverage (ScatterQA, n={n12})")
print(f"   HE={np.mean(hy_cov[:n12]):.4f}  BM25={np.mean(bm_cov[:n12]):.4f}  diff={r2.observed_diff:+.4f}")
print(f"   p={r2.p_value:.4f}  {significance_annotation(r2.p_value)}  95%CI=[{r2.ci_lower:+.4f},{r2.ci_upper:+.4f}]")
print()

# ── 3: EntityFirst vs Semantic — Attr F1 (per-record) ─────────────────────
per    = attr["per_record"]
ef_f1  = [r["EntityFirst_attr_f1"]      for r in per if "EntityFirst_attr_f1"      in r]
sem_f1 = [r["StandardSemantic_attr_f1"] for r in per if "StandardSemantic_attr_f1" in r]
n3 = min(len(ef_f1), len(sem_f1))

r3 = paired_bootstrap_test(ef_f1[:n3], sem_f1[:n3])
print(f"3. EntityFirst vs Semantic — Attr F1 (n={n3})")
print(f"   EF={np.mean(ef_f1[:n3]):.4f}  Sem={np.mean(sem_f1[:n3]):.4f}  diff={r3.observed_diff:+.4f}")
print(f"   p={r3.p_value:.4f}  {significance_annotation(r3.p_value)}  95%CI=[{r3.ci_lower:+.4f},{r3.ci_upper:+.4f}]")
print()

# ── 4: Full System vs No Diversity (γ=0) — Scatter Coverage (ablation) ────
full  = [r["scatter_coverage"] for r in exp2_2["Full System"]]
nodiv = [r["scatter_coverage"] for r in exp2_2["No Diversity (\u03b3=0)"]]
n4 = min(len(full), len(nodiv))

r4 = paired_bootstrap_test(full[:n4], nodiv[:n4])
print(f"4. Full System vs No Diversity (γ=0) — Scatter Coverage (n={n4})")
print(f"   Full={np.mean(full[:n4]):.4f}  NoDiv={np.mean(nodiv[:n4]):.4f}  diff={r4.observed_diff:+.4f}")
print(f"   p={r4.p_value:.4f}  {significance_annotation(r4.p_value)}  95%CI=[{r4.ci_lower:+.4f},{r4.ci_upper:+.4f}]")
