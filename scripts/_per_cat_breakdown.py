"""Quick per-category breakdown of attribute metrics."""
import json
import numpy as np
from collections import defaultdict

with open("results/attr_eval/attribute_metrics.json") as f:
    data = json.load(f)

records = data["per_record"]
strategies = ["BM25", "StandardSemantic", "EntityExpanded", "EntityFirst", "Iterative", "HybridEntity"]
metrics = ["attr_recall", "attr_precision", "attr_f1", "scatter_coverage"]

by_cat = defaultdict(list)
for r in records:
    by_cat[r["scatter_category"]].append(r)

print(f"Categories: {list(by_cat.keys())} (counts: {[len(v) for v in by_cat.values()]})\n")

for cat in sorted(by_cat.keys()):
    recs = by_cat[cat]
    print(f"{cat} (n={len(recs)}):")
    hdr = f"  {'Strategy':<22} {'AttrRec':>8} {'AttrPre':>8} {'AttrF1':>8} {'ScatCov':>8}"
    print(hdr)
    for s in strategies:
        vals = {m: np.mean([r[f"{s}_{m}"] for r in recs]) for m in metrics}
        row = f"  {s:<22} {vals['attr_recall']:>8.4f} {vals['attr_precision']:>8.4f} {vals['attr_f1']:>8.4f} {vals['scatter_coverage']:>8.4f}"
        print(row)
    print()
