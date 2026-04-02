#!/usr/bin/env python3
"""Generate 11 new paper figures. Run from workspace root:
    PYTHONPATH=. .venv/bin/python scripts/gen_paper_figures.py

All output figures are saved with a 'new_' prefix so they are instantly
distinguishable from earlier figures in results/figures/.
"""
import json
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

ROOT    = Path(__file__).parent.parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
FIGURES.mkdir(exist_ok=True)

STYLE = {
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}
plt.rcParams.update(STYLE)

# Wong 2011 colorblind-safe palette
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00"]

STRATS_4 = ["BM25", "StandardSemantic", "EntityFirst", "HybridEntity"]
LABELS_4  = ["BM25", "Semantic", "EntityFirst", "HybridEntity"]


def load(rel: str):
    with open(ROOT / rel) as f:
        return json.load(f)


def save(fig: plt.Figure, stem: str) -> None:
    """Save figure as both PDF and PNG inside results/figures/."""
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{stem}.{ext}")
    plt.close(fig)
    print(f"  ✓  {stem}.{{pdf,png}}")


# ─── Figure 1 ─── Attribute-Level Performance ────────────────────────────────
def fig_attr_performance():
    summary = load("results/attr_eval/attribute_metrics.json")["summary"]
    n = 515  # ScatterQA evaluation set size

    metrics   = ["attr_precision", "attr_recall", "attr_f1"]
    mlabels   = ["Precision", "Recall", "F1"]
    std_keys  = ["attr_precision_std", "attr_recall_std", "attr_f1_std"]

    x     = np.arange(len(STRATS_4))
    width = 0.22
    offs  = np.array([-1, 0, 1]) * width

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (m, ml, sk) in enumerate(zip(metrics, mlabels, std_keys)):
        vals = [summary[s][m] for s in STRATS_4]
        errs = [summary[s][sk] / np.sqrt(n) for s in STRATS_4]
        ax.bar(x + offs[i], vals, width, label=ml,
               color=PALETTE[i], alpha=0.87,
               yerr=errs, capsize=3,
               error_kw={"linewidth": 1.2, "ecolor": "black"})

    # Highlight best strategies
    ax.axvspan(1.62, 2.38, color="#FFD700", alpha=0.13, zorder=0)
    ax.axvspan(2.62, 3.38, color="#FFD700", alpha=0.13, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS_4)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Attribute-Level Retrieval Performance (ScatterQA, n=515)")
    ax.legend(ncol=3)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Annotate best F1
    f1_vals = [summary[s]["attr_f1"] for s in STRATS_4]
    best_f1 = max(f1_vals)
    for xi, v in enumerate(f1_vals):
        if v == best_f1:
            ax.annotate(f"F1={v:.3f}",
                        xy=(xi + offs[2], v + 0.025),
                        ha="center", fontsize=8.5,
                        fontweight="bold", color="#8B6914")

    fig.tight_layout()
    save(fig, "new_fig_attr_performance")


# ─── Figure 2 ─── Metric Misalignment: ROUGE vs Attribute F1 ─────────────────
def fig_metric_misalignment():
    summary = load("results/attr_eval/attribute_metrics.json")["summary"]
    exp2_1  = load("results/exp2_1/exp2_1_results.json")

    rouge_acc: dict = defaultdict(list)
    for recs in exp2_1.values():
        for r in recs:
            for s in STRATS_4:
                v = r.get(f"{s}_rougeL")
                if v is not None:
                    rouge_acc[s].append(v)
    rouge_means = {s: float(np.mean(rouge_acc[s])) for s in STRATS_4}
    attr_f1     = {s: summary[s]["attr_f1"]        for s in STRATS_4}

    label_offsets = {
        "BM25":             (-0.0004,  0.012),
        "StandardSemantic": ( 0.0003,  0.012),
        "EntityFirst":      ( 0.0003, -0.018),
        "HybridEntity":     (-0.0002, -0.018),
    }

    fig, ax = plt.subplots(figsize=(7, 6))
    for s, lbl, col in zip(STRATS_4, LABELS_4, PALETTE):
        ax.scatter(rouge_means[s], attr_f1[s], color=col, s=130, zorder=5, label=lbl)
        dx, dy = label_offsets.get(s, (0.0002, 0.010))
        ax.annotate(
            lbl,
            xy=(rouge_means[s], attr_f1[s]),
            xytext=(rouge_means[s] + dx, attr_f1[s] + dy),
            fontsize=10,
            ha="left" if dx >= 0 else "right",
            fontweight="bold" if s == "HybridEntity" else "normal",
        )

    # Annotate spreads
    rouge_vals = list(rouge_means.values())
    f1_vals    = list(attr_f1.values())
    mid_rouge  = (min(rouge_vals) + max(rouge_vals)) / 2
    ax.annotate(
        "",
        xy=(max(rouge_vals), 0.17),
        xytext=(min(rouge_vals), 0.17),
        arrowprops=dict(arrowstyle="<->", color="gray", lw=1.5),
    )
    ax.text(mid_rouge, 0.152,
            f"ROUGE spread: {max(rouge_vals)-min(rouge_vals):.4f}",
            ha="center", fontsize=8, color="gray")

    left_x = min(rouge_vals) - 0.001
    mid_f1 = (min(f1_vals) + max(f1_vals)) / 2
    ax.annotate(
        "",
        xy=(left_x, max(f1_vals)),
        xytext=(left_x, min(f1_vals)),
        arrowprops=dict(arrowstyle="<->", color="#D55E00", lw=1.5),
    )
    ax.text(left_x - 0.0006, mid_f1,
            f"Attr F1\nspread:\n{max(f1_vals)-min(f1_vals):.3f}",
            ha="right", fontsize=8, color="#D55E00")

    ax.set_xlabel("ROUGE-L F1 (mean, 800 queries × 4 datasets)")
    ax.set_ylabel("Attribute F1 (ScatterQA, n=515)")
    ax.set_title("Metric Misalignment: ROUGE-L vs. Attribute F1")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    save(fig, "new_fig_metric_misalignment")


# ─── Figure 3 ─── RAG Failure: Localized vs Scattered (fixed) ────────────────
def fig_rag_failure():
    data = load("results/exp1_2/exp1_2_results.json")
    loc  = data["localized"]
    sct  = data["scattered"]

    bm25_loc = float(np.mean([r["bm25_rougeL"]     for r in loc]))
    bm25_sct = float(np.mean([r["bm25_rougeL"]     for r in sct]))
    sem_loc  = float(np.mean([r["semantic_rougeL"] for r in loc]))
    sem_sct  = float(np.mean([r["semantic_rougeL"] for r in sct]))

    categories = ["Localized Queries", "Scattered Queries"]
    x = np.arange(2)
    w = 0.30

    fig, ax = plt.subplots(figsize=(7, 5))
    bars_b = ax.bar(x - w / 2, [bm25_loc, bm25_sct], w,
                    label="BM25", color=PALETTE[0], alpha=0.87)
    bars_s = ax.bar(x + w / 2, [sem_loc,  sem_sct],  w,
                    label="Semantic", color=PALETTE[1], alpha=0.87)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylabel("ROUGE-L F1")
    ax.set_title("Standard RAG Failure on Scattered Queries")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Annotate all bars with value; scattered bars also get % drop
    for bars, loc_val, sct_val in [
        (bars_b, bm25_loc, bm25_sct),
        (bars_s, sem_loc,  sem_sct),
    ]:
        for idx, bar in enumerate(bars):
            h  = bar.get_height()
            xp = bar.get_x() + bar.get_width() / 2
            ax.text(xp, h + 0.002, f"{h:.4f}",
                    ha="center", va="bottom", fontsize=8.5)
            if idx == 1 and loc_val > 0:           # scattered bar
                pct = (loc_val - sct_val) / loc_val * 100
                ax.text(xp, h + 0.013, f"↓ {pct:.0f}%",
                        ha="center", fontsize=8.5,
                        color="darkred", fontweight="bold")

    fig.tight_layout()
    save(fig, "new_fig_rag_failure_v2")


# ─── Figure 4 ─── Scatter Coverage by Strategy (X=strategies, groups=datasets)
def fig_scatter_coverage_strategies():
    data     = load("results/exp2_1/exp2_1_results.json")
    datasets = ["quality", "cuad", "qasper", "scatterqa"]
    dlabels  = ["QuALITY", "CUAD", "QASPER", "ScatterQA"]

    x    = np.arange(len(STRATS_4))
    n_ds = len(datasets)
    w    = 0.18
    offs = np.linspace(-(n_ds - 1) / 2 * w, (n_ds - 1) / 2 * w, n_ds)

    fig, ax = plt.subplots(figsize=(10, 5))
    for di, (ds, dlbl) in enumerate(zip(datasets, dlabels)):
        means = [
            float(np.nanmean([r.get(f"{s}_scatter_coverage", np.nan)
                               for r in data[ds]]))
            for s in STRATS_4
        ]
        is_scatter = ds == "scatterqa"
        ax.bar(x + offs[di], means, w,
               label=dlbl,
               color=PALETTE[di],
               alpha=1.0 if is_scatter else 0.72,
               edgecolor="black" if is_scatter else "none",
               linewidth=1.2 if is_scatter else 0)

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS_4)
    ax.set_ylabel("Scatter Coverage@15")
    ax.set_ylim(0, 1.15)
    ax.set_title("Scatter Coverage by Retrieval Strategy (Grouped by Dataset)")
    ax.legend(ncol=2)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.axhline(1.0, linestyle=":", color="gray", linewidth=1, alpha=0.7)

    fig.tight_layout()
    save(fig, "new_fig_scatter_coverage_strategies")


# ─── Figure 5 ─── Effect of Retrieval Depth K vs Scatter Factor ──────────────
def fig_kdepth_vs_sf():
    records = load("results/exp2_3/exp2_3_results.json")
    k_vals  = [1, 3, 5, 7, 10, 15, 20, 30]

    bins: dict = defaultdict(list)
    for r in records:
        bins[r["sf_bin"]].append(r)

    bin_cfg = [
        ("low",    "Low SF (sf < 0.1)",    PALETTE[2]),
        ("medium", "Medium SF (0.1–0.3)",  PALETTE[0]),
        ("high",   "High SF (sf ≥ 0.3)",   PALETTE[1]),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    for key, lbl, col in bin_cfg:
        if key not in bins:
            continue
        grp   = bins[key]
        means = [
            float(np.nanmean([r.get(f"rougeL_k{k}", np.nan) for r in grp]))
            for k in k_vals
        ]
        ax.plot(k_vals, means, marker="o", label=lbl, color=col, linewidth=2.2)
        pk = int(np.argmax(means))
        ax.annotate(
            f"peak k={k_vals[pk]}",
            xy=(k_vals[pk], means[pk]),
            xytext=(k_vals[pk] + 1.5, means[pk] + 0.003),
            fontsize=8, color=col,
            arrowprops=dict(arrowstyle="->", color=col, lw=0.9),
        )

    ax.set_xlabel("Retrieval Depth k (chunks)")
    ax.set_ylabel("ROUGE-L F1")
    ax.set_title("Effect of Retrieval Depth on Performance by Scatter Factor Bin")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_xticks(k_vals)
    fig.tight_layout()
    save(fig, "new_fig_kdepth_vs_sf")


# ─── Figure 6 ─── Scatter Pattern Taxonomy Performance ───────────────────────
def fig_taxonomy_perf():
    records = load("results/exp3_1/exp3_1_results.json")

    by_type: dict = defaultdict(lambda: defaultdict(list))
    for r in records:
        stype = r.get("scatter_type", "unknown")
        for s in STRATS_4:
            v = r.get(f"{s}_rougeL")
            if v is not None:
                by_type[stype][s].append(v)

    # Canonical ordering + display labels (only types present in data)
    canonical_order = [
        "progressive_accumulation",
        "distributed_attributes",
        "cross_reference",
        "implicit",
        "contradictory_evolution",  # may be absent
    ]
    disp = {
        "progressive_accumulation": "Progressive\nAccumulation",
        "distributed_attributes":   "Distributed\nAttributes",
        "cross_reference":          "Cross\nReference",
        "implicit":                 "Implicit",
        "contradictory_evolution":  "Contradictory\nEvolution",
    }
    present = [t for t in canonical_order if t in by_type]
    xlabels = [disp[t] for t in present]

    x    = np.arange(len(present))
    n    = len(STRATS_4)
    w    = 0.18
    offs = np.linspace(-(n - 1) / 2 * w, (n - 1) / 2 * w, n)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (s, lbl) in enumerate(zip(STRATS_4, LABELS_4)):
        means = [
            float(np.mean(by_type[t][s])) if by_type[t][s] else 0.0
            for t in present
        ]
        ax.bar(x + offs[i], means, w, label=lbl,
               color=PALETTE[i], alpha=0.87)

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=9.5)
    ax.set_ylabel("ROUGE-L F1")
    ax.set_title("Performance by Scatter Pattern Type")
    ax.legend(ncol=4)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Annotate sample counts
    for xi, t in enumerate(present):
        n_rec = len(next(iter(by_type[t].values()), []))
        ax.text(xi, -0.004, f"n={n_rec}", ha="center", fontsize=7.5, color="gray")

    fig.tight_layout()
    save(fig, "new_fig_taxonomy_perf")


# ─── Figure 7 ─── Quality vs Cost Tradeoff ───────────────────────────────────
def fig_quality_cost():
    GPT4O_MINI_IN,  GPT4O_MINI_OUT  = 0.15, 0.60   # $/1M tokens
    GPT4O_IN,       GPT4O_OUT       = 2.50, 10.00

    # Scatter-aware RAG: k=15 chunks × ~200 tokens/chunk + 500 instruction → 3500 in, 300 out
    in_rag  = 15 * 200 + 500
    cost_rag  = (in_rag  * GPT4O_MINI_IN + 300 * GPT4O_MINI_OUT) / 1e6
    # Full-context: avg document ~8500 tok + 600 instruction
    in_full = 9100
    cost_mini = (in_full * GPT4O_MINI_IN + 400 * GPT4O_MINI_OUT) / 1e6
    cost_4o   = (in_full * GPT4O_IN      + 400 * GPT4O_OUT)      / 1e6

    # ROUGE-L for scatter-aware: HybridEntity mean from exp2_1
    exp2_1 = load("results/exp2_1/exp2_1_results.json")
    rag_scores = [
        r.get("HybridEntity_rougeL", np.nan)
        for recs in exp2_1.values()
        for r in recs
    ]
    rouge_rag = float(np.nanmean(rag_scores))

    exp2_4 = load("results/exp2_4/exp2_4_results.json")
    rouge_mini = float(np.mean([r["rougeL"] for r in exp2_4.get("gpt-4o-mini", [])]))
    rouge_4o   = float(np.mean([r["rougeL"] for r in exp2_4.get("gpt-4o", [])]))

    points = [
        (cost_rag,  rouge_rag,  "Scatter-Aware\nRAG (k=15)", PALETTE[2]),
        (cost_mini, rouge_mini, "Full-Context\nGPT-4o-mini",  PALETTE[0]),
        (cost_4o,   rouge_4o,   "Full-Context\nGPT-4o",       PALETTE[1]),
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    for xv, yv, lbl, col in points:
        ax.scatter(xv, yv, color=col, s=200, zorder=5)
        ax.annotate(
            lbl,
            xy=(xv, yv),
            xytext=(xv * 3.0, yv + 0.004),
            fontsize=9.5,
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.9),
        )

    ax.set_xscale("log")
    ax.set_xlabel("Estimated Cost per Query (USD, log scale)")
    ax.set_ylabel("ROUGE-L F1")
    ax.set_title("Quality–Cost Tradeoff Across System Configurations")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    fig.tight_layout()
    save(fig, "new_fig_quality_cost")


# ─── Figure 8 ─── Cross-Domain Transfer Heatmap ──────────────────────────────
def fig_cross_domain():
    raw     = load("results/exp3_2/exp3_2_results.json")
    domains = ["quality", "cuad", "qasper"]
    dnames  = ["QuALITY", "CUAD", "QASPER"]
    cols    = ["In-Domain", "Cross-Domain"]

    mat = np.array([
        [raw[d]["in_domain_mean"], raw[d]["cross_domain_mean"]]
        for d in domains
    ])

    fig, ax = plt.subplots(figsize=(6, 4))
    vmax = max(mat.max() * 1.1, 0.2)
    im   = ax.imshow(mat, cmap="YlGnBu", aspect="auto", vmin=0, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax, fraction=0.035)
    cbar.set_label("ROUGE-L F1")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(cols, fontsize=11)
    ax.set_yticks(range(len(domains)))
    ax.set_yticklabels(dnames)
    ax.set_title("Cross-Domain Transfer: Entity-Aware Retrieval (HybridEntity)")

    for i in range(len(domains)):
        for j in range(2):
            v     = mat[i, j]
            color = "white" if v > vmax * 0.55 else "black"
            delta = mat[i, 1] - mat[i, 0]
            if j == 0:
                note = f"{v:.4f}"
            else:
                sign = "↑" if delta >= 0 else "↓"
                note = f"{v:.4f}\n({sign}{abs(delta):.4f})"
            ax.text(j, i, note, ha="center", va="center",
                    fontsize=9.5, color=color)

    fig.tight_layout()
    save(fig, "new_fig_cross_domain")


# ─── Figure 9 ─── Scatter Factor Distribution (Violin) ───────────────────────
def fig_sf_violin():
    raw      = load("results/exp1_1/scatter_factors.json")
    datasets = sorted(raw.keys())

    disp = {
        "narrativeqa": "NarrativeQA",
        "qasper":      "QASPER",
        "cuad":        "CUAD",
    }

    sf_lists = [np.array([r["scatter_factor"] for r in raw[d]]) for d in datasets]

    fig, ax = plt.subplots(figsize=(8, 5))
    parts = ax.violinplot(sf_lists, positions=range(len(datasets)),
                          showmedians=True, showmeans=False, showextrema=True)

    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(PALETTE[i % len(PALETTE)])
        pc.set_alpha(0.72)
    for part in ("cmedians", "cmins", "cmaxes", "cbars"):
        parts[part].set_color("black")
        parts[part].set_linewidth(1.3)

    # Mean diamond
    for i, vals in enumerate(sf_lists):
        ax.scatter(i, float(np.mean(vals)), marker="D",
                   color="white", edgecolor="black", s=50, zorder=10, linewidth=1.3)

    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels([disp.get(d, d.upper()) for d in datasets])
    ax.set_ylabel("Scatter Factor")
    ax.set_ylim(-0.05, 1.12)
    ax.set_title("Scatter Factor Distribution by Dataset")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for i, vals in enumerate(sf_lists):
        ax.text(i, 1.07, f"μ={np.mean(vals):.2f}",
                ha="center", fontsize=8.5, color="gray")

    fig.tight_layout()
    save(fig, "new_fig_sf_violin")


# ─── Figure 10 ─── Scatter Factor vs Chunk Count (trend + colour by dataset) ─
def fig_sf_vs_chunks():
    raw      = load("results/exp1_1/scatter_factors.json")
    datasets = sorted(raw.keys())

    disp = {
        "narrativeqa": "NarrativeQA",
        "qasper":      "QASPER",
        "cuad":        "CUAD",
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    all_x, all_y = [], []

    for i, d in enumerate(datasets):
        xs = [r["n_chunks"]       for r in raw[d]]
        ys = [r["scatter_factor"] for r in raw[d]]
        ax.scatter(xs, ys, alpha=0.35, s=20,
                   color=PALETTE[i], label=disp.get(d, d.upper()))
        all_x.extend(xs)
        all_y.extend(ys)

    # Trend line via scipy pearsonr + numpy polyfit
    r_val, p_val = pearsonr(all_x, all_y)
    coeffs  = np.polyfit(all_x, all_y, 1)
    x_range = np.linspace(min(all_x), max(all_x), 300)
    y_fit   = np.polyval(coeffs, x_range)

    p_str = f"p={p_val:.3f}" if p_val >= 0.001 else "p<0.001"
    ax.plot(x_range, y_fit, color="black", linewidth=2, linestyle="--",
            label=f"Trend (r={r_val:.2f}, {p_str})")

    ax.set_xlabel("Number of Document Chunks Containing Entity")
    ax.set_ylabel("Scatter Factor")
    ax.set_title("Entity Dispersion vs. Document Scatter Complexity")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    save(fig, "new_fig_sf_vs_chunks")


# ─── Figure 11 ─── Qualitative Comparison Table ──────────────────────────────
def fig_qualitative_table():
    data = load("results/exp2_1/exp2_1_results.json")

    # Auto-select record with highest HybridEntity ROUGE-L advantage over BM25
    best_delta, best_rec = -1.0, None
    for recs in data.values():
        for r in recs:
            hy = r.get("HybridEntity_rougeL", 0.0)
            bm = r.get("BM25_rougeL", 0.0)
            if (hy - bm) > best_delta and r.get("reference", ""):
                best_delta = hy - bm
                best_rec   = r

    if best_rec is None:
        print("  ✗  No suitable record found for fig_qualitative_table")
        return

    def trunc(text: str, n: int = 140) -> str:
        text = str(text)
        return (text[:n].rstrip() + "…") if len(text) > n else text

    def wrap_cell(text: str, width: int = 42) -> str:
        return "\n".join(textwrap.wrap(str(text), width))

    q_disp   = textwrap.fill(best_rec["query"], 100)
    ref_disp = trunc(best_rec["reference"], 85)

    rows = [
        ["BM25",
         wrap_cell(trunc(best_rec["BM25_answer"],            140)),
         f"{best_rec.get('BM25_rougeL', 0):.3f}"],
        ["Semantic",
         wrap_cell(trunc(best_rec["StandardSemantic_answer"], 140)),
         f"{best_rec.get('StandardSemantic_rougeL', 0):.3f}"],
        ["HybridEntity",
         wrap_cell(trunc(best_rec["HybridEntity_answer"],    140)),
         f"{best_rec.get('HybridEntity_rougeL', 0):.3f}"],
    ]

    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.axis("off")

    ax.text(0.5, 0.99, f"Query: {q_disp}",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=9.5, style="italic")
    ax.text(0.5, 0.88, f"Reference: {ref_disp}",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=9, color="#555555")

    tbl = ax.table(
        cellText=rows,
        colLabels=["Method", "Answer Excerpt", "ROUGE-L"],
        cellLoc="left",
        loc="lower center",
        colWidths=[0.12, 0.74, 0.09],
        bbox=[0.0, 0.0, 1.0, 0.82],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)

    header_color = "#2C7BB6"
    for j in range(3):
        tbl[0, j].set_facecolor(header_color)
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Highlight HybridEntity row (row 3 = header[0] + data rows 1,2,3)
    for j in range(3):
        tbl[3, j].set_facecolor("#E8F4E8")

    ax.set_title(
        "Qualitative Comparison: Retrieval Methods on a Scattered Query",
        fontsize=12, pad=4,
    )
    fig.tight_layout()
    save(fig, "new_fig_qualitative_table")


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    steps = [
        ("1  Attribute-Level Performance",   fig_attr_performance),
        ("2  Metric Misalignment",           fig_metric_misalignment),
        ("3  RAG Failure (fixed)",           fig_rag_failure),
        ("4  Scatter Coverage Strategies",   fig_scatter_coverage_strategies),
        ("5  K-Depth vs Scatter Factor",     fig_kdepth_vs_sf),
        ("6  Taxonomy Performance",          fig_taxonomy_perf),
        ("7  Quality vs Cost",               fig_quality_cost),
        ("8  Cross-Domain Heatmap",          fig_cross_domain),
        ("9  SF Distribution Violin",        fig_sf_violin),
        ("10 SF vs Chunk Count",             fig_sf_vs_chunks),
        ("11 Qualitative Table",             fig_qualitative_table),
    ]
    for name, fn in steps:
        print(f"→ Figure {name}")
        fn()

    # Verify
    new_files = sorted(FIGURES.glob("new_fig_*.png"))
    print(f"\n{'='*50}")
    print(f"Done. {len(new_files)}/11 new PNG figures in results/figures/")
    for f in new_files:
        print(f"  {f.name}")
