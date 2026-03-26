"""Publication-quality figure generation for all 13 experiment figures."""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.utils.config import results_dir

# Publication defaults
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

# Colorblind-friendly palette (Wong 2011)
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00"]
sns.set_palette(PALETTE)


def create_figure(figsize: tuple[float, float] = (8, 5)) -> tuple[plt.Figure, plt.Axes]:
    """Create a publication-quality figure with a single axes."""
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def save_figure(fig: plt.Figure, name: str, formats: tuple[str, ...] = ("png", "pdf")):
    """Save figure to results/figures/ in multiple formats."""
    out_dir = results_dir("figures")
    name = Path(name)
    # Strip extension if already present (e.g. "fig1.pdf" -> "fig1")
    stem = name.with_suffix("").name if name.suffix else name.name
    parent = name.parent if name.is_absolute() or str(name.parent) != "." else out_dir
    for fmt in formats:
        path = parent / f"{stem}.{fmt}"
        fig.savefig(path)
    plt.close(fig)
    return parent / f"{stem}.{formats[0]}"
