"""Plotting for Experiment 1: loads each benchmark's decomposition.py
outputs from the observations folder
(observations/{MODEL_DIR}/{dataset}[_complete]/aggregate/) and produces four PNGs under
experiment_1/plots/ comparing mathvista vs hallusionbench:

  1. per_step_trends.png   - mean M_s vs T_s over normalized step position,
                             binned, +/- 1 SE band, all datasets on one axes (T_s in
                             the dataset's color, M_s a darker shade of it).
  2. beta_ci.png           - beta_M / beta_T / delta_beta point + bootstrap
                             CI, grouped by dataset.
  3. beta_scatter.png      - per-example beta_M vs beta_T, colored by
                             dataset, with a y=x reference line.
  4. redundancy_trend.png  - mean R_s over normalized step position, binned,
                             +/- 1 SE band, one line per dataset.

Purely descriptive: axis/legend labels only, no narrative interpretation
baked into the plots. summary.json's own "interpretation" string is
printed to the console instead, once per dataset.

Usage:
    python experiment_1/plot_results.py --model qwen3vl [--datasets mathvista hallusionbench] [--n-bins 20]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from lib.config import DATASET_PRESETS, MODEL_PRESETS, resolve_path

# Folder name under experiment_1/observations/ for each model key.
MODEL_DIRS = {
    "qwen3vl": "QWEN_3_VL_2B",
}

# Categorical palette slots 1-5 (blue, orange, aqua, yellow, magenta) - the
# default order's first five slots (see dataviz skill references/palette.md).
# NOTE: only the first 3 slots validate CVD-safe under "all pairs" (the
# check beta_scatter.png's plot needs, since every dataset there is visible
# at once); slots 4-5 are only adjacent-pair-safe. If all 5 benchmarks are
# ever passed to plot_beta_scatter() together, re-run the dataviz skill's
# validator on the resulting 5-way pairing before trusting that one plot's
# color-only distinctions - per_step_trends.png/beta_ci.png/
# redundancy_trend.png are unaffected since they never overlay more than
# one dataset's points at the same position.
COLORS = {
    "mathvista": "#2a78d6",
    "hallusionbench": "#eb6834",
    "chartqa": "#1baf7a",
    "mmmu": "#eda100",
    "realworldqa": "#e87ba4",
}
ZERO_LINE_COLOR = "#c3c2b7"  # palette's "baseline/axis" role
GRID_COLOR = "#e1e0d9"       # palette's "gridline (hairline)" role
N_BINS = 10


def find_run_dir(obs_dir: Path, dataset: str) -> Path:
    """Observation folders are named either "{dataset}" or "{dataset}_complete"."""
    for name in (dataset, f"{dataset}_complete"):
        if (obs_dir / name / "aggregate").is_dir():
            return obs_dir / name
    raise FileNotFoundError(f"no {dataset!r} or {dataset + '_complete'!r} folder with aggregate/ in {obs_dir}")


def load_dataset(dataset: str, obs_dir: Path) -> dict:
    agg_dir = find_run_dir(obs_dir, dataset) / "aggregate"

    step_df = pd.read_csv(agg_dir / "per_step_measures.csv")
    slopes_df = pd.read_csv(agg_dir / "per_example_slopes.csv")
    with open(agg_dir / "summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)

    return {"step_df": step_df, "slopes_df": slopes_df, "summary": summary}


def bin_stats(df: pd.DataFrame, value_col: str, n_bins: int = N_BINS) -> pd.DataFrame:
    """Bin s_hat into n_bins equal-width bins and return each bin's center,
    mean, and standard error of value_col.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(df["s_hat"], bins) - 1, 0, n_bins - 1)
    centers = (bins[:-1] + bins[1:]) / 2

    rows = []
    for b in range(n_bins):
        vals = df.loc[bin_idx == b, value_col].dropna()
        if len(vals) == 0:
            continue
        se = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        rows.append({"s_hat_center": centers[b], "mean": vals.mean(), "se": se, "n": len(vals)})
    return pd.DataFrame(rows)


def darken(color: str, amount: float = 0.45) -> str:
    """Mix `color` toward black by `amount` (0 = unchanged, 1 = black)."""
    r, g, b = mcolors.to_rgb(color)
    return mcolors.to_hex((r * (1 - amount), g * (1 - amount), b * (1 - amount)))


def plot_per_step_trends(data: dict, datasets: list[str], out_dir: Path, n_bins: int = N_BINS) -> None:
    """One axes for all benchmarks. Each benchmark's T_s is its palette color
    (solid, squares); its M_s is a darker shade of the same color (dashed,
    circles).
    """
    fig, ax = plt.subplots(figsize=(18, 11))
    ax.axhline(0, color=ZERO_LINE_COLOR, linewidth=1, zorder=1)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    for dataset in datasets:
        step_df = data[dataset]["step_df"]
        m_bins = bin_stats(step_df, "M_s", n_bins)
        t_bins = bin_stats(step_df, "T_s", n_bins)
        t_color = COLORS[dataset]
        m_color = darken(t_color)

        ax.plot(t_bins["s_hat_center"], t_bins["mean"], color=t_color, linewidth=2,
                marker="s", markersize=5, label=rf"{dataset}: $T_s$ (total)")
        ax.fill_between(t_bins["s_hat_center"], t_bins["mean"] - t_bins["se"],
                         t_bins["mean"] + t_bins["se"], color=t_color, alpha=0.15)

        ax.plot(m_bins["s_hat_center"], m_bins["mean"], color=m_color, linewidth=2,
                linestyle="--", marker="o", markersize=5,
                label=rf"{dataset}: $M_s$ (marginal)")
        ax.fill_between(m_bins["s_hat_center"], m_bins["mean"] - m_bins["se"],
                         m_bins["mean"] + m_bins["se"], color=m_color, alpha=0.15)

    ax.set_xlabel(r"step position (0 = start, 1 = end)")
    ax.set_ylabel("mean nats/token (± 1 SE)")
    ax.set_title(r"$M_s$ vs $T_s$ over trajectory position, all benchmarks")
    ax.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), frameon=False, ncol=2)

    fig.tight_layout()
    fig.savefig(out_dir / "per_step_trends.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_beta_ci(data: dict, datasets: list[str], out_dir: Path) -> None:
    coeffs = ["beta_M", "beta_T", "delta_beta"]
    coeff_labels = [
        r"$\beta_M$" + "\n(marginal slope)",
        r"$\beta_T$" + "\n(total slope)",
        r"$\Delta\beta$" + "\n" + r"($\beta_T - \beta_M$)",
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0, color=ZERO_LINE_COLOR, linewidth=1, zorder=1)

    n_datasets = len(datasets)
    group_width = 0.6
    offsets = np.linspace(-group_width / 2, group_width / 2, n_datasets) if n_datasets > 1 else [0.0]

    for offset, dataset in zip(offsets, datasets):
        summary = data[dataset]["summary"]
        means = [summary[c]["mean"] for c in coeffs]
        lo = [summary[c]["mean"] - summary[c]["ci_low"] for c in coeffs]
        hi = [summary[c]["ci_high"] - summary[c]["mean"] for c in coeffs]
        x = np.arange(len(coeffs)) + offset

        ax.errorbar(x, means, yerr=[lo, hi], fmt="o", markersize=8, capsize=5,
                     color=COLORS[dataset], label=dataset, linewidth=2)

    ax.set_xticks(np.arange(len(coeffs)))
    ax.set_xticklabels(coeff_labels)
    ax.set_ylabel("slope (nats/token per unit step position)")
    ax.set_title("Per-example OLS slopes with bootstrap 95% CI")
    ax.grid(True, axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="best", frameon=False, title="dataset")

    fig.tight_layout()
    fig.savefig(out_dir / "beta_ci.png", dpi=150)
    plt.close(fig)


def plot_beta_scatter(data: dict, datasets: list[str], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))

    all_vals = []
    for dataset in datasets:
        slopes_df = data[dataset]["slopes_df"]
        ax.scatter(slopes_df["beta_M"], slopes_df["beta_T"], color=COLORS[dataset],
                    alpha=0.6, s=30, edgecolors="none", label=dataset)
        all_vals.extend(slopes_df["beta_M"].dropna().tolist())
        all_vals.extend(slopes_df["beta_T"].dropna().tolist())

    if all_vals:
        lo, hi = min(all_vals), max(all_vals)
        pad = 0.05 * (hi - lo) if hi > lo else 1.0
        line_range = [lo - pad, hi + pad]
        ax.plot(line_range, line_range, color=ZERO_LINE_COLOR, linewidth=1,
                 linestyle="--", zorder=1, label=r"$y = x$")
        ax.set_xlim(line_range)
        ax.set_ylim(line_range)

    ax.axhline(0, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.axvline(0, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_xlabel(r"$\beta_M$ (marginal slope, per example)")
    ax.set_ylabel(r"$\beta_T$ (total slope, per example)")
    ax.set_title(r"Per-example $\beta_M$ vs $\beta_T$")
    ax.set_aspect("equal", adjustable="box")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="best", frameon=False)

    fig.tight_layout()
    fig.savefig(out_dir / "beta_scatter.png", dpi=150)
    plt.close(fig)


def plot_redundancy_trend(data: dict, datasets: list[str], out_dir: Path, n_bins: int = N_BINS) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    for dataset in datasets:
        step_df = data[dataset]["step_df"]
        r_bins = bin_stats(step_df, "R_s", n_bins)
        ax.plot(r_bins["s_hat_center"], r_bins["mean"], color=COLORS[dataset], linewidth=2,
                 marker="o", markersize=5, label=dataset)
        ax.fill_between(r_bins["s_hat_center"], r_bins["mean"] - r_bins["se"],
                         r_bins["mean"] + r_bins["se"], color=COLORS[dataset], alpha=0.2)

    ax.set_xlabel(r"step position (0 = start, 1 = end)")
    ax.set_ylabel("mean $R_s$, redundancy ratio (0–1) (± 1 SE)")
    ax.set_title(r"Redundancy ratio over trajectory position")
    ax.set_ylim(0, 1)
    ax.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False, ncol=len(datasets), title="dataset")

    fig.tight_layout()
    fig.savefig(out_dir / "redundancy_trend.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", nargs="+", default=["mathvista", "hallusionbench"],
        choices=sorted(DATASET_PRESETS),
        help="Benchmarks to load and plot together. Default: mathvista hallusionbench.",
    )
    parser.add_argument(
        "--model", required=True, choices=sorted(MODEL_PRESETS),
        help="Which model's results to plot.",
    )
    parser.add_argument(
        "--obs-dir", default=None,
        help="Folder holding the per-dataset run folders. "
             "Default: experiment_1/observations/{MODEL_DIR for --model}.",
    )
    parser.add_argument(
        "--n-bins", type=int, default=N_BINS,
        help=f"Number of equal-width step-position bins for per_step_trends.png and "
             f"redundancy_trend.png. Default: {N_BINS}.",
    )
    parser.add_argument("--out-dir", default=None, help="Default: experiment_1/plots/{model}.")
    args = parser.parse_args()
    if args.n_bins < 1:
        parser.error("--n-bins must be >= 1")

    out_dir = resolve_path(args.out_dir or f"experiment_1/plots/{args.model}")
    out_dir.mkdir(parents=True, exist_ok=True)

    obs_dir = resolve_path(args.obs_dir or f"experiment_1/observations/{MODEL_DIRS[args.model]}")

    data = {}
    for dataset in args.datasets:
        try:
            data[dataset] = load_dataset(dataset, obs_dir)
        except FileNotFoundError as e:
            raise SystemExit(f"Missing decomposition output for {dataset!r}: {e}")

    plot_per_step_trends(data, args.datasets, out_dir, args.n_bins)
    plot_beta_ci(data, args.datasets, out_dir)
    plot_beta_scatter(data, args.datasets, out_dir)
    plot_redundancy_trend(data, args.datasets, out_dir, args.n_bins)

    print(f"Wrote 4 plots to {out_dir}\n")
    for dataset in args.datasets:
        print(f"[{dataset}] interpretation: {data[dataset]['summary']['interpretation']}")


if __name__ == "__main__":
    main()
