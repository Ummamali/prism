"""Compute the marginal/total dependence decomposition (proposal §5.1) from
the per-example four-condition scores produced by inference.py, and fit the
headline test (proposal §5.2): slopes beta_M, beta_T of M_s and T_s on
normalised step index s_hat = s/S, and Delta-beta = beta_T - beta_M.

Per-step quantities (all length-normalised log-likelihood differences, in
nats/token, following the proposal exactly):
    M_s = ell_1 - ell_2            controlled direct effect ("marginal")
    T_s = ell_1 - ell_4            total visual dependence
    C_s = T_s - M_s = ell_2 - ell_4   prefix-carried / indirect dependence
    S_s = ell_3 - ell_4            visual sufficiency
    R_s = C_s / max(T_s, epsilon)  redundancy ratio, clipped to [0, 1]

Aggregation: per-example (per-trajectory) OLS slope of M_s (resp. T_s) on
s_hat, then the mean slope across examples is reported as beta_M (resp.
beta_T), with a percentile bootstrap CI over examples. This is a simple
Fama-MacBeth-style estimator rather than the proposal's full mixed-effects
model with random intercepts by item/model (§5.2) -- reasonable for a
~75-example single-model pilot; flagged here rather than silently
presented as the full analysis.

Usage:
    python experiment_1/decomposition.py [--config experiment_1/config.yaml]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from common import load_config, load_jsonl, resolve_path


def compute_step_measures(step: dict, epsilon: float) -> dict:
    ell_1, ell_2, ell_3, ell_4 = step["ell_1"], step["ell_2"], step["ell_3"], step["ell_4"]
    M = ell_1 - ell_2
    T = ell_1 - ell_4
    C = T - M
    S = ell_3 - ell_4
    R = float(np.clip(C / max(T, epsilon), 0.0, 1.0)) if T > 0 else float("nan")
    return {"M_s": M, "T_s": T, "C_s": C, "S_s": S, "R_s": R}


def ols_slope(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or np.allclose(x, x[0]):
        return None
    slope, _intercept = np.polyfit(x, y, 1)
    return float(slope)


def bootstrap_ci(values: np.ndarray, iters: int, seed: int, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    boot_means = np.empty(iters)
    n = len(values)
    for i in range(iters):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[i] = sample.mean()
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "mean": float(values.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n": int(n),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_cfg = cfg["output"]
    analysis_cfg = cfg["analysis"]
    epsilon = float(analysis_cfg["epsilon"])

    raw_dir = resolve_path(out_cfg["raw_dir"])
    agg_dir = resolve_path(out_cfg["aggregate_dir"])
    agg_dir.mkdir(parents=True, exist_ok=True)

    example_files = sorted(raw_dir.glob("*.json"))
    if not example_files:
        raise SystemExit(f"No per-example results found in {raw_dir}. Run inference.py first.")

    all_step_rows = []
    per_example_slopes = []

    for path in example_files:
        with open(path, "r", encoding="utf-8") as f:
            ex = json.load(f)

        S = ex["n_steps"]
        if S == 0:
            continue

        step_rows = []
        for step in ex["steps"]:
            measures = compute_step_measures(step, epsilon)
            row = {
                "pid": ex["pid"],
                "step_index": step["step_index"],
                "s_hat": step["step_index"] / S,
                **measures,
            }
            step_rows.append(row)
            all_step_rows.append(row)

        df = pd.DataFrame(step_rows)
        beta_M_i = ols_slope(df["s_hat"].to_numpy(), df["M_s"].to_numpy())
        beta_T_i = ols_slope(df["s_hat"].to_numpy(), df["T_s"].to_numpy())
        delta_beta_i = (
            (beta_T_i - beta_M_i) if (beta_M_i is not None and beta_T_i is not None) else None
        )

        per_example_slopes.append(
            {
                "pid": ex["pid"],
                "n_steps": S,
                "beta_M": beta_M_i,
                "beta_T": beta_T_i,
                "delta_beta": delta_beta_i,
                "mean_R_s": float(df["R_s"].mean(skipna=True)),
            }
        )

    step_df = pd.DataFrame(all_step_rows)
    slopes_df = pd.DataFrame(per_example_slopes)

    step_df.to_csv(agg_dir / "per_step_measures.csv", index=False)
    slopes_df.to_csv(agg_dir / "per_example_slopes.csv", index=False)

    boot_iters = int(analysis_cfg["bootstrap_iters"])
    boot_seed = int(analysis_cfg["bootstrap_seed"])

    beta_M_stats = bootstrap_ci(slopes_df["beta_M"].dropna().to_numpy(), boot_iters, boot_seed)
    beta_T_stats = bootstrap_ci(slopes_df["beta_T"].dropna().to_numpy(), boot_iters, boot_seed + 1)
    delta_beta_stats = bootstrap_ci(
        slopes_df["delta_beta"].dropna().to_numpy(), boot_iters, boot_seed + 2
    )

    # Secondary check: pooled OLS ignoring item clustering (proposal's
    # headline test is the mixed-effects/per-item version above; this is
    # a fast sanity companion, not a substitute).
    pooled_beta_M = ols_slope(step_df["s_hat"].to_numpy(), step_df["M_s"].to_numpy())
    pooled_beta_T = ols_slope(step_df["s_hat"].to_numpy(), step_df["T_s"].to_numpy())

    summary = {
        "n_examples": int(slopes_df.shape[0]),
        "n_examples_with_valid_slope": int(slopes_df["beta_M"].notna().sum()),
        "beta_M": beta_M_stats,
        "beta_T": beta_T_stats,
        "delta_beta": delta_beta_stats,
        "pooled_ols_beta_M": pooled_beta_M,
        "pooled_ols_beta_T": pooled_beta_T,
        "mean_redundancy_ratio_R_s": float(step_df["R_s"].mean(skipna=True)),
        "interpretation": interpret(beta_M_stats, beta_T_stats, delta_beta_stats),
    }

    with open(agg_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nWrote per-step measures, per-example slopes, and summary to {agg_dir}")


def interpret(beta_M_stats: dict, beta_T_stats: dict, delta_beta_stats: dict) -> str:
    """Proposal §5.2's three pre-registered outcome buckets, applied
    mechanically to the point estimates. Read the bootstrap CIs before
    trusting this label -- it does not itself test significance.
    """
    bm, bt, db = beta_M_stats["mean"], beta_T_stats["mean"], delta_beta_stats["mean"]
    if np.isnan(bm) or np.isnan(bt):
        return "insufficient data"
    if bm < 0 and bt < 0 and abs(db) < 0.5 * abs(bm):
        return "decay is real: beta_M < 0 and beta_T < 0 with similar magnitude"
    if bm < 0 and abs(bt) < abs(bm):
        return "decay is substantially a redundancy artifact: beta_M < 0, beta_T ~= 0"
    return "mixed / regime-dependent result -- inspect per-example slopes and R_s trajectories"


if __name__ == "__main__":
    main()
