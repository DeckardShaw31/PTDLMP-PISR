import os
import sys
import argparse
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure code root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Same policy order and colors as Figure 2 (Deadline darkened so its line stays visible)
ORDERED_KEYS = ["Random", "Deadline", "RB", "Slack", "AB", "RA"]
DISPLAY_NAMES = {"Random": "Random", "Deadline": "Deadline", "RB": "RB (Risk)", "Slack": "Slack",
                 "AB": "AB (Action.)", "RA": "RA (Proposed)"}
COLORS = {"Random": "#95a5a6", "Deadline": "#7f8c8d", "RB": "#e67e22", "Slack": "#3498db",
          "AB": "#9b59b6", "RA": "#27ae60"}
EPS_MIN = 1e-6


def _load_scenario_runs(scenario_csv_path: str) -> pd.DataFrame:
    if not os.path.exists(scenario_csv_path):
        raise FileNotFoundError(f"Scenario results '{scenario_csv_path}' not found. Run run_travel_time_scenarios.py first.")
    return pd.read_csv(scenario_csv_path)


def _cv_labels(cvs) -> List[str]:
    return [f"{cv:.0%}" for cv in cvs]


def plot_scenario_robustness(stats: pd.DataFrame, out_path: str, sample_desc: str):
    """
    M8 robustness figure from per-(policy, CV) statistics with columns mean, p05, p95, share_neg (percent).
    Panel A: mean Delta TT with 5-95% range per policy and CV (pooled over routes, cells and scenarios).
    Panel B: share of route-cell-scenario outcomes in which R' is worse than R0 (Delta TT < 0).
    """
    cvs = sorted(stats.index.get_level_values("cv").unique())
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    x = np.arange(len(cvs))
    offsets = np.linspace(-0.25, 0.25, len(ORDERED_KEYS))

    # Panel A: mean Delta TT with 5-95% range
    for off, key in zip(offsets, ORDERED_KEYS):
        s = stats.loc[key].reindex(cvs)
        yerr = np.vstack([s["mean"] - s["p05"], s["p95"] - s["mean"]])
        ax1.errorbar(x + off, s["mean"], yerr=yerr, fmt="o-", color=COLORS[key], capsize=3,
                     linewidth=1.4, markersize=5, label=DISPLAY_NAMES[key])
    ax1.axhline(0.0, color="black", linewidth=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(_cv_labels(cvs))
    ax1.set_xlabel("Travel-time coefficient of variation (CV)", fontsize=11)
    ax1.set_ylabel("Tardiness Reduction $\\Delta TT$ (min)", fontsize=12, fontweight="bold")
    ax1.set_title(f"A. Frozen-Plan Tardiness Reduction under Travel-Time Scenarios\n(Mean and 5-95% Range, {sample_desc})",
                  fontsize=12, fontweight="bold", pad=12)
    ax1.legend(loc="upper left", frameon=True, fontsize=9, ncol=2)

    # Panel B: share of outcomes where the intervention increased tardiness
    width = 0.8 / len(ORDERED_KEYS)
    for i, key in enumerate(ORDERED_KEYS):
        s = stats.loc[key].reindex(cvs)
        bars = ax2.bar(x + (i - (len(ORDERED_KEYS) - 1) / 2) * width, s["share_neg"], width=width,
                       color=COLORS[key], edgecolor="black", linewidth=0.6, label=DISPLAY_NAMES[key])
        for bar in bars:
            yval = bar.get_height()
            if yval >= 1.0:
                ax2.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.3, f"{yval:.0f}", ha='center', va='bottom', fontsize=7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(_cv_labels(cvs))
    ax2.set_xlabel("Travel-time coefficient of variation (CV)", fontsize=11)
    ax2.set_ylabel("Outcomes with $\\Delta TT < 0$ (%)", fontsize=12, fontweight="bold")
    ax2.set_title("B. Share of Outcomes where R' Is Worse than R0\n(Planned on Mean Travel Times, Re-evaluated per Scenario)",
                  fontsize=12, fontweight="bold", pad=12)
    ax2.set_ylim(0, max(5.0, stats["share_neg"].max() * 1.25))
    ax2.legend(loc="upper left", frameon=True, fontsize=9, ncol=2)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[Figures] Saved scenario robustness figure to '{out_path}'.")


def plot_vrisk(v_by_cv: Dict[float, np.ndarray], route_means: pd.DataFrame, out_path: str, cell_desc: str):
    """
    M8 RA-vs-AB figure: V_risk = Delta TT(RA) - Delta TT(AB) on identical scenarios.
    v_by_cv: CV -> all route x cell x scenario V_risk values; route_means: route x CV mean V_risk.
    Panel A: distribution of V_risk per CV. Panel B: route-level mean V_risk (lines for a few routes,
    otherwise its distribution across routes).
    """
    cvs = sorted(v_by_cv)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    box_style = dict(showfliers=False, widths=0.5, medianprops=dict(color="black", linewidth=1.4),
                     boxprops=dict(facecolor="#d5f5e3", edgecolor="#27ae60"), patch_artist=True)

    # Panel A: V_risk distribution per CV
    data = [v_by_cv[cv] for cv in cvs]
    ax1.boxplot(data, whis=(5, 95), **box_style)
    means = [d.mean() for d in data]
    ax1.plot(np.arange(1, len(cvs) + 1), means, "D", color="#27ae60", markersize=7, label="Mean $V_{risk}$")
    for i, (d, m) in enumerate(zip(data, means), start=1):
        ra_better = (d > EPS_MIN).mean() * 100.0
        ab_better = (d < -EPS_MIN).mean() * 100.0
        ax1.annotate(f"mean {m:+.1f}\nRA {ra_better:.0f}% / AB {ab_better:.0f}%", xy=(i, np.percentile(d, 95)),
                     xytext=(0, 6), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    ax1.axhline(0.0, color="black", linewidth=0.8)
    ax1.set_xticks(np.arange(1, len(cvs) + 1))
    ax1.set_xticklabels(_cv_labels(cvs))
    ax1.set_xlabel("Travel-time coefficient of variation (CV)", fontsize=11)
    ax1.set_ylabel("$V_{risk} = \\Delta TT(RA) - \\Delta TT(AB)$ (min)", fontsize=12, fontweight="bold")
    ax1.set_title("A. Incremental Value of Predicted Risk (RA vs AB)\n(Box = IQR, Whiskers = 5-95%, Positive Favors RA)",
                  fontsize=12, fontweight="bold", pad=12)
    ax1.margins(y=0.15)
    ax1.legend(loc="lower left", frameon=True, fontsize=9)

    # Panel B: route-level mean V_risk
    n_routes = len(route_means)
    if n_routes <= 5:
        route_colors = ["#2980b9", "#c0392b", "#8e44ad", "#d35400", "#16a085"]
        for color, (route_id, row) in zip(route_colors, route_means.iterrows()):
            short_id = str(route_id).replace("RouteID_", "")[:8]
            ax2.plot(range(len(cvs)), row.reindex(cvs).to_numpy(), "o-", color=color, linewidth=1.6, markersize=6, label=f"Route {short_id}")
        ax2.plot(range(len(cvs)), [v_by_cv[cv].mean() for cv in cvs], "D--", color="black",
                 linewidth=1.2, markersize=6, label="All routes")
        ax2.set_xticks(range(len(cvs)))
        ax2.legend(loc="center right", bbox_to_anchor=(0.98, 0.3), frameon=True, fontsize=9)
        title_b = "B. Mean $V_{risk}$ by Held-Out Route"
    else:
        cols = [route_means[cv].to_numpy() for cv in cvs]
        ax2.boxplot(cols, whis=(5, 95), positions=range(len(cvs)), **box_style)
        for i, col in enumerate(cols):
            ra = (col > EPS_MIN).mean() * 100.0
            ab = (col < -EPS_MIN).mean() * 100.0
            ax2.annotate(f"RA {ra:.0f}% / AB {ab:.0f}% of routes", xy=(i, np.percentile(col, 95)),
                         xytext=(0, 6), textcoords="offset points", ha='center', va='bottom', fontsize=8)
        ax2.margins(y=0.15)
        title_b = f"B. Route-Level Mean $V_{{risk}}$ across {n_routes:,} Held-Out Routes"
    ax2.axhline(0.0, color="black", linewidth=0.8)
    ax2.set_xticks(range(len(cvs)))
    ax2.set_xticklabels(_cv_labels(cvs))
    ax2.set_xlabel("Travel-time coefficient of variation (CV)", fontsize=11)
    ax2.set_ylabel("Mean $V_{risk}$ per route (min)", fontsize=12, fontweight="bold")
    ax2.set_title(f"{title_b}\n({cell_desc})", fontsize=12, fontweight="bold", pad=12)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[Figures] Saved V_risk figure to '{out_path}'.")


def create_scenario_robustness_figure(scenario_csv_path: str, out_path: str):
    """Pilot version: statistics computed from the scenario-level runs CSV."""
    runs = _load_scenario_runs(scenario_csv_path)
    grouped = runs.groupby(["policy", "cv"])["delta_tt"]
    stats = pd.DataFrame({
        "mean": grouped.mean(),
        "p05": grouped.quantile(0.05),
        "p95": grouped.quantile(0.95),
        "share_neg": grouped.apply(lambda s: float((s < -EPS_MIN).mean()) * 100.0),
    })
    n_routes = runs["route_id"].nunique()
    n_scen = runs["scenario_id"].nunique()
    plot_scenario_robustness(stats, out_path, f"{n_routes} Routes x 16 Cells x {n_scen} Scenarios")


def create_vrisk_figure(scenario_csv_path: str, out_path: str):
    """Pilot version: V_risk computed from the scenario-level runs CSV."""
    runs = _load_scenario_runs(scenario_csv_path)
    keys = ["route_id", "cv", "scenario_id", "budget", "delta"]
    wide = runs[runs["policy"].isin(["RA", "AB"])].pivot_table(index=keys, columns="policy", values="delta_tt").reset_index()
    wide["v_risk"] = wide["RA"] - wide["AB"]
    v_by_cv = {cv: g["v_risk"].to_numpy() for cv, g in wide.groupby("cv")}
    route_means = wide.groupby(["route_id", "cv"])["v_risk"].mean().unstack("cv")
    plot_vrisk(v_by_cv, route_means, out_path, f"Averaged over 16 (B, $\\delta$) Cells and {wide['scenario_id'].nunique()} Scenarios")


def create_full_cohort_figures(output_dir: str, figures_dir: str):
    """Full-cohort versions from code/outputs/full_cohort (scenario_policy_summary.csv and scenario_delta_arrays.npz)."""
    pooled = pd.read_csv(os.path.join(output_dir, "scenario_policy_summary.csv"))
    stats = pooled.rename(columns={"mean_delta_tt": "mean", "p05_delta_tt": "p05", "p95_delta_tt": "p95"})
    stats["share_neg"] = stats["share_delta_tt_neg"] * 100.0
    stats = stats.set_index(["policy", "cv"])[["mean", "p05", "p95", "share_neg"]]

    arr = np.load(os.path.join(output_dir, "scenario_delta_arrays.npz"))
    keys = [tuple(k.split("|")) for k in arr["cell_keys"]]
    ra = [i for i, k in enumerate(keys) if k[0] == "RA"]
    ab = [keys.index(("AB",) + keys[i][1:]) for i in ra]
    v = arr["delta_tt"][..., ra] - arr["delta_tt"][..., ab]  # (routes, scenarios, CV, cells)
    cvs = [float(c) for c in arr["cv_levels"]]
    n_routes, n_scen = v.shape[0], v.shape[1]
    v_by_cv = {cv: v[:, :, c, :].reshape(-1) for c, cv in enumerate(cvs)}
    route_means = pd.DataFrame({cv: v[:, :, c, :].mean(axis=(1, 2)) for c, cv in enumerate(cvs)}, index=arr["route_ids"])

    os.makedirs(figures_dir, exist_ok=True)
    plot_scenario_robustness(stats, os.path.join(figures_dir, "fig_m8_scenario_robustness.png"),
                             f"{n_routes:,} Routes x 16 Cells x {n_scen} Scenarios")
    plot_vrisk(v_by_cv, route_means, os.path.join(figures_dir, "fig_m8_vrisk_ra_vs_ab.png"),
               f"Averaged over 16 (B, $\\delta$) Cells and {n_scen} Scenarios")


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-cohort", action="store_true", help="Plot code/outputs/full_cohort into figures/full_cohort.")
    args = parser.parse_args()
    figures_dir = os.path.abspath(os.path.join(base_dir, "..", "figures"))

    if args.full_cohort:
        create_full_cohort_figures(os.path.join(base_dir, "outputs", "full_cohort"), os.path.join(figures_dir, "full_cohort"))
    else:
        scenario_csv = os.path.join(base_dir, "outputs", "scenario_level_runs.csv.gz")
        os.makedirs(figures_dir, exist_ok=True)
        create_scenario_robustness_figure(scenario_csv, os.path.join(figures_dir, "fig_m8_scenario_robustness.png"))
        create_vrisk_figure(scenario_csv, os.path.join(figures_dir, "fig_m8_vrisk_ra_vs_ab.png"))
