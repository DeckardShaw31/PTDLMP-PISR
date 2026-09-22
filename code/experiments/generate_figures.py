import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np

# Ensure code root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.amazon_adapter import load_official_amazon_dataset
from src.routing.baseline import build_nearest_neighbor_baseline
from src.routing.schedule import propagate_schedule
from src.routing.actionability import compute_actionability_scores, rank_candidates, select_intervention_set
from src.routing.sfr import selective_forward_relocation

def create_policy_comparison_figure(benchmark_json_path: str, out_path: str):
    """
    Creates Figure 2: Quantitative policy comparison dynamically loaded from benchmark_summary.json.
    Plots Mean Total Tardiness Reduction (%) and Mean Accepted Relocations.
    """
    if not os.path.exists(benchmark_json_path):
        raise FileNotFoundError(f"Benchmark summary file '{benchmark_json_path}' not found. Run benchmark first.")

    with open(benchmark_json_path, "r", encoding="utf-8") as f:
        summary_data = json.load(f)

    p_summary = summary_data.get("policy_summary", [])
    if not p_summary:
        raise ValueError("No policy_summary found in benchmark summary JSON.")

    policy_map = {item["policy"]: item for item in p_summary}
    # Display order
    ordered_keys = ["Random", "Deadline", "RB", "Slack", "AB", "RA"]
    display_names = ["Random", "Deadline", "RB (Risk)", "Slack", "AB (Action.)", "RA (Proposed)"]
    
    tt_reduction = [policy_map[k]["tt_red_pct"] for k in ordered_keys]
    accepted_moves = [policy_map[k]["accepted_moves"] for k in ordered_keys]
    colors = ["#95a5a6", "#bdc3c7", "#e67e22", "#3498db", "#9b59b6", "#27ae60"]

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # Bar chart 1: % TT Reduction
    bars1 = ax1.bar(display_names, tt_reduction, color=colors, width=0.55, edgecolor="black", linewidth=0.8)
    ax1.set_ylabel("Total Tardiness Reduction (%)", fontsize=12, fontweight="bold")
    ax1.set_title("A. Mean Tardiness Reduction by Targeting Policy\n(Held-out Amazon Routes, 288 Factorial Runs)", fontsize=12, fontweight="bold", pad=12)
    max_tt = max(tt_reduction) if tt_reduction else 15.0
    ax1.set_ylim(0, max_tt * 1.3)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f"{yval:.1f}%", ha='center', va='bottom', fontsize=10, fontweight="bold")

    # Bar chart 2: Accepted Relocations
    bars2 = ax2.bar(display_names, accepted_moves, color=colors, width=0.55, edgecolor="black", linewidth=0.8)
    ax2.set_ylabel("Mean Relocations Accepted per Route", fontsize=12, fontweight="bold")
    ax2.set_title("B. Intervention Efficiency (Accepted Moves)\n(Budget B=5% to 30%, Delta=0% to 10%)", fontsize=12, fontweight="bold", pad=12)
    max_acc = max(accepted_moves) if accepted_moves else 20.0
    ax2.set_ylim(0, max_acc * 1.3)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f"{yval:.1f}", ha='center', va='bottom', fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[Figures] Saved dynamic policy comparison to '{out_path}'.")

def create_route_resequencing_map_figure(data_dir: str, out_path: str):
    """
    Creates Figure 3: Visual inspection of route before and after Selective Forward Relocation (SFR)
    using official Amazon Challenge data.
    """
    instances = load_official_amazon_dataset(data_dir, strict_mode=True, default_sla_hours=4.0)
    # Select first route with active tardiness
    target_r_id = None
    target_inst = None
    target_base = None
    target_sched = None

    for r_id, inst in instances.items():
        base = build_nearest_neighbor_baseline(inst)
        sched = propagate_schedule(inst, base)
        if sched.total_tardiness > 500.0:
            target_r_id = r_id
            target_inst = inst
            target_base = base
            target_sched = sched
            break

    if target_inst is None:
        target_r_id, target_inst = list(instances.items())[0]
        target_base = build_nearest_neighbor_baseline(target_inst)
        target_sched = propagate_schedule(target_inst, target_base)

    # Perform relocation at standard setting (B=0.20, delta=0.05)
    g_scores, _ = compute_actionability_scores(target_inst, target_base, delta=0.05)
    ranked = rank_candidates(target_inst, target_base, policy="RA", g_scores=g_scores)
    num_cust = len(target_base) - 1
    selected = select_intervention_set(ranked, budget_b=0.20, num_customers=num_cust)
    final_route, logs = selective_forward_relocation(target_inst, target_base, selected, delta=0.05)
    sched_final = propagate_schedule(target_inst, final_route)

    accepted_stops = {log.customer_id for log in logs if log.accepted}

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

    # Subplot 1: Baseline Route
    lats_b = [target_inst.stops[s].lat for s in target_base]
    lngs_b = [target_inst.stops[s].lng for s in target_base]
    ax1.plot(lngs_b, lats_b, color="#2980b9", linestyle="--", linewidth=1.2, zorder=1, alpha=0.7, label="Initial Dispatch Sequence")

    depot = target_inst.depot_id
    d_lat, d_lng = target_inst.stops[depot].lat, target_inst.stops[depot].lng
    ax1.scatter([d_lng], [d_lat], color="#e74c3c", s=180, marker="s", zorder=4, label="Station Depot")

    late_b_lngs = [target_inst.stops[s].lng for s in target_base if s != depot and sched_base_is_late(target_inst, target_sched, s)]
    late_b_lats = [target_inst.stops[s].lat for s in target_base if s != depot and sched_base_is_late(target_inst, target_sched, s)]
    ontime_b_lngs = [target_inst.stops[s].lng for s in target_base if s != depot and not sched_base_is_late(target_inst, target_sched, s)]
    ontime_b_lats = [target_inst.stops[s].lat for s in target_base if s != depot and not sched_base_is_late(target_inst, target_sched, s)]

    ax1.scatter(ontime_b_lngs, ontime_b_lats, color="#3498db", s=40, marker="o", zorder=2, alpha=0.8, label="On-Time Delivery")
    if late_b_lngs:
        ax1.scatter(late_b_lngs, late_b_lats, color="#c0392b", s=75, marker="^", zorder=3, label=f"Late Delivery (N={len(late_b_lngs)})")

    ax1.set_title(f"A. Initial Nearest Neighbor Route $R^0$ ({target_inst.station_code})\n"
                  f"Total Tardiness = {target_sched.total_tardiness:.1f} min | Late = {target_sched.total_lateness} | Dist = {target_sched.total_distance:.1f} km",
                  fontsize=11, fontweight="bold", pad=10)
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    ax1.legend(loc="best", frameon=True)

    # Subplot 2: Final Resequenced Route
    lats_f = [target_inst.stops[s].lat for s in final_route]
    lngs_f = [target_inst.stops[s].lng for s in final_route]
    ax2.plot(lngs_f, lats_f, color="#27ae60", linestyle="-", linewidth=1.4, zorder=1, alpha=0.8, label="Resequenced Route $R'$")
    ax2.scatter([d_lng], [d_lat], color="#e74c3c", s=180, marker="s", zorder=4, label="Station Depot")

    repositioned_lngs = [target_inst.stops[s].lng for s in accepted_stops]
    repositioned_lats = [target_inst.stops[s].lat for s in accepted_stops]

    ax2.scatter(ontime_b_lngs, ontime_b_lats, color="#2ecc71", s=40, marker="o", zorder=2, alpha=0.6, label="Regular Stop")
    if repositioned_lngs:
        ax2.scatter(repositioned_lngs, repositioned_lats, color="#8e44ad", s=90, marker="*", zorder=3, label=f"Relocated Stop (N={len(accepted_stops)})")

    d_tt_val = target_sched.total_tardiness - sched_final.total_tardiness
    tt_pct = (d_tt_val / target_sched.total_tardiness) * 100.0 if target_sched.total_tardiness > 0 else 0.0

    ax2.set_title(f"B. Resequenced Route $R'$ after PISR SFR\n"
                  f"Total Tardiness = {sched_final.total_tardiness:.1f} min (-{tt_pct:.1f}%) | Late = {sched_final.total_lateness} | Dist = {sched_final.total_distance:.1f} km",
                  fontsize=11, fontweight="bold", pad=10)
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    ax2.legend(loc="best", frameon=True)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[Figures] Saved official route map to '{out_path}'.")

def sched_base_is_late(inst, sched, s_id):
    if s_id not in sched.completion_times:
        return False
    prom = inst.stops[s_id].promised_time
    if prom is None:
        return False
    return sched.completion_times[s_id] > prom

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    raw_dir = os.path.join(base_dir, "data", "raw")
    bm_json = os.path.join(base_dir, "outputs", "benchmark_summary.json")
    figures_dir = os.path.abspath(os.path.join(base_dir, "..", "figures"))
    os.makedirs(figures_dir, exist_ok=True)

    create_policy_comparison_figure(bm_json, os.path.join(figures_dir, "fig2_policy_comparison.png"))
    create_route_resequencing_map_figure(raw_dir, os.path.join(figures_dir, "fig3_route_before_after.png"))
