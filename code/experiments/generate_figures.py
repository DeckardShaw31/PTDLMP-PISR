"""
generate_figures.py
Generates publication figures from frozen canonical experimental results:
- Figure 2: Policy performance and operational trade-offs (routing_benchmark_runs.csv)
- Figure 3: Prediction discrimination and empirical calibration (rq1_stop_level_predictions.csv)
- Figure 4: Budget-distance sensitivity heatmaps (routing_benchmark_runs.csv)
- Supplementary Figure S1: Route completion-time trajectory (held_out_route_trace.csv)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import precision_recall_curve, auc, roc_curve, roc_auc_score

# Set overall publication style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 12

def generate_figure2_policy_comparison(runs_csv_path: str, out_path: str):
    """
    Figure 2: Policy performance and trade-offs.
    Panel A: Route-level policy comparison (one dot per held-out route, overlaid 3-route mean).
    Panel B: Operational trade-off (Delta D % vs Delta TT %, bubble size = accepted moves).
    """
    df = pd.read_csv(runs_csv_path)
    
    # 1. Route-level means (averaging the 16 B, delta settings for each route)
    route_means = df.groupby(['policy', 'route_id'])[['tt_red_pct', 'delta_tt', 'dist_inc_pct', 'accepted_moves', 'delta_nl']].mean().reset_index()
    
    # 2. Overall 3-route means
    grand_means = route_means.groupby('policy')[['tt_red_pct', 'delta_tt', 'dist_inc_pct', 'accepted_moves', 'delta_nl']].mean().reset_index()
    
    policy_order = ['RA', 'AB', 'Slack', 'RB', 'Deadline', 'Random']
    policy_labels = ['RA (Proposed)', 'AB (Action.)', 'Slack', 'RB (Risk)', 'Deadline', 'Random']
    
    colors = {
        'RA': '#1b9e77',        # Emerald Green
        'AB': '#2980b9',        # Deep Blue
        'Slack': '#8e44ad',     # Purple
        'RB': '#d95f02',        # Dark Orange
        'Deadline': '#e7298a',  # Magenta / Terracotta
        'Random': '#7570b3'     # Slate Purple / Gray
    }
    
    routes = sorted(route_means['route_id'].unique())
    route_markers = {'RouteID_693060a6-88bb-4324-9e9c-925d5240263c': ('o', 'Route 1 (6930)'),
                     'RouteID_7f5d87f0-c39f-434f-bf3f-b159ef321909': ('s', 'Route 2 (7f5d)'),
                     'RouteID_9475872b-287f-4c2c-8e29-887766a4e090': ('^', 'Route 3 (9475)')}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    fig.patch.set_facecolor('white')

    # ------------------ PANEL A ------------------
    ax1.set_facecolor('#fafafa')
    ax1.grid(True, linestyle='--', alpha=0.5, color='#ccc', zorder=0)
    
    x_positions = np.arange(len(policy_order))
    
    # Plot light background guide bars
    for x in x_positions:
        ax1.axvline(x, color='#e0e0e0', linestyle=':', linewidth=1.0, zorder=1)

    # Plot route points and grand mean
    for i, p in enumerate(policy_order):
        p_sub = route_means[route_means['policy'] == p]
        gm_row = grand_means[grand_means['policy'] == p].iloc[0]
        gm_tt = gm_row['tt_red_pct']
        
        # Plot individual route points
        jitter = [-0.15, 0.0, 0.15]
        for j, r_id in enumerate(routes):
            r_row = p_sub[p_sub['route_id'] == r_id]
            if not r_row.empty:
                val = r_row['tt_red_pct'].values[0]
                marker, lbl = route_markers[r_id]
                ax1.scatter(i + jitter[j], val, marker=marker, s=85, color=colors[p], edgecolors='black', linewidth=0.8, alpha=0.85, zorder=4)
        
        # Plot 3-route grand mean marker (diamond with horizontal bar)
        ax1.scatter(i, gm_tt, marker='D', s=130, color=colors[p], edgecolors='black', linewidth=1.5, zorder=5)
        ax1.hlines(gm_tt, i - 0.28, i + 0.28, colors=colors[p], linestyles='-', linewidth=2.5, zorder=5)
        
        # Annotate mean value
        offset = 0.55 if p in ['RA', 'AB'] else 0.45
        ax1.text(i, gm_tt + offset, f"{gm_tt:.1f}%", ha='center', va='bottom', fontsize=9.5, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=colors[p], alpha=0.9, linewidth=1.0), zorder=6)

    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(policy_labels, fontsize=9.5, fontweight='bold')
    ax1.set_ylabel(r"Mean Total Tardiness Reduction $\Delta TT$ (%)", fontsize=10.5, fontweight='bold')
    ax1.set_title(r"A. Route-Level Policy Comparison" "\n" r"(Each point averages 16 prespecified $B, \delta$ settings; $N=3$ routes)", fontsize=11, fontweight='bold', pad=12)
    ax1.set_ylim(0, 15.0)

    # Panel A Legend
    legend_elements_a = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#555', markeredgecolor='black', markersize=8, label='Route 1 (6930)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#555', markeredgecolor='black', markersize=8, label='Route 2 (7f5d)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#555', markeredgecolor='black', markersize=8, label='Route 3 (9475)'),
        Line2D([0], [0], marker='D', color='black', markerfacecolor='#555', markeredgecolor='black', markersize=9, linewidth=2, label='Three-Route Mean')
    ]
    ax1.legend(handles=legend_elements_a, loc='upper right', frameon=True, facecolor='white', framealpha=0.95, edgecolor='#ccc', fontsize=8.5)

    # ------------------ PANEL B ------------------
    ax2.set_facecolor('#fafafa')
    ax2.grid(True, linestyle='--', alpha=0.5, color='#ccc', zorder=0)

    # Coordinated label offsets to avoid overlapping
    label_configs = {
        'RA': (-0.18, 0.70),    # top-left of RA bubble
        'AB': (0.05, -0.85),     # bottom-right of AB bubble
        'Slack': (0.04, 0.45),   # top-right of Slack bubble
        'RB': (0.05, 0.65),      # top-right of RB bubble
        'Deadline': (0.04, 0.35),# right of Deadline bubble
        'Random': (-0.22, -0.75) # bottom-left of Random bubble
    }

    for p in policy_order:
        gm = grand_means[grand_means['policy'] == p].iloc[0]
        x_dist = gm['dist_inc_pct']
        y_tt = gm['tt_red_pct']
        moves = gm['accepted_moves']
        size = moves * 30  # Bubble size scaling

        ax2.scatter(x_dist, y_tt, s=size, color=colors[p], edgecolors='black', linewidth=1.2, alpha=0.75, zorder=4)

        # Label directly beside point
        dx, dy = label_configs.get(p, (0.03, 0.3))
        ax2.annotate(f"{p}\n(+{y_tt:.1f}%, {moves:.1f} mv)", xy=(x_dist, y_tt), xytext=(x_dist + dx, y_tt + dy),
                     fontsize=9, fontweight='bold', color='#222', zorder=6,
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=colors[p], alpha=0.9, linewidth=1.0),
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1', color='#555', lw=0.9))

    ax2.set_xlabel(r"Mean Distance Increase $\Delta D$ (%)", fontsize=10.5, fontweight='bold')
    ax2.set_ylabel(r"Mean Total Tardiness Reduction $\Delta TT$ (%)", fontsize=10.5, fontweight='bold')
    ax2.set_title(r"B. Operational Trade-Off: Tardiness vs. Distance" "\n" r"(Bubble size $\propto$ Mean Accepted Moves per route)", fontsize=11, fontweight='bold', pad=12)
    ax2.set_xlim(1.2, 2.5)
    ax2.set_ylim(0, 15.0)

    # Bubble size legend
    bubble_moves = [5, 10, 15, 20]
    legend_bubbles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#bbb', markeredgecolor='black', markersize=np.sqrt(m * 30), label=f"{m} moves")
        for m in bubble_moves
    ]
    ax2.legend(handles=legend_bubbles, loc='lower left', title="Accepted Moves", frameon=True, facecolor='white', framealpha=0.95, edgecolor='#ccc', fontsize=8.5, title_fontsize=8.5)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[Figure 2] Successfully generated and saved to: {out_path}")


def generate_figure3_prediction_calibration(pred_csv_path: str, out_path: str):
    """
    Figure 3: Prediction and empirical calibration on held-out test stops (N=534, H=4h).
    Panel A: Reliability diagram (10 uniform bins matching ECE definition, with bin counts).
    Panel B: Precision-Recall curves (with test prevalence baseline 0.539, PR-AUC, and ROC inset).
    """
    df = pd.read_csv(pred_csv_path)
    test = df[df['split'] == 'test']
    
    y_true = test['y_true'].values
    p_log = test['p_logistic_calibrated'].values
    p_rf = test['p_random_forest'].values
    n_test = len(y_true)
    prevalence = np.mean(y_true)

    # 1. ECE and Reliability Diagram Calculation (10 uniform bins on [0, 1])
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    
    def get_bin_stats(y, p):
        confs, accs, counts, centers = [], [], [], []
        ece = 0.0
        for i in range(n_bins):
            low, high = bin_edges[i], bin_edges[i+1]
            mask = (p >= low) & (p <= high) if i == n_bins - 1 else (p >= low) & (p < high)
            n_k = np.sum(mask)
            counts.append(n_k)
            centers.append((low + high) / 2.0)
            if n_k > 0:
                acc_k = np.mean(y[mask])
                conf_k = np.mean(p[mask])
                confs.append(conf_k)
                accs.append(acc_k)
                ece += (n_k / len(y)) * np.abs(acc_k - conf_k)
            else:
                confs.append(np.nan)
                accs.append(np.nan)
        return np.array(confs), np.array(accs), np.array(counts), np.array(centers), ece

    log_confs, log_accs, log_counts, bin_centers, ece_log = get_bin_stats(y_true, p_log)
    rf_confs, rf_accs, rf_counts, _, ece_rf = get_bin_stats(y_true, p_rf)

    # PR Curve & AUC
    p_log_pr, r_log_pr, _ = precision_recall_curve(y_true, p_log)
    pr_auc_log = auc(r_log_pr, p_log_pr)
    p_rf_pr, r_rf_pr, _ = precision_recall_curve(y_true, p_rf)
    pr_auc_rf = auc(r_rf_pr, p_rf_pr)

    # ROC Curve & AUC
    fpr_log, tpr_log, _ = roc_curve(y_true, p_log)
    roc_auc_log = roc_auc_score(y_true, p_log)
    fpr_rf, tpr_rf, _ = roc_curve(y_true, p_rf)
    roc_auc_rf = roc_auc_score(y_true, p_rf)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    fig.patch.set_facecolor('white')

    # ------------------ PANEL A: RELIABILITY DIAGRAM ------------------
    ax1.set_facecolor('#fafafa')
    ax1.grid(True, linestyle='--', alpha=0.5, color='#ccc', zorder=0)

    # Perfect calibration line
    ax1.plot([0, 1], [0, 1], linestyle='--', color='#7f8c8d', linewidth=1.5, label='Perfect Calibration ($y = x$)', zorder=2)

    # Logistic Regression curve
    mask_log = ~np.isnan(log_confs)
    ax1.plot(log_confs[mask_log], log_accs[mask_log], marker='o', markersize=8, linewidth=2.2, color='#1b9e77',
             label=f'Logistic Regression (Primary, ECE = {ece_log:.4f})', zorder=5)

    # Random Forest curve
    mask_rf = ~np.isnan(rf_confs)
    ax1.plot(rf_confs[mask_rf], rf_accs[mask_rf], marker='s', markersize=7, linewidth=1.8, linestyle='-.', color='#d95f02',
             label=f'Random Forest (Comparator, ECE = {ece_rf:.4f})', zorder=4)

    # Annotate bin sample sizes cleanly without clipping bottom
    for c, a, cnt in zip(log_confs[mask_log], log_accs[mask_log], log_counts[mask_log]):
        if cnt > 0:
            if c < 0.2:
                # low confidence bins: place text above to avoid bottom margin
                ax1.annotate(f"n={cnt}", (c, a), xytext=(c + 0.01, a + 0.06),
                             fontsize=8, fontweight='bold', color='#1b9e77',
                             bbox=dict(boxstyle='round,pad=0.15', facecolor='#e8f8f5', edgecolor='#1b9e77', alpha=0.9, lw=0.6),
                             arrowprops=dict(arrowstyle='->', color='#1b9e77', lw=0.7))
            else:
                ax1.annotate(f"n={cnt}", (c, a), xytext=(c + 0.02, a - 0.05),
                             fontsize=8, fontweight='bold', color='#1b9e77',
                             bbox=dict(boxstyle='round,pad=0.15', facecolor='#e8f8f5', edgecolor='#1b9e77', alpha=0.9, lw=0.6))

    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_xlabel(r"Mean Predicted Probability $\hat{p}$ (per bin)", fontsize=10.5, fontweight='bold')
    ax1.set_ylabel("Observed Lateness Frequency", fontsize=10.5, fontweight='bold')
    ax1.set_title(r"A. Empirical Reliability Diagram" "\n" r"(10 Prespecified Uniform Bins, Held-Out Test Cohort $N=534$, $H=4$h)", fontsize=11, fontweight='bold', pad=12)
    ax1.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.95, edgecolor='#ccc', fontsize=8.5)

    # ------------------ PANEL B: PRECISION-RECALL CURVE WITH ROC INSET ------------------
    ax2.set_facecolor('#fafafa')
    ax2.grid(True, linestyle='--', alpha=0.5, color='#ccc', zorder=0)

    # Test Prevalence baseline
    ax2.axhline(prevalence, linestyle=':', color='#c0392b', linewidth=1.5,
                label=f'Test Prevalence Baseline ({prevalence:.3f})', zorder=2)

    # Logistic Regression PR curve
    ax2.plot(r_log_pr, p_log_pr, color='#1b9e77', linewidth=2.2,
             label=f'Logistic Regression (PR-AUC = {pr_auc_log:.3f})', zorder=5)

    # Random Forest PR curve
    ax2.plot(r_rf_pr, p_rf_pr, color='#d95f02', linewidth=1.8, linestyle='-.',
             label=f'Random Forest (PR-AUC = {pr_auc_rf:.3f})', zorder=4)

    ax2.set_xlim(0.0, 1.02)
    ax2.set_ylim(0.40, 1.02)
    ax2.set_xlabel("Recall", fontsize=10.5, fontweight='bold')
    ax2.set_ylabel("Precision", fontsize=10.5, fontweight='bold')
    ax2.set_title(r"B. Precision–Recall Curve & ROC Inset" "\n" r"(Held-Out Test Cohort $N=534$, Threshold $H=4$h)", fontsize=11, fontweight='bold', pad=12)
    ax2.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.95, edgecolor='#ccc', fontsize=8.5)

    # INSET: ROC CURVES
    ax_inset = ax2.inset_axes([0.55, 0.52, 0.41, 0.42])
    ax_inset.set_facecolor('#f4f6f7')
    ax_inset.grid(True, linestyle=':', alpha=0.6, color='#aaa')
    ax_inset.plot([0, 1], [0, 1], linestyle='--', color='#7f8c8d', linewidth=1.0)
    ax_inset.plot(fpr_log, tpr_log, color='#1b9e77', linewidth=1.8, label=f'Log. (AUC={roc_auc_log:.3f})')
    ax_inset.plot(fpr_rf, tpr_rf, color='#d95f02', linewidth=1.5, linestyle='-.', label=f'RF (AUC={roc_auc_rf:.3f})')
    ax_inset.set_xlim(-0.02, 1.02)
    ax_inset.set_ylim(-0.02, 1.02)
    ax_inset.set_title("ROC Curves", fontsize=8.5, fontweight='bold', pad=3)
    ax_inset.set_xlabel("FPR", fontsize=7.5, labelpad=1)
    ax_inset.set_ylabel("TPR", fontsize=7.5, labelpad=1)
    ax_inset.tick_params(axis='both', which='major', labelsize=7)
    ax_inset.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9, edgecolor='#ccc', fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[Figure 3] Successfully generated and saved to: {out_path}")


def generate_figure4_budget_distance_sensitivity(runs_csv_path: str, out_path: str):
    """
    Figure 4: Budget-distance sensitivity heatmaps.
    Three 4x4 heatmaps:
    - Panel A: RA mean Delta TT reduction (%) (sequential colormap)
    - Panel B: RA minus RB percentage-point difference (diverging colormap centered at 0)
    - Panel C: RA minus AB percentage-point difference (diverging colormap centered at 0)
    Rows: B in {5%, 10%, 20%, 30%}, Cols: delta in {0%, 2%, 5%, 10%}.
    """
    df = pd.read_csv(runs_csv_path)
    
    budgets = [0.05, 0.10, 0.20, 0.30]
    deltas = [0.00, 0.02, 0.05, 0.10]
    
    budget_labels = ['5%', '10%', '20%', '30%']
    delta_labels = ['0%', '2%', '5%', '10%']

    # For each policy, compute 4x4 table of mean across 3 routes
    route_means = df.groupby(['policy', 'budget', 'delta', 'route_id'])['tt_red_pct'].mean().reset_index()
    pivots = {}
    for p in ['RA', 'RB', 'AB']:
        sub = route_means[route_means['policy'] == p]
        piv = sub.groupby(['budget', 'delta'])['tt_red_pct'].mean().unstack()
        pivots[p] = piv.loc[budgets, deltas].values

    mat_ra = pivots['RA']
    mat_diff_rb = pivots['RA'] - pivots['RB']
    mat_diff_ab = pivots['RA'] - pivots['AB']

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
    fig.patch.set_facecolor('white')

    # ------------------ PANEL A: RA MEAN DELTA TT % ------------------
    im1 = ax1.imshow(mat_ra, cmap='YlGnBu', vmin=4.0, vmax=20.0, aspect='auto')
    ax1.set_title(r"A. RA Mean Tardiness Reduction $\Delta TT$ (%)" "\n" r"(Higher = More Lateness Avoided)", fontsize=11, fontweight='bold', pad=12)
    ax1.set_xticks(np.arange(len(delta_labels)))
    ax1.set_yticks(np.arange(len(budget_labels)))
    ax1.set_xticklabels(delta_labels, fontweight='bold')
    ax1.set_yticklabels(budget_labels, fontweight='bold')
    ax1.set_xlabel(r"Distance Tolerance Bound $\delta$", fontsize=10, fontweight='bold')
    ax1.set_ylabel(r"Relocation Budget $B$", fontsize=10, fontweight='bold')

    for i in range(len(budgets)):
        for j in range(len(deltas)):
            val = mat_ra[i, j]
            text_color = "white" if val > 13.0 else "black"
            ax1.text(j, i, f"{val:.1f}%", ha='center', va='center', color=text_color, fontweight='bold', fontsize=10.5)

    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label(r"Mean $\Delta TT$ (%)", fontsize=9.5, fontweight='bold')

    # ------------------ PANEL B: RA MINUS RB (PP) ------------------
    im2 = ax2.imshow(mat_diff_rb, cmap='coolwarm', vmin=-14.0, vmax=14.0, aspect='auto')
    ax2.set_title("B. Risk + Actionability vs. Pure Risk\n(RA minus RB, Percentage-Point Difference)", fontsize=11, fontweight='bold', pad=12)
    ax2.set_xticks(np.arange(len(delta_labels)))
    ax2.set_yticks(np.arange(len(budget_labels)))
    ax2.set_xticklabels(delta_labels, fontweight='bold')
    ax2.set_yticklabels(budget_labels, fontweight='bold')
    ax2.set_xlabel(r"Distance Tolerance Bound $\delta$", fontsize=10, fontweight='bold')
    ax2.set_ylabel(r"Relocation Budget $B$", fontsize=10, fontweight='bold')

    for i in range(len(budgets)):
        for j in range(len(deltas)):
            val = mat_diff_rb[i, j]
            text_color = "white" if abs(val) > 9.0 else "black"
            ax2.text(j, i, f"+{val:.1f} pp", ha='center', va='center', color=text_color, fontweight='bold', fontsize=10)

    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Difference (pp)', fontsize=9.5, fontweight='bold')

    # ------------------ PANEL C: RA MINUS AB (PP) ------------------
    im3 = ax3.imshow(mat_diff_ab, cmap='coolwarm', vmin=-3.0, vmax=3.0, aspect='auto')
    ax3.set_title("C. Value of Risk Information vs. Actionability Alone\n(RA minus AB, Percentage-Point Difference)", fontsize=11, fontweight='bold', pad=12)
    ax3.set_xticks(np.arange(len(delta_labels)))
    ax3.set_yticks(np.arange(len(budget_labels)))
    ax3.set_xticklabels(delta_labels, fontweight='bold')
    ax3.set_yticklabels(budget_labels, fontweight='bold')
    ax3.set_xlabel(r"Distance Tolerance Bound $\delta$", fontsize=10, fontweight='bold')
    ax3.set_ylabel(r"Relocation Budget $B$", fontsize=10, fontweight='bold')

    for i in range(len(budgets)):
        for j in range(len(deltas)):
            val = mat_diff_ab[i, j]
            text_color = "white" if abs(val) > 2.0 else "black"
            sign_str = "+" if val > 0 else ""
            ax3.text(j, i, f"{sign_str}{val:.1f} pp", ha='center', va='center', color=text_color, fontweight='bold', fontsize=10)

    cbar3 = fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cbar3.set_label('Difference (pp)', fontsize=9.5, fontweight='bold')

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[Figure 4] Successfully generated and saved to: {out_path}")


def generate_figure_supp_route_trace(trace_csv_path: str, out_path: str):
    """
    Supplementary Figure S1: Route completion-time trajectory on actual held-out route.
    RouteID_693060a6-88bb-4324-9e9c-925d5240263c under RA (B=0.20, delta=0.05).
    Baseline TT = 7,735.5 min | Final TT = 6,284.1 min (Delta TT = +1,451.4 min, -18.8%)
    Late stops: 77 -> 72 (Delta NL = +5 stops) | Delta D = +0.77% | 25 accepted moves.
    """
    df = pd.read_csv(trace_csv_path)
    
    # Sort by baseline sequence position
    df_sorted_base = df.sort_values('baseline_position').reset_index(drop=True)
    df_sorted_final = df.sort_values('final_position').reset_index(drop=True)
    
    fig, ax = plt.subplots(figsize=(15, 6.8), dpi=300)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#fafafa')
    ax.grid(True, linestyle='--', alpha=0.5, color='#ccc', zorder=0)

    # 4-hour threshold reference line (240 min)
    ax.axhline(240.0, color='#c0392b', linestyle='--', linewidth=2.0, label='Derived 4-Hour Service Threshold (240 min)', zorder=2)
    ax.fill_between([0, 170], 240.0, 480.0, color='#fdebd0', alpha=0.35, label='Lateness Region ($T > 240$ min)', zorder=1)

    # Baseline completion-time curve (position 1 to 167)
    x_base = df_sorted_base['baseline_position'].values
    y_base = df_sorted_base['baseline_completion_minutes'].values
    ax.plot(x_base, y_base, color='#2980b9', linewidth=2.2, linestyle='-', label=r'Baseline Route Sequence $R^0$ (Nearest Neighbor)', zorder=4)

    # Resequenced completion-time curve
    x_final = df_sorted_final['final_position'].values
    y_final = df_sorted_final['final_completion_minutes'].values
    ax.plot(x_final, y_final, color='#1b9e77', linewidth=2.2, linestyle='-', label=r"Resequenced Route Sequence $R'$ (PISR Proposed RA)", zorder=5)

    # Identify relocated stops (where baseline_position != final_position)
    relocated = df[df['baseline_position'] != df['final_position']].copy()
    
    # Advanced stops: final_position < baseline_position
    advanced = relocated[relocated['final_position'] < relocated['baseline_position']].copy()
    
    # Highlight relocated stops with markers
    ax.scatter(advanced['baseline_position'], advanced['baseline_completion_minutes'],
               color='#c0392b', marker='x', s=45, linewidth=1.5, zorder=6, label=f'Relocated Stops: Baseline Position ($N={len(advanced)}$)')
    ax.scatter(advanced['final_position'], advanced['final_completion_minutes'],
               color='#8e44ad', marker='o', s=55, edgecolors='black', linewidth=1.0, zorder=7, label=f'Relocated Stops: Resequenced Position ($N={len(advanced)}$)')

    # Draw direct clean connector lines from baseline to resequenced position for key relocated stops
    # Sample every 3rd stop to keep visual clean and uncluttered
    for _, row in advanced.iloc[::3].iterrows():
        bp, fp = row['baseline_position'], row['final_position']
        bc, fc = row['baseline_completion_minutes'], row['final_completion_minutes']
        ax.plot([bp, fp], [bc, fc], color='#8e44ad', linestyle=':', linewidth=1.1, alpha=0.7, zorder=6)

    # Highlight stops that crossed the 240-minute threshold into on-time status
    crossed = df[(df['baseline_completion_minutes'] > 240.0) & (df['final_completion_minutes'] <= 240.0)]
    if not crossed.empty:
        ax.scatter(crossed['final_position'], crossed['final_completion_minutes'],
                   color='#f39c12', marker='*', s=140, edgecolors='black', linewidth=1.2, zorder=8,
                   label=f'Late Deliveries Recovered to On-Time ($N={len(crossed)}$ stops)')

    # Annotation box summarizing held-out route parameters and results
    info_text = (
        "Held-Out Route Verification Run:\n"
        "- Route ID: RouteID_693060a6 (167 customer stops)\n"
        "- Resequencing Policy: RA (B = 20%, delta = 5%)\n"
        "- Baseline Tardiness TT(R0): 7,735.5 min (77 late stops)\n"
        "- Resequenced Tardiness TT(R'): 6,284.1 min (72 late stops)\n"
        "- Total Tardiness Reduction: Delta TT = +1,451.4 min (-18.8%)\n"
        "- Late Deliveries Eliminated: Delta NL = +5 stops\n"
        "- Distance Detour: Delta D = +0.77% (within delta = 5.0% bound)\n"
        "- Route Resequencing Latency: 0.231 s (Feasibility: 100%)"
    )
    ax.text(0.02, 0.44, info_text, transform=ax.transAxes, fontsize=8.8, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#2c3e50', alpha=0.95, lw=1.2), zorder=10)

    ax.set_xlim(0, 169)
    ax.set_ylim(0, 480)
    ax.set_xlabel("Stop Sequence Position along Route (1 to 167)", fontsize=10.5, fontweight='bold')
    ax.set_ylabel("Elapsed Completion Time after Departure (Minutes)", fontsize=10.5, fontweight='bold')
    ax.set_title("Supplementary Figure S1: Stop Completion-Time Trajectory under PISR Resequencing\n"
                 "(Reproducible trace from held-out Amazon route RouteID_693060a6 under RA targeting)",
                 fontsize=11.5, fontweight='bold', pad=12)
    ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.95, edgecolor='#ccc', fontsize=8.5)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[Supplementary Figure S1] Successfully generated and saved to: {out_path}")


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    outputs_dir = os.path.join(base_dir, "outputs")
    figures_dir = os.path.abspath(os.path.join(base_dir, "..", "figures"))
    os.makedirs(figures_dir, exist_ok=True)

    runs_csv = os.path.join(outputs_dir, "routing_benchmark_runs.csv")
    preds_csv = os.path.join(outputs_dir, "rq1_stop_level_predictions.csv")
    trace_csv = os.path.join(outputs_dir, "held_out_route_trace.csv")

    fig2_out = os.path.join(figures_dir, "fig2_policy_comparison.png")
    fig3_out = os.path.join(figures_dir, "fig3_prediction_calibration.png")
    fig4_out = os.path.join(figures_dir, "fig4_budget_distance_sensitivity.png")
    supp_fig_out = os.path.join(figures_dir, "fig_supp_route_trace.png")

    print("--- Generating Figure 2 ---")
    generate_figure2_policy_comparison(runs_csv, fig2_out)

    print("\n--- Generating Figure 3 ---")
    generate_figure3_prediction_calibration(preds_csv, fig3_out)

    print("\n--- Generating Figure 4 ---")
    generate_figure4_budget_distance_sensitivity(runs_csv, fig4_out)

    print("\n--- Generating Supplementary Figure S1 ---")
    generate_figure_supp_route_trace(trace_csv, supp_fig_out)
    
    print("\nAll publication figures successfully created!")
