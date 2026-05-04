"""
Per-Scenario Visualization — Option D + Paper Tables

For each scenario, loads 5 Q-tables (one per seed), averages Q-values,
and produces:

  Figure: 6-panel "Policy + Value Overview"
    (a) Culling Heatmap (open, healthy)
    (b) Culling Heatmap (open, mastitis)
    (c) Culling Boundary Line Plot
    (d) Line plot: Parity 1 (healthy + mastitis + pregnancy)
    (e) Line plot: Parity 3
    (f) Line plot: Parity 5

  Tables (CSV, saved to outputs/tables/):
    - table_pregnancy_value_<scenario>.csv
    - table_mastitis_cost_<scenario>.csv

  Supplementary figures:
    - <scenario>_heatmap_parity_mac.png  (healthy + mastitis side by side)
    - <scenario>_heatmap_mip_mac.png     (pregnancy effect for parities 1,3,5,8)
    - <scenario>_line_plots_all.png      (all 12 parities, healthy vs mastitis)

Usage:
    python visualize_scenario.py --collected_dir collected/
    python visualize_scenario.py --collected_dir collected/ --scenario 2025
"""

import argparse
import csv
import os
import glob
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.ndimage import uniform_filter1d

import utility

FIGURES_ROOT = Path("outputs") / "figures"
TABLES_ROOT = Path("outputs") / "tables"

SCENARIOS = ['2025', 'OG', 'OB', 'UG', 'UB']
SEEDS = [42, 123, 456, 789, 1024]

SCENARIO_NAMES = {
    "2025": "2025 Baseline",
    "OG": "Oversupply, Good Market",
    "OB": "Oversupply, Bad Market",
    "UG": "Undersupply, Good Market",
    "UB": "Undersupply, Bad Market",
}

parity_range = range(13)
mac_range = range(21)
mip_range = range(10)
disease_range = range(2)

# Conception MACs: cow conceives at MAC-1, first observed pregnant (MIP=1) at this MAC
TRACK_START_MACS = (4, 7, 10)
MAX_OPEN_MAC = 19

PREG_LEGEND_LABELS = {
    4: "Conceived MAC 3",
    7: "Conceived MAC 6",
    10: "Conceived MAC 9",
}


def load_pkl(filename):
    with open(filename, "rb") as f:
        q_table, rewards, epsilon = pickle.load(f)
    return q_table, rewards, epsilon


def add_subplot_label(ax, label, fontsize=16, x=-0.08, y=1.05):
    ax.text(x, y, f"({label})", transform=ax.transAxes,
            fontsize=fontsize, fontweight='bold', va='top', ha='left')


def auto_discover_files(collected_dir, scenario=None):
    scenarios = [scenario] if scenario else SCENARIOS
    scenario_files = {}
    for sc in scenarios:
        files = []
        for seed in SEEDS:
            for pattern in [
                os.path.join(collected_dir, f"DQN_{sc}_seed{seed}.pkl"),
                os.path.join(collected_dir, f"DQN_{sc}_{seed}.pkl"),
            ]:
                if os.path.exists(pattern):
                    files.append(pattern)
                    break
        if files:
            scenario_files[sc] = files
    return scenario_files


# =============================================================================
# Q-value extraction helpers
# =============================================================================

def compute_q_diff_stats(q_tables, state):
    """Mean and SE of Q(keep) - Q(replace) across seeds."""
    diffs = []
    for qt in q_tables:
        if state in qt:
            diffs.append(qt[state].get('keep', 0) - qt[state].get('replace', 0))
    if diffs:
        n = len(diffs)
        return np.mean(diffs), np.std(diffs, ddof=1) / np.sqrt(n) if n > 1 else 0, n
    return np.nan, 0, 0


def compute_q_keep_stats(q_tables, state):
    """Mean and SE of Q(keep) across seeds."""
    vals = []
    for qt in q_tables:
        if state in qt:
            vals.append(qt[state].get('keep', 0))
    if vals:
        n = len(vals)
        return np.mean(vals), np.std(vals, ddof=1) / np.sqrt(n) if n > 1 else 0, n
    return np.nan, 0, 0


def extract_heatmap_parity_mac(q_tables, disease_status=0, fixed_mip=0):
    """Heatmap (Parity x MAC) of mean Q(keep) - Q(replace)."""
    parities = list(range(1, 13))
    macs = list(range(1, 21))
    heatmap = np.full((len(parities), len(macs)), np.nan)

    for i, parity in enumerate(parities):
        for j, mac in enumerate(macs):
            state = (parity, mac, fixed_mip, disease_status)
            if not utility.possible_state2(state, parity_range, mac_range,
                                           mip_range, disease_range):
                continue
            mean_diff, _, _ = compute_q_diff_stats(q_tables, state)
            heatmap[i, j] = mean_diff

    return heatmap, parities, macs


def extract_heatmap_mip_mac(q_tables, fixed_parity=1, disease_status=0):
    """Heatmap (MIP x MAC) of mean Q(keep) - Q(replace)."""
    mips = list(range(0, 10))
    macs = list(range(1, 21))
    heatmap = np.full((len(mips), len(macs)), np.nan)

    for i, mip in enumerate(mips):
        for j, mac in enumerate(macs):
            state = (fixed_parity, mac, mip, disease_status)
            if not utility.possible_state2(state, parity_range, mac_range,
                                           mip_range, disease_range):
                continue
            mean_diff, _, _ = compute_q_diff_stats(q_tables, state)
            heatmap[i, j] = mean_diff

    return heatmap, mips, macs


def track_pregnant_cow(q_tables, parity, starting_mac, disease, starting_mip=1):
    """Track a cow that gets pregnant at starting_mac through pregnancy."""
    mac_list, mip_list, mean_list, se_list = [], [], [], []
    mac, mip = starting_mac, starting_mip

    while mip <= 9 and mac <= 20:
        state = (parity, mac, mip, disease)
        mean_diff, se_diff, n = compute_q_diff_stats(q_tables, state)
        if not np.isnan(mean_diff):
            mac_list.append(mac)
            mip_list.append(mip)
            mean_list.append(mean_diff)
            se_list.append(se_diff)
        mac += 1
        mip += 1

    return mac_list, mip_list, mean_list, se_list


def track_open_cow(q_tables, parity, disease, max_mac=MAX_OPEN_MAC):
    """Track an open cow (MIP=0) from MAC=1 to max_mac."""
    mac_list, mean_list, se_list = [], [], []

    for mac in range(1, max_mac + 1):
        state = (parity, mac, 0, disease)
        mean_diff, se_diff, n = compute_q_diff_stats(q_tables, state)
        if not np.isnan(mean_diff):
            mac_list.append(mac)
            mean_list.append(mean_diff)
            se_list.append(se_diff)

    return mac_list, mean_list, se_list


def find_culling_boundary(q_tables, disease=0, pregnant_start_mac=None):
    """For each parity, find MAC where Q(keep)-Q(replace) first < 0."""
    boundaries = {}
    for parity in range(1, 13):
        if pregnant_start_mac is not None:
            mac_list, _, mean_list, _ = track_pregnant_cow(
                q_tables, parity, pregnant_start_mac, disease)
        else:
            mac_list, mean_list, _ = track_open_cow(q_tables, parity, disease, max_mac=20)

        boundary = None
        for mac, val in zip(mac_list, mean_list):
            if val < 0:
                boundary = mac
                break
        boundaries[parity] = boundary

    return boundaries


# =============================================================================
# Pregnancy Value and Mastitis Cost Tables
# =============================================================================

def compute_pregnancy_value_table(q_tables):
    """
    Compute new pregnancy value: Q_keep(pregnant) - Q_keep(open)
    for each (parity, conception MAC), averaged over both health statuses.

    For each (parity, conceived_mac, disease) triple, the value is:
      Q_keep(parity, mac, MIP=1, disease) - Q_keep(parity, mac, MIP=0, disease)

    The result is then averaged across disease in {0, 1} so that the
    comparison differs **only** in pregnancy status (MIP=0 vs MIP=1),
    consistent with how compute_mastitis_cost_table averages over MIP.

    Returns dict: {parity: {conceived_mac: (mean, se)}}
    """
    table = {}
    # All conception MACs from 3 to 10, consistent with
    # compute_pregnancy_value_by_mac() which uses range(3, 11).
    conceived_macs_list = list(range(3, 11))  # MAC 3, 4, 5, 6, 7, 8, 9, 10

    for parity in range(1, 13):
        table[parity] = {}
        for conceived_mac in conceived_macs_list:
            mac = conceived_mac + 1  # month the cow is first observed pregnant
            diffs = []
            for disease in range(0, 2):  # 0=healthy, 1=mastitis
                state_preg = (parity, mac, 1, disease)  # MIP=1
                state_open = (parity, mac, 0, disease)  # MIP=0

                if (not utility.possible_state2(state_preg, parity_range, mac_range,
                                                mip_range, disease_range) or
                    not utility.possible_state2(state_open, parity_range, mac_range,
                                                mip_range, disease_range)):
                    continue

                mean_preg, _, _ = compute_q_keep_stats(q_tables, state_preg)
                mean_open, _, _ = compute_q_keep_stats(q_tables, state_open)

                if not np.isnan(mean_preg) and not np.isnan(mean_open):
                    diffs.append(mean_preg - mean_open)

            if diffs:
                table[parity][conceived_mac] = (
                    np.mean(diffs),
                    np.std(diffs) / np.sqrt(len(diffs)) if len(diffs) > 1 else 0
                )
            else:
                table[parity][conceived_mac] = (np.nan, 0)

    return table


def compute_mastitis_cost_table(q_tables):
    """
    Compute mastitis cost: Q_diff(healthy) - Q_diff(mastitis)
    for each (parity, lactation stage).
    Positive = mastitis reduces the keep advantage.

    Lactation stages are defined by DIM (days in milk), mapped to MAC:
      Early       0-100 DIM  -> MAC 1-3
      Mid       100-200 DIM  -> MAC 4-7
      Late      200-305 DIM  -> MAC 7-10
      Extended  >=305 DIM    -> MAC 11-20

    For each (parity, mac), all valid MIP values (0=open, 1-9=pregnant)
    are included so that both open and pregnant cows contribute to the
    stage average.  Only the disease flag differs between the paired
    healthy and mastitic states.

    Returns dict: {parity: {stage_label: (mean, se)}}
    """
    table = {}
    # MAC ranges derived from DIM boundaries (1 MAC ~ 30 DIM)
    stages = {
        'Early (0-100 DIM)':    range(1, 4),    # MAC 1-3
        'Mid (100-200 DIM)':    range(4, 8),    # MAC 4-7
        'Late (200-305 DIM)':   range(8, 11),   # MAC 8-10
        'Extended (>=305 DIM)': range(11, 21),  # MAC 11-20
    }

    for parity in range(1, 13):
        table[parity] = {}
        for stage_label, mac_range_stage in stages.items():
            diffs = []
            for mac in mac_range_stage:
                # Iterate over all valid MIP values to include both open
                # (mip=0) and pregnant (mip=1..9) cows.  The parity, mac,
                # and mip are held constant; only disease status differs.
                for mip in range(0, 10):
                    state_h = (parity, mac, mip, 0)  # healthy
                    state_d = (parity, mac, mip, 1)  # mastitis

                    if (not utility.possible_state2(state_h, parity_range, mac_range,
                                                    mip_range, disease_range) or
                        not utility.possible_state2(state_d, parity_range, mac_range,
                                                    mip_range, disease_range)):
                        continue

                    mean_h, _, _ = compute_q_diff_stats(q_tables, state_h)
                    mean_d, _, _ = compute_q_diff_stats(q_tables, state_d)

                    if not np.isnan(mean_h) and not np.isnan(mean_d):
                        diffs.append(mean_h - mean_d)

            if diffs:
                table[parity][stage_label] = (np.mean(diffs), np.std(diffs) / np.sqrt(len(diffs)))
            else:
                table[parity][stage_label] = (np.nan, 0)

    return table


def save_pregnancy_value_csv(table, scenario_code, output_dir):
    """Save pregnancy value table as CSV.

    Columns are generated dynamically from all conception MACs present in the
    table, so the output automatically expands when conceived_macs_list in
    compute_pregnancy_value_table() is changed.
    """
    # Collect all conception MACs stored in the table (sorted)
    all_conceived_macs = sorted(
        {cm for parity_data in table.values() for cm in parity_data.keys()}
    )

    path = os.path.join(output_dir, f"table_pregnancy_value_{scenario_code}.csv")
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Dynamic header: one pair of (value, SE) columns per conception MAC
        header = ['Parity']
        for conceived_mac in all_conceived_macs:
            header.extend([f'Conceived MAC {conceived_mac} ($)', 'SE'])
        writer.writerow(header)

        for parity in range(1, 13):
            row = [str(parity)]
            for conceived_mac in all_conceived_macs:
                mean, se = table[parity].get(conceived_mac, (np.nan, 0))
                if not np.isnan(mean):
                    row.extend([f"{mean:.0f}", f"{se:.0f}"])
                else:
                    row.extend(['', ''])
            writer.writerow(row)

        # Add average row
        row_avg = ['Average']
        for conceived_mac in all_conceived_macs:
            vals = [table[p][conceived_mac][0] for p in range(1, 13)
                    if not np.isnan(table[p].get(conceived_mac, (np.nan,))[0])]
            if vals:
                row_avg.extend([f"{np.mean(vals):.0f}", ''])
            else:
                row_avg.extend(['', ''])
        writer.writerow(row_avg)

    print(f"  Table saved: {path}")
    return path


def save_mastitis_cost_csv(table, scenario_code, output_dir):
    """Save mastitis cost table as CSV."""
    stages = ['Early (0-100 DIM)', 'Mid (100-200 DIM)',
              'Late (200-305 DIM)', 'Extended (>=305 DIM)']
    path = os.path.join(output_dir, f"table_mastitis_cost_{scenario_code}.csv")
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['Parity']
        for s in stages:
            header.extend([f'{s} ($)', 'SE'])
        writer.writerow(header)

        for parity in range(1, 13):
            row = [str(parity)]
            for stage in stages:
                mean, se = table[parity].get(stage, (np.nan, 0))
                if not np.isnan(mean):
                    row.extend([f"{mean:.0f}", f"{se:.0f}"])
                else:
                    row.extend(['', ''])
            writer.writerow(row)

        # Add average row
        row_avg = ['Average']
        for stage in stages:
            vals = [table[p][stage][0] for p in range(1, 13)
                    if not np.isnan(table[p].get(stage, (np.nan,))[0])]
            if vals:
                row_avg.extend([f"{np.mean(vals):.0f}", ''])
            else:
                row_avg.extend(['', ''])
        writer.writerow(row_avg)

    print(f"  Table saved: {path}")
    return path


# =============================================================================
# Plotting helpers
# =============================================================================

def plot_heatmap(ax, heatmap, parities, macs, title, cmap="RdYlGn",
                 ylabel="Parity", xlabel="Month After Calving (MAC)",
                 cbar_label="Q(Keep) \u2212 Q(Replace)", symmetric=True):
    """Shared heatmap plotting."""
    if np.any(~np.isnan(heatmap)):
        if symmetric:
            vmax = np.nanmax(np.abs(heatmap))
            vmin = -vmax
        else:
            vmin = np.nanmin(heatmap)
            vmax = np.nanmax(heatmap)
    else:
        vmin, vmax = -1, 1

    im = ax.imshow(heatmap, aspect="auto", cmap=cmap, origin="lower",
                   vmin=vmin, vmax=vmax)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks(np.arange(len(macs)))
    ax.set_xticklabels(macs, fontsize=7)
    ax.set_yticks(np.arange(len(parities)))
    ax.set_yticklabels(parities, fontsize=9)
    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(cbar_label, fontsize=9)
    
    # Add range text box in upper-left corner
    actual_min = np.nanmin(heatmap)
    actual_max = np.nanmax(heatmap)
    range_text = f"Range: [{actual_min:.0f}, {actual_max:.0f}]"
    ax.text(0.02, 0.98, range_text, transform=ax.transAxes,
            fontsize=8, verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='lightyellow', 
                     edgecolor='black', alpha=0.7))
    
    return im


def plot_line_chart_combined(q_tables, parity, title, ax, smooth=True):
    """Line plot with healthy (solid) and mastitis (dashed) overlaid."""
    track_colors = ["#1f77b4", "#e377c2", "#f28e2b"]
    open_color = "#555555"

    disease_styles = {
        0: ("-",  "o", "full",  1.0),
        1: ("--", "o", "none",  0.70),
    }

    for disease in [0, 1]:
        ls, marker, fillstyle, alpha = disease_styles[disease]
        health_label = "" if disease == 0 else ", Mastitis"

        # Pregnant cow tracks
        for i, start_mac in enumerate(TRACK_START_MACS):
            mac_list, mip_list, mean_list, se_list = track_pregnant_cow(
                q_tables, parity, start_mac, disease)
            if len(mac_list) == 0:
                continue

            color = track_colors[i % len(track_colors)]
            means = np.array(mean_list)
            ses = np.array(se_list)

            if smooth and len(means) > 3:
                smoothed = uniform_filter1d(means, size=3, mode="nearest")
            else:
                smoothed = means

            conceived_mac = start_mac - 1
            label = PREG_LEGEND_LABELS.get(start_mac, f"Conceived MAC {conceived_mac}")
            label += health_label

            ax.plot(mac_list, smoothed, linestyle=ls, marker=marker,
                    fillstyle=fillstyle, linewidth=2, markersize=3.5,
                    color=color, alpha=alpha, label=label)
            ax.fill_between(mac_list, smoothed - ses, smoothed + ses,
                           alpha=0.08 if disease == 1 else 0.12, color=color)

        # Open cow track
        mac_list, mean_list, se_list = track_open_cow(q_tables, parity, disease)
        if len(mac_list) > 0:
            means = np.array(mean_list)
            ses = np.array(se_list)

            if smooth and len(means) > 3:
                smoothed = uniform_filter1d(means, size=3, mode="nearest")
            else:
                smoothed = means

            label = "Open" + health_label
            ax.plot(mac_list, smoothed, linestyle=ls, marker='s',
                    fillstyle=fillstyle, linewidth=2, markersize=3.5,
                    color=open_color, alpha=alpha, label=label)
            ax.fill_between(mac_list, smoothed - ses, smoothed + ses,
                           alpha=0.06 if disease == 1 else 0.08, color=open_color)

    ax.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Month After Calving (MAC)", fontsize=10)
    ax.set_xticks(range(0, 21, 2))
    ax.set_xlim(0.5, 20.5)
    ax.set_ylabel("Q(Keep) \u2212 Q(Replace)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=6.5, loc="best", ncol=2, columnspacing=0.8,
             handlelength=2.5, handletextpad=0.4)
    ax.grid(True, alpha=0.3)
    # ax.text(0.98, 0.02,
    #         "Solid = Healthy | Dashed = Mastitis\nAbove 0: Keep | Below 0: Cull",
    #         transform=ax.transAxes, fontsize=7,
    #         verticalalignment="bottom", horizontalalignment="right",
    #         bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.4))


# =============================================================================
# Main figure: Option D (6-panel)
# =============================================================================

def plot_option_d(scenario_code, q_tables, output_dir):
    """Option D: 6-panel Policy + Value Overview."""
    scenario_name = SCENARIO_NAMES.get(scenario_code, scenario_code)
    n_seeds = len(q_tables)

    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.32)

    # (a) Replacement Heatmap: Healthy
    ax_a = fig.add_subplot(gs[0, 0])
    heatmap_h, parities, macs = extract_heatmap_parity_mac(q_tables, disease_status=0)
    plot_heatmap(ax_a, heatmap_h, parities, macs,
                 "Replacement Decision Heatmap\n(Open, Healthy)", cbar_label="Green=Keep, Red=Replace")
    add_subplot_label(ax_a, 'a')

    # (b) Replacement Heatmap: Mastitis
    ax_b = fig.add_subplot(gs[0, 1])
    heatmap_d, parities, macs = extract_heatmap_parity_mac(q_tables, disease_status=1)
    plot_heatmap(ax_b, heatmap_d, parities, macs,
                 "Replacement Decision Heatmap\n(Open, Mastitis)", cbar_label="Green=Keep, Red=Replace")
    add_subplot_label(ax_b, 'b')

    # (c) Replacement Boundary Line Plot
    ax_c = fig.add_subplot(gs[0, 2])

    bound_healthy_open = find_culling_boundary(q_tables, disease=0)
    bound_mastitis_open = find_culling_boundary(q_tables, disease=1)

    parities_plot = list(range(1, 13))

    def get_bv(boundaries, plist):
        vals, valid_p = [], []
        for p in plist:
            if boundaries.get(p) is not None:
                vals.append(boundaries[p])
                valid_p.append(p)
        return valid_p, vals

    styles = [
        (bound_healthy_open, '#2ca02c', '-o', 'Healthy, Open'),
        (bound_mastitis_open, '#d62728', '-s', 'Mastitis, Open'),
    ]
    for bounds, color, style, label in styles:
        p_vals, b_vals = get_bv(bounds, parities_plot)
        if p_vals:
            ax_c.plot(p_vals, b_vals, style, color=color, linewidth=2,
                      markersize=6, label=label)

    ax_c.set_xlabel("Parity", fontsize=10)
    ax_c.set_ylabel("Month After Calving", fontsize=10)
    ax_c.set_title("Replacement Boundary\n(where Q(keep) < Q(replace))",
                   fontsize=11, fontweight="bold")
    ax_c.set_xticks(parities_plot)
    ax_c.legend(fontsize=9, loc="best")
    ax_c.grid(True, alpha=0.3)
    # ax_c.text(0.98, 0.98, "Lower = earlier culling",
    #           transform=ax_c.transAxes, fontsize=8,
    #           va='top', ha='right',
    #           bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.4))
    add_subplot_label(ax_c, 'c')

    # (d)(e)(f) Line plots: Parity 1, 3, 5
    for idx, (parity, title, label) in enumerate([
        (1, "Parity 1", 'd'),
        (3, "Parity 3", 'e'),
        (5, "Parity 5", 'f'),
    ]):
        ax = fig.add_subplot(gs[1, idx])
        plot_line_chart_combined(q_tables, parity, title, ax, smooth=True)
        add_subplot_label(ax, label)

    fig.suptitle(f"{scenario_name} \u2014 Optimal Replacement Policy\n"
                 f"(Mean across {n_seeds} seeds)",
                 fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()

    path = output_dir / f"{scenario_code}_policy_overview.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# =============================================================================
# Supplementary figures
# =============================================================================

def plot_standalone_heatmaps(scenario_code, q_tables, output_dir):
    """Standalone heatmap figures for supplementary material."""
    scenario_name = SCENARIO_NAMES.get(scenario_code, scenario_code)
    n_seeds = len(q_tables)

    # Parity x MAC heatmaps (healthy + diseased side by side)
    fig1, axes1 = plt.subplots(1, 2, figsize=(18, 7))
    for ax_idx, (disease, disease_label) in enumerate([(0, "Healthy"), (1, "Mastitis")]):
        ax = axes1[ax_idx]
        heatmap, parities, macs = extract_heatmap_parity_mac(q_tables, disease_status=disease)
        plot_heatmap(ax, heatmap, parities, macs, f"Open Cows, {disease_label}")
        add_subplot_label(ax, chr(ord('a') + ax_idx))

    fig1.suptitle(f"{scenario_name} \u2014 Replacement Decision Heatmaps\n"
                  f"(Mean across {n_seeds} seeds)",
                  fontsize=14, fontweight="bold", y=1.02)
    fig1.tight_layout()
    path1 = output_dir / f"{scenario_code}_heatmap_parity_mac.png"
    fig1.savefig(path1, dpi=300, bbox_inches="tight")
    plt.close(fig1)
    print(f"  Saved: {path1}")

    # MIP x MAC heatmaps for selected parities
    selected_parities = [1, 3, 5, 8]
    fig2, axes2 = plt.subplots(1, len(selected_parities),
                                figsize=(5 * len(selected_parities), 5))
    for ax_idx, par in enumerate(selected_parities):
        ax = axes2[ax_idx]
        heatmap_mip, mips, macs_mip = extract_heatmap_mip_mac(q_tables, fixed_parity=par)
        plot_heatmap(ax, heatmap_mip, mips, macs_mip,
                     f"Parity {par}, Healthy",
                     ylabel="Month In Pregnancy (MIP)")
        add_subplot_label(ax, chr(ord('a') + ax_idx))

    fig2.suptitle(f"{scenario_name} \u2014 Pregnancy Effect on Replacement Decisions\n"
                  f"(Mean across {n_seeds} seeds)",
                  fontsize=14, fontweight="bold", y=1.02)
    fig2.tight_layout()
    path2 = output_dir / f"{scenario_code}_heatmap_mip_mac.png"
    fig2.savefig(path2, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {path2}")


def plot_standalone_line_plots(scenario_code, q_tables, output_dir):
    """All 12 parities, healthy vs mastitis on same subplot."""
    scenario_name = SCENARIO_NAMES.get(scenario_code, scenario_code)
    n_seeds = len(q_tables)

    fig, axes = plt.subplots(4, 3, figsize=(18, 20))
    axes_flat = axes.flatten()

    for idx, parity in enumerate(range(1, 13)):
        ax = axes_flat[idx]
        plot_line_chart_combined(q_tables, parity, f"Parity {parity}", ax, smooth=True)

    fig.suptitle(f"{scenario_name} \u2014 Replacement Decision: Healthy vs Mastitis\n"
                 f"(Mean \u00b1 SE across {n_seeds} seeds)",
                 fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = output_dir / f"{scenario_code}_line_plots_all.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================================
# New-pregnancy value and mastitis cost line figures
# =============================================================================

# Parities to highlight and their visual styles (marker, color)
_PREG_FIG_PARITIES = [
    (1,  'o', '#1f77b4', 'Parity 1'),
    (2,  's', '#2ca02c', 'Parity 2'),
    (3,  '^', '#d62728', 'Parity 3'),
    (4,  'D', '#9467bd', 'Parity 4'),
    (5, 'v', '#8c564b', 'Parity 5'),
    # (12, 'P', '#7f7f7f', 'Parity 12'),
]

_MAST_FIG_PARITIES = [
    (1,  'o', '#1f77b4', 'Parity 1'),
    (3,  's', '#2ca02c', 'Parity 3'),
    (5,  '^', '#d62728', 'Parity 5'),
    (8,  'D', '#9467bd', 'Parity 8'),
    (10, 'v', '#8c564b', 'Parity 10'),
    (12, 'P', '#7f7f7f', 'Parity 12'),
]


def compute_pregnancy_value_by_mac(q_tables, parity):
    """
    For a given parity, compute the new-pregnancy value at every possible
    conception MAC, averaged over both health statuses (disease=0 and disease=1).

    For each (conceived_mac, disease):
      diff = Q_keep(parity, mac+1, MIP=1, disease) - Q_keep(parity, mac+1, MIP=0, disease)

    The two diffs (healthy and mastitic) are averaged so that the comparison
    differs **only** in pregnancy status (MIP=0 vs MIP=1).

    Returns arrays: conceived_macs, means, ses
    """
    conceived_macs, means, ses = [], [], []
    for conceived_mac in range(3, 11):
        mac = conceived_mac + 1  # month the cow is first observed pregnant
        diffs = []
        for disease in range(0, 2):  # 0=healthy, 1=mastitis
            state_preg = (parity, mac, 1, disease)
            state_open = (parity, mac, 0, disease)
            if (not utility.possible_state2(state_preg, parity_range, mac_range,
                                            mip_range, disease_range) or
                    not utility.possible_state2(state_open, parity_range, mac_range,
                                                mip_range, disease_range)):
                continue
            mean_p, _, _ = compute_q_keep_stats(q_tables, state_preg)
            mean_o, _, _ = compute_q_keep_stats(q_tables, state_open)
            if np.isnan(mean_p) or np.isnan(mean_o):
                continue
            diffs.append(mean_p - mean_o)
        if not diffs:
            continue
        conceived_macs.append(conceived_mac)
        means.append(np.mean(diffs))
        ses.append(np.std(diffs) / np.sqrt(len(diffs)) if len(diffs) > 1 else 0)
    return np.array(conceived_macs), np.array(means), np.array(ses)


def compute_mastitis_cost_by_mac(q_tables, parity):
    """
    For a given parity, compute the per-MAC mastitis cost averaged over all
    valid MIP values (0=open, 1-9=pregnant).

    For each (mac, mip):
      diff = Q_keep(parity, mac, mip, CM=0) - Q_keep(parity, mac, mip, CM=1)

    Diffs are averaged across all valid mip values so that the comparison
    differs **only** in health status (CM=0 vs CM=1).

    Returns arrays: macs, means, ses
    """
    macs_out, means, ses = [], [], []
    for mac in range(1, 21):
        diffs = []
        for mip in range(0, 10):
            state_h = (parity, mac, mip, 0)
            state_d = (parity, mac, mip, 1)
            if (not utility.possible_state2(state_h, parity_range, mac_range,
                                            mip_range, disease_range) or
                    not utility.possible_state2(state_d, parity_range, mac_range,
                                                mip_range, disease_range)):
                continue
            mean_h, _, _ = compute_q_keep_stats(q_tables, state_h)
            mean_d, _, _ = compute_q_keep_stats(q_tables, state_d)
            if np.isnan(mean_h) or np.isnan(mean_d):
                continue
            diffs.append(mean_h - mean_d)
        if not diffs:
            continue
        macs_out.append(mac)
        means.append(np.mean(diffs))
        ses.append(np.std(diffs) / np.sqrt(len(diffs)) if len(diffs) > 1 else 0)
    return np.array(macs_out), np.array(means), np.array(ses)


def plot_pregnancy_value_figure(scenario_code, q_tables, output_dir):
    """
    Line figure: new-pregnancy value vs. MAC at conception,
    one line per selected parity.  Styled after De Vries (2006) Fig. 3/4.
    """
    scenario_name = SCENARIO_NAMES.get(scenario_code, scenario_code)

    fig, ax = plt.subplots(figsize=(9, 6))

    for parity, marker, color, label in _PREG_FIG_PARITIES:
        conceived_macs, means, ses = compute_pregnancy_value_by_mac(q_tables, parity)
        if len(conceived_macs) == 0:
            continue
        ax.plot(conceived_macs, means, marker=marker, color=color,
                linewidth=2, markersize=7, label=label)
        ax.fill_between(conceived_macs,
                        means - ses, means + ses,
                        color=color, alpha=0.12)

    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.6)
    ax.set_xlabel('Month After Calving at Conception (MAC)', fontsize=12)
    ax.set_ylabel('Value of New Pregnancy ($)', fontsize=12)
    ax.set_title(
        f'Value of a New Pregnancy by Conception Timing\n'
        f'{scenario_name} (Mean \u00b1 SE across seeds)',
        fontsize=13, fontweight='bold')
    ax.set_xticks(range(3, 11))
    ax.legend(fontsize=10, loc='upper right', framealpha=0.8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = output_dir / f'{scenario_code}_pregnancy_value.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')
    return path


def plot_mastitis_cost_figure(scenario_code, q_tables, output_dir):
    """
    Line figure: mastitis cost vs. MAC,
    one line per selected parity.  Styled after De Vries (2006) Fig. 3/4.
    """
    scenario_name = SCENARIO_NAMES.get(scenario_code, scenario_code)

    fig, ax = plt.subplots(figsize=(9, 6))

    for parity, marker, color, label in _MAST_FIG_PARITIES:
        macs, means, ses = compute_mastitis_cost_by_mac(q_tables, parity)
        if len(macs) == 0:
            continue
        ax.plot(macs, means, marker=marker, color=color,
                linewidth=2, markersize=7, label=label)
        ax.fill_between(macs,
                        means - ses, means + ses,
                        color=color, alpha=0.12)

    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.6)
    ax.set_xlabel('Month After Calving (MAC)', fontsize=12)
    ax.set_ylabel('Cost of Clinical Mastitis ($)', fontsize=12)
    ax.set_title(
        f'Cost of Clinical Mastitis by Lactation Stage\n'
        f'{scenario_name} (Mean \u00b1 SE across seeds)',
        fontsize=13, fontweight='bold')
    ax.set_xticks(range(1, 21, 2))
    ax.legend(fontsize=10, loc='upper right', framealpha=0.8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = output_dir / f'{scenario_code}_mastitis_cost.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')
    return path


# =============================================================================
# Master function per scenario
# =============================================================================

def plot_scenario(scenario_code, q_tables, figures_dir, tables_dir):
    """Generate all outputs for one scenario."""
    scenario_name = SCENARIO_NAMES.get(scenario_code, scenario_code)
    n_seeds = len(q_tables)
    print(f"\nProcessing {scenario_name} ({n_seeds} seeds)...")

    scenario_fig_dir = figures_dir / scenario_code
    scenario_fig_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Main figure: Option D
    plot_option_d(scenario_code, q_tables, scenario_fig_dir)

    # Supplementary figures
    plot_standalone_heatmaps(scenario_code, q_tables, scenario_fig_dir)
    plot_standalone_line_plots(scenario_code, q_tables, scenario_fig_dir)

    # New line figures: pregnancy value and mastitis cost
    # (both averaged over the complementary dimension so only one variable differs)
    plot_pregnancy_value_figure(scenario_code, q_tables, scenario_fig_dir)
    plot_mastitis_cost_figure(scenario_code, q_tables, scenario_fig_dir)

    # Paper tables: Pregnancy Value
    preg_table = compute_pregnancy_value_table(q_tables)
    save_pregnancy_value_csv(preg_table, scenario_code, str(tables_dir))

    # Print pregnancy value summary to terminal (all conception MACs)
    all_conceived_macs = sorted(
        {cm for parity_data in preg_table.values() for cm in parity_data.keys()}
    )
    col_w = 14
    header_str = f"  {'Parity':<10}" + "".join(
        f" {'Conc.MAC'+str(cm):>{col_w}}" for cm in all_conceived_macs
    )
    print(f"\n  Pregnancy Value Summary ({scenario_code}):")
    print(header_str)
    print(f"  {'-' * (10 + (col_w + 1) * len(all_conceived_macs))}")
    all_vals = []
    for parity in range(1, 13):
        row = f"  {parity:<10}"
        for cm in all_conceived_macs:
            mean, se = preg_table[parity].get(cm, (np.nan, 0))
            if not np.isnan(mean):
                row += f" ${mean:>{col_w - 1},.0f}"
                all_vals.append(mean)
            else:
                row += f" {'--':>{col_w}}"
        print(row)
    if all_vals:
        print(f"  {'Average':<10} ${np.mean(all_vals):>11,.0f}")

    # Paper tables: Mastitis Cost
    mast_table = compute_mastitis_cost_table(q_tables)
    save_mastitis_cost_csv(mast_table, scenario_code, str(tables_dir))

    # Print mastitis cost summary to terminal
    stages = ['Early (0-100 DIM)', 'Mid (100-200 DIM)',
              'Late (200-305 DIM)', 'Extended (>=305 DIM)']
    print(f"\n  Mastitis Cost Summary ({scenario_code}):")
    print(f"  {'Parity':<10} {'Early':>10} {'Mid':>10} {'Late':>10} {'Extended':>10}")
    print(f"  {'-'*50}")
    all_costs = []
    for parity in range(1, 13):
        row = f"  {parity:<10}"
        for stage in stages:
            mean, se = mast_table[parity].get(stage, (np.nan, 0))
            if not np.isnan(mean):
                row += f" ${mean:>8,.0f}"
                all_costs.append(mean)
            else:
                row += f" {'--':>10}"
        print(row)
    if all_costs:
        print(f"  {'Average':<10} ${np.mean(all_costs):>8,.0f}")

    print(f"\n  All outputs for {scenario_name} complete.")


# =============================================================================
# Cross-scenario pregnancy value figure
# =============================================================================

# Visual encoding: scenario → line style/color; parity → marker/linestyle
_CROSS_SCENARIO_STYLES = {
    '2025': ('#1f77b4', '-',  '2025 Baseline'),
    'OG':   ('#2ca02c', '--', 'Oversupply, Good'),
    'OB':   ('#ff7f0e', ':',  'Oversupply, Bad'),
    'UG':   ('#9467bd', '-.',  'Undersupply, Good'),
    'UB':   ('#e377c2', (0,(3,1,1,1)), 'Undersupply, Bad'),
}

_CROSS_PARITY_MARKERS = {
    1: ('o', 'solid',  'Parity 1'),
    2: ('s', 'solid',  'Parity 2'),
    3: ('^', 'solid',  'Parity 3'),
}


def plot_cross_scenario_pregnancy_value(all_scenario_qtables, output_dir):
    """
    Single figure with 3 subplots (one per parity: 1, 2, 3).
    Each subplot shows 5 lines — one per scenario — plotting the
    new-pregnancy value vs. MAC at conception (MAC 3–10).

    Parameters
    ----------
    all_scenario_qtables : dict
        {scenario_code: [q_table, ...]}  — all 5 seeds per scenario.
    output_dir : Path
        Directory where the figure is saved.
    """
    parities_to_plot = [1, 2, 3]
    n_panels = len(parities_to_plot)

    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 5.5),
                             sharey=True)

    for ax_idx, parity in enumerate(parities_to_plot):
        ax = axes[ax_idx]
        marker, _, parity_label = _CROSS_PARITY_MARKERS[parity]

        for scenario_code, q_tables in all_scenario_qtables.items():
            color, linestyle, sc_label = _CROSS_SCENARIO_STYLES.get(
                scenario_code, ('#333333', '-', scenario_code))

            conceived_macs, means, ses = compute_pregnancy_value_by_mac(
                q_tables, parity)
            if len(conceived_macs) == 0:
                continue

            ax.plot(conceived_macs, means,
                    linestyle=linestyle, marker=marker,
                    color=color, linewidth=2, markersize=7,
                    label=sc_label)
            ax.fill_between(conceived_macs,
                            means - ses, means + ses,
                            color=color, alpha=0.10)

        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Month After Calving at Conception (MAC)', fontsize=11)
        ax.set_ylabel('Value of New Pregnancy ($)', fontsize=11)
        ax.set_title(f'Parity {parity}', fontsize=13, fontweight='bold')
        ax.set_xticks(range(3, 11))
        ax.grid(True, alpha=0.3)

        # Only add legend to the first panel to avoid repetition
        if ax_idx == 0:
            ax.legend(fontsize=9, loc='best', framealpha=0.85)

    fig.suptitle(
        'Value of a New Pregnancy by Conception Timing\n'
        'Across Scenarios (Mean \u00b1 SE across seeds)',
        fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()

    out_path = output_dir / 'cross_scenario_pregnancy_value.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')
    return out_path


# =============================================================================
# Flipped layout: 5 subplots (one per scenario), 3 parity lines each
# =============================================================================

# Parity visual encoding for the flipped figure
_PARITY_STYLES = {
    1: ('o', '#1f77b4', '-',  'Parity 1'),
    2: ('s', '#d62728', '--', 'Parity 2'),
    3: ('^', '#2ca02c', ':',  'Parity 3'),
}


def plot_scenario_pregnancy_value_by_parity(all_scenario_qtables, output_dir):
    """
    Figure with 5 subplots arranged in a single row (or 2-row grid),
    one subplot per scenario.  Each subplot shows 3 lines — one per
    parity (1, 2, 3) — plotting new-pregnancy value vs. MAC at conception.

    Parameters
    ----------
    all_scenario_qtables : dict
        {scenario_code: [q_table, ...]}  — all 5 seeds per scenario.
    output_dir : Path
        Directory where the figure is saved.
    """
    scenario_order = [sc for sc in SCENARIOS if sc in all_scenario_qtables]
    n_scenarios = len(scenario_order)
    parities_to_plot = [1, 2, 3]

    # Lay out as a 1×5 row when all 5 scenarios present; gracefully handles fewer
    ncols = min(n_scenarios, 5)
    nrows = 1
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.0 * ncols, 5.5),
                             sharey=True)
    if n_scenarios == 1:
        axes = [axes]
    else:
        axes = list(axes.flatten())

    for ax_idx, scenario_code in enumerate(scenario_order):
        ax = axes[ax_idx]
        q_tables = all_scenario_qtables[scenario_code]
        scenario_name = SCENARIO_NAMES.get(scenario_code, scenario_code)

        for parity in parities_to_plot:
            marker, color, linestyle, parity_label = _PARITY_STYLES[parity]
            conceived_macs, means, ses = compute_pregnancy_value_by_mac(
                q_tables, parity)
            if len(conceived_macs) == 0:
                continue

            ax.plot(conceived_macs, means,
                    linestyle=linestyle, marker=marker,
                    color=color, linewidth=2, markersize=7,
                    label=parity_label)
            ax.fill_between(conceived_macs,
                            means - ses, means + ses,
                            color=color, alpha=0.12)

        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('MAC at Conception', fontsize=11)
        ax.set_ylabel('Value of New Pregnancy ($)', fontsize=11)
        ax.set_title(scenario_name, fontsize=12, fontweight='bold')
        ax.set_xticks(range(3, 11))
        ax.grid(True, alpha=0.3)

        # Legend only on the first subplot
        if ax_idx == 0:
            ax.legend(fontsize=10, loc='best', framealpha=0.85)

    fig.suptitle(
        'Value of a New Pregnancy by Conception Timing\n'
        'Parities 1, 2, 3 — by Scenario (Mean \u00b1 SE across seeds)',
        fontsize=14, fontweight='bold', y=1.03)
    fig.tight_layout()

    out_path = output_dir / 'cross_scenario_pregnancy_value_by_parity.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')
    return out_path


# =============================================================================
# Cross-scenario replacement boundary figure
# =============================================================================

def plot_cross_scenario_culling_boundary(all_scenario_qtables, output_dir):
    """
    Summary line figure: replacement boundary (MAC where Q(keep) < Q(replace))
    for all scenarios, for both healthy-open and mastitis-open cows.

    Layout: 2 panels side by side.
      Left  panel: Healthy, Open  (disease=0, MIP=0)
      Right panel: Mastitis, Open (disease=1, MIP=0)
    Each panel shows one line per scenario (parity on x-axis).

    To avoid overlapping lines when scenarios share the same integer MAC value,
    a small vertical jitter is applied: scenarios are evenly spread across
    [-jitter_range, +jitter_range] MAC units so coincident lines are separated
    without distorting the underlying trend.

    Parameters
    ----------
    all_scenario_qtables : dict
        {scenario_code: [q_table, ...]}  — all seeds per scenario.
    output_dir : Path
        Directory where the figure is saved.
    """
    scenario_order = [sc for sc in ['OG', 'OB','2025', 'UG', 'UB'] if sc in all_scenario_qtables]
    n_sc = len(scenario_order)
    parities_plot = list(range(1, 13))

    # Evenly-spaced vertical offsets across [-0.2, +0.2] MAC units
    jitter_range = 0.2
    if n_sc > 1:
        jitters = np.linspace(-jitter_range, jitter_range, n_sc)
    else:
        jitters = [0.0]

    panel_specs = [
        (0, 'Healthy, Open',  '(a)'),
        (1, 'Mastitis, Open', '(b)'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for disease, panel_title, panel_label in panel_specs:
        ax = axes[disease]
        for sc_idx, scenario_code in enumerate(scenario_order):
            q_tables = all_scenario_qtables[scenario_code]
            color, linestyle, sc_label = _CROSS_SCENARIO_STYLES.get(
                scenario_code, ('#333333', '-', scenario_code))

            boundaries = find_culling_boundary(q_tables, disease=disease)
            jitter = jitters[sc_idx]

            p_vals, b_vals = [], []
            for p in parities_plot:
                val = boundaries.get(p)
                if val is not None:
                    p_vals.append(p)
                    b_vals.append(val + jitter)   # apply vertical jitter

            if p_vals:
                ax.plot(p_vals, b_vals,
                        linestyle=linestyle, marker='o',
                        color=color, linewidth=2, markersize=6,
                        label=sc_label)

        ax.set_xlabel('Parity', fontsize=12)
        ax.set_ylabel('Month After Calving (MAC)', fontsize=12)
        ax.set_title(panel_title, fontsize=13, fontweight='bold')
        ax.set_xticks(parities_plot)
        # Y-ticks at integer MAC values; jitter is sub-unit so labels stay clean
        ymin, ymax = ax.get_ylim()
        ax.set_yticks(range(max(1, int(np.floor(ymin))),
                            int(np.ceil(ymax)) + 1))
        ax.legend(fontsize=10, loc='best', framealpha=0.85)
        ax.grid(True, alpha=0.3)
        ax.text(-0.08, 1.05, panel_label, transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top', ha='left')

    fig.suptitle(
        'Replacement Boundary Across Scenarios\n'
        '(MAC where Q(keep) < Q(replace), open cows; lines offset \u00b10.3 MAC for clarity)',
        fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()

    out_path = output_dir / 'cross_scenario_culling_boundary.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')
    return out_path


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Per-scenario visualization (Option D) + paper tables")
    parser.add_argument("--collected_dir", type=str, default="collected")
    parser.add_argument("--scenario", type=str, default=None, choices=SCENARIOS)
    parser.add_argument("--figures_dir", type=str, default=None)
    parser.add_argument("--tables_dir", type=str, default=None)
    args = parser.parse_args()

    figures_dir = Path(args.figures_dir) if args.figures_dir else FIGURES_ROOT / "scenarios"
    tables_dir = Path(args.tables_dir) if args.tables_dir else TABLES_ROOT
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    scenario_files = auto_discover_files(args.collected_dir, args.scenario)

    if not scenario_files:
        print("ERROR: No Q-table files found. Check --collected_dir path.")
        return

    all_scenario_qtables = {}  # accumulate for cross-scenario figure

    for scenario, files in scenario_files.items():
        q_tables = []
        for f in files:
            if os.path.exists(f):
                print(f"Loading {scenario}: {os.path.basename(f)}")
                qt, _, _ = load_pkl(f)
                q_tables.append(qt)

        if q_tables:
            plot_scenario(scenario, q_tables, figures_dir, tables_dir)
            all_scenario_qtables[scenario] = q_tables

    # Cross-scenario figures
    if len(all_scenario_qtables) > 1:
        print("\nGenerating cross-scenario replacement boundary figure...")
        plot_cross_scenario_culling_boundary(all_scenario_qtables, figures_dir)
        print("Generating cross-scenario pregnancy value figure (3 panels by scenario)...")
        plot_cross_scenario_pregnancy_value(all_scenario_qtables, figures_dir)
        print("Generating cross-scenario pregnancy value figure (5 panels by parity)...")
        plot_scenario_pregnancy_value_by_parity(all_scenario_qtables, figures_dir)

    print(f"\n{'='*70}")
    print(f"Figures saved to: {figures_dir}")
    print(f"Tables saved to:  {tables_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
