"""
Comprehensive Visualization Script for DQN Results - FIXED VERSION

Usage:
    # Single scenario analysis (figures saved under outputs/figures/<stem>)
    python visualize_dqn_results.py --mode single --file outputs/BL_dqn.pkl
    
    # Multi-scenario comparison
    python visualize_dqn_results.py --mode multi --files outputs/BL_dqn.pkl outputs/OG_dqn.pkl outputs/OB_dqn.pkl outputs/UG_dqn.pkl outputs/UB_dqn.pkl
    
    # With smoothing disabled for Q-value plots
    python visualize_dqn_results.py --mode single --file outputs/BL_dqn.pkl --no-smooth
"""

import argparse
import os
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d
import utility


FIGURES_ROOT = Path("outputs") / "figures"
FIGURES_ROOT.mkdir(parents=True, exist_ok=True)

# Configuration for tracking pregnant cows
TRACK_START_MACS = (4, 7, 10)  # MAC values where cows get pregnant
MAX_OPEN_MAC = 19  # Maximum MAC for open cow trajectory


def _slugify(text: str) -> str:
    """Return a filesystem-friendly version of the provided text."""
    cleaned = [ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in str(text)]
    slug = "".join(cleaned).strip("_")
    return slug or "figure"


def _prepare_output_dir(name_parts) -> Path:
    """Create and return the directory under FIGURES_ROOT for the figure."""
    if isinstance(name_parts, (list, tuple)):
        raw_name = "_vs_".join(str(part) for part in name_parts if part)
    else:
        raw_name = str(name_parts)
    target = FIGURES_ROOT / _slugify(raw_name)
    target.mkdir(parents=True, exist_ok=True)
    return target


def _save_and_log(fig: plt.Figure, output_path: Path) -> None:
    """Persist the Matplotlib figure and emit a short log message."""
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {output_path}")


def load_pkl(filename: str):
    """Load Q-table, rewards, and epsilon from pickle file."""
    with open(filename, "rb") as f:
        q_table, rewards, epsilon = pickle.load(f)
    return q_table, rewards, epsilon


def print_q_table(
    q_table: dict,
    max_rows: int | None = None,
    parity_filter: int | None = None,
    disease_filter: int | None = None,
) -> None:
    """Print the Q-table sorted by state; optionally filter and limit rows."""
    rows = []
    for state in sorted(q_table.keys()):
        if parity_filter is not None and state[0] != parity_filter:
            continue
        if disease_filter is not None and state[3] != disease_filter:
            continue
        q_keep = q_table[state].get("keep", 0)
        q_replace = q_table[state].get("replace", 0)
        rows.append((state, q_keep, q_replace, q_keep - q_replace))

    if max_rows is not None:
        rows = rows[:max_rows]

    print("state -> keep, replace, diff")
    for state, q_keep, q_replace, diff in rows:
        print(f"{state} -> {q_keep:.3f}, {q_replace:.3f}, {diff:.3f}")


def _axis_values_from_qtable(q_table, idx: int, fallback_range) -> list:
    """Return sorted unique values for a state index, or the fallback range."""
    vals = sorted({state[idx] for state in q_table})
    return vals if vals else list(fallback_range)


def extract_q_diff_heatmap_by_parity(
    q_table,
    disease_status: int,
    fixed_mip: int = 0,
    parity_range=range(13),
    mac_range=range(21),
    mip_range=range(10),
    disease_range=range(2),
):
    """Return heatmap (parity x MAC) of Q(keep) - Q(replace) at a fixed MIP.

    Rows correspond to parity values; columns correspond to month-after-calving (MAC).
    Impossible states (via utility.possible_state2) are masked with NaN.
    """

    parities_all = _axis_values_from_qtable(q_table, 0, parity_range)
    parities = [p for p in parities_all if p >= 1] or [p for p in parity_range if p >= 1]
    macs_all = _axis_values_from_qtable(q_table, 1, mac_range)
    macs = [m for m in macs_all if m >= 1] or [m for m in mac_range if m >= 1]
    heatmap = np.full((len(parities), len(macs)), np.nan)

    for i_parity, parity in enumerate(parities):
        for j_mac, mac in enumerate(macs):
            state = (parity, mac, fixed_mip, disease_status)
            if not utility.possible_state2(state, parity_range, mac_range, mip_range, disease_range):
                continue
            if state in q_table:
                q_keep = q_table[state].get("keep", 0)
                q_replace = q_table[state].get("replace", 0)
                heatmap[i_parity, j_mac] = q_keep - q_replace

    return heatmap, parities, macs


def extract_q_diff_heatmap_by_mip(
    q_table,
    fixed_parity: int = 1,
    disease_status: int = 0,
    mac_range=range(21),
    mip_range=range(10),
    parity_range=range(13),
    disease_range=range(2),
):
    """Return heatmap (MIP x MAC) of Q(keep) - Q(replace) at a fixed parity.

    Rows correspond to months-in-pregnancy (MIP); columns correspond to MAC.
    Impossible states (via utility.possible_state2) are masked with NaN.
    """

    mips = _axis_values_from_qtable(q_table, 2, mip_range)
    macs_all = _axis_values_from_qtable(q_table, 1, mac_range)
    macs = [m for m in macs_all if m >= 1] or [m for m in mac_range if m >= 1]
    heatmap = np.full((len(mips), len(macs)), np.nan)

    for i_mip, mip in enumerate(mips):
        for j_mac, mac in enumerate(macs):
            state = (fixed_parity, mac, mip, disease_status)
            if not utility.possible_state2(state, parity_range, mac_range, mip_range, disease_range):
                continue
            if state in q_table:
                q_keep = q_table[state].get("keep", 0)
                q_replace = q_table[state].get("replace", 0)
                heatmap[i_mip, j_mac] = q_keep - q_replace

    return heatmap, mips, macs


def smooth_line(data, window: int = 3):
    """Apply simple moving-average smoothing."""
    if len(data) < window:
        return data
    return uniform_filter1d(data, size=window, mode="nearest")


def get_scenario_name(filename: str) -> str:
    """Extract scenario identifier from filename."""
    stem_tokens = Path(filename).stem.upper().split("_")
    if {"BL", "BASELINE", "2025"} & set(stem_tokens):
        return "Baseline"
    if "OG" in stem_tokens:
        return "Oversupply, Good Market"
    if "OB" in stem_tokens:
        return "Oversupply, Bad Market"
    if "UG" in stem_tokens:
        return "Undersupply, Good Market"
    if "UB" in stem_tokens:
        return "Undersupply, Bad Market"
    basename = os.path.basename(filename)
    return basename.replace("_dqn.pkl", "").replace(".pkl", "")


def get_scenario_color(scenario_name: str) -> str:
    """Map scenario names to consistent colors."""
    colors = {
        "Baseline": "#7A7A7A",
        "Oversupply, Bad Market": "#FAA43A",
        "Oversupply, Good Market": "#60BD68",
        "Undersupply, Bad Market": "#F17CB0",
        "Undersupply, Good Market": "#B276B2",
    }
    return colors.get(scenario_name, "#999999")


def track_cow_from_preg(q_table, parity: int, starting_mac: int, disease: int, starting_mip: int = 1):
    """
    Track a cow that gets pregnant at starting_mac.
    Returns lists of (mac, mip, q_diff) as the cow progresses through pregnancy.
    """
    mac_list = []
    mip_list = []
    q_diff_list = []
    
    mac = starting_mac
    mip = starting_mip
    
    # Track through pregnancy (MIP 1-9)
    while mip <= 9 and mac <= 20:
        state = (parity, mac, mip, disease)
        if state in q_table:
            q_keep = q_table[state].get("keep", 0)
            q_replace = q_table[state].get("replace", 0)
            mac_list.append(mac)
            mip_list.append(mip)
            q_diff_list.append(q_keep - q_replace)
        mac += 1
        mip += 1
    
    return mac_list, mip_list, q_diff_list


def track_cow_open(q_table, parity: int, disease: int, max_mac: int = MAX_OPEN_MAC):
    """
    Track an open cow (MIP=0) from MAC=0 to max_mac.
    Returns lists of (mac, q_diff).
    """
    mac_list = []
    q_diff_list = []
    
    for mac in range(0, max_mac + 1):
        state = (parity, mac, 0, disease)  # MIP=0 for open cow
        if state in q_table:
            q_keep = q_table[state].get("keep", 0)
            q_replace = q_table[state].get("replace", 0)
            mac_list.append(mac)
            q_diff_list.append(q_keep - q_replace)
    
    return mac_list, q_diff_list


def plot_qvalue_diff_correct(q_table, parity: int, disease: int, title: str,
                            smooth: bool = True, ax=None):
    """
    Plot Q-value difference correctly:
    - Lines for cows that get pregnant at different MAC values (tracking through pregnancy)
    - Reference line for open cow (MIP=0)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure
    mac_range = range(0, 21)
    
    # Color palette for pregnant cow tracks
    track_colors = ["#1f77b4", "#f4a2cd", "#f28e2b"]
    
    # Plot pregnant cow tracks
    for i, start_mac in enumerate(TRACK_START_MACS):
        mac_list, mip_list, q_diff_list = track_cow_from_preg(q_table, parity, start_mac, disease)
        
        if len(mac_list) > 0:
            color = track_colors[i % len(track_colors)]
            
            # Plot raw points
            ax.plot(mac_list, q_diff_list, "o", alpha=0.3, markersize=3, color=color)
            
            # Plot smoothed line if requested
            if smooth and len(q_diff_list) > 3:
                smoothed = smooth_line(np.array(q_diff_list), window=3)
                ax.plot(mac_list, smoothed, "-", linewidth=2, color=color,
                       label=f"at MIP = {start_mac}")
    
    # Plot open cow reference line
    mac_list_open, q_diff_list_open = track_cow_open(q_table, parity, disease)
    
    if len(mac_list_open) > 0:
        # Plot raw points
        ax.plot(mac_list_open, q_diff_list_open, "o", alpha=0.3, markersize=3, color="gray")
        
        # Plot smoothed line if requested
        if smooth and len(q_diff_list_open) > 3:
            smoothed_open = smooth_line(np.array(q_diff_list_open), window=3)
            ax.plot(mac_list_open, smoothed_open, "--", linewidth=2, color="black",
                   label=f"Open cow")
    
    # Styling
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Month After Calving (MAC)", fontsize=10)
    ax.set_xticks(mac_range)
    ax.set_xlim(0, 18.5)
    ax.set_ylim(-1100, 1200)
    ax.set_ylabel("Q-value difference\n(Keep - Replace)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    ax.text(0.98, 0.02, "Above 0: Keep\nBelow 0: Cull",
            transform=ax.transAxes, fontsize=8,
            verticalalignment="bottom", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.3))
    
    return fig


def _create_combined_overview(q_table, rewards, scenario_name: str, smooth: bool) -> plt.Figure:
    """Create the combined overview figure with corrected Q-value difference plots."""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Reward convergence (top row)
    ax_conv = fig.add_subplot(gs[0, :])
    episodes = np.arange(len(rewards))
    ax_conv.plot(episodes, rewards, alpha=0.3, linewidth=0.5, color="gray", label="Raw rewards")
    if len(rewards) >= 100:
        ma_100 = uniform_filter1d(rewards, size=min(100, len(rewards)), mode="nearest")
        ax_conv.plot(episodes, ma_100, linewidth=2, color="blue", label="MA(100)")
    if len(rewards) >= 1000:
        ma_1000 = uniform_filter1d(rewards, size=min(1000, len(rewards)), mode="nearest")
        ax_conv.plot(episodes, ma_1000, linewidth=3, color="red", label="MA(1000)")
    ax_conv.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax_conv.set_xlabel("Episode", fontsize=14)
    ax_conv.set_ylabel("Total Reward ($)", fontsize=14)
    ax_conv.set_title(f"{scenario_name} - Reward Convergence", fontsize=16, fontweight="bold")
    ax_conv.legend(fontsize=12)
    ax_conv.grid(True, alpha=0.3)
    if len(rewards):
        span = rewards[-min(1000, len(rewards)) :]
        stats_text = f"Final 1K: Mean=${np.mean(span):.0f}, Std=${np.std(span):.0f}"
        ax_conv.text(0.02, 0.98, stats_text, transform=ax_conv.transAxes,
                     fontsize=12, verticalalignment="top",
                     bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Heatmap (middle row, center)
    ax_heatmap = fig.add_subplot(gs[1, 1])
    heatmap, parities, macs = extract_q_diff_heatmap_by_parity(q_table, disease_status=0, fixed_mip=0)
    vmax = np.nanmax(np.abs(heatmap)) if np.any(~np.isnan(heatmap)) else 1
    im = ax_heatmap.imshow(heatmap, aspect="auto", cmap="RdYlGn", origin="lower",
                           vmin=-vmax, vmax=vmax)
    ax_heatmap.set_xlabel("Month After Calving (MAC)", fontsize=12)
    ax_heatmap.set_ylabel("Parity", fontsize=12)
    ax_heatmap.set_title(f"{scenario_name}\n(Open Cows, Healthy)", fontsize=14, fontweight="bold")
    ax_heatmap.set_xticks(np.arange(len(macs)))
    ax_heatmap.set_xticklabels(macs)
    ax_heatmap.set_yticks(np.arange(len(parities)))  # Adjusted to match the number of parities
    ax_heatmap.set_yticklabels(parities)
    cbar = fig.colorbar(im, ax=ax_heatmap)
    cbar.set_label("Q-value\n(Keep - Replace)", fontsize=11)
    if np.any(~np.isnan(heatmap)):
        range_text = f"Range: [{np.nanmin(heatmap):.0f}, {np.nanmax(heatmap):.0f}]"
        ax_heatmap.text(0.02, 0.98, range_text, transform=ax_heatmap.transAxes,
                        fontsize=10, verticalalignment="top",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # Line charts around the heatmap - CORRECTED VERSION
    line_axes = [
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 2]),
        fig.add_subplot(gs[2, 0]),
        fig.add_subplot(gs[2, 1]),
        fig.add_subplot(gs[2, 2]),
    ]
    configs = [
        (1, 0, "Parity 1, Healthy"),
        (1, 1, "Parity 1, Sick (Mastitis)"),
        (2, 0, "Parity 2, Healthy"),
        (2, 1, "Parity 2, Sick (Mastitis)"),
        (3, 0, "Parity 3, Healthy"),
    ]
    
    for ax, (parity, disease, title) in zip(line_axes, configs):
        plot_qvalue_diff_correct(q_table, parity, disease, title, smooth=smooth, ax=ax)

    fig.suptitle(f"DQN Analysis: {scenario_name}", fontsize=18, fontweight="bold", y=0.995)
    return fig


def plot_single_scenario(filename: str, smooth: bool = True) -> None:
    """Create per-panel analysis for a single scenario and save each figure separately."""
    print(f"\nAnalyzing {filename}...")

    q_table, rewards, epsilon = load_pkl(filename)
    scenario_name = get_scenario_name(filename)
    file_stem = Path(filename).stem
    slug_base = _slugify(file_stem)
    output_dir = _prepare_output_dir(file_stem)
    saved_paths = []

    def save_panel(fig: plt.Figure, suffix: str) -> None:
        path = output_dir / f"{slug_base}_{suffix}.png"
        _save_and_log(fig, path)
        saved_paths.append(path)

    # Reward convergence -------------------------------------------------
    fig_conv, ax_conv = plt.subplots(figsize=(12, 5))
    episodes = np.arange(len(rewards))
    ax_conv.plot(episodes, rewards, alpha=0.3, linewidth=0.5, color="gray", label="Raw rewards")

    if len(rewards) >= 100:
        ma_100 = uniform_filter1d(rewards, size=min(100, len(rewards)), mode="nearest")
        ax_conv.plot(episodes, ma_100, linewidth=2, color="blue", label="MA(100)")

    if len(rewards) >= 1000:
        ma_1000 = uniform_filter1d(rewards, size=min(1000, len(rewards)), mode="nearest")
        ax_conv.plot(episodes, ma_1000, linewidth=3, color="red", label="MA(1000)")

    ax_conv.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax_conv.set_xlabel("Episode", fontsize=14)
    ax_conv.set_ylabel("Total Reward ($)", fontsize=14)
    ax_conv.set_title(f"{scenario_name} - Reward Convergence", fontsize=16, fontweight="bold")
    ax_conv.legend(fontsize=12)
    ax_conv.grid(True, alpha=0.3)

    final_1000 = rewards[-min(1000, len(rewards)):] if len(rewards) else []
    if len(final_1000):
        stats_text = f"Final 1K: Mean=${np.mean(final_1000):.0f}, Std=${np.std(final_1000):.0f}"
        ax_conv.text(0.02, 0.98, stats_text, transform=ax_conv.transAxes,
                     fontsize=12, verticalalignment="top",
                     bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    save_panel(fig_conv, "reward_convergence")

    # Confidence heatmap (parity x MAC, MIP fixed) ----------------------
    fig_heat, ax_heatmap = plt.subplots(figsize=(9, 7))
    heatmap, parities, macs = extract_q_diff_heatmap_by_parity(q_table, disease_status=0, fixed_mip=0)
    if np.any(~np.isnan(heatmap)):
        vmax = np.nanmax(np.abs(heatmap))
    else:
        vmax = 1
    im = ax_heatmap.imshow(heatmap, aspect="auto", cmap="RdYlGn", origin="lower",
                           vmin=-vmax, vmax=vmax)

    ax_heatmap.set_xlabel("Month After Calving (MAC)", fontsize=12)
    ax_heatmap.set_ylabel("Parity", fontsize=12)
    ax_heatmap.set_title(f"{scenario_name}\n(MIP = 0, Healthy)", fontsize=14, fontweight="bold")
    ax_heatmap.set_xticks(np.arange(len(macs)))
    ax_heatmap.set_xticklabels(macs)
    ax_heatmap.set_yticks(np.arange(len(parities)))
    ax_heatmap.set_yticklabels(parities)

    cbar = fig_heat.colorbar(im, ax=ax_heatmap)
    cbar.set_label("Q-value\n(Keep - Replace)", fontsize=11)

    if np.any(~np.isnan(heatmap)):
        range_text = f"Range: [{np.nanmin(heatmap):.0f}, {np.nanmax(heatmap):.0f}]"
        ax_heatmap.text(0.02, 0.98, range_text, transform=ax_heatmap.transAxes,
                        fontsize=10, verticalalignment="top",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    save_panel(fig_heat, "heatmap_open_healthy")

    # Confidence heatmap (MIP x MAC, parity fixed) ----------------------
    fig_heat_mip, ax_heatmap_mip = plt.subplots(figsize=(9, 7))
    heatmap_mip, mips, macs_mip = extract_q_diff_heatmap_by_mip(q_table, fixed_parity=1, disease_status=0)
    if np.any(~np.isnan(heatmap_mip)):
        vmax_mip = np.nanmax(np.abs(heatmap_mip))
    else:
        vmax_mip = 1
    im_mip = ax_heatmap_mip.imshow(heatmap_mip, aspect="auto", cmap="RdYlGn", origin="lower",
                                   vmin=-vmax_mip, vmax=vmax_mip)

    ax_heatmap_mip.set_xlabel("Month After Calving (MAC)", fontsize=12)
    ax_heatmap_mip.set_ylabel("Month In Pregnancy (MIP)", fontsize=12)
    ax_heatmap_mip.set_title(f"{scenario_name}\n(Parity = 1, Healthy)", fontsize=14, fontweight="bold")
    ax_heatmap_mip.set_xticks(np.arange(len(macs_mip)))
    ax_heatmap_mip.set_xticklabels(macs_mip)
    ax_heatmap_mip.set_yticks(np.arange(len(mips)))
    ax_heatmap_mip.set_yticklabels(mips)

    cbar_mip = fig_heat_mip.colorbar(im_mip, ax=ax_heatmap_mip)
    cbar_mip.set_label("Q-value\n(Keep - Replace)", fontsize=11)

    if np.any(~np.isnan(heatmap_mip)):
        range_text_mip = f"Range: [{np.nanmin(heatmap_mip):.0f}, {np.nanmax(heatmap_mip):.0f}]"
        ax_heatmap_mip.text(0.02, 0.98, range_text_mip, transform=ax_heatmap_mip.transAxes,
                            fontsize=10, verticalalignment="top",
                            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    save_panel(fig_heat_mip, "heatmap_mip_parity1_healthy")

    # Q-value difference line plots - CORRECTED VERSION ------------------
    configs = [
        (1, 0, "Parity 1, Healthy"),
        (1, 1, "Parity 1, Sick (Mastitis)"),
        (2, 0, "Parity 2, Healthy"),
        (2, 1, "Parity 2, Sick (Mastitis)"),
        (3, 0, "Parity 3, Healthy"),
        (3, 1, "Parity 3, Sick (Mastitis)"),
        # (4, 0, "Parity 4, Healthy"),
        # (4, 1, "Parity 4, Sick (Mastitis)"),
        # (5, 0, "Parity 5, Healthy"),
        # (5, 1, "Parity 5, Sick (Mastitis)"),
        # (10, 0, "Parity 10, Healthy"),
        # (10, 1, "Parity 10, Sick (Mastitis)"),
        # (11, 0, "Parity 11, Healthy"),
        # (11, 1, "Parity 11, Sick (Mastitis)"),
        # (12, 0, "Parity 12, Healthy"),
        # (12, 1, "Parity 12, Sick (Mastitis)"),
    ]

    for parity, disease, title in configs:
        fig_line = plot_qvalue_diff_correct(q_table, parity, disease, title, smooth=smooth)
        save_panel(fig_line, f"qvalue_diff_parity{parity}_disease{disease}")

    # Combined overview figure -------------------------------------------
    fig_combined = _create_combined_overview(q_table, rewards, scenario_name, smooth)
    save_panel(fig_combined, "combined_overview")

    print(f"\nSaved {len(saved_paths)} figures to {output_dir}")
    print("=" * 70)


def plot_multi_scenario(filenames: list, smooth: bool = True) -> None:
    """Create multi-scenario comparison plots."""
    print(f"\nComparing {len(filenames)} scenarios...")
    
    # Load all data
    data = []
    for filename in filenames:
        q_table, rewards, epsilon = load_pkl(filename)
        scenario_name = get_scenario_name(filename)
        data.append({
            "filename": filename,
            "scenario": scenario_name,
            "q_table": q_table,
            "rewards": rewards,
            "epsilon": epsilon,
            "color": get_scenario_color(scenario_name)
        })
    
    # Create output directory
    scenario_names = [d["scenario"] for d in data]
    output_dir = _prepare_output_dir(scenario_names)
    slug_base = _slugify("_vs_".join([Path(f).stem for f in filenames]))
    
    # Create 2x2 comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    
    # Top left: Raw rewards (first 10K episodes)
    ax = axes[0, 0]
    max_episodes = min(10000, min(len(d["rewards"]) for d in data))
    for d in data:
        episodes = np.arange(max_episodes)
        ax.plot(episodes, d["rewards"][:max_episodes], alpha=0.6, linewidth=0.5,
               color=d["color"], label=d["scenario"])
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Episode", fontsize=14)
    ax.set_ylabel("Reward ($)", fontsize=14)
    ax.set_title("Raw Rewards - First 10K Episodes (High Variance is Normal)", fontsize=16, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Top right: Distribution of rewards (last 100K episodes)
    ax = axes[0, 1]
    for d in data:
        last_100k = d["rewards"][-min(100000, len(d["rewards"])):]
        ax.hist(last_100k, bins=50, alpha=0.6, color=d["color"], label=d["scenario"])
    ax.set_xlabel("Reward ($)", fontsize=14)
    ax.set_ylabel("Frequency", fontsize=14)
    ax.set_title("Distribution of Rewards - Last 100K Episodes", fontsize=16, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Bottom left: Convergence comparison (moving averages)
    ax = axes[1, 0]
    for d in data:
        episodes = np.arange(len(d["rewards"]))
        if len(d["rewards"]) >= 1000:
            ma_1000 = uniform_filter1d(d["rewards"], size=1000, mode="nearest")
            ax.plot(episodes, ma_1000, linewidth=2, color=d["color"], label=d["scenario"])
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Episode", fontsize=14)
    ax.set_ylabel("Reward ($)", fontsize=14)
    ax.set_title("Convergence Comparison (MA 1000)", fontsize=16, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Bottom right: Final performance ranking
    ax = axes[1, 1]
    final_means = []
    final_stds = []
    labels = []
    colors = []
    for d in data:
        last_100k = d["rewards"][-min(100000, len(d["rewards"])):]
        final_means.append(np.mean(last_100k))
        final_stds.append(np.std(last_100k))
        labels.append(d["scenario"])
        colors.append(d["color"])
    
    # Sort by mean (descending)
    sorted_indices = np.argsort(final_means)[::-1]
    final_means = [final_means[i] for i in sorted_indices]
    final_stds = [final_stds[i] for i in sorted_indices]
    labels = [labels[i] for i in sorted_indices]
    colors = [colors[i] for i in sorted_indices]
    
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, final_means, xerr=final_stds, color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel("Mean Reward ($)", fontsize=14)
    ax.set_title("Final Performance Ranking", fontsize=16, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    
    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(final_means, final_stds)):
        ax.text(mean, i, f"  ${mean:.0f} ± ${std:.0f}", va="center", fontsize=10)
    
    fig.suptitle("Multi-Scenario Comparison", fontsize=20, fontweight="bold", y=0.995)
    fig.tight_layout()
    
    output_path = output_dir / f"{slug_base}_comparison.png"
    _save_and_log(fig, output_path)
    
    print(f"\nSaved comparison figure to {output_dir}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Visualize DQN training results")
    parser.add_argument("--mode", choices=["single", "multi"], required=True,
                       help="Visualization mode: single scenario or multi-scenario comparison")
    parser.add_argument("--file", type=str, help="Path to single .pkl file (for single mode)")
    parser.add_argument("--files", nargs="+", help="Paths to multiple .pkl files (for multi mode)")
    parser.add_argument("--no-smooth", action="store_true", help="Disable smoothing for Q-value plots")
    
    args = parser.parse_args()
    
    smooth = not args.no_smooth
    
    if args.mode == "single":
        if not args.file:
            parser.error("--file is required for single mode")
        q_table, _, _ = load_pkl(args.file)
        print_q_table(q_table, max_rows=100, parity_filter=1, disease_filter=0)
        print_q_table(q_table, max_rows=100, parity_filter=11, disease_filter=0)
        print_q_table(q_table, max_rows=200, parity_filter=12, disease_filter=0)
        plot_single_scenario(args.file, smooth=smooth)
    
    elif args.mode == "multi":
        if not args.files or len(args.files) < 2:
            parser.error("--files requires at least 2 .pkl files for multi mode")
        plot_multi_scenario(args.files, smooth=smooth)



if __name__ == "__main__":
    main()
