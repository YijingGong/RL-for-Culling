"""
Enhanced Multi-Run Visualization Script for DQN Results

This script handles multiple independent runs per scenario and computes proper statistics.
It aggregates runs by scenario and shows mean ± confidence intervals.

Usage:
    # Compare 5 scenarios, each with 5 runs (25 files total)
    python visualize_dqn_multi_runs.py \
        --scenario-files \
        2025:outputs/DQN_2025_500k_run1.pkl,outputs/DQN_2025_500k_run2.pkl,outputs/DQN_2025_500k_run3.pkl,outputs/DQN_2025_500k_run4.pkl,outputs/DQN_2025_500k_run5.pkl \
        OG:outputs/DQN_OG_500k_run1.pkl,outputs/DQN_OG_500k_run2.pkl,outputs/DQN_OG_500k_run3.pkl,outputs/DQN_OG_500k_run4.pkl,outputs/DQN_OG_500k_run5.pkl \
        OB:outputs/DQN_OB_500k_run1.pkl,outputs/DQN_OB_500k_run2.pkl,outputs/DQN_OB_500k_run3.pkl,outputs/DQN_OB_500k_run4.pkl,outputs/DQN_OB_500k_run5.pkl \
        UG:outputs/DQN_UG_500k_run1.pkl,outputs/DQN_UG_500k_run2.pkl,outputs/DQN_UG_500k_run3.pkl,outputs/DQN_UG_500k_run4.pkl,outputs/DQN_UG_500k_run5.pkl \
        UB:outputs/DQN_UB_500k_run1.pkl,outputs/DQN_UB_500k_run2.pkl,outputs/DQN_UB_500k_run3.pkl,outputs/DQN_UB_500k_run4.pkl,outputs/DQN_UB_500k_run5.pkl

    python visualize_dqn_multi_runs.py \
        --scenario-files \
        2025:outputs/DQN_2025_500k_run1.pkl,outputs/DQN_2025_500k_run2.pkl,outputs/DQN_2025_500k_run3.pkl \
        OG:outputs/DQN_OG_500k_run1.pkl,outputs/DQN_OG_500k_run2.pkl,outputs/DQN_OG_500k_run3.pkl \
        OB:outputs/DQN_OB_500k_run1.pkl,outputs/DQN_OB_500k_run2.pkl,outputs/DQN_OB_500k_run3.pkl \
        UG:outputs/DQN_UG_500k_run1.pkl,outputs/DQN_UG_500k_run2.pkl,outputs/DQN_UG_500k_run3.pkl \
        UB:outputs/DQN_UB_500k_run1.pkl,outputs/DQN_UB_500k_run2.pkl,outputs/DQN_UB_500k_run3.pkl
"""

import argparse
import os
from pathlib import Path
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy import stats


FIGURES_ROOT = Path("outputs") / "figures"
FIGURES_ROOT.mkdir(parents=True, exist_ok=True)


def load_pkl(filename: str):
    """Load Q-table, rewards, and epsilon from pickle file."""
    with open(filename, "rb") as f:
        q_table, rewards, epsilon = pickle.load(f)
    return q_table, rewards, epsilon


def get_scenario_display_name(scenario_code: str) -> str:
    """Map scenario codes to display names."""
    name_map = {
        "2025": "Baseline",
        "BL": "Baseline",
        "OG": "Oversupply, Good Market",
        "OB": "Oversupply, Bad Market",
        "UG": "Undersupply, Good Market",
        "UB": "Undersupply, Bad Market",
    }
    return name_map.get(scenario_code.upper(), scenario_code)


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


def compute_confidence_interval(data, confidence=0.95):
    """
    Compute mean and confidence interval for data.
    
    Args:
        data: array-like, shape (n_runs, n_points)
        confidence: confidence level (default 0.95 for 95% CI)
    
    Returns:
        mean, lower_bound, upper_bound
    """
    data = np.array(data)
    n = len(data)
    mean = np.mean(data, axis=0)
    sem = stats.sem(data, axis=0)  # Standard error of the mean
    ci = sem * stats.t.ppf((1 + confidence) / 2., n - 1)
    return mean, mean - ci, mean + ci


def align_rewards_to_shortest(rewards_list):
    """
    Align multiple reward arrays to the shortest length.
    
    Args:
        rewards_list: list of reward arrays (possibly different lengths)
    
    Returns:
        numpy array of shape (n_runs, min_length)
    """
    min_length = min(len(r) for r in rewards_list)
    aligned = np.array([r[:min_length] for r in rewards_list])
    return aligned


def plot_multi_scenario_with_runs(scenario_data: dict, output_dir: Path):
    """
    Create comprehensive comparison plots for multiple scenarios with multiple runs each.
    
    Args:
        scenario_data: dict mapping scenario_code -> list of (q_table, rewards, epsilon) tuples
        output_dir: directory to save figures
    """
    
    # Prepare data structures
    scenario_stats = {}
    
    for scenario_code, runs in scenario_data.items():
        scenario_name = get_scenario_display_name(scenario_code)
        color = get_scenario_color(scenario_name)
        
        # Extract rewards from all runs
        all_rewards = [run[1] for run in runs]  # run[1] is rewards
        
        # Align to shortest run
        aligned_rewards = align_rewards_to_shortest(all_rewards)
        
        # Compute statistics for convergence plot (MA 1000)
        ma_1000_list = []
        for rewards in all_rewards:
            if len(rewards) >= 1000:
                ma = uniform_filter1d(rewards, size=1000, mode="nearest")
                ma_1000_list.append(ma)
        
        if ma_1000_list:
            aligned_ma = align_rewards_to_shortest(ma_1000_list)
            ma_mean, ma_lower, ma_upper = compute_confidence_interval(aligned_ma)
        else:
            ma_mean = ma_lower = ma_upper = None
        
        # Compute final performance statistics (last 100k episodes)
        final_means = []
        final_stds = []
        for rewards in all_rewards:
            last_100k = rewards[-min(100000, len(rewards)):]
            final_means.append(np.mean(last_100k))
            final_stds.append(np.std(last_100k))
        
        # Aggregate final performance across runs
        overall_mean = np.mean(final_means)
        overall_std = np.std(final_means)  # Std across runs (not within run)
        
        scenario_stats[scenario_code] = {
            "name": scenario_name,
            "color": color,
            "aligned_rewards": aligned_rewards,
            "all_rewards": all_rewards,
            "ma_mean": ma_mean,
            "ma_lower": ma_lower,
            "ma_upper": ma_upper,
            "final_means": final_means,
            "final_stds": final_stds,
            "overall_mean": overall_mean,
            "overall_std": overall_std,
            "n_runs": len(runs)
        }
    
    # Create the comprehensive comparison figure
    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
    
    # ========================================================================
    # Plot 1: Raw rewards - First 10K episodes (all runs, all scenarios)
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    for scenario_code, stats in scenario_stats.items():
        for i, rewards in enumerate(stats["all_rewards"]):
            max_ep = min(10000, len(rewards))
            episodes = np.arange(max_ep)
            label = stats["name"] if i == 0 else None  # Only label first run
            ax1.plot(episodes, rewards[:max_ep], alpha=0.3, linewidth=0.5,
                    color=stats["color"], label=label)
    
    ax1.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax1.set_xlabel("Episode", fontsize=12)
    ax1.set_ylabel("Reward ($)", fontsize=12)
    ax1.set_title("Raw Rewards - First 10K Episodes\n(All Runs, High Variance is Normal)",
                 fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10, loc="lower right")
    ax1.grid(True, alpha=0.3)
    
    # ========================================================================
    # Plot 2: Convergence with confidence intervals (MA 1000)
    # ========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    for scenario_code, stats in scenario_stats.items():
        if stats["ma_mean"] is not None:
            episodes = np.arange(len(stats["ma_mean"]))
            ax2.plot(episodes, stats["ma_mean"], linewidth=2.5,
                    color=stats["color"], label=stats["name"])
            ax2.fill_between(episodes, stats["ma_lower"], stats["ma_upper"],
                            alpha=0.2, color=stats["color"])
    
    ax2.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax2.set_xlabel("Episode", fontsize=12)
    ax2.set_ylabel("Reward ($)", fontsize=12)
    ax2.set_title("Convergence Comparison (MA 1000)\nMean ± 95% CI across runs",
                 fontsize=14, fontweight="bold")
    ax2.legend(fontsize=10, loc="lower right")
    ax2.grid(True, alpha=0.3)
    
    # ========================================================================
    # Plot 3: Distribution of rewards - Last 100K episodes (overlapping histograms)
    # ========================================================================
    ax3 = fig.add_subplot(gs[0, 2])
    for scenario_code, stats in scenario_stats.items():
        # Pool all rewards from last 100k across all runs
        pooled_rewards = []
        for rewards in stats["all_rewards"]:
            last_100k = rewards[-min(100000, len(rewards)):]
            pooled_rewards.extend(last_100k)
        
        ax3.hist(pooled_rewards, bins=60, alpha=0.5, color=stats["color"],
                label=stats["name"], density=False)
    
    ax3.set_xlabel("Reward ($)", fontsize=12)
    ax3.set_ylabel("Frequency", fontsize=12)
    ax3.set_title("Distribution of Rewards - Last 100K Episodes\n(Pooled across all runs)",
                 fontsize=14, fontweight="bold")
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # ========================================================================
    # Plot 4: Final performance ranking (mean across runs with error bars)
    # ========================================================================
    ax4 = fig.add_subplot(gs[1, :])
    
    # Sort scenarios by overall mean
    sorted_scenarios = sorted(scenario_stats.items(),
                             key=lambda x: x[1]["overall_mean"],
                             reverse=True)
    
    labels = [stats["name"] for _, stats in sorted_scenarios]
    means = [stats["overall_mean"] for _, stats in sorted_scenarios]
    stds = [stats["overall_std"] for _, stats in sorted_scenarios]
    colors = [stats["color"] for _, stats in sorted_scenarios]
    n_runs = [stats["n_runs"] for _, stats in sorted_scenarios]
    
    y_pos = np.arange(len(labels))
    ax4.barh(y_pos, means, xerr=stds, color=colors, alpha=0.7, capsize=5)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(labels, fontsize=12)
    ax4.set_xlabel("Mean Reward ($)", fontsize=14)
    ax4.set_title("Final Performance Ranking\n(Mean ± Std across runs, last 100K episodes)",
                 fontsize=16, fontweight="bold")
    ax4.grid(True, alpha=0.3, axis="x")
    
    # Add value labels on bars
    for i, (mean, std, n) in enumerate(zip(means, stds, n_runs)):
        ax4.text(mean, i, f"  ${mean:.0f} ± ${std:.0f} (n={n})",
                va="center", fontsize=11, fontweight="bold")
    
    # ========================================================================
    # Plot 5: Violin plot of final performance distribution
    # ========================================================================
    ax5 = fig.add_subplot(gs[2, 0])
    
    violin_data = []
    violin_labels = []
    violin_colors = []
    
    for scenario_code, stats in sorted_scenarios:
        violin_data.append(stats["final_means"])
        violin_labels.append(stats["name"])
        violin_colors.append(stats["color"])
    
    parts = ax5.violinplot(violin_data, positions=range(len(violin_data)),
                          showmeans=True, showmedians=True, vert=False)
    
    # Color the violins
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(violin_colors[i])
        pc.set_alpha(0.6)
    
    ax5.set_yticks(range(len(violin_labels)))
    ax5.set_yticklabels(violin_labels, fontsize=10)
    ax5.set_xlabel("Mean Reward ($)", fontsize=12)
    ax5.set_title("Distribution of Final Performance\n(Across independent runs)",
                 fontsize=14, fontweight="bold")
    ax5.grid(True, alpha=0.3, axis="x")
    
    # ========================================================================
    # Plot 6: Statistical summary table
    # ========================================================================
    ax6 = fig.add_subplot(gs[2, 1:])
    ax6.axis("off")
    
    # Create summary table
    table_data = []
    table_data.append(["Scenario", "N Runs", "Mean Reward", "Std (across runs)", "Min", "Max"])
    
    for scenario_code, stats in sorted_scenarios:
        row = [
            stats["name"],
            f"{stats['n_runs']}",
            f"${stats['overall_mean']:.0f}",
            f"${stats['overall_std']:.0f}",
            f"${min(stats['final_means']):.0f}",
            f"${max(stats['final_means']):.0f}"
        ]
        table_data.append(row)
    
    table = ax6.table(cellText=table_data, cellLoc="center", loc="center",
                     colWidths=[0.3, 0.1, 0.15, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(len(table_data[0])):
        cell = table[(0, i)]
        cell.set_facecolor("#4CAF50")
        cell.set_text_props(weight="bold", color="white")
    
    # Color rows by scenario
    for i, (scenario_code, stats) in enumerate(sorted_scenarios, start=1):
        for j in range(len(table_data[0])):
            cell = table[(i, j)]
            cell.set_facecolor(stats["color"])
            cell.set_alpha(0.3)
    
    ax6.set_title("Statistical Summary\n(Last 100K episodes per run)",
                 fontsize=14, fontweight="bold", pad=20)
    
    # ========================================================================
    # Final touches
    # ========================================================================
    fig.suptitle("Multi-Scenario Comparison with Multiple Runs per Scenario",
                fontsize=20, fontweight="bold", y=0.995)
    
    # Save figure
    output_path = output_dir / "multi_scenario_multi_run_comparison.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n✓ Saved comprehensive comparison to: {output_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Visualize DQN results with multiple runs per scenario",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python visualize_dqn_multi_runs.py --scenario-files \\
    2025:outputs/DQN_2025_500k_run1.pkl,outputs/DQN_2025_500k_run2.pkl,outputs/DQN_2025_500k_run3.pkl \\
    OG:outputs/DQN_OG_500k_run1.pkl,outputs/DQN_OG_500k_run2.pkl,outputs/DQN_OG_500k_run3.pkl
        """
    )
    
    parser.add_argument(
        "--scenario-files",
        nargs="+",
        required=True,
        help="Scenario files in format 'SCENARIO_CODE:file1.pkl,file2.pkl,file3.pkl'. "
             "Example: 2025:run1.pkl,run2.pkl OG:run1.pkl,run2.pkl"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for figures (default: outputs/figures/multi_run_comparison)"
    )
    
    args = parser.parse_args()
    
    # Parse scenario files
    scenario_data = {}
    
    for scenario_spec in args.scenario_files:
        if ":" not in scenario_spec:
            print(f"Error: Invalid format '{scenario_spec}'. Expected 'SCENARIO:file1,file2,...'")
            return
        
        scenario_code, files_str = scenario_spec.split(":", 1)
        files = [f.strip() for f in files_str.split(",")]
        
        # Load all runs for this scenario
        runs = []
        for filename in files:
            if not os.path.exists(filename):
                print(f"Warning: File not found: {filename}")
                continue
            
            print(f"Loading {scenario_code}: {filename}")
            q_table, rewards, epsilon = load_pkl(filename)
            runs.append((q_table, rewards, epsilon))
        
        if runs:
            scenario_data[scenario_code] = runs
            print(f"  → Loaded {len(runs)} runs for scenario '{scenario_code}'")
    
    if not scenario_data:
        print("Error: No valid scenario data loaded.")
        return
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = FIGURES_ROOT / "multi_run_comparison"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    print(f"\nGenerating comparison plots...")
    print(f"Scenarios: {list(scenario_data.keys())}")
    print(f"Total runs: {sum(len(runs) for runs in scenario_data.values())}")
    
    plot_multi_scenario_with_runs(scenario_data, output_dir)
    
    print(f"\n{'='*70}")
    print(f"All figures saved to: {output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
