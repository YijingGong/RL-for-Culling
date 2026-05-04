"""
Cross-Scenario Summary Visualization — Option B
(4-panel, no embedded table; table goes in paper body)

Layout:
  (a) Training Convergence (MA 1000, Mean +/- 95% CI)
  (b) Annualized Cow Stall Value (bar chart)
  (c) Per-Parity Discounted Stall Value
  (d) Steady-State Herd Parity Distribution

Usage:
    python visualize_summary.py --collected_dir collected/
"""

import argparse
import os
import glob
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy import stats


FIGURES_ROOT = Path("outputs") / "figures"

SCENARIOS = ['2025', 'OG', 'OB', 'UG', 'UB']
SEEDS = [42, 123, 456, 789, 1024]

SCENARIO_SHORT = {
    "2025": "2025 Baseline",
    "OG": "Oversupply, Good",
    "OB": "Oversupply, Bad",
    "UG": "Undersupply, Good",
    "UB": "Undersupply, Bad",
}

SCENARIO_COLORS = {
    "2025": "#5B8DB8",
    "OG": "#60BD68",
    "OB": "#FAA43A",
    "UG": "#B276B2",
    "UB": "#F17CB0",
}


def load_pkl(filename):
    with open(filename, "rb") as f:
        q_table, rewards, epsilon = pickle.load(f)
    return q_table, rewards, epsilon


def load_eval_pkl(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data


def compute_ci(data, confidence=0.95):
    data = np.array(data)
    n = len(data)
    mean = np.mean(data, axis=0)
    if n > 1:
        sem = stats.sem(data, axis=0)
        ci = sem * stats.t.ppf((1 + confidence) / 2., n - 1)
    else:
        ci = np.zeros_like(mean)
    return mean, mean - ci, mean + ci


def add_subplot_label(ax, label, fontsize=16, x=-0.08, y=1.05):
    ax.text(x, y, f"({label})", transform=ax.transAxes,
            fontsize=fontsize, fontweight='bold', va='top', ha='left')


def auto_discover_files(collected_dir):
    scenario_files = {}
    for scenario in SCENARIOS:
        files = []
        for seed in SEEDS:
            for pattern in [
                os.path.join(collected_dir, f"DQN_{scenario}_seed{seed}.pkl"),
                os.path.join(collected_dir, f"DQN_{scenario}_{seed}.pkl"),
            ]:
                if os.path.exists(pattern):
                    files.append(pattern)
                    break
        if files:
            scenario_files[scenario] = files
    return scenario_files


def auto_discover_eval_files(collected_dir):
    scenario_eval_files = {}
    for scenario in SCENARIOS:
        pattern = os.path.join(collected_dir, f"DQN_{scenario}_seed*_eval.pkl")
        files = sorted(glob.glob(pattern))
        if files:
            scenario_eval_files[scenario] = files
    return scenario_eval_files


def main():
    parser = argparse.ArgumentParser(
        description="Cross-scenario summary visualization (Option B)")
    parser.add_argument("--collected_dir", type=str, default="collected")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else FIGURES_ROOT / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load training data
    scenario_files = auto_discover_files(args.collected_dir)
    scenario_data = {}
    for scenario, files in scenario_files.items():
        runs = []
        for f in files:
            if os.path.exists(f):
                print(f"Loading {scenario}: {os.path.basename(f)}")
                runs.append(load_pkl(f))
        if runs:
            scenario_data[scenario] = runs

    # Load evaluation data
    eval_files = auto_discover_eval_files(args.collected_dir)
    eval_data = {}
    for scenario, files in eval_files.items():
        evals = []
        for f in files:
            print(f"Loading eval {scenario}: {os.path.basename(f)}")
            evals.append(load_eval_pkl(f))
        if evals:
            eval_data[scenario] = evals

    if not scenario_data:
        print("ERROR: No training data found. Check --collected_dir path.")
        return

    scenario_order = [s for s in SCENARIOS if s in scenario_data]
    print(f"\nGenerating Option B summary for: {scenario_order}")

    # =====================================================================
    # Build the 2x2 figure
    # =====================================================================
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.30)

    # ----- (a) Training Convergence -----
    ax_a = fig.add_subplot(gs[0, 0])
    for scenario in scenario_order:
        runs = scenario_data[scenario]
        color = SCENARIO_COLORS[scenario]
        name = SCENARIO_SHORT[scenario]

        ma_list = []
        for q_table, rewards, eps in runs:
            if len(rewards) >= 1000:
                ma = uniform_filter1d(np.array(rewards, dtype=float),
                                      size=1000, mode="nearest")
                ma_list.append(ma)

        if ma_list:
            min_len = min(len(m) for m in ma_list)
            aligned = np.array([m[:min_len] for m in ma_list])  # shape: (n_seeds, n_episodes)
            mean, lo, hi = compute_ci(aligned, confidence=0.95)
            episodes = np.arange(min_len)
            ax_a.fill_between(episodes, lo, hi, color=color, alpha=0.25, zorder=1)
            ax_a.plot(episodes, mean, linewidth=2, color=color, label=name, zorder=2)

    ax_a.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_a.set_xlabel("Training Episode", fontsize=12)
    ax_a.set_ylabel("Cumulative Reward per Episode ($)", fontsize=12)
    ax_a.set_title("Training Convergence\n(1,000-episode moving average \u00b1 95% CI across seeds)",
                   fontsize=13, fontweight="bold")
    ax_a.legend(fontsize=10, loc="lower right")
    ax_a.grid(True, alpha=0.3)
    add_subplot_label(ax_a, 'a')

    # ----- (b) Annualized Cow Stall Value -----
    ax_b = fig.add_subplot(gs[0, 1])
    annual_means, annual_ses, annual_labels, annual_colors = [], [], [], []

    for scenario in scenario_order:
        if scenario not in eval_data:
            continue
        seed_means = []
        for ed in eval_data[scenario]:
            if 'overall' in ed:
                seed_means.append(np.mean(ed['overall']['episode_rewards']) / 15.0)
        if seed_means:
            annual_means.append(np.mean(seed_means))
            annual_ses.append(np.std(seed_means, ddof=1) / np.sqrt(len(seed_means))
                              if len(seed_means) > 1 else 0)
            annual_labels.append(SCENARIO_SHORT[scenario])
            annual_colors.append(SCENARIO_COLORS[scenario])

    x_pos = np.arange(len(annual_labels))
    ax_b.bar(x_pos, annual_means, yerr=annual_ses, color=annual_colors,
             alpha=0.8, capsize=5, width=0.6)
    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels(annual_labels, fontsize=10, rotation=15, ha='right')
    ax_b.set_ylabel("Annual Return ($/stall/year)", fontsize=12)
    ax_b.set_title("Annual Return per Cow Stall",
                   fontsize=13, fontweight="bold")
    ax_b.grid(True, alpha=0.3, axis="y")

    if annual_means:
        max_height = max(m + s for m, s in zip(annual_means, annual_ses))
        ax_b.set_ylim(0, max_height + 500)

    for i, (m, s) in enumerate(zip(annual_means, annual_ses)):
        ax_b.text(i, m + s + max(annual_means) * 0.02, f"${m:,.0f}",
                  ha="center", va="bottom", fontsize=11, fontweight="bold")
    add_subplot_label(ax_b, 'b')

    # ----- (c) Per-Parity Discounted Value -----
    ax_c = fig.add_subplot(gs[1, 0])
    parities = list(range(0, 13))
    parity_labels = ['Spr'] + [str(p) for p in range(1, 13)]

    for scenario in scenario_order:
        if scenario not in eval_data:
            continue
        color = SCENARIO_COLORS[scenario]
        name = SCENARIO_SHORT[scenario]

        parity_means_across_seeds = {p: [] for p in parities}
        for ed in eval_data[scenario]:
            if 'by_parity' in ed:
                for p in parities:
                    if p in ed['by_parity']:
                        parity_means_across_seeds[p].append(
                            ed['by_parity'][p]['mean_discounted'])

        plot_means, plot_ses = [], []
        for p in parities:
            vals = parity_means_across_seeds[p]
            if vals:
                plot_means.append(np.mean(vals))
                plot_ses.append(np.std(vals, ddof=1) / np.sqrt(len(vals))
                                if len(vals) > 1 else 0)
            else:
                plot_means.append(np.nan)
                plot_ses.append(0)

        x = np.arange(len(parities))
        ax_c.errorbar(x, plot_means, yerr=plot_ses, marker='o', markersize=5,
                      linewidth=2, color=color, label=name, capsize=3)

    ax_c.set_xticks(np.arange(len(parities)))
    ax_c.set_xticklabels(parity_labels, fontsize=10)
    ax_c.set_xlabel("Parity", fontsize=12)
    ax_c.set_ylabel("Expected Stall Value ($)", fontsize=12)
    ax_c.set_title("Expected Stall Value by Cow Parity",
                   fontsize=13, fontweight="bold")
    ax_c.legend(fontsize=10, loc="upper right")
    ax_c.grid(True, alpha=0.3)
    add_subplot_label(ax_c, 'c')

    # ----- (d) Steady-State Herd Parity Distribution -----
    ax_d = fig.add_subplot(gs[1, 1])
    n_scenarios = len(scenario_order)
    bar_width = 0.8 / max(n_scenarios, 1)
    x = np.arange(13)

    for idx, scenario in enumerate(scenario_order):
        if scenario not in eval_data:
            continue
        color = SCENARIO_COLORS[scenario]
        name = SCENARIO_SHORT[scenario]

        parity_fracs = {p: [] for p in range(13)}
        for ed in eval_data[scenario]:
            if 'steady_state' in ed and ed['steady_state'] is not None:
                ss = ed['steady_state']
                if 'parity_fractions' in ss:
                    for p in range(13):
                        if p in ss['parity_fractions']:
                            parity_fracs[p].append(ss['parity_fractions'][p])

        if not any(parity_fracs[p] for p in range(13)):
            continue

        frac_means, frac_ses = [], []
        for p in range(13):
            vals = parity_fracs[p]
            if vals:
                frac_means.append(np.mean(vals) * 100)
                frac_ses.append(np.std(vals, ddof=1) / np.sqrt(len(vals)) * 100
                                if len(vals) > 1 else 0)
            else:
                frac_means.append(0)
                frac_ses.append(0)

        offset = (idx - n_scenarios / 2 + 0.5) * bar_width
        ax_d.bar(x + offset, frac_means, bar_width, yerr=frac_ses,
                 color=color, alpha=0.8, label=name, capsize=2)

    ax_d.set_xticks(x)
    ax_d.set_xticklabels(['Spr'] + [str(p) for p in range(1, 13)], fontsize=10)
    ax_d.set_xlabel("Parity", fontsize=12)
    ax_d.set_ylabel("Proportion of Herd (%)", fontsize=12)
    ax_d.set_title("Steady-State Herd Parity Distribution",
                   fontsize=13, fontweight="bold")
    ax_d.legend(fontsize=10, loc="upper right")
    ax_d.grid(True, alpha=0.3, axis="y")
    add_subplot_label(ax_d, 'd')

    # fig.suptitle("DQN Cow Replacement Policy \u2014 Cross-Scenario Comparison\n"
    #              "(5 scenarios \u00d7 5 seeds)",
    #              fontsize=16, fontweight="bold", y=0.995)

    output_path = output_dir / "cross_scenario_summary.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
