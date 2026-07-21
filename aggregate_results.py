"""
Aggregate DQN evaluation results across seeds for each scenario.

Reads the _eval.csv files from all 25 jobs and produces:
  1. Terminal printout of all metrics
  2. aggregated_results.csv — long-format CSV of all metrics
  3. outputs/tables/ — individual CSV tables ready for copy-paste into paper:
       - table_cross_scenario_summary.csv
       - table_culling_rates.csv
       - table_parity_distribution.csv
       - table_replacement_distribution.csv
       - table_per_parity_culling.csv

BACKWARD COMPATIBILITY:
  If the _eval.csv was produced by the OLD local_evaluate.py (without granular
  culling fields), the script back-calculates total culling rates from the
  existing steady-state metrics.

Usage:
    python3 aggregate_results.py --results_dir collected/
"""

import argparse
import csv
import os
import glob
import numpy as np


SCENARIOS_ORDER = ['2025', 'OG', 'OB', 'UG', 'UB']
MODEL_PREFIX = "DQN_prod"   # production (5-state) model file prefix; use "DQN" for the base model
SCENARIO_NAMES = {
    '2025': '2025 Baseline',
    'OG': 'Oversupply, Good',
    'OB': 'Oversupply, Bad',
    'UG': 'Undersupply, Good',
    'UB': 'Undersupply, Bad',
}


# =============================================================================
# CSV Parsing
# =============================================================================

def parse_eval_csv(csv_path):
    """Parse an evaluation CSV file and return all sections."""
    overall = {}
    parity_data = {}
    steady_state_dist = {}
    steady_state_meta = {}
    parity_culling = {}

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)

    # Parse overall metrics (rows 1+, format: metric, value)
    i = 1
    while i < len(rows) and len(rows[i]) == 2 and rows[i][0] != '':
        metric, value = rows[i]
        overall[metric] = float(value)
        i += 1

    # Skip blank row
    i += 1

    # Skip header row for per-parity section
    if i < len(rows):
        i += 1

    # Parse per-parity data
    while i < len(rows) and len(rows[i]) >= 7:
        row = rows[i]
        try:
            p = int(row[0])
            parity_data[p] = {
                'mean_reward': float(row[1]),
                'std_reward': float(row[2]),
                'se_reward': float(row[3]),
                'mean_discounted': float(row[4]),
                'std_discounted': float(row[5]),
                'se_discounted': float(row[6]),
            }
            i += 1
        except (ValueError, IndexError):
            break

    # Skip blank rows
    while i < len(rows) and (len(rows[i]) == 0 or rows[i][0] == ''):
        i += 1

    # Parse steady-state distribution section
    if (i < len(rows) and len(rows[i]) >= 5 and
            rows[i][0] == 'parity' and 'fraction_of_time' in rows[i][1]):
        header_row = rows[i]
        has_granular = len(header_row) >= 8 and 'voluntary_count' in header_row[5]
        i += 1
        while i < len(rows) and len(rows[i]) >= 5 and rows[i][0] != '':
            try:
                p = int(rows[i][0])
                steady_state_dist[p] = {
                    'fraction_of_time': float(rows[i][1]),
                    'months_count': int(float(rows[i][2])),
                    'replacement_fraction': float(rows[i][3]),
                    'replacement_count': int(float(rows[i][4])),
                }
                if has_granular and len(rows[i]) >= 8:
                    steady_state_dist[p]['voluntary_count'] = int(float(rows[i][5]))
                    steady_state_dist[p]['death_count'] = int(float(rows[i][6]))
                    steady_state_dist[p]['involuntary_count'] = int(float(rows[i][7]))
                i += 1
            except (ValueError, IndexError):
                break

    # Skip blank rows
    while i < len(rows) and (len(rows[i]) == 0 or rows[i][0] == ''):
        i += 1

    # Parse steady-state meta
    if (i < len(rows) and len(rows[i]) >= 2 and
            rows[i][0] == 'steady_state_metric'):
        i += 1
        while i < len(rows) and len(rows[i]) >= 2 and rows[i][0] != '':
            try:
                steady_state_meta[rows[i][0]] = float(rows[i][1])
                i += 1
            except (ValueError, IndexError):
                break

    # Skip blank rows
    while i < len(rows) and (len(rows[i]) == 0 or rows[i][0] == ''):
        i += 1

    # Parse per-parity culling rates
    if (i < len(rows) and len(rows[i]) >= 3 and rows[i][0] == 'parity'):
        header_row = rows[i]
        has_granular_rates = len(header_row) >= 5 and 'annual_rate_voluntary' in header_row[2]
        i += 1
        while i < len(rows) and len(rows[i]) >= 3 and rows[i][0] != '':
            try:
                p = int(rows[i][0])
                if has_granular_rates and len(rows[i]) >= 5:
                    parity_culling[p] = {
                        'annual_rate_total': float(rows[i][1]),
                        'annual_rate_voluntary': float(rows[i][2]),
                        'annual_rate_death': float(rows[i][3]),
                        'annual_rate_involuntary': float(rows[i][4]),
                    }
                else:
                    parity_culling[p] = {
                        'annual_rate_total': float(rows[i][2]) if len(rows[i]) >= 3 else 0,
                        'annual_rate_voluntary': 0.0,
                        'annual_rate_death': 0.0,
                        'annual_rate_involuntary': 0.0,
                    }
                i += 1
            except (ValueError, IndexError):
                break

    # Back-calculate culling rates if not explicitly present
    if not parity_culling and steady_state_dist and steady_state_meta:
        replacements_per_15yr = steady_state_meta.get('replacements_per_15yr', 0)

        if 'annual_culling_rate_total' not in steady_state_meta:
            annual_rate = replacements_per_15yr / 15 if replacements_per_15yr > 0 else 0
            steady_state_meta['annual_culling_rate_total'] = annual_rate

        for p in range(13):
            if p in steady_state_dist:
                frac_time = steady_state_dist[p]['fraction_of_time']
                repl_frac = steady_state_dist[p]['replacement_fraction']
                if frac_time > 0 and replacements_per_15yr > 0:
                    annual_rate_p = (repl_frac * replacements_per_15yr) / (frac_time * 15)
                else:
                    annual_rate_p = 0.0
                parity_culling[p] = {
                    'annual_rate_total': annual_rate_p,
                    'annual_rate_voluntary': 0.0,
                    'annual_rate_death': 0.0,
                    'annual_rate_involuntary': 0.0,
                }
            else:
                parity_culling[p] = {
                    'annual_rate_total': 0.0,
                    'annual_rate_voluntary': 0.0,
                    'annual_rate_death': 0.0,
                    'annual_rate_involuntary': 0.0,
                }

    if steady_state_meta and 'annual_culling_rate_total' not in steady_state_meta:
        replacements_per_15yr = steady_state_meta.get('replacements_per_15yr', 0)
        steady_state_meta['annual_culling_rate_total'] = (
            replacements_per_15yr / 15 if replacements_per_15yr > 0 else 0)

    return overall, parity_data, steady_state_dist, steady_state_meta, parity_culling


# =============================================================================
# Aggregation
# =============================================================================

def aggregate_scenario(csv_files):
    """Aggregate results from multiple seeds for one scenario."""
    all_overall = []
    all_parity = {}
    all_steady_dist = {}
    all_steady_meta = []
    all_parity_culling = {}

    for csv_path in csv_files:
        overall, parity_data, steady_dist, steady_meta, parity_culling = parse_eval_csv(csv_path)
        all_overall.append(overall)

        for p, data in parity_data.items():
            if p not in all_parity:
                all_parity[p] = []
            all_parity[p].append(data)

        if steady_dist:
            for p, data in steady_dist.items():
                if p not in all_steady_dist:
                    all_steady_dist[p] = []
                all_steady_dist[p].append(data)

        if steady_meta:
            all_steady_meta.append(steady_meta)

        if parity_culling:
            for p, data in parity_culling.items():
                if p not in all_parity_culling:
                    all_parity_culling[p] = []
                all_parity_culling[p].append(data)

    n_seeds = len(all_overall)
    mean_rewards = [o['mean_reward'] for o in all_overall]
    mean_discounted = [o['mean_discounted'] for o in all_overall]

    agg_overall = {
        'n_seeds': n_seeds,
        'mean_reward': np.mean(mean_rewards),
        'std_across_seeds': np.std(mean_rewards, ddof=1) if n_seeds > 1 else 0,
        'se_across_seeds': np.std(mean_rewards, ddof=1) / np.sqrt(n_seeds) if n_seeds > 1 else 0,
        'mean_discounted': np.mean(mean_discounted),
        'std_disc_across_seeds': np.std(mean_discounted, ddof=1) if n_seeds > 1 else 0,
        'se_disc_across_seeds': np.std(mean_discounted, ddof=1) / np.sqrt(n_seeds) if n_seeds > 1 else 0,
        'per_seed_rewards': mean_rewards,
        'per_seed_discounted': mean_discounted,
    }

    agg_parity = {}
    for p in sorted(all_parity.keys()):
        disc_values = [d['mean_discounted'] for d in all_parity[p]]
        reward_values = [d['mean_reward'] for d in all_parity[p]]
        n = len(disc_values)
        agg_parity[p] = {
            'mean_reward': np.mean(reward_values),
            'std_reward_across_seeds': np.std(reward_values, ddof=1) if n > 1 else 0,
            'se_reward_across_seeds': np.std(reward_values, ddof=1) / np.sqrt(n) if n > 1 else 0,
            'mean_discounted': np.mean(disc_values),
            'std_disc_across_seeds': np.std(disc_values, ddof=1) if n > 1 else 0,
            'se_disc_across_seeds': np.std(disc_values, ddof=1) / np.sqrt(n) if n > 1 else 0,
        }

    agg_steady = None
    if all_steady_dist:
        agg_steady = {}
        for p in range(13):
            if p in all_steady_dist:
                fracs = [d['fraction_of_time'] for d in all_steady_dist[p]]
                repl_fracs = [d['replacement_fraction'] for d in all_steady_dist[p]]
                n = len(fracs)
                agg_steady[p] = {
                    'mean_fraction': np.mean(fracs),
                    'se_fraction': np.std(fracs, ddof=1) / np.sqrt(n) if n > 1 else 0,
                    'mean_repl_fraction': np.mean(repl_fracs),
                    'se_repl_fraction': np.std(repl_fracs, ddof=1) / np.sqrt(n) if n > 1 else 0,
                }

    agg_steady_meta = None
    if all_steady_meta:
        agg_steady_meta = {}
        for key in ['mean_parity', 'mean_production', 'mean_replacement_parity',
                     'replacements_per_15yr', 'std_replacements_per_15yr']:
            values = [m[key] for m in all_steady_meta if key in m]
            if values:
                n = len(values)
                agg_steady_meta[key] = {
                    'mean': np.mean(values),
                    'se': np.std(values, ddof=1) / np.sqrt(n) if n > 1 else 0,
                }
        for key in ['annual_culling_rate_total', 'annual_culling_rate_voluntary',
                     'annual_culling_rate_death', 'annual_culling_rate_involuntary']:
            values = [m[key] for m in all_steady_meta if key in m]
            if values:
                n = len(values)
                agg_steady_meta[key] = {
                    'mean': np.mean(values),
                    'se': np.std(values, ddof=1) / np.sqrt(n) if n > 1 else 0,
                }

    agg_parity_culling = None
    if all_parity_culling:
        agg_parity_culling = {}
        for p in range(13):
            if p in all_parity_culling:
                n = len(all_parity_culling[p])
                agg_parity_culling[p] = {}
                for key in ['annual_rate_total', 'annual_rate_voluntary',
                            'annual_rate_death', 'annual_rate_involuntary']:
                    values = [d[key] for d in all_parity_culling[p] if key in d]
                    if values:
                        agg_parity_culling[p][key] = {
                            'mean': np.mean(values),
                            'se': np.std(values, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0,
                        }
                    else:
                        agg_parity_culling[p][key] = {'mean': 0.0, 'se': 0.0}

    return agg_overall, agg_parity, agg_steady, agg_steady_meta, agg_parity_culling


# =============================================================================
# Terminal Printing
# =============================================================================

def print_results(scenario_results):
    """Print formatted aggregation results to terminal."""
    scenarios_present = [s for s in SCENARIOS_ORDER if s in scenario_results]

    print("\n" + "=" * 80)
    print("AGGREGATED DQN EVALUATION RESULTS")
    print("=" * 80)

    # Overall summary
    print(f"\n{'Scenario':<12} {'Seeds':>6} {'Mean Reward':>14} {'SE':>10} "
          f"{'Mean Disc.':>14} {'SE':>10}")
    print("-" * 66)
    for scenario in scenarios_present:
        o = scenario_results[scenario]['overall']
        print(f"{scenario:<12} {o['n_seeds']:>6} "
              f"${o['mean_reward']:>12,.2f} ${o['se_across_seeds']:>8,.2f} "
              f"${o['mean_discounted']:>12,.2f} ${o['se_disc_across_seeds']:>8,.2f}")

    # Annualized
    print(f"\n{'':=<80}")
    print("ANNUALIZED REWARDS (Mean Reward / 15 years)")
    print(f"{'':=<80}")
    print(f"{'Scenario':<12} {'Annual $/stall':>16} {'SE':>10}")
    print("-" * 40)
    for scenario in scenarios_present:
        o = scenario_results[scenario]['overall']
        annual = o['mean_reward'] / 15
        annual_se = o['se_across_seeds'] / 15
        print(f"{scenario:<12} ${annual:>14,.2f} ${annual_se:>8,.2f}")

    # Per-parity discounted
    print(f"\n{'':=<80}")
    print("PER-PARITY DISCOUNTED REWARDS (Mean +/- SE across seeds)")
    print(f"{'':=<80}")
    header = f"{'Parity':<10}"
    for scenario in scenarios_present:
        header += f" {scenario:>14}"
    print(header)
    print("-" * (10 + 15 * len(scenarios_present)))
    for p in range(0, 13):
        label = 'Springer' if p == 0 else f'Parity {p}'
        row = f"{label:<10}"
        for scenario in scenarios_present:
            pr = scenario_results[scenario]['parity']
            if p in pr:
                row += f" ${pr[p]['mean_discounted']:>10,.2f}"
                row += f" +/-{pr[p]['se_disc_across_seeds']:>3.0f}"
            else:
                row += f" {'N/A':>14}"
        print(row)

    # Steady-state sections
    has_steady = any(scenario_results[s].get('steady_state') is not None
                     for s in scenarios_present)
    if has_steady:
        # Parity distribution
        print(f"\n{'':=<80}")
        print("STEADY-STATE HERD PARITY DISTRIBUTION (% of herd at each parity)")
        print(f"{'':=<80}")
        header = f"{'Parity':<10}"
        for scenario in scenarios_present:
            header += f" {scenario:>10}"
        print(header)
        print("-" * (10 + 11 * len(scenarios_present)))
        for p in range(0, 13):
            label = 'Springer' if p == 0 else f'Parity {p}'
            row = f"{label:<10}"
            for scenario in scenarios_present:
                ss = scenario_results[scenario].get('steady_state')
                if ss and p in ss:
                    row += f" {ss[p]['mean_fraction']*100:>9.1f}%"
                else:
                    row += f" {'--':>10}"
            print(row)

        # Replacement distribution
        print(f"\n{'':=<80}")
        print("REPLACEMENT DISTRIBUTION (% of all replacements at each parity)")
        print(f"{'':=<80}")
        header = f"{'Parity':<10}"
        for scenario in scenarios_present:
            header += f" {scenario:>10}"
        print(header)
        print("-" * (10 + 11 * len(scenarios_present)))
        for p in range(0, 13):
            label = 'Springer' if p == 0 else f'Parity {p}'
            row = f"{label:<10}"
            for scenario in scenarios_present:
                ss = scenario_results[scenario].get('steady_state')
                if ss and p in ss:
                    row += f" {ss[p]['mean_repl_fraction']*100:>9.1f}%"
                else:
                    row += f" {'--':>10}"
            print(row)

        # Replacement statistics
        print(f"\n{'':=<80}")
        print("REPLACEMENT STATISTICS")
        print(f"{'':=<80}")
        header = f"{'Metric':<32}"
        for scenario in scenarios_present:
            header += f" {scenario:>10}"
        print(header)
        print("-" * (32 + 11 * len(scenarios_present)))

        for key, label in [
                ('mean_parity', 'Mean herd parity'),
                ('mean_replacement_parity', 'Mean replacement parity'),
                ('replacements_per_15yr', 'Replacements per 15 yr')]:
            row = f"{label:<32}"
            for scenario in scenarios_present:
                meta = scenario_results[scenario].get('steady_state_meta')
                if meta and key in meta:
                    row += f" {meta[key]['mean']:>10.2f}"
                else:
                    row += f" {'--':>10}"
            print(row)

        # Avg cow lifespan
        row = f"{'Avg cow lifespan (yr)':<32}"
        for scenario in scenarios_present:
            meta = scenario_results[scenario].get('steady_state_meta')
            if meta and 'replacements_per_15yr' in meta and meta['replacements_per_15yr']['mean'] > 0:
                lifespan_yr = (180 / meta['replacements_per_15yr']['mean']) / 12
                row += f" {lifespan_yr:>10.1f}"
            else:
                row += f" {'--':>10}"
        print(row)

        # Culling rates
        has_granular = any(
            scenario_results[s].get('steady_state_meta') and
            'annual_culling_rate_voluntary' in scenario_results[s]['steady_state_meta']
            for s in scenarios_present
        )

        print(f"\n{'':=<80}")
        print("ANNUAL CULLING RATES")
        print("  Total = Voluntary + Death + Involuntary")
        print(f"{'':=<80}")
        header = f"{'Category':<32}"
        for scenario in scenarios_present:
            header += f" {scenario:>10}"
        print(header)
        print("-" * (32 + 11 * len(scenarios_present)))

        rate_keys = [
            ('annual_culling_rate_total', 'Total culling rate (%)'),
            ('annual_culling_rate_voluntary', 'Voluntary (%)'),
            ('annual_culling_rate_death', 'Death (%)'),
            ('annual_culling_rate_involuntary', 'Involuntary overflow (%)'),
        ]
        for key, label in rate_keys:
            if not has_granular and key != 'annual_culling_rate_total':
                continue
            row = f"{label:<32}"
            for scenario in scenarios_present:
                meta = scenario_results[scenario].get('steady_state_meta')
                if meta and key in meta:
                    row += f" {meta[key]['mean']*100:>9.1f}%"
                else:
                    row += f" {'--':>10}"
            print(row)

        # Per-parity culling rate
        has_culling = any(scenario_results[s].get('parity_culling') is not None
                          for s in scenarios_present)
        if has_culling:
            print(f"\n{'':=<80}")
            print("PER-PARITY ANNUAL CULLING RATE — TOTAL (%)")
            print(f"{'':=<80}")
            header = f"{'Parity':<10}"
            for scenario in scenarios_present:
                header += f" {scenario:>10}"
            print(header)
            print("-" * (10 + 11 * len(scenarios_present)))
            for p in range(0, 13):
                label = 'Springer' if p == 0 else f'Parity {p}'
                row = f"{label:<10}"
                for scenario in scenarios_present:
                    pc = scenario_results[scenario].get('parity_culling')
                    if pc and p in pc and 'annual_rate_total' in pc[p]:
                        row += f" {pc[p]['annual_rate_total']['mean']*100:>9.1f}%"
                    else:
                        row += f" {'--':>10}"
                print(row)

            if has_granular:
                for rate_key, rate_label in [
                        ('annual_rate_voluntary', 'VOLUNTARY'),
                        ('annual_rate_death', 'DEATH'),
                        ('annual_rate_involuntary', 'INVOLUNTARY (OVERFLOW)')]:
                    print(f"\n{'':=<80}")
                    print(f"PER-PARITY ANNUAL CULLING RATE — {rate_label} (%)")
                    print(f"{'':=<80}")
                    header = f"{'Parity':<10}"
                    for scenario in scenarios_present:
                        header += f" {scenario:>10}"
                    print(header)
                    print("-" * (10 + 11 * len(scenarios_present)))
                    for p in range(0, 13):
                        label = 'Springer' if p == 0 else f'Parity {p}'
                        row = f"{label:<10}"
                        for scenario in scenarios_present:
                            pc = scenario_results[scenario].get('parity_culling')
                            if pc and p in pc and rate_key in pc[p]:
                                row += f" {pc[p][rate_key]['mean']*100:>9.1f}%"
                            else:
                                row += f" {'--':>10}"
                        print(row)

    # Per-seed detail
    print(f"\n{'':=<80}")
    print("PER-SEED DETAIL (Mean Reward per seed)")
    print(f"{'':=<80}")
    for scenario in scenarios_present:
        o = scenario_results[scenario]['overall']
        seeds_str = ", ".join([f"${r:,.2f}" for r in o['per_seed_rewards']])
        print(f"  {scenario}: [{seeds_str}]")

    print("=" * 80)


# =============================================================================
# Save long-format CSV (all metrics)
# =============================================================================

def save_aggregated_csv(scenario_results, output_path):
    """Save aggregated results to a long-format CSV file."""
    scenarios_present = [s for s in SCENARIOS_ORDER if s in scenario_results]

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['scenario', 'metric', 'parity',
                         'mean', 'std_across_seeds', 'se_across_seeds'])

        for scenario in scenarios_present:
            o = scenario_results[scenario]['overall']
            writer.writerow([scenario, 'overall_reward', 'all',
                             o['mean_reward'], o['std_across_seeds'],
                             o['se_across_seeds']])
            writer.writerow([scenario, 'overall_discounted', 'all',
                             o['mean_discounted'], o['std_disc_across_seeds'],
                             o['se_disc_across_seeds']])
            writer.writerow([scenario, 'annualized_reward', 'all',
                             o['mean_reward'] / 15, o['std_across_seeds'] / 15,
                             o['se_across_seeds'] / 15])

            for p in sorted(scenario_results[scenario]['parity'].keys()):
                pr = scenario_results[scenario]['parity'][p]
                writer.writerow([scenario, 'parity_reward', p,
                                 pr['mean_reward'],
                                 pr['std_reward_across_seeds'],
                                 pr['se_reward_across_seeds']])
                writer.writerow([scenario, 'parity_discounted', p,
                                 pr['mean_discounted'],
                                 pr['std_disc_across_seeds'],
                                 pr['se_disc_across_seeds']])

            ss = scenario_results[scenario].get('steady_state')
            if ss:
                for p in range(13):
                    if p in ss:
                        writer.writerow([scenario, 'parity_fraction', p,
                                         ss[p]['mean_fraction'], '', ss[p]['se_fraction']])
                        writer.writerow([scenario, 'replacement_fraction', p,
                                         ss[p]['mean_repl_fraction'], '', ss[p]['se_repl_fraction']])

            meta = scenario_results[scenario].get('steady_state_meta')
            if meta:
                for key in ['mean_parity', 'mean_replacement_parity',
                            'replacements_per_15yr',
                            'annual_culling_rate_total',
                            'annual_culling_rate_voluntary',
                            'annual_culling_rate_death',
                            'annual_culling_rate_involuntary']:
                    if key in meta:
                        writer.writerow([scenario, key, 'all',
                                         meta[key]['mean'], '', meta[key]['se']])
                if ('replacements_per_15yr' in meta and
                        meta['replacements_per_15yr']['mean'] > 0):
                    lifespan_months = 180 / meta['replacements_per_15yr']['mean']
                    writer.writerow([scenario, 'avg_cow_lifespan_months', 'all',
                                     lifespan_months, '', ''])

            pc = scenario_results[scenario].get('parity_culling')
            if pc:
                for p in range(13):
                    if p in pc:
                        for key in ['annual_rate_total', 'annual_rate_voluntary',
                                    'annual_rate_death', 'annual_rate_involuntary']:
                            if key in pc[p]:
                                writer.writerow([scenario, f'parity_{key}', p,
                                                 pc[p][key]['mean'], '',
                                                 pc[p][key]['se']])

    print(f"\nAggregated CSV saved to: {output_path}")


# =============================================================================
# Save paper-ready tables as individual CSVs
# =============================================================================

def save_paper_tables(scenario_results, output_dir):
    """Save individual CSV tables ready for copy-paste into paper."""
    os.makedirs(output_dir, exist_ok=True)
    scenarios_present = [s for s in SCENARIOS_ORDER if s in scenario_results]

    # ---- Table 1: Cross-Scenario Summary ----
    path1 = os.path.join(output_dir, "table_cross_scenario_summary.csv")
    with open(path1, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Scenario', 'Annual $/stall', 'SE',
                         'Mean Herd Parity', 'Mean Production (retained herd)',
                         'Mean Repl. Parity',
                         'Repl/15yr', 'Avg Life (yr)',
                         'Culling Rate Total (%)',
                         'Culling Rate Voluntary (%)',
                         'Culling Rate Death (%)',
                         'Culling Rate Involuntary (%)'])
        for scenario in scenarios_present:
            o = scenario_results[scenario]['overall']
            meta = scenario_results[scenario].get('steady_state_meta', {})
            if meta is None:
                meta = {}  # Ensure meta is always a dict, not None

            annual = o['mean_reward'] / 15
            annual_se = o['se_across_seeds'] / 15
            mean_par = meta['mean_parity']['mean'] if 'mean_parity' in meta else ''
            mean_prod = meta['mean_production']['mean'] if 'mean_production' in meta else ''
            mean_repl = meta['mean_replacement_parity']['mean'] if 'mean_replacement_parity' in meta else ''
            repl_15 = meta['replacements_per_15yr']['mean'] if 'replacements_per_15yr' in meta else ''
            avg_life = (180 / meta['replacements_per_15yr']['mean'] / 12
                        if 'replacements_per_15yr' in meta and meta['replacements_per_15yr']['mean'] > 0 else '')

            def get_rate(key):
                if key in meta:
                    return f"{meta[key]['mean']*100:.1f}"
                return ''

            writer.writerow([
                SCENARIO_NAMES.get(scenario, scenario),
                f"{annual:.0f}", f"{annual_se:.0f}",
                f"{mean_par:.2f}" if mean_par != '' else '',
                f"{mean_prod:.3f}" if mean_prod != '' else '',
                f"{mean_repl:.2f}" if mean_repl != '' else '',
                f"{repl_15:.1f}" if repl_15 != '' else '',
                f"{avg_life:.1f}" if avg_life != '' else '',
                get_rate('annual_culling_rate_total'),
                get_rate('annual_culling_rate_voluntary'),
                get_rate('annual_culling_rate_death'),
                get_rate('annual_culling_rate_involuntary'),
            ])
    print(f"  Table 1 saved: {path1}")

    # ---- Table 2: Culling Rates (overall + per-parity) ----
    path2 = os.path.join(output_dir, "table_culling_rates.csv")
    with open(path2, 'w', newline='') as f:
        writer = csv.writer(f)

        # Overall section
        writer.writerow(['--- Overall Annual Culling Rates (%) ---'])
        header = ['Category'] + [SCENARIO_NAMES.get(s, s) for s in scenarios_present]
        writer.writerow(header)

        for key, label in [
                ('annual_culling_rate_total', 'Total'),
                ('annual_culling_rate_voluntary', 'Voluntary'),
                ('annual_culling_rate_death', 'Death'),
                ('annual_culling_rate_involuntary', 'Involuntary')]:
            row = [label]
            for scenario in scenarios_present:
                meta = scenario_results[scenario].get('steady_state_meta', {})
                if meta is None:
                    meta = {}
                if key in meta:
                    row.append(f"{meta[key]['mean']*100:.1f}")
                else:
                    row.append('')
            writer.writerow(row)

        writer.writerow([])

        # Per-parity section
        writer.writerow(['--- Per-Parity Annual Culling Rate: Total (%) ---'])
        header = ['Parity'] + [SCENARIO_NAMES.get(s, s) for s in scenarios_present]
        writer.writerow(header)
        for p in range(0, 13):
            label = 'Springer' if p == 0 else str(p)
            row = [label]
            for scenario in scenarios_present:
                pc = scenario_results[scenario].get('parity_culling', {})
                if pc and p in pc and 'annual_rate_total' in pc[p]:
                    row.append(f"{pc[p]['annual_rate_total']['mean']*100:.1f}")
                else:
                    row.append('')
            writer.writerow(row)

    print(f"  Table 2 saved: {path2}")

    # ---- Table 3: Parity Distribution ----
    path3 = os.path.join(output_dir, "table_parity_distribution.csv")
    with open(path3, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['Parity'] + [SCENARIO_NAMES.get(s, s) for s in scenarios_present]
        writer.writerow(header)
        for p in range(0, 13):
            label = 'Springer' if p == 0 else str(p)
            row = [label]
            for scenario in scenarios_present:
                ss = scenario_results[scenario].get('steady_state', {})
                if ss and p in ss:
                    row.append(f"{ss[p]['mean_fraction']*100:.1f}")
                else:
                    row.append('')
            writer.writerow(row)
    print(f"  Table 3 saved: {path3}")

    # ---- Table 4: Replacement Distribution ----
    path4 = os.path.join(output_dir, "table_replacement_distribution.csv")
    with open(path4, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['Parity'] + [SCENARIO_NAMES.get(s, s) for s in scenarios_present]
        writer.writerow(header)
        for p in range(0, 13):
            label = 'Springer' if p == 0 else str(p)
            row = [label]
            for scenario in scenarios_present:
                ss = scenario_results[scenario].get('steady_state', {})
                if ss and p in ss:
                    row.append(f"{ss[p]['mean_repl_fraction']*100:.1f}")
                else:
                    row.append('')
            writer.writerow(row)
    print(f"  Table 4 saved: {path4}")

    # ---- Table 5: Per-Parity Culling Breakdown ----
    has_granular = any(
        scenario_results[s].get('steady_state_meta') and
        'annual_culling_rate_voluntary' in scenario_results[s]['steady_state_meta']
        for s in scenarios_present
    )
    if has_granular:
        path5 = os.path.join(output_dir, "table_per_parity_culling_breakdown.csv")
        with open(path5, 'w', newline='') as f:
            writer = csv.writer(f)
            for rate_key, rate_label in [
                    ('annual_rate_voluntary', 'Voluntary'),
                    ('annual_rate_death', 'Death'),
                    ('annual_rate_involuntary', 'Involuntary')]:
                writer.writerow([f'--- Per-Parity Annual Culling Rate: {rate_label} (%) ---'])
                header = ['Parity'] + [SCENARIO_NAMES.get(s, s) for s in scenarios_present]
                writer.writerow(header)
                for p in range(0, 13):
                    label = 'Springer' if p == 0 else str(p)
                    row = [label]
                    for scenario in scenarios_present:
                        pc = scenario_results[scenario].get('parity_culling', {})
                        if pc and p in pc and rate_key in pc[p]:
                            row.append(f"{pc[p][rate_key]['mean']*100:.1f}")
                        else:
                            row.append('')
                    writer.writerow(row)
                writer.writerow([])
        print(f"  Table 5 saved: {path5}")

    print(f"\n  All paper tables saved to: {output_dir}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate DQN evaluation results across seeds")
    parser.add_argument(
        "--results_dir", type=str, default="collected",
        help="Directory containing the _eval.csv files (default: collected/)")
    parser.add_argument(
        "--output", type=str, default="outputs/aggregated_results.csv",
        help="Output CSV path (default: outputs/aggregated_results.csv)")
    parser.add_argument(
        "--tables_dir", type=str, default="outputs/tables",
        help="Directory for paper-ready table CSVs (default: outputs/tables)")
    args = parser.parse_args()

    scenario_results = {}

    for scenario in SCENARIOS_ORDER:
        pattern = os.path.join(args.results_dir, f"{MODEL_PREFIX}_{scenario}_seed*_eval.csv")
        csv_files = sorted(glob.glob(pattern))

        if not csv_files:
            print(f"WARNING: No evaluation files found for scenario {scenario}")
            continue

        print(f"Scenario {scenario}: found {len(csv_files)} seed results")
        agg = aggregate_scenario(csv_files)
        scenario_results[scenario] = {
            'overall': agg[0],
            'parity': agg[1],
            'steady_state': agg[2] if agg[2] is not None else {},
            'steady_state_meta': agg[3] if agg[3] is not None else {},
            'parity_culling': agg[4] if agg[4] is not None else {},
        }

    if scenario_results:
        print_results(scenario_results)

        output_dir = os.path.dirname(args.output) or '.'
        os.makedirs(output_dir, exist_ok=True)
        save_aggregated_csv(scenario_results, args.output)

        save_paper_tables(scenario_results, args.tables_dir)
    else:
        print("ERROR: No results found to aggregate.")


if __name__ == "__main__":
    main()