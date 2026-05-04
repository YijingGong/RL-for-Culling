"""
Local Evaluation Script for trained DQN models.

Runs Monte Carlo policy evaluation + steady-state distribution analysis
on all 25 trained Q-tables (5 scenarios × 5 seeds) locally on your Mac.

No training is performed — this only loads existing Q-tables and evaluates.

Culling is tracked in three granular categories:
  1. Voluntary:   Agent chose action='replace'
  2. Death:       Agent chose 'keep' but cow died (stochastic mortality)
  3. Involuntary: Agent chose 'keep' but cow was forced out
                  (parity overflow after calving, or MAC overflow)

Usage:
    # Evaluate all 25 Q-tables in the collected/ directory
    python local_evaluate.py --collected_dir collected/

    # Evaluate a single file
    python local_evaluate.py --single collected/DQN_2025_seed42.pkl --scenario 2025

    # Custom episode counts
    python local_evaluate.py --collected_dir collected/ --eval_episodes 2000 --parity_episodes 1000
"""

import argparse
import numpy as np
import pickle
import os
import csv
import time
import glob
import random

import cow_environment2
import utility

parity_range = range(13)
mim_range = range(21)
mip_range = range(10)
disease_range = range(2)

SCENARIOS = ['2025', 'OG', 'OB', 'UG', 'UB']
SEEDS = [42, 123, 456, 789, 1024]


def load_q_table(pkl_path):
    """Load the Q-table from a saved .pkl file."""
    with open(pkl_path, 'rb') as f:
        q_table, rewards_per_episode, epsilon = pickle.load(f)
    return q_table, rewards_per_episode


def get_action(q_table, state):
    """Get the greedy action from the Q-table for a given state."""
    if state in q_table:
        q_keep = q_table[state]['keep']
        q_replace = q_table[state]['replace']
        return 'keep' if q_keep >= q_replace else 'replace'
    else:
        return 'keep'


def evaluate_policy(q_table, env, num_episodes=1000, max_steps=180, gamma=0.95,
                    seed=None):
    """
    Evaluate a policy by running Monte Carlo episodes from random starts.
    
    Also tracks parity distribution and replacement events to compute
    steady-state herd structure metrics.
    
    Returns: (overall_results, steady_state_results)
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    episode_rewards = []
    discounted_rewards = []
    
    # Steady-state tracking
    parity_counts = {p: 0 for p in range(13)}
    replacement_parity_counts = {p: 0 for p in range(13)}
    vol_parity_counts = {p: 0 for p in range(13)}
    death_parity_counts = {p: 0 for p in range(13)}
    invol_parity_counts = {p: 0 for p in range(13)}
    
    total_replacements = 0
    total_voluntary = 0
    total_death = 0
    total_involuntary = 0
    total_steps = 0
    replacements_per_ep = []

    for ep in range(num_episodes):
        state = env.reset()  # Random start
        total_reward = 0.0
        total_discounted = 0.0
        discount = 1.0
        ep_replacements = 0

        for step in range(max_steps):
            current_parity = state[0]
            parity_counts[current_parity] += 1
            total_steps += 1
            
            action = get_action(q_table, state)
            next_state, reward = env.step(action)
            total_reward += reward
            total_discounted += discount * reward
            discount *= gamma
            
            # Check for replacement
            repl_type = classify_replacement(action, state, next_state)
            if repl_type is not None:
                replacement_parity_counts[current_parity] += 1
                total_replacements += 1
                ep_replacements += 1
                
                if repl_type == 'voluntary':
                    vol_parity_counts[current_parity] += 1
                    total_voluntary += 1
                elif repl_type == 'death':
                    death_parity_counts[current_parity] += 1
                    total_death += 1
                elif repl_type == 'involuntary':
                    invol_parity_counts[current_parity] += 1
                    total_involuntary += 1

            if not (next_state[0] in parity_range and
                    next_state[1] in mim_range and
                    next_state[2] in mip_range):
                break

            state = next_state

        episode_rewards.append(total_reward)
        discounted_rewards.append(total_discounted)
        replacements_per_ep.append(ep_replacements)
    
    # Compute steady-state metrics
    parity_fractions = {}
    replacement_fractions = {}
    for p in range(13):
        parity_fractions[p] = parity_counts[p] / total_steps if total_steps > 0 else 0
        replacement_fractions[p] = (replacement_parity_counts[p] / total_replacements
                                   if total_replacements > 0 else 0)
    
    mean_parity = sum(p * parity_counts[p] for p in range(13)) / total_steps if total_steps > 0 else 0
    mean_replacement_parity = (sum(p * replacement_parity_counts[p] for p in range(13)) / total_replacements
                              if total_replacements > 0 else 0)
    
    replacements_per_15yr = np.mean(replacements_per_ep)
    std_replacements = np.std(replacements_per_ep)
    
    # Culling rates
    total_cow_years = total_steps / 12
    annual_culling_rate_total = total_replacements / total_cow_years if total_cow_years > 0 else 0
    annual_culling_rate_voluntary = total_voluntary / total_cow_years if total_cow_years > 0 else 0
    annual_culling_rate_death = total_death / total_cow_years if total_cow_years > 0 else 0
    annual_culling_rate_involuntary = total_involuntary / total_cow_years if total_cow_years > 0 else 0
    
    # Per-parity annual culling rates
    parity_annual_rate_total = {}
    parity_annual_rate_voluntary = {}
    parity_annual_rate_death = {}
    parity_annual_rate_involuntary = {}
    
    for p in range(13):
        cow_years_at_p = parity_counts[p] / 12 if parity_counts[p] > 0 else 0
        if cow_years_at_p > 0:
            parity_annual_rate_total[p] = replacement_parity_counts[p] / cow_years_at_p
            parity_annual_rate_voluntary[p] = vol_parity_counts[p] / cow_years_at_p
            parity_annual_rate_death[p] = death_parity_counts[p] / cow_years_at_p
            parity_annual_rate_involuntary[p] = invol_parity_counts[p] / cow_years_at_p
        else:
            parity_annual_rate_total[p] = 0.0
            parity_annual_rate_voluntary[p] = 0.0
            parity_annual_rate_death[p] = 0.0
            parity_annual_rate_involuntary[p] = 0.0

    overall_results = {
        'num_episodes': num_episodes,
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'se_reward': float(np.std(episode_rewards) / np.sqrt(num_episodes)),
        'median_reward': float(np.median(episode_rewards)),
        'min_reward': float(np.min(episode_rewards)),
        'max_reward': float(np.max(episode_rewards)),
        'mean_discounted': float(np.mean(discounted_rewards)),
        'std_discounted': float(np.std(discounted_rewards)),
        'se_discounted': float(np.std(discounted_rewards) / np.sqrt(num_episodes)),
        'episode_rewards': episode_rewards,
        'discounted_rewards': discounted_rewards,
    }
    
    steady_state_results = {
        'parity_counts': parity_counts,
        'parity_fractions': parity_fractions,
        'replacement_parity_counts': replacement_parity_counts,
        'replacement_parity_fractions': replacement_fractions,
        'vol_parity_counts': vol_parity_counts,
        'death_parity_counts': death_parity_counts,
        'invol_parity_counts': invol_parity_counts,
        'mean_parity': float(mean_parity),
        'mean_replacement_parity': float(mean_replacement_parity),
        'replacements_per_episode': float(replacements_per_15yr),
        'std_replacements_per_episode': float(std_replacements),
        'total_replacements': total_replacements,
        'total_voluntary': total_voluntary,
        'total_death': total_death,
        'total_involuntary': total_involuntary,
        'annual_culling_rate_total': float(annual_culling_rate_total),
        'annual_culling_rate_voluntary': float(annual_culling_rate_voluntary),
        'annual_culling_rate_death': float(annual_culling_rate_death),
        'annual_culling_rate_involuntary': float(annual_culling_rate_involuntary),
        'parity_annual_rate_total': parity_annual_rate_total,
        'parity_annual_rate_voluntary': parity_annual_rate_voluntary,
        'parity_annual_rate_death': parity_annual_rate_death,
        'parity_annual_rate_involuntary': parity_annual_rate_involuntary,
    }
    
    return overall_results, steady_state_results


def evaluate_by_starting_parity(q_table, env, num_episodes_per_parity=500,
                                max_steps=180, gamma=0.95, seed=None):
    """
    Evaluate the policy starting from each parity level.
    For each parity p, starts at (p, 1, 0, 0) for p>=1 or (0, 0, 9, 0) for springer.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    parity_results = {}

    for p in range(0, 13):
        ep_rewards = []
        disc_rewards = []

        for ep in range(num_episodes_per_parity):
            if p == 0:
                state = (0, 0, 9, 0)  # springer
            else:
                state = (p, 1, 0, 0)  # start of lactation, healthy

            env.state = state
            total_reward = 0.0
            total_discounted = 0.0
            discount = 1.0

            for step in range(max_steps):
                action = get_action(q_table, state)
                next_state, reward = env.step(action)
                total_reward += reward
                total_discounted += discount * reward
                discount *= gamma

                if not (next_state[0] in parity_range and
                        next_state[1] in mim_range and
                        next_state[2] in mip_range):
                    break

                state = next_state

            ep_rewards.append(total_reward)
            disc_rewards.append(total_discounted)

        parity_results[p] = {
            'mean_reward': float(np.mean(ep_rewards)),
            'std_reward': float(np.std(ep_rewards)),
            'se_reward': float(np.std(ep_rewards) / np.sqrt(num_episodes_per_parity)),
            'mean_discounted': float(np.mean(disc_rewards)),
            'std_discounted': float(np.std(disc_rewards)),
            'se_discounted': float(np.std(disc_rewards) / np.sqrt(num_episodes_per_parity)),
        }

    return parity_results


def classify_replacement(action, state, next_state, max_parity=12, max_mac=20):
    """
    Classify a replacement event into one of three categories.

    Checks whether a replacement occurred this step and returns its type.
    A replacement is detected when next_state == (0, 0, 9, 0) (springer)
    and the current state was NOT already a springer about to calve naturally.

    Returns:
        str or None: 'voluntary', 'death', 'involuntary', or None (no replacement)
    """
    parity, mac, mip, disease = state
    next_p, next_mac, next_mip, next_d = next_state

    # Check if a replacement happened: next state is a springer
    is_replacement = (next_p == 0 and next_mac == 0 and next_mip == 9)

    if not is_replacement:
        return None

    # If the agent chose 'replace', it's voluntary
    if action == 'replace':
        return 'voluntary'

    # Agent chose 'keep' but replacement happened anyway — involuntary
    # Sub-classify:

    # Death: cow died stochastically. This happens BEFORE any transition logic.
    # Detection: action='keep', cow was NOT at a boundary condition
    # (i.e., not about to overflow parity or MAC), so it must have been death.
    #
    # Parity overflow: mip==9 (about to calve) and parity+1 > max_parity
    # MAC overflow: mac+1 > max_mac

    # Check parity overflow: cow was about to calve (mip==9) and would exceed max parity
    if mip == 9 and parity + 1 > max_parity:
        return 'involuntary'  # forced out due to parity overflow after calving

    # Check MAC overflow: next month would exceed max MAC
    if mac + 1 > max_mac:
        return 'involuntary'  # forced out due to MAC overflow (failed to conceive in time)

    # Otherwise it must be death (stochastic mortality)
    return 'death'


def evaluate_steady_state_distribution(q_table, env, num_episodes=200,
                                       max_steps=180, seed=None):
    """
    Estimate the steady-state parity distribution under the learned policy.

    Runs long episodes starting from a springer (0, 0, 9, 0) and tracks
    what parity the cow is in at each time step. This reveals the herd
    age structure that would emerge if the farmer follows the DQN policy.

    Tracks three categories of replacement:
      - Voluntary:   Agent chose action='replace'
      - Death:       Agent chose 'keep' but cow died (stochastic mortality)
      - Involuntary: Agent chose 'keep' but cow was forced out
                     (parity overflow or MAC overflow)

    Also computes overall and per-parity culling rates for each category.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    max_parity = max(env.parity_range)
    max_mac = max(env.mac_range)

    parity_counts = {p: 0 for p in range(13)}
    total_steps_all = 0

    # Granular replacement tracking
    vol_parity_counts = {p: 0 for p in range(13)}     # voluntary
    death_parity_counts = {p: 0 for p in range(13)}    # death
    invol_parity_counts = {p: 0 for p in range(13)}    # involuntary (overflow)

    total_voluntary = 0
    total_death = 0
    total_involuntary = 0
    total_replacements = 0

    # Also track combined replacement parity for backward compat
    replacement_parity_counts = {p: 0 for p in range(13)}
    replacements_per_ep = []

    for ep in range(num_episodes):
        state = (0, 0, 9, 0)
        env.state = state
        ep_replacements = 0

        for step in range(max_steps):
            current_parity = state[0]
            parity_counts[current_parity] += 1
            total_steps_all += 1

            action = get_action(q_table, state)
            next_state, reward = env.step(action)

            # Classify the replacement type (if any)
            repl_type = classify_replacement(action, state, next_state,
                                             max_parity, max_mac)

            if repl_type is not None:
                replacement_parity_counts[current_parity] += 1
                total_replacements += 1
                ep_replacements += 1

                if repl_type == 'voluntary':
                    vol_parity_counts[current_parity] += 1
                    total_voluntary += 1
                elif repl_type == 'death':
                    death_parity_counts[current_parity] += 1
                    total_death += 1
                elif repl_type == 'involuntary':
                    invol_parity_counts[current_parity] += 1
                    total_involuntary += 1

            if not (next_state[0] in parity_range and
                    next_state[1] in mim_range and
                    next_state[2] in mip_range):
                break

            state = next_state

        replacements_per_ep.append(ep_replacements)

    # =========================================================================
    # Compute all metrics
    # =========================================================================

    # Parity fractions (steady-state herd distribution)
    parity_fractions = {}
    for p in range(13):
        parity_fractions[p] = parity_counts[p] / total_steps_all if total_steps_all > 0 else 0

    # Replacement fractions (combined)
    replacement_parity_fractions = {}
    for p in range(13):
        replacement_parity_fractions[p] = (replacement_parity_counts[p] / total_replacements
                                           if total_replacements > 0 else 0)

    # Per-category replacement fractions
    vol_parity_fractions = {}
    death_parity_fractions = {}
    invol_parity_fractions = {}
    for p in range(13):
        vol_parity_fractions[p] = vol_parity_counts[p] / total_voluntary if total_voluntary > 0 else 0
        death_parity_fractions[p] = death_parity_counts[p] / total_death if total_death > 0 else 0
        invol_parity_fractions[p] = invol_parity_counts[p] / total_involuntary if total_involuntary > 0 else 0

    # Summary statistics
    mean_parity = sum(p * parity_counts[p] for p in range(13)) / total_steps_all if total_steps_all > 0 else 0
    mean_replacement_parity = (sum(p * replacement_parity_counts[p] for p in range(13)) / total_replacements
                               if total_replacements > 0 else 0)

    # =========================================================================
    # Culling rates: simple formula = events per year / herd size
    #   Annual rate = (total_events / num_episodes) / 15
    #   This is equivalent to total_events / total_cow_years
    # =========================================================================
    total_cow_years = total_steps_all / 12  # convert cow-months to cow-years

    # Overall rates (events per cow per year)
    annual_culling_rate_total = total_replacements / total_cow_years if total_cow_years > 0 else 0
    annual_culling_rate_voluntary = total_voluntary / total_cow_years if total_cow_years > 0 else 0
    annual_culling_rate_death = total_death / total_cow_years if total_cow_years > 0 else 0
    annual_culling_rate_involuntary = total_involuntary / total_cow_years if total_cow_years > 0 else 0

    # Per-parity annual culling rates (for each category)
    parity_annual_rate_total = {}
    parity_annual_rate_voluntary = {}
    parity_annual_rate_death = {}
    parity_annual_rate_involuntary = {}
    for p in range(13):
        cow_years_at_p = parity_counts[p] / 12 if parity_counts[p] > 0 else 0
        if cow_years_at_p > 0:
            parity_annual_rate_total[p] = replacement_parity_counts[p] / cow_years_at_p
            parity_annual_rate_voluntary[p] = vol_parity_counts[p] / cow_years_at_p
            parity_annual_rate_death[p] = death_parity_counts[p] / cow_years_at_p
            parity_annual_rate_involuntary[p] = invol_parity_counts[p] / cow_years_at_p
        else:
            parity_annual_rate_total[p] = 0.0
            parity_annual_rate_voluntary[p] = 0.0
            parity_annual_rate_death[p] = 0.0
            parity_annual_rate_involuntary[p] = 0.0

    results = {
        # Parity distribution
        'parity_counts': parity_counts,
        'parity_fractions': parity_fractions,
        'total_steps': total_steps_all,

        # Combined replacement stats (backward compatible)
        'replacements_per_episode': float(np.mean(replacements_per_ep)),
        'std_replacements_per_episode': float(np.std(replacements_per_ep)),
        'total_replacements': total_replacements,
        'replacement_parity_counts': replacement_parity_counts,
        'replacement_parity_fractions': replacement_parity_fractions,
        'mean_parity': float(mean_parity),
        'mean_replacement_parity': float(mean_replacement_parity),

        # Granular replacement counts
        'total_voluntary': total_voluntary,
        'total_death': total_death,
        'total_involuntary': total_involuntary,
        'vol_parity_counts': vol_parity_counts,
        'death_parity_counts': death_parity_counts,
        'invol_parity_counts': invol_parity_counts,
        'vol_parity_fractions': vol_parity_fractions,
        'death_parity_fractions': death_parity_fractions,
        'invol_parity_fractions': invol_parity_fractions,

        # Annual culling rates (simple: events / cow-years)
        'annual_culling_rate_total': float(annual_culling_rate_total),
        'annual_culling_rate_voluntary': float(annual_culling_rate_voluntary),
        'annual_culling_rate_death': float(annual_culling_rate_death),
        'annual_culling_rate_involuntary': float(annual_culling_rate_involuntary),

        # Per-parity annual culling rates
        'parity_annual_rate_total': parity_annual_rate_total,
        'parity_annual_rate_voluntary': parity_annual_rate_voluntary,
        'parity_annual_rate_death': parity_annual_rate_death,
        'parity_annual_rate_involuntary': parity_annual_rate_involuntary,
    }

    return results


def save_evaluation_csv(results, parity_results, csv_path, steady_state=None):
    """Save a summary CSV for easy aggregation across runs."""
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Overall results
        writer.writerow(['metric', 'value'])
        writer.writerow(['mean_reward', results['mean_reward']])
        writer.writerow(['std_reward', results['std_reward']])
        writer.writerow(['se_reward', results['se_reward']])
        writer.writerow(['mean_discounted', results['mean_discounted']])
        writer.writerow(['std_discounted', results['std_discounted']])
        writer.writerow(['se_discounted', results['se_discounted']])
        writer.writerow([])

        # Per-parity results
        writer.writerow(['parity', 'mean_reward', 'std_reward', 'se_reward',
                         'mean_discounted', 'std_discounted', 'se_discounted'])
        for p in sorted(parity_results.keys()):
            pr = parity_results[p]
            writer.writerow([p, pr['mean_reward'], pr['std_reward'],
                             pr['se_reward'], pr['mean_discounted'],
                             pr['std_discounted'], pr['se_discounted']])

        # Steady-state parity distribution
        if steady_state is not None:
            writer.writerow([])
            writer.writerow(['parity', 'fraction_of_time', 'months_count',
                             'replacement_fraction', 'replacement_count',
                             'voluntary_count', 'death_count', 'involuntary_count'])
            for p in range(13):
                writer.writerow([
                    p,
                    steady_state['parity_fractions'][p],
                    steady_state['parity_counts'][p],
                    steady_state['replacement_parity_fractions'][p],
                    steady_state['replacement_parity_counts'][p],
                    steady_state['vol_parity_counts'][p],
                    steady_state['death_parity_counts'][p],
                    steady_state['invol_parity_counts'][p],
                ])
            writer.writerow([])
            writer.writerow(['steady_state_metric', 'value'])
            writer.writerow(['mean_parity', steady_state['mean_parity']])
            writer.writerow(['mean_replacement_parity', steady_state['mean_replacement_parity']])
            writer.writerow(['replacements_per_15yr', steady_state['replacements_per_episode']])
            writer.writerow(['std_replacements_per_15yr', steady_state['std_replacements_per_episode']])
            writer.writerow(['total_replacements', steady_state['total_replacements']])
            writer.writerow(['total_voluntary', steady_state['total_voluntary']])
            writer.writerow(['total_death', steady_state['total_death']])
            writer.writerow(['total_involuntary', steady_state['total_involuntary']])
            writer.writerow(['annual_culling_rate_total', steady_state['annual_culling_rate_total']])
            writer.writerow(['annual_culling_rate_voluntary', steady_state['annual_culling_rate_voluntary']])
            writer.writerow(['annual_culling_rate_death', steady_state['annual_culling_rate_death']])
            writer.writerow(['annual_culling_rate_involuntary', steady_state['annual_culling_rate_involuntary']])

            # Per-parity culling rates (all categories)
            writer.writerow([])
            writer.writerow(['parity', 'annual_rate_total', 'annual_rate_voluntary',
                             'annual_rate_death', 'annual_rate_involuntary'])
            for p in range(13):
                writer.writerow([
                    p,
                    steady_state['parity_annual_rate_total'][p],
                    steady_state['parity_annual_rate_voluntary'][p],
                    steady_state['parity_annual_rate_death'][p],
                    steady_state['parity_annual_rate_involuntary'][p],
                ])


def save_evaluation_pkl(results, parity_results, rewards_per_episode,
                        output_path, steady_state=None):
    """Save all evaluation results to a single .pkl file."""
    eval_data = {
        'overall': results,
        'by_parity': parity_results,
        'training_rewards': rewards_per_episode,
    }
    if steady_state is not None:
        eval_data['steady_state'] = steady_state

    with open(output_path, 'wb') as f:
        pickle.dump(eval_data, f)


def evaluate_single(pkl_path, scenario, eval_episodes=1000, parity_episodes=500,
                    eval_seed=99999):
    """
    Run full evaluation on a single trained Q-table.
    Returns (results, parity_results, steady_state).
    """
    # Set scenario
    cow_environment2.set_scenario(scenario)

    # Load trained model
    q_table, rewards_per_episode = load_q_table(pkl_path)
    print(f"  Q-table has {len(q_table)} states, trained for {len(rewards_per_episode)} episodes")

    # Create environment
    env = cow_environment2.CowEnv(parity_range, mim_range, mip_range, disease_range)

    # Overall evaluation + steady-state (combined)
    print(f"  Running overall evaluation + steady-state ({eval_episodes} episodes)...", end='', flush=True)
    start = time.time()
    results, steady_state = evaluate_policy(q_table, env, num_episodes=eval_episodes,
                                           max_steps=180, gamma=0.95, seed=eval_seed)
    print(f" {time.time()-start:.1f}s")

    # Per-parity evaluation
    print(f"  Running per-parity evaluation ({parity_episodes} ep/parity)...", end='', flush=True)
    start = time.time()
    parity_results = evaluate_by_starting_parity(q_table, env,
                                                  num_episodes_per_parity=parity_episodes,
                                                  max_steps=180, gamma=0.95,
                                                  seed=eval_seed + 1)
    print(f" {time.time()-start:.1f}s")

    # Save results
    base = pkl_path.replace('.pkl', '')
    csv_path = f"{base}_eval.csv"
    pkl_out_path = f"{base}_eval.pkl"

    save_evaluation_csv(results, parity_results, csv_path, steady_state)
    save_evaluation_pkl(results, parity_results, rewards_per_episode,
                        pkl_out_path, steady_state)

    print(f"  Saved: {csv_path}")
    print(f"  Saved: {pkl_out_path}")

    return results, parity_results, steady_state


def print_summary(results, parity_results, steady_state):
    """Print a compact summary of evaluation results."""
    print(f"    Mean Reward:      ${results['mean_reward']:>12,.2f} ± ${results['se_reward']:,.2f}")
    print(f"    Mean Discounted:  ${results['mean_discounted']:>12,.2f} ± ${results['se_discounted']:,.2f}")
    print(f"    Mean Herd Parity: {steady_state['mean_parity']:.2f}")
    print(f"    Mean Repl Parity: {steady_state['mean_replacement_parity']:.2f}")
    print(f"    Replacements/15yr: {steady_state['replacements_per_episode']:.1f}")
    avg_lifespan = 180 / steady_state['replacements_per_episode'] if steady_state['replacements_per_episode'] > 0 else float('inf')
    print(f"    Avg Cow Lifespan: {avg_lifespan:.1f} months ({avg_lifespan/12:.1f} years)")
    print(f"    --- Culling Rates (annual) ---")
    print(f"    Total:       {steady_state['annual_culling_rate_total']*100:.1f}%")
    print(f"    Voluntary:   {steady_state['annual_culling_rate_voluntary']*100:.1f}%")
    print(f"    Death:        {steady_state['annual_culling_rate_death']*100:.1f}%")
    print(f"    Involuntary: {steady_state['annual_culling_rate_involuntary']*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Local evaluation of trained DQN Q-tables (no training needed)")
    parser.add_argument(
        "--collected_dir", type=str, default=None,
        help="Directory containing all 25 DQN_<scenario>_seed<N>.pkl files")
    parser.add_argument(
        "--single", type=str, default=None,
        help="Path to a single .pkl file to evaluate")
    parser.add_argument(
        "--scenario", type=str, default=None,
        choices=SCENARIOS,
        help="Scenario (required for --single mode)")
    parser.add_argument(
        "--eval_episodes", type=int, default=1000,
        help="Number of overall evaluation episodes (default: 1000)")
    parser.add_argument(
        "--parity_episodes", type=int, default=500,
        help="Number of episodes per parity (default: 500)")  # dist_episodes removed
    parser.add_argument(
        "--eval_seed", type=int, default=99999,
        help="Random seed for evaluation reproducibility (default: 99999)")
    parser.add_argument(
        "--skip_existing", action="store_true",
        help="Skip evaluation if _eval.csv and _eval.pkl already exist")
    args = parser.parse_args()

    if args.single:
        # Single file mode
        if not args.scenario:
            parser.error("--scenario is required when using --single")
        print(f"\n{'='*70}")
        print(f"Evaluating: {args.single} (scenario: {args.scenario})")
        print(f"{'='*70}")

        results, parity_results, steady_state = evaluate_single(
            args.single, args.scenario,
            eval_episodes=args.eval_episodes,
            parity_episodes=args.parity_episodes,
            eval_seed=args.eval_seed
        )
        print_summary(results, parity_results, steady_state)

    elif args.collected_dir:
        # Batch mode: evaluate all 25 Q-tables
        print(f"\n{'='*70}")
        print(f"BATCH EVALUATION - All scenarios and seeds")
        print(f"Directory: {args.collected_dir}")
        print(f"Settings: {args.eval_episodes} eval eps (also used for steady-state), "
              f"{args.parity_episodes} parity eps")
        print(f"{'='*70}")

        total_start = time.time()
        completed = 0
        failed = 0

        for scenario in SCENARIOS:
            for seed in SEEDS:
                # Try common naming patterns
                patterns = [
                    f"DQN_{scenario}_seed{seed}.pkl",
                    f"DQN_{scenario}_{seed}.pkl",
                ]

                pkl_path = None
                for pattern in patterns:
                    candidate = os.path.join(args.collected_dir, pattern)
                    if os.path.exists(candidate):
                        pkl_path = candidate
                        break

                if pkl_path is None:
                    # Try glob
                    matches = glob.glob(os.path.join(args.collected_dir,
                                                     f"DQN_{scenario}*seed{seed}*.pkl"))
                    # Exclude _eval.pkl files
                    matches = [m for m in matches if '_eval' not in m]
                    if matches:
                        pkl_path = matches[0]

                if pkl_path is None:
                    print(f"\n  WARNING: No Q-table found for {scenario}/seed{seed}")
                    failed += 1
                    continue

                print(f"\n[{completed+1}/25] {scenario} seed={seed}: {os.path.basename(pkl_path)}")

                # Check if evaluation files already exist
                if args.skip_existing:
                    eval_csv = os.path.join(args.collected_dir,
                                            f"DQN_{scenario}_seed{seed}_eval.csv")
                    eval_pkl = os.path.join(args.collected_dir,
                                            f"DQN_{scenario}_seed{seed}_eval.pkl")
                    if os.path.exists(eval_csv) and os.path.exists(eval_pkl):
                        print(f"  SKIPPING: _eval files already exist (use without --skip_existing to re-evaluate)")
                        completed += 1
                        continue

                try:
                    results, parity_results, steady_state = evaluate_single(
                        pkl_path, scenario,
                        eval_episodes=args.eval_episodes,
                        parity_episodes=args.parity_episodes,
                        eval_seed=args.eval_seed
                    )
                    print_summary(results, parity_results, steady_state)
                    completed += 1
                except Exception as e:
                    print(f"  ERROR: {e}")
                    failed += 1

        total_time = time.time() - total_start
        print(f"\n{'='*70}")
        print(f"BATCH EVALUATION COMPLETE")
        print(f"  Completed: {completed}/25")
        print(f"  Failed:    {failed}")
        print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"{'='*70}")
        print(f"\nNext step: run aggregate_results.py to aggregate across seeds")
        print(f"  python aggregate_results.py --results_dir {args.collected_dir}")

    else:
        parser.error("Either --collected_dir or --single is required")


if __name__ == "__main__":
    main()
