"""
Monte Carlo Policy Evaluation for trained DQN models.

Evaluates a trained DQN policy by running many episodes in the environment
and recording the actual discounted rewards obtained.

Usage:
    python evaluate_dqn.py --model outputs/DQN_2025_seed42.pkl --scenario 2025 --eval_episodes 1000
"""

import argparse
import numpy as np
import pickle
import os
import csv
import time
import cow_environment2
import utility

parity_range = range(13)
mim_range = range(21)
mip_range = range(10)
disease_range = range(2)


def load_q_table(pkl_path):
    """Load the Q-table from a saved .pkl file."""
    with open(pkl_path, 'rb') as f:
        q_table, rewards_per_episode, epsilon = pickle.load(f)
    return q_table, rewards_per_episode


def get_action(q_table, state):
    """
    Get the greedy action from the Q-table for a given state.
    Returns 'keep' or 'replace'.
    """
    if state in q_table:
        q_keep = q_table[state]['keep']
        q_replace = q_table[state]['replace']
        return 'keep' if q_keep >= q_replace else 'replace'
    else:
        # If state not in Q-table (shouldn't happen), default to keep
        return 'keep'


def evaluate_policy(q_table, env, num_episodes=1000, max_steps=180, gamma=0.95,
                    seed=None):
    """
    Evaluate a policy by running Monte Carlo episodes.

    Args:
        q_table: Dictionary mapping states to action Q-values.
        env: CowEnv environment instance.
        num_episodes: Number of evaluation episodes.
        max_steps: Maximum steps per episode.
        gamma: Discount factor (must match training).
        seed: Random seed for reproducibility of evaluation.

    Returns:
        Dictionary with evaluation results.
    """
    if seed is not None:
        np.random.seed(seed)
        import random
        random.seed(seed)

    episode_rewards = []          # undiscounted total reward per episode
    discounted_rewards = []       # discounted total reward per episode

    for ep in range(num_episodes):
        state = env.reset()
        total_reward = 0.0
        total_discounted = 0.0
        discount = 1.0

        for step in range(max_steps):
            action = get_action(q_table, state)
            next_state, reward = env.step(action)
            total_reward += reward
            total_discounted += discount * reward
            discount *= gamma

            # Check if episode should end (same logic as training)
            if not (next_state[0] in parity_range and
                    next_state[1] in mim_range and
                    next_state[2] in mip_range):
                break

            state = next_state

        episode_rewards.append(total_reward)
        discounted_rewards.append(total_discounted)

    results = {
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
    return results


def evaluate_by_starting_parity(q_table, env, num_episodes_per_parity=500,
                                max_steps=180, gamma=0.95, seed=None):
    """
    Evaluate the policy starting from each parity level.

    For each parity p (1..12), starts the cow at state (p, 1, 0, 0) and
    runs num_episodes_per_parity episodes.

    Returns:
        Dictionary mapping parity -> evaluation results dict.
    """
    if seed is not None:
        np.random.seed(seed)
        import random
        random.seed(seed)

    parity_results = {}

    for p in range(0, 13):  # 0 = springer, 1..12 = productive parities
        ep_rewards = []
        disc_rewards = []

        for ep in range(num_episodes_per_parity):
            if p == 0:
                state = (0, 0, 9, 0)  # springer
            else:
                state = (p, 1, 0, 0)  # start of lactation

            env.state = state  # manually set the starting state
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


def save_evaluation_results(results, parity_results, q_table, rewards_per_episode,
                            output_path):
    """
    Save all evaluation results to a single .pkl file.
    """
    eval_data = {
        'overall': results,
        'by_parity': parity_results,
        'training_rewards': rewards_per_episode,
    }

    with open(output_path, 'wb') as f:
        pickle.dump(eval_data, f)

    print(f"Evaluation results saved to {output_path}")


def save_evaluation_csv(results, parity_results, csv_path):
    """
    Save a summary CSV for easy aggregation across runs.
    """
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

    print(f"Evaluation CSV saved to {csv_path}")


def print_evaluation_summary(results, parity_results):
    """Print a formatted summary of evaluation results."""
    print("\n" + "=" * 70)
    print("MONTE CARLO POLICY EVALUATION RESULTS")
    print("=" * 70)

    print(f"\nOverall Performance ({results['num_episodes']} episodes):")
    print(f"  Mean Total Reward:      ${results['mean_reward']:,.2f} "
          f"(+/- ${results['se_reward']:,.2f})")
    print(f"  Std Total Reward:       ${results['std_reward']:,.2f}")
    print(f"  Mean Discounted Reward: ${results['mean_discounted']:,.2f} "
          f"(+/- ${results['se_discounted']:,.2f})")
    print(f"  Min / Max Reward:       ${results['min_reward']:,.2f} / "
          f"${results['max_reward']:,.2f}")

    print(f"\nPer-Parity Performance (discounted rewards):")
    print(f"  {'Parity':<10} {'Mean':>12} {'Std':>12} {'SE':>12}")
    print(f"  {'-'*46}")
    for p in sorted(parity_results.keys()):
        pr = parity_results[p]
        label = 'Springer' if p == 0 else f'Parity {p}'
        print(f"  {label:<10} ${pr['mean_discounted']:>10,.2f} "
              f"${pr['std_discounted']:>10,.2f} "
              f"${pr['se_discounted']:>10,.2f}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Monte Carlo Policy Evaluation for DQN")
    parser.add_argument(
        "--model", required=True,
        help="Path to the trained DQN .pkl file (Q-table)")
    parser.add_argument(
        "--scenario", type=str, default="2025",
        choices=["2025", "OG", "OB", "UG", "UB"],
        help="Scenario for animal constants (default: 2025)")
    parser.add_argument(
        "--eval_episodes", type=int, default=1000,
        help="Number of evaluation episodes for overall performance (default: 1000)")
    parser.add_argument(
        "--parity_episodes", type=int, default=500,
        help="Number of episodes per parity for per-parity evaluation (default: 500)")
    parser.add_argument(
        "--eval_seed", type=int, default=99999,
        help="Random seed for evaluation reproducibility (default: 99999)")
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for evaluation results .pkl "
             "(default: <model>_eval.pkl)")
    args = parser.parse_args()

    # Set scenario
    cow_environment2.set_scenario(args.scenario)

    # Determine output path
    if args.output is None:
        base = args.model.replace('.pkl', '')
        output_pkl = f"{base}_eval.pkl"
        output_csv = f"{base}_eval.csv"
    else:
        output_pkl = args.output
        output_csv = args.output.replace('.pkl', '.csv')

    # Load trained model
    print(f"Loading trained model from {args.model}")
    q_table, rewards_per_episode = load_q_table(args.model)
    print(f"  Q-table has {len(q_table)} states")
    print(f"  Training ran for {len(rewards_per_episode)} episodes")

    # Create environment
    env = cow_environment2.CowEnv(parity_range, mim_range, mip_range,
                                  disease_range)

    # Run overall evaluation
    print(f"\nRunning overall evaluation ({args.eval_episodes} episodes)...")
    start = time.time()
    results = evaluate_policy(q_table, env,
                              num_episodes=args.eval_episodes,
                              max_steps=180, gamma=0.95,
                              seed=args.eval_seed)
    elapsed_overall = time.time() - start
    print(f"  Completed in {elapsed_overall:.1f}s")

    # Run per-parity evaluation
    print(f"\nRunning per-parity evaluation "
          f"({args.parity_episodes} episodes per parity)...")
    start = time.time()
    parity_results = evaluate_by_starting_parity(
        q_table, env,
        num_episodes_per_parity=args.parity_episodes,
        max_steps=180, gamma=0.95,
        seed=args.eval_seed + 1)
    elapsed_parity = time.time() - start
    print(f"  Completed in {elapsed_parity:.1f}s")

    # Print summary
    print_evaluation_summary(results, parity_results)

    # Save results
    save_evaluation_results(results, parity_results, q_table,
                            rewards_per_episode, output_pkl)
    save_evaluation_csv(results, parity_results, output_csv)

    print(f"\nTotal evaluation time: {elapsed_overall + elapsed_parity:.1f}s")


if __name__ == "__main__":
    main()
