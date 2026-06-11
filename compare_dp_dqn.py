"""
Compare the exact DP solution against a trained DQN policy for one scenario.

Produces the head-to-head numbers requested by the reviewer:
  - policy agreement: fraction of states where DP and DQN choose the same action
  - replacement-boundary agreement for healthy open cows (per parity)
  - agreement of the cow-value SIGN (keep vs replace preference)
  - correlation and mean absolute gap of cow value (Q_keep - Q_replace)

Usage:
  python compare_dp_dqn.py --scenario 2025 \
      --dqn outputs/DQN_2025_seed42.pkl \
      --dp  outputs/dp/dp_2025_qtable.pkl

The DQN pickle is the repo's standard format: (q_table, rewards_per_episode, epsilon),
where q_table[state] = {'keep':float, 'replace':float}. The DP pickle is
{state: {'keep':float, 'replace':float}}.
"""
import argparse, pickle
import numpy as np


def load_dqn(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj[0] if isinstance(obj, tuple) else obj


def load_dp(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def act(q):
    return 'keep' if q['keep'] >= q['replace'] else 'replace'


def boundary(qt, parity):
    """First MAC at which a healthy open cow switches keep->replace (None if never)."""
    for mac in range(1, 21):
        s = (parity, mac, 0, 0)
        if s in qt and act(qt[s]) == 'replace':
            return mac
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', required=True)
    ap.add_argument('--dqn', required=True)
    ap.add_argument('--dp', required=True)
    args = ap.parse_args()

    dqn = load_dqn(args.dqn)
    dp = load_dp(args.dp)
    common = sorted(set(dqn) & set(dp))
    n = len(common)

    agree = sum(1 for s in common if act(dqn[s]) == act(dp[s]))
    cv_dp = np.array([dp[s]['keep'] - dp[s]['replace'] for s in common])
    cv_dqn = np.array([dqn[s]['keep'] - dqn[s]['replace'] for s in common])
    sign_agree = np.mean(np.sign(cv_dp) == np.sign(cv_dqn))
    corr = float(np.corrcoef(cv_dp, cv_dqn)[0, 1])
    mae = float(np.mean(np.abs(cv_dp - cv_dqn)))

    print(f"\n=== DP vs DQN comparison ({args.scenario}) ===")
    print(f"common states: {n}")
    print(f"policy agreement (keep/replace): {agree}/{n} = {100*agree/n:.1f}%")
    print(f"cow-value sign agreement:        {100*sign_agree:.1f}%")
    print(f"cow-value correlation:           {corr:.3f}")
    print(f"mean |cow_value_DP - cow_value_DQN|: ${mae:.0f}")
    print("\nReplacement boundary (healthy open cow), MAC at which replace becomes optimal:")
    print(f"{'parity':>6} {'DP':>5} {'DQN':>5}")
    for p in range(1, 13):
        bd, bq = boundary(dp, p), boundary(dqn, p)
        print(f"{p:>6} {str(bd):>5} {str(bq):>5}")


if __name__ == '__main__':
    main()
