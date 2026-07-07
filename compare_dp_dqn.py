"""
Compare the exact DP solution against trained DQN policies for one scenario.

The DQN is stochastic, so a single run can land near the DP optimum by luck.
This script therefore aggregates over MULTIPLE seeds and reports each head-to-head
metric as mean +/- SD (and range), so the claim becomes "the DQN reliably
recovers the optimum" rather than "one run happened to".

Metrics (requested by the reviewer), computed per seed then aggregated:
  - policy agreement: fraction of states where DP and DQN choose the same action
  - agreement of the cow-value SIGN (keep vs replace preference)
  - correlation and mean absolute gap of cow value (Q_keep - Q_replace)
  - replacement boundary for healthy open cows (per parity): the DP value and the
    across-seed DQN mean/range

Usage:
  # aggregate over all seeds (defaults: --dqn-dir CHTC_DQN_results, seeds 42,123,456,789,1024):
  python compare_dp_dqn.py --scenario 2025 --dp dp_results/dp_2025_qtable.pkl

  # or point at a different folder / subset of seeds:
  python compare_dp_dqn.py --scenario 2025 --dqn-dir CHTC_DQN_results \
      --seeds 42 123 --dp dp_results/dp_2025_qtable.pkl

  # or pass explicit DQN pickles (one or more):
  python compare_dp_dqn.py --scenario 2025 \
      --dqn CHTC_DQN_results/DQN_2025_seed42.pkl \
      --dp  dp_results/dp_2025_qtable.pkl

The DQN pickle is the repo's standard format: (q_table, rewards_per_episode, epsilon),
where q_table[state] = {'keep':float, 'replace':float}. The DP pickle is
{state: {'keep':float, 'replace':float}}.
"""
import argparse, os, pickle
import numpy as np

DEFAULT_SEEDS = [42, 123, 456, 789, 1024]
DEFAULT_DQN_DIR = 'CHTC_DQN_results'


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


def compare_one(dqn, dp):
    """All per-seed metrics against the DP Q* on the common state set."""
    common = sorted(set(dqn) & set(dp))
    n = len(common)
    agree = sum(1 for s in common if act(dqn[s]) == act(dp[s]))
    cv_dp = np.array([dp[s]['keep'] - dp[s]['replace'] for s in common])
    cv_dqn = np.array([dqn[s]['keep'] - dqn[s]['replace'] for s in common])
    return {
        'n': n,
        'policy_agree_pct': 100.0 * agree / n,
        'corr': float(np.corrcoef(cv_dp, cv_dqn)[0, 1]),
        'mae': float(np.mean(np.abs(cv_dp - cv_dqn))),
        'boundaries': {p: boundary(dqn, p) for p in range(1, 13)},
    }


def fmt_mean_sd(vals, unit='', prec=1):
    a = np.array(vals, dtype=float)
    if len(a) == 1:
        return f"{unit}{a[0]:.{prec}f}"
    return (f"{unit}{a.mean():.{prec}f} +/- {a.std(ddof=1):.{prec}f}  "
            f"[{unit}{a.min():.{prec}f}, {unit}{a.max():.{prec}f}]")


def resolve_paths(args):
    """Return list of (label, path) for the DQN pickles to compare."""
    if args.dqn:
        return [(os.path.basename(p), p) for p in args.dqn]
    if not args.dqn_dir:
        raise SystemExit("Provide either --dqn <paths...> or --dqn-dir <folder>.")
    pairs = []
    for seed in args.seeds:
        p = os.path.join(args.dqn_dir, f"DQN_{args.scenario}_seed{seed}.pkl")
        if os.path.exists(p):
            pairs.append((f"seed{seed}", p))
        else:
            print(f"  (skip) missing: {p}")
    if not pairs:
        raise SystemExit(f"No DQN pickles found for scenario {args.scenario} in {args.dqn_dir}")
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', required=True)
    ap.add_argument('--dqn', nargs='*', help='explicit DQN pickle path(s)')
    ap.add_argument('--dqn-dir', default=DEFAULT_DQN_DIR,
                    help='folder holding DQN_<scenario>_seed<seed>.pkl files')
    ap.add_argument('--seeds', nargs='*', type=int, default=DEFAULT_SEEDS,
                    help='seeds to aggregate when using --dqn-dir')
    ap.add_argument('--dp', required=True)
    args = ap.parse_args()

    dp = load_dp(args.dp)
    pairs = resolve_paths(args)
    results = [(label, compare_one(load_dqn(path), dp)) for label, path in pairs]
    n_states = results[0][1]['n']

    print(f"\n=== DP vs DQN comparison ({args.scenario}) ===")
    print(f"DQN runs compared: {len(results)}  ({', '.join(l for l, _ in results)})")
    print(f"common states per run: {n_states}\n")

    print("cow value = Q_keep - Q_replace ($); corr & MAE are on that quantity across states.\n")

    # ---- per-seed table ----
    print(f"{'run':>10} {'policy%':>9} {'cv corr':>8} {'cv MAE($)':>10}")
    for label, r in results:
        print(f"{label:>10} {r['policy_agree_pct']:>9.1f} "
              f"{r['corr']:>8.3f} {r['mae']:>10.0f}")

    # ---- aggregated across seeds ----
    print("\n--- aggregate over seeds (mean +/- SD [min, max]) ---")
    print(f"policy agreement:      {fmt_mean_sd([r['policy_agree_pct'] for _, r in results], '', 1)} %")
    print(f"cow-value correlation: {fmt_mean_sd([r['corr'] for _, r in results], '', 3)}")
    print(f"cow-value MAE:         {fmt_mean_sd([r['mae'] for _, r in results], '$', 0)}")

    # ---- replacement boundary per parity: DP vs across-seed DQN ----
    print("\nReplacement boundary (healthy open cow), MAC at which replace becomes optimal:")
    print("(exact = seeds hitting DP's MAC exactly; +/-1 = seeds within one MAC of DP)")
    print(f"{'parity':>6} {'DP':>5} {'DQN mean':>9} {'DQN range':>12} {'exact':>7} {'+/-1':>6}")
    for p in range(1, 13):
        bd = boundary(dp, p)
        vals = [r['boundaries'][p] for _, r in results]
        present = [v for v in vals if v is not None]
        if present:
            mean_s = f"{np.mean(present):.1f}"
            rng_s = f"[{min(present)}, {max(present)}]"
        else:
            mean_s, rng_s = "None", "-"
        n_never = len(vals) - len(present)
        if n_never:
            rng_s += f" (+{n_never} None)"
        exact = sum(1 for v in vals if v == bd)
        within1 = sum(1 for v in present if bd is not None and abs(v - bd) <= 1)
        print(f"{p:>6} {str(bd):>5} {mean_s:>9} {rng_s:>12} "
              f"{exact:>3}/{len(vals)} {within1:>3}/{len(vals)}")


if __name__ == '__main__':
    main()
