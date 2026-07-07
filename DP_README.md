# Dynamic-programming (DP) benchmark

This branch adds an **exact** dynamic-programming solution of the same dairy-cow
replacement MDP that the DQN learns, to serve as ground truth (addresses
Reviewer 1's request for a direct DP-vs-DRL comparison).

## What it does

`dp_solver.py` solves the infinite-horizon discounted MDP (γ = 0.95) by value
iteration. It is consistent with the DQN by construction:

- It **imports the same parameters** (`animal_constants_*.py`) and the **same
  reward functions** (`CowEnv.calculate_milk_income`, `calculate_feed_cost`,
  `calculate_slaughter_income`).
- It reproduces the **exact transition logic of `CowEnv.step()`** analytically:
  every stochastic branch (on-farm death, conception success, clinical-mastitis
  incidence/recovery) and every forced-replacement rule (death, max parity at
  calving, max lactation length) is enumerated with its probability, instead of
  being sampled. The result is the exact optimal value function `V*`,
  action-value function `Q*`, and greedy policy.
- The DP `Q*` is saved in the **same dict format** as the DQN `q_table`
  (`{state: {'keep':…, 'replace':…}}`), so the two are directly comparable.

State space: 2,209 valid states (`possible_state2`, DNB window 3–10).
Value iteration converges in ~420 sweeps (final Bellman residual < 1e-7) in a
few seconds per scenario.

## Run

```bash
python dp_solver.py --scenario all --outdir dp_results
python build_comparison.py            # writes dp_results/dp_vs_dqn_summary.md

# state-by-state comparison against the trained DQN models. By default this
# aggregates over all 5 seeds (42,123,456,789,1024) found in CHTC_DQN_results/
# and reports each metric as mean +/- SD, so the agreement reflects the DQN's
# reliability, not a single lucky run:
python compare_dp_dqn.py --scenario 2025 --dp dp_results/dp_2025_qtable.pkl

# other scenarios (swap both the --scenario and the matching --dp table):
python compare_dp_dqn.py --scenario OB --dp dp_results/dp_OB_qtable.pkl
python compare_dp_dqn.py --scenario OG --dp dp_results/dp_OG_qtable.pkl
python compare_dp_dqn.py --scenario UB --dp dp_results/dp_UB_qtable.pkl
python compare_dp_dqn.py --scenario UG --dp dp_results/dp_UG_qtable.pkl

# a single run, or a custom folder / seed subset:
python compare_dp_dqn.py --scenario 2025 --dqn CHTC_DQN_results/DQN_2025_seed42.pkl \
    --dp dp_results/dp_2025_qtable.pkl
python compare_dp_dqn.py --scenario 2025 --dqn-dir CHTC_DQN_results --seeds 42 123 \
    --dp dp_results/dp_2025_qtable.pkl
```

## Outputs (`dp_results/`)

| File | Contents |
|---|---|
| `dp_<scn>_qtable.pkl` | exact `Q*` (DQN-compatible format) |
| `dp_<scn>_policy.csv` | per-state V, Q_keep, Q_replace, cow value, optimal action |
| `dp_<scn>_pregnancy.csv` | value of a new pregnancy by parity × conception MAC |
| `dp_<scn>_cm_cost.csv` | cost of clinical mastitis by parity × MAC |
| `dp_<scn>_herd.json` | annual return/stall, culling decomposition, mean parity, productive life |
| `dp_summary.json` | all scenarios combined |
| `dp_vs_dqn_summary.md` | side-by-side table vs. the manuscript's DRL numbers |

## Headline comparison (exact DP vs. DRL reported in the paper)

| Scenario | DRL return | DP return | DRL mean parity | DP mean parity |
|---|---|---|---|---|
| BL | $3,082 | $3,101 (+0.6%) | 2.71 | 2.69 |
| OB | $1,036 | $1,061 (+2.4%) | 3.43 | 3.39 |
| UG | $4,545 | $4,580 (+0.8%) | 3.27 | 3.31 |

The DRL agent recovers the exact optimum closely: annual return within ~2.5%
and steady-state herd structure within ~0.1 parity across scenarios. Baseline
value of a new pregnancy in parity 1 matches almost exactly ($415 DP vs $418
DRL).

## State-by-state DP-vs-DRL agreement (5 seeds per scenario)

`compare_dp_dqn.py` compares each trained DQN `Q`-table against the exact DP `Q*`
over all 2,209 common states. Because the DQN is stochastic, we run all five
seeds (42, 123, 456, 789, 1024) per scenario and report mean ± SD. Metrics:
**policy agreement** = fraction of states where DQN and DP choose the same
keep/replace action; **cow-value correlation / MAE** = Pearson correlation and
mean absolute gap of the cow value `Q_keep − Q_replace` (in $) across states.

| Scenario | Policy agreement | Cow-value corr. | Cow-value MAE |
|---|---|---|---|
| BL (2025) | 93.5% ± 2.0 | 0.874 ± 0.018 | $152 ± 31 |
| OB        | 94.7% ± 0.8 | 0.977 ± 0.005 | $52 ± 10 |
| OG        | 94.8% ± 1.0 | 0.973 ± 0.004 | $92 ± 18 |
| UB        | 93.3% ± 2.1 | 0.950 ± 0.018 | $129 ± 35 |
| UG        | 94.5% ± 1.5 | 0.950 ± 0.014 | $165 ± 45 |

Across all scenarios and every seed, the DQN agrees with the exact optimal
action on **~93–95% of states**, and cow-value correlation is **0.87–0.98**.
The tight SDs show this is a reliable property of the learning, not a
single lucky run.

**Replacement boundary (healthy open cow).** For each parity, the DP gives the
MAC at which replacement becomes optimal. Across the five seeds the DQN boundary
lands **within ±1 MAC of DP at essentially every parity** in every scenario
(e.g. BL: 5/5 seeds within ±1 at 11 of 12 parities). The DQN tends to replace
slightly *earlier* than optimal at the highest parities. The full per-parity
`DP | DQN mean | range | exact | ±1` table is printed by `compare_dp_dqn.py` for
each scenario.

## Known definitional differences (to reconcile when writing)

- **Annual culling rate** is ~2 percentage points lower under DP than under the
  DRL Monte-Carlo evaluation, even though mean parity matches closely. This is a
  bookkeeping difference in how exits are counted (DP uses 12 × stationary
  monthly exit probability), not a policy difference. The DRL culling rate is
  computed from simulation; we should compute the DP culling rate the same way,
  or report both under a single explicit definition.
- **Cost of CM** and **value of a new pregnancy** here use a value-function
  definition (V difference between health/pregnancy states). For early lactation
  and parity 1 these match the paper well; for late/extended lactation and higher
  parities they diverge, likely because the manuscript's quantities use a
  different Q-difference convention. The exact convention used in the paper should
  be applied identically to the DP `Q*` (the saved `dp_<scn>_qtable.pkl` supports
  any such recomputation).
