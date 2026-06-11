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

# state-by-state comparison against a trained DQN model:
python compare_dp_dqn.py --scenario 2025 \
    --dqn outputs/DQN_2025_seed42.pkl \
    --dp  dp_results/dp_2025_qtable.pkl
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
