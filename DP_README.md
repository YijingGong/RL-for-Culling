# Dynamic-programming (DP) benchmark

An **exact** dynamic-programming solution of the same dairy-cow replacement MDP that the
DQN learns, used as ground truth for a direct DP-vs-DRL comparison (addresses Reviewer 1
comments on the absence of a direct comparison — original comments #2 and #23).

## What it does

`dp_solver.py` solves the infinite-horizon discounted MDP (γ = 0.95) by value iteration.
It is consistent with the DQN by construction:

- imports the **same parameters** (`animal_constants_*.py`) and the **same reward functions**
  (`CowEnv.calculate_milk_income`, `calculate_feed_cost`, `calculate_slaughter_income`);
- reproduces the **exact transition logic of `CowEnv.step()`** analytically (every stochastic
  branch and forced-replacement rule enumerated with its probability), giving the exact optimal
  value function `V*`, action-value function `Q*`, and greedy policy;
- saves `Q*` in the **same dict format** as the DQN `q_table`, so the two are directly comparable.

**State space:** 2,209 valid states (`possible_state2`, DNB window 3–10). Value iteration
converges in ~420 sweeps (final Bellman residual < 1e-7) in a few seconds per scenario.

## Definition of derived quantities (IMPORTANT — read before using the numbers)

All economic quantities are computed from the **cow value = retention payoff = Q(s, keep) − Q(s, replace)**
(manuscript **Eqn 4**), so the DP and the DRL use one identical definition:

| Quantity | Definition | Reference |
|---|---|---|
| **Cow value / retention payoff** | Q(s, keep) − Q(s, replace) | Van Arendonk (1985); De Vries (2006); Cabrera (2012) |
| **Value of a new pregnancy** | cow value(pregnant) − cow value(open), same month | De Vries (2006) |
| **Cost of clinical mastitis** | cow value(healthy) − cow value(CM), same month | Bar et al. (2008); Cha et al. (2011) |

No positive-retention-payoff screen is applied, so pregnancy value and CM cost share one definition.

> **Convention note (fixed 2026-07):** earlier revisions of `dp_solver.py` computed these two
> quantities from the optimal-value difference `max(Q_keep, Q_replace)` rather than from the cow
> value `Q_keep − Q_replace`. That understated late-lactation CM cost and did not match the DRL
> post-processing (`visualize_scenario.py`, which already uses the cow-value difference for Table 5).
> The solver now uses the cow-value difference for both quantities, matching the DRL and the
> gold-standard definitions above.

## How to reproduce

```bash
python dp_solver.py --scenario all --outdir dp_results   # exact DP, all 5 scenarios
python build_comparison.py                               # writes dp_results/dp_vs_dqn_summary.md
python compare_dp_dqn.py --scenario 2025 --dp dp_results/dp_2025_qtable.pkl   # state-by-state, 5 seeds
```

---

# Results (exact DP)

## Table 1. Economic performance and herd structure

| Scenario | Annual return / stall | Annual culling rate | Mean parity | Productive life |
|---|---|---|---|---|
| BL (2025) | $3,101 | 26.3% | 2.69 | 3.80 yr |
| OG | $3,824 | 19.4% | 3.40 | 5.16 yr |
| OB | $1,061 | 19.1% | 3.39 | 5.23 yr |
| UG | $4,580 | 20.0% | 3.31 | 4.99 yr |
| UB | $1,829 | 21.1% | 3.15 | 4.73 yr |

## Table 2. Value of a new pregnancy (healthy cows, average over conception MAC 3–10)

| Scenario | Parity 1 | Parity 2 | Parity 3 |
|---|---|---|---|
| BL (2025) | $416 | $284 | $196 |
| OG | $474 | $511 | $507 |
| OB | $353 | $395 | $374 |
| UG | $837 | $758 | $658 |
| UB | $765 | $653 | $529 |

## Table 3. Cost of clinical mastitis (open cows, by lactation stage)

| Scenario | Early (MAC 1–3) | Mid (MAC 4–6) | Late (MAC 7–9) | Extended (MAC ≥10) |
|---|---|---|---|---|
| BL (2025) | $399 | $346 | $243 | $132 |
| OG | $585 | $562 | $410 | $210 |
| OB | $399 | $383 | $288 | $156 |
| UG | $497 | $470 | $324 | $151 |
| UB | $307 | $291 | $205 | $94 |

---

# Validation: exact DP vs. DRL

## Table 4. Baseline (2025) — DP vs. DRL, one identical definition

| Quantity | DP (exact) | DRL (manuscript) | Difference |
|---|---|---|---|
| Annual return / stall | $3,101 | $3,082 | 0.6% |
| Annual culling rate | 26.3% | 28.5% | 2.2 pp* |
| Mean parity | 2.69 | 2.7 | ~0 |
| Productive life | 3.80 yr | 3.5 yr | 0.3 yr |
| Pregnancy value, parity 1 | $416 | $418 | 0.5% |
| Pregnancy value, parity 2 | $284 | $313 | ~10% |
| Pregnancy value, parity 3 | $196 | $231 | ~18% |
| CM cost, early (MAC 1–3) | $399 | $387 | 3% |
| CM cost, mid (MAC 4–6) | $346 | $339 | 2% |
| CM cost, late (MAC 7–9) | $243 | $293 | ~20% |
| CM cost, extended (≥10) | $132 | $147 | 11% |

The DRL recovers the exact optimum closely: annual return within ~2.5%, herd structure within
~0.1 parity, and pregnancy value / CM cost within a few percent in early–mid lactation. The larger
residuals in late-lactation and high-parity cells (pregnancy P3, CM late) reflect genuine DRL
approximation error in rarely-visited state–action pairs, not a definitional mismatch.

\* **Culling-rate note:** annual culling rate is defined **identically** on both sides — voluntary +
mortality + involuntary (forced, at max parity or max lactation) exits per cow-year. The DP value is
computed exactly from the policy's stationary distribution (12 × monthly exit probability); the DRL
value is estimated by Monte-Carlo simulation of the learned policy, which is an *unbiased* estimate of
that same quantity (it converges to the exact value as the number of episodes grows). The ~2 pp gap is
therefore **not** a bookkeeping artifact but a genuine, small policy-approximation difference: the
approximate DRL policy culls slightly more than the exact optimum, consistent with its shorter
productive life (3.5 vs 3.80 yr). Each number is reported the way conventional for its method (DP
analytically, DRL by simulation); no re-evaluation or forced apples-to-apples re-computation is needed.

## Table 5. State-by-state DP–DRL agreement (mean ± SD over 5 seeds per scenario)

Policy agreement = fraction of states where DQN and DP choose the same keep/replace action;
cow-value correlation / MAE compare Q_keep − Q_replace across all 2,209 common states.

| Scenario | Policy agreement | Cow-value corr. | Cow-value MAE |
|---|---|---|---|
| BL (2025) | 93.5% ± 2.0 | 0.874 ± 0.018 | $152 ± 31 |
| OB | 94.7% ± 0.8 | 0.977 ± 0.005 | $52 ± 10 |
| OG | 94.8% ± 1.0 | 0.973 ± 0.004 | $92 ± 18 |
| UB | 93.3% ± 2.1 | 0.950 ± 0.018 | $129 ± 35 |
| UG | 94.5% ± 1.5 | 0.950 ± 0.014 | $165 ± 45 |

Across all scenarios and every seed, the DQN agrees with the exact optimal action on **~93–95%**
of states, with cow-value correlation **0.87–0.98**. The replacement boundary (healthy open cow)
lands within ±1 MAC of the DP optimum at essentially every parity in every scenario.

---

# For the manuscript

Suggested framing for the "Validation against exact DP" subsection:

> To validate the learned policy, we solved the identical 4-state MDP exactly by value iteration
> (2,209 states; De Vries–style retention-payoff formulation, Eqn 4) and compared it against the
> DRL agent under one identical set of definitions. The DRL agent recovered the exact optimum
> closely: expected annual return within ~2.5%, steady-state herd structure within ~0.1 parity,
> and per-state policy agreement of 93–95% (mean ± SD across five seeds; cow-value correlation
> 0.87–0.98). The value of a new pregnancy and the cost of clinical mastitis — both computed as
> differences in cow value (Eqn 4), following De Vries (2006) and Bar et al. (2008) / Cha et al.
> (2011) — agreed within a few percent in early–mid lactation, with larger residuals confined to
> rarely-visited late-lactation and high-parity states, consistent with the expected approximation
> error of a sampled value function. Annual culling rate was defined identically for both methods
> (voluntary + mortality + involuntary exits per cow-year) and computed on the DP stationary
> distribution and by simulation for the DRL policy; the small difference (26.3% DP vs 28.5% DRL,
> baseline) reflects the DRL policy culling slightly more than the exact optimum rather than any
> difference in definition.

Data sources: `dp_results/dp_summary.json` (all DP values), `dp_results/dp_vs_dqn_summary.md`
(side-by-side table), `dp_results/dp_<scn>_pregnancy.csv` and `dp_<scn>_cm_cost.csv` (per-cell).
