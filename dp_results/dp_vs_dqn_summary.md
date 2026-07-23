# Exact DP benchmark vs. DRL (DQN) — comparison

Exact value-iteration solution of the same MDP (identical parameters and reward functions; gamma = 0.95) compared against the DRL results reported in the manuscript. "n/r" = not reported numerically in the current draft.

## Annual return per stall ($/yr)

| Scenario | DRL (paper) | DP (exact) | Difference |
|---|---|---|---|
| BL | $3,082 | $3,101 | +0.6% |
| OG | n/r | $3,824 | — |
| OB | $1,036 | $1,061 | +2.4% |
| UG | $4,545 | $4,580 | +0.8% |
| UB | n/r | $1,829 | — |

## Annual culling rate (%) and herd structure

| Scenario | DRL cull | DP cull | DRL parity | DP parity | DRL life | DP life |
|---|---|---|---|---|---|---|
| BL | 28.5 | 26.3 | 2.7 | 2.69 | 3.5 | 3.80 |
| OG | 21.4 | 19.4 | 3.4 | 3.40 | 4.7 | 5.16 |
| OB | 21.1 | 19.1 | 3.4 | 3.39 | 4.7 | 5.23 |
| UG | 22.8 | 20.0 | 3.3 | 3.31 | 4.4 | 4.99 |
| UB | 23.9 | 21.1 | 3.1 | 3.15 | 4.2 | 4.73 |

## Value of a new pregnancy, baseline (avg MAC 3-10, $)

| Parity | DRL (paper) | DP (exact) |
|---|---|---|
| 1 | $418 | $416 |
| 2 | $313 | $284 |
| 3 | $231 | $196 |

## Cost of clinical mastitis by lactation stage, baseline ($)

| Stage | DRL (paper) | DP (exact) |
|---|---|---|
| Early (MAC 1-3) | $387 | $399 |
| Mid (MAC 4-6) | $339 | $346 |
| Late (MAC 7-9) | $293 | $243 |
| Extended (MAC>=10) | $147 | $132 |
