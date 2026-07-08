# Branch `production-state-variable` — updates vs. `main`

`main` is the manuscript version of the model: state = **(parity, MAC, MIP, CM)**, 4 discrete
variables. This branch adds a **fifth state variable — a persistent per-cow production level `M`**
(a continuous milk-yield multiplier). Everything else about the MDP, reward, and DQN is unchanged.

## Why

Addresses Reviewer 1's core criticisms (state too simple; can't cull low producers; the
curse-of-dimensionality advantage of DRL over DP is never demonstrated) and Reviewer 2's realism
concern. `M` is continuous, so the DQN takes it as a network input and generalizes over it, whereas
exact tabular DP would have to discretize it — the intended demonstration of DRL's scaling advantage.

## The production process

- `M` is a **persistent per-cow multiplier** on the whole Wood's lactation curve.
- Drawn once when a cow enters the herd from a truncated Normal(mean = 1.0, SD = 0.11), held
  **constant for her entire life**, and **resampled for each replacement heifer**.
- SD = 0.11 comes from the between-cow (animal) variance component of 305-d milk yield in US
  Holsteins: √1,236,877 / 10,210 = 1,112 / 10,210 = 0.11 (**Li et al., 2022, Table 6**). This is the
  persistent, within-herd spread (implied repeatability ≈ 0.47), so it already excludes herd and
  temporary variation. No within-lactation noise and no genetic gain in this version.
- `M` scales monthly milk production, and therefore both **milk income** and **DMI/feed cost**
  (feed derives from milk). Slaughter value, conception, and mortality are unaffected (v1 simplification).

## File-by-file changes

| File | Change |
|---|---|
| `animal_constants_2025/OG/OB/UG/UB.py` | Added `PRODUCTION_MULT_MEAN=1.0`, `PRODUCTION_MULT_SD=0.11`, `PRODUCTION_MULT_MIN=0.67`, `PRODUCTION_MULT_MAX=1.33` (identical in all five; it's a biological trait). |
| `cow_environment2.py` | State is now a 5-tuple `(parity, mac, mip, disease, M)`. New `sample_production_level()` (truncated normal). `reset()` samples `M`; `step()` carries `M` on keep+survive and draws a fresh `M` at every replacement (voluntary, death, max-parity, max-MAC). Milk/DMI/feed methods take an `M` argument (default 1.0) and scale milk by `M`. |
| `dqn_learning.py` | Network input `state_dim` 4→5. `state_to_tensor` now returns 5 **normalized** inputs (parity/12, mac/20, mip/9, disease, (M−1)/0.11). `extract_q_table_from_dqn` evaluates the network on an **M-grid** (0.7…1.3) for visualization. The trained **network weights** (`<file>_model.pth`) are the source of truth; the pickled Q-table is now a gridded convenience artifact. |
| `evaluate_dqn.py`, `local_evaluate.py` | Action selection now uses the **network** (handles continuous `M`) via a polymorphic `get_action`; added `load_policy_net()`. Controlled per-parity start states carry `M=1.0` (average producer). `classify_replacement()` unpacks the first 4 state elements so replacement/culling detection still works with 5-tuples. |
| `utility.py` | Unchanged — `possible_state2` validates the first four elements, so it accepts 5-tuples as-is. |

**Backwards compatibility:** the milk/feed methods default `M=1.0`, so any external caller (e.g.
the visualization scripts) that passes only four state fields behaves as an "average producer."

## Not yet done (follow-ups)

- `visualize_scenario.py` / `visualize_summary.py` / `aggregate_results.py` were **not** updated to
  plot the production dimension (they will run, treating M via the gridded Q-table, but new
  M-specific figures — e.g., culling threshold vs. production level — still need to be added).
- No genetic gain (replacements drawn at mean 1.0). Noted as a future extension.
- Hyperparameters unchanged; may want to revisit hidden width if learning the extra dimension is hard.

## How to run

```bash
# train one scenario/seed (production model)
python dqn_learning.py --filename outputs/DQN_prod_2025_seed42.pkl --episodes 500000 --scenario 2025 --seed 42

# evaluate (uses the saved network, continuous M)
python evaluate_dqn.py --model outputs/DQN_prod_2025_seed42.pkl --scenario 2025 --eval_episodes 1000
```

## What was verified

- **Environment (no torch needed):** verified end-to-end — 5-tuple states, milk scales exactly with
  `M` (×1.2 input → ×1.2 milk), feed rises with `M`, `M` persists on keep and is resampled within
  [0.67, 1.33] on replace, and a full 180-step episode runs.
- **Logic:** `classify_replacement` correctly handles 5-tuples (voluntary/death/involuntary/none);
  input normalization maps M=1.11 → 1.0 as intended.
- **All files compile** (`py_compile`).
- **Not run here:** the actual DQN training/evaluation, because this environment cannot install
  PyTorch (network policy blocks the CPU wheel index; the full PyPI wheel is too large to fetch).
  Run the two commands above on your machine or CHTC (which have torch) as the final smoke test.
