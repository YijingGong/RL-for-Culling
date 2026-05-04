# RL-for-Culling

Code for **"Optimizing Dairy Cow Replacement Decisions Using Deep Reinforcement Learning in an Evolving Stochastic Environment"** (Gong, da Silva, Cabrera; University of Wisconsin–Madison, Department of Animal Science).

A Deep Q-Network (DQN) agent is trained to make monthly keep-vs-replace decisions for individual dairy cows, with the goal of maximizing the expected discounted sum of net returns over a 15-year horizon. The agent is trained from scratch under five economic scenarios and evaluated against the dairy replacement literature.

## Project layout

```
RL-for-Culling/
├── cow_environment2.py         # Simulated dairy-cow MDP environment (state, transitions, reward)
├── animal_constants_2025.py    # Economic + biological parameters: 2025 baseline (BL)
├── animal_constants_OG.py      # Oversupply heifer market, good milk market
├── animal_constants_OB.py      # Oversupply heifer market, bad milk market
├── animal_constants_UG.py      # Undersupply heifer market, good milk market
├── animal_constants_UB.py      # Undersupply heifer market, bad milk market
├── utility.py                  # State-validity helpers (possible_state, possible_state2)
│
├── dqn_learning.py             # DQN training loop (entry point on CHTC)
├── evaluate_dqn.py             # Single-model Monte Carlo policy evaluation
│
├── run_dqn.sh                  # CHTC executable: unpack, train one (scenario, seed)
├── dqn_jobs.sub                # HTCondor submit file: 25 jobs (5 scenarios × 5 seeds)
├── arg_matrix.csv              # (scenario, seed) pairs consumed by dqn_jobs.sub
│
├── local_evaluate.py           # Batch Monte Carlo evaluation of all 25 trained Q-tables
├── aggregate_results.py        # Aggregate _eval files into long-format CSV + paper tables
├── visualize_summary.py        # 4-panel cross-scenario summary figure
├── visualize_scenario.py       # Per-scenario figures (policy maps, parity dist, etc.)
├── run_local_pipeline.sh       # Driver: extract → evaluate → aggregate → plot
│
└── outputs/                    # Local training/eval artifacts (gitignored)
```

## MDP formulation

| Component | Definition |
|---|---|
| **State** | `(parity, MAC, MIP, CM)` where parity ∈ {0…12}, month after calving (MAC) ∈ {0…20}, month in pregnancy (MIP) ∈ {0…9}, clinical mastitis (CM) ∈ {0, 1} |
| **Action** | `keep` or `replace` |
| **Reward** | Monthly net return: milk + calf − feed − breeding − mastitis treatment, with replacement reward = slaughter − heifer cost |
| **Transitions** | Stochastic: conception, mastitis incidence/recovery, on-farm mortality (parity- and CM-dependent) |
| **Forced replacement** | (1) on-farm mortality, (2) parity > 12 at calving, (3) MAC > 20 while open |
| **Horizon / discount** | 180 months (15 yr) per episode, γ = 0.95 |

The agent is trained tabula rasa — it does not observe the underlying transition probabilities, only `(s, a, r, s')` tuples sampled from the environment. Exploration is ε-greedy with ε decayed from 1.0 to 0.01.

## Scenarios

Five economic scenarios are evaluated. The 2025 baseline (BL) uses 12-month average USDA market prices (Nov 2024 – Oct 2025). The four hypothetical scenarios cross heifer supply (over vs under) with milk market quality (good vs bad):

| Code | Heifer supply | Milk market |
|---|---|---|
| `2025` | baseline (USDA) | baseline (USDA) |
| `OG` | oversupply | good |
| `OB` | oversupply | bad |
| `UG` | undersupply | good |
| `UB` | undersupply | bad |

Five random seeds per scenario (42, 123, 456, 789, 1024) ⇒ 25 independent training runs.

## How to run

### 1. Train on CHTC (UW–Madison Center for High Throughput Computing)

The training is parallelized as 25 batch jobs. Each job runs `run_dqn.sh` with one `(scenario, seed)` pair from `arg_matrix.csv` and trains for 500,000 episodes.

Bundle the project once:

```bash
tar -czf project.tar.gz cow_environment2.py utility.py dqn_learning.py evaluate_dqn.py animal_constants_*.py
```

You also need a `packages.tar.gz` containing the Python deps used on the execute node (numpy, scipy, torch). Then on the CHTC submit node:

```bash
condor_submit dqn_jobs.sub
```

Each job emits `DQN_<SCENARIO>_seed<SEED>_results.tar.gz` containing the trained Q-table, model weights, and run log.

### 2. Local post-processing pipeline

After downloading all 25 result tarballs from CHTC into a single directory, run:

```bash
bash run_local_pipeline.sh /path/to/tarballs/
```

This script (idempotent — safe to re-run):

1. Extracts tarballs into `collected/`
2. Verifies all 25 Q-tables are present
3. Runs Monte Carlo policy evaluation (`local_evaluate.py`) — 1,000 overall episodes + 500 per starting parity per Q-table
4. Aggregates results across seeds (`aggregate_results.py`) → `outputs/aggregated_results.csv` + paper-ready tables in `outputs/tables/`
5. Builds a cross-scenario summary figure (`visualize_summary.py`)
6. Builds per-scenario figures (`visualize_scenario.py`)

Useful flags:

```bash
bash run_local_pipeline.sh .                # cached extraction + cached eval
bash run_local_pipeline.sh . --force        # re-evaluate all Q-tables
bash run_local_pipeline.sh . --re-extract   # re-extract tarballs from scratch
```

### 3. Run a single training locally (for debugging)

```bash
python3 dqn_learning.py \
    --filename outputs/DQN_2025_seed42.pkl \
    --episodes 500000 \
    --scenario 2025 \
    --seed 42 \
    --restart
```

Argument summary:

| Flag | Description |
|---|---|
| `--scenario` | One of `2025`, `OG`, `OB`, `UG`, `UB` |
| `--seed` | Random seed for numpy / torch / Python |
| `--episodes` | Training episodes (default 1,000,000; CHTC runs use 500,000) |
| `--filename` | Output `.pkl` (also drives the `_model.pth` and `_run.log` paths) |
| `--restart` | Ignore any existing checkpoint and start fresh |

## Outputs of evaluation

For each trained Q-table the pipeline produces:

- `*_eval.csv` — overall metrics (annualized stall value, culling rates, parity distribution, mastitis cost by lactation stage)
- `*_eval.pkl` — full evaluation artifacts (per-parity discounted values, steady-state distributions, etc.)

`aggregate_results.py` then collapses the five seeds per scenario into mean ± 95% CI for every metric, and writes the per-paper tables:

- `table_cross_scenario_summary.csv`
- `table_culling_rates.csv` (voluntary / death / forced / total)
- `table_parity_distribution.csv`
- `table_replacement_distribution.csv`
- `table_pregnancy_value_<SC>.csv` — economic value of pregnancy per parity
- `table_mastitis_cost_<SC>.csv` — cost of CM by lactation stage

## Dependencies

- Python 3.9+
- `numpy`, `scipy`, `matplotlib`, `pickle` (stdlib)
- `torch` (PyTorch) — used by `dqn_learning.py`
- HTCondor (only for CHTC training)

## Reproducibility

All randomness (Python, numpy, torch CPU/CUDA) is seeded in `dqn_learning.py` via `set_all_seeds(seed)`. PyTorch is forced into deterministic mode (`cudnn.deterministic=True`, `cudnn.benchmark=False`). With the same `(scenario, seed)` pair, training is reproducible.

## Citation

If you use this code, please cite:

> Gong, Y., da Silva, L. H., & Cabrera, V. E. *Optimizing Dairy Cow Replacement Decisions Using Deep Reinforcement Learning in an Evolving Stochastic Environment.*

## License

MIT — see [LICENSE](LICENSE).

## Contact

Yijing Gong — gong44@wisc.edu — Department of Animal Science, University of Wisconsin–Madison.
