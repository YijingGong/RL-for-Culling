#!/bin/bash
###############################################################################
#  run_local_pipeline.sh
#
#  Complete local post-processing pipeline for DQN cow replacement results.
#  Run this AFTER you have downloaded all .tar.gz result files from CHTC.
#
#  Usage:
#    bash run_local_pipeline.sh <tarball_dir> [options]
#
#  Examples:
#    # First time (extracts, evaluates, aggregates, plots):
#    bash run_local_pipeline.sh .
#
#    # Re-run after tweaking visualization code (no extraction, no eval):
#    bash run_local_pipeline.sh .
#
#    # Force re-evaluation only (keep extracted files, redo evaluation):
#    bash run_local_pipeline.sh . --force
#
#    # Re-extract tarballs only (e.g., got new tarballs from CHTC):
#    bash run_local_pipeline.sh . --re-extract
#
#    # Nuclear option — redo everything from scratch:
#    bash run_local_pipeline.sh . --re-extract --force
#
#  Flags:
#    --re-extract   Delete collected/ and re-extract all tarballs.
#                   Without this, extraction is skipped if collected/ has Q-tables.
#    --force        Re-evaluate all Q-tables (ignore existing _eval files).
#                   Without this, evaluation is skipped for Q-tables that
#                   already have _eval.csv and _eval.pkl.
#    (no flags)     Skip extraction if collected/ exists,
#                   skip evaluation if _eval files exist,
#                   always re-aggregate and re-plot.
#
#  Prerequisites:
#    - Python 3 with numpy, matplotlib, scipy, pickle installed
#    - All project files in the same directory:
#        cow_environment2.py, utility.py, animal_constants_*.py
#        local_evaluate.py, aggregate_results.py
#        visualize_summary.py, visualize_scenario.py
###############################################################################

set -e  # Exit on any error

# ── Parse arguments ─────────────────────────────────────────────────────────
TARBALL_DIR=""
FORCE=false
RE_EXTRACT=false

for arg in "$@"; do
    case "$arg" in
        --force|-f)
            FORCE=true
            ;;
        --re-extract|--reextract|-x)
            RE_EXTRACT=true
            ;;
        *)
            if [ -z "$TARBALL_DIR" ]; then
                TARBALL_DIR="$arg"
            fi
            ;;
    esac
done

TARBALL_DIR="${TARBALL_DIR:-.}"          # Default: current directory

# ── Configuration ────────────────────────────────────────────────────────────
COLLECTED_DIR="collected"                # Where extracted results go
OUTPUT_DIR="outputs"                     # Where figures and aggregated CSV go
EVAL_EPISODES=1000                       # Overall evaluation episodes per Q-table (also used for steady-state)
PARITY_EPISODES=500                      # Episodes per starting parity

SCENARIOS=("2025" "OG" "OB" "UG" "UB")
SEEDS=(42 123 456 789 1024)

# ── Colors for terminal output ───────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'  # No Color

echo ""
echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}  DQN Cow Replacement — Local Post-Processing Pipeline${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""
echo "  Tarball directory: ${TARBALL_DIR}"
echo "  --re-extract:      ${RE_EXTRACT}"
echo "  --force (re-eval): ${FORCE}"
echo ""

###############################################################################
# STEP 1: Extract tarballs
#   - SKIP if collected/ already has Q-tables (unless --re-extract)
#   - With --re-extract: delete collected/ and re-extract everything
###############################################################################
echo -e "${YELLOW}STEP 1: Extraction${NC}"

# Count existing Q-tables in collected/
existing_pkl=0
if [ -d "${COLLECTED_DIR}" ]; then
    existing_pkl=$(find "${COLLECTED_DIR}" -maxdepth 1 -name "DQN_*.pkl" ! -name "*_eval.pkl" 2>/dev/null | wc -l | tr -d ' ')
fi

if [ "$RE_EXTRACT" = true ]; then
    echo "  --re-extract: Removing old collected/ and re-extracting..."
    rm -rf "${COLLECTED_DIR}"
    existing_pkl=0
fi

if [ "$existing_pkl" -gt 0 ]; then
    echo -e "${GREEN}  collected/ already has ${existing_pkl} Q-table(s). Skipping extraction.${NC}"
    echo "  (Use --re-extract to re-extract from tarballs)"
else
    echo "  Extracting .tar.gz files from ${TARBALL_DIR}..."
    mkdir -p "${COLLECTED_DIR}"

    tarball_count=0
    for tarball in "${TARBALL_DIR}"/DQN_*_results.tar.gz; do
        if [ -f "$tarball" ]; then
            echo "    $(basename $tarball)"
            tar -xzf "$tarball" -C "${COLLECTED_DIR}/"
            tarball_count=$((tarball_count + 1))
        fi
    done

    # Also try without _results suffix
    for tarball in "${TARBALL_DIR}"/DQN_*.tar.gz; do
        if [ -f "$tarball" ] && [[ ! "$tarball" == *"_results.tar.gz" ]]; then
            echo "    $(basename $tarball)"
            tar -xzf "$tarball" -C "${COLLECTED_DIR}/"
            tarball_count=$((tarball_count + 1))
        fi
    done

    if [ $tarball_count -eq 0 ]; then
        echo -e "${RED}  ERROR: No .tar.gz files found in ${TARBALL_DIR}${NC}"
        echo "  Usage: bash run_local_pipeline.sh /path/to/tarballs/"
        exit 1
    fi

    echo -e "${GREEN}  Extracted ${tarball_count} tarballs.${NC}"
fi
echo ""

###############################################################################
# STEP 2: Verify Q-tables
###############################################################################
echo -e "${YELLOW}STEP 2: Verifying Q-tables${NC}"

missing=0
found=0
for scenario in "${SCENARIOS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        if [ -f "${COLLECTED_DIR}/DQN_${scenario}_seed${seed}.pkl" ] || \
           [ -f "${COLLECTED_DIR}/DQN_${scenario}_${seed}.pkl" ]; then
            found=$((found + 1))
        else
            echo -e "${RED}  MISSING: DQN_${scenario}_seed${seed}.pkl${NC}"
            missing=$((missing + 1))
        fi
    done
done

echo -e "${GREEN}  Found: ${found}/25 Q-tables${NC}"
if [ $missing -gt 0 ]; then
    echo -e "${YELLOW}  Missing: ${missing} (will be skipped)${NC}"
fi
echo ""

###############################################################################
# STEP 3: Evaluate Q-tables
#   - SKIP individual Q-tables if _eval.csv + _eval.pkl exist (unless --force)
#   - With --force: re-evaluate all Q-tables
###############################################################################
echo -e "${YELLOW}STEP 3: Evaluation${NC}"

# Count how many need evaluation
need_eval=0
skip_eval=0
for scenario in "${SCENARIOS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        # Check Q-table exists
        if [ ! -f "${COLLECTED_DIR}/DQN_${scenario}_seed${seed}.pkl" ] && \
           [ ! -f "${COLLECTED_DIR}/DQN_${scenario}_${seed}.pkl" ]; then
            continue
        fi

        eval_csv="${COLLECTED_DIR}/DQN_${scenario}_seed${seed}_eval.csv"
        eval_pkl="${COLLECTED_DIR}/DQN_${scenario}_seed${seed}_eval.pkl"

        if [ "$FORCE" = true ]; then
            need_eval=$((need_eval + 1))
        elif [ -f "$eval_csv" ] && [ -f "$eval_pkl" ]; then
            skip_eval=$((skip_eval + 1))
        else
            need_eval=$((need_eval + 1))
        fi
    done
done

echo "  Already evaluated: ${skip_eval}"
echo "  Need evaluation:   ${need_eval}"

if [ $need_eval -eq 0 ]; then
    echo -e "${GREEN}  All evaluations up to date. Skipping.${NC}"
    echo "  (Use --force to re-evaluate anyway)"
else
    est_minutes=$(( need_eval * 2 ))
    echo ""
    echo -e "${YELLOW}  Evaluating ${need_eval} Q-tables (~${est_minutes} min)...${NC}"
    echo "  Episodes: ${EVAL_EPISODES} eval, ${PARITY_EPISODES} parity, ${DIST_EPISODES} dist"
    echo ""

    eval_flags=""
    if [ "$FORCE" = false ]; then
        eval_flags="--skip_existing"
    fi

    python3 local_evaluate.py \
        --collected_dir "${COLLECTED_DIR}/" \
        --eval_episodes ${EVAL_EPISODES} \
        --parity_episodes ${PARITY_EPISODES} \
        ${eval_flags}

    echo -e "${GREEN}  Evaluation complete.${NC}"
fi
echo ""

###############################################################################
# STEP 4: Aggregate results (always runs)
###############################################################################
echo -e "${YELLOW}STEP 4: Aggregating results${NC}"

mkdir -p "${OUTPUT_DIR}"

python3 aggregate_results.py \
    --results_dir "${COLLECTED_DIR}/" \
    --output "${OUTPUT_DIR}/aggregated_results.csv" \
    --tables_dir "${OUTPUT_DIR}/tables"

echo -e "${GREEN}  Done.${NC}"
echo ""

###############################################################################
# STEP 5: Cross-scenario summary figure (always runs)
###############################################################################
echo -e "${YELLOW}STEP 5: Cross-scenario summary figure${NC}"

python3 visualize_summary.py \
    --collected_dir "${COLLECTED_DIR}/"

echo -e "${GREEN}  Done.${NC}"
echo ""

###############################################################################
# STEP 6: Per-scenario figures (always runs)
###############################################################################
echo -e "${YELLOW}STEP 6: Per-scenario figures${NC}"

python3 visualize_scenario.py \
    --collected_dir "${COLLECTED_DIR}/"

echo -e "${GREEN}  Done.${NC}"
echo ""

###############################################################################
# DONE
###############################################################################
echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}  PIPELINE COMPLETE${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""
echo "  ${OUTPUT_DIR}/"
echo "  ├── aggregated_results.csv"
echo "  ├── tables/"
echo "  │   ├── table_cross_scenario_summary.csv"
echo "  │   ├── table_culling_rates.csv"
echo "  │   ├── table_parity_distribution.csv"
echo "  │   ├── table_replacement_distribution.csv"
echo "  │   ├── table_pregnancy_value_<SC>.csv"
echo "  │   └── table_mastitis_cost_<SC>.csv"
echo "  └── figures/"
echo "      ├── summary/cross_scenario_summary.png"
echo "      └── scenarios/<SC>/<SC>_*.png"
echo ""
if [ $skip_eval -gt 0 ] && [ "$FORCE" = false ]; then
    echo -e "${CYAN}  NOTE: ${skip_eval} Q-tables used cached evaluations.${NC}"
fi
echo ""
echo "  Quick reference:"
echo "    bash run_local_pipeline.sh .                     # Re-aggregate & re-plot only"
echo "    bash run_local_pipeline.sh . --force             # Re-evaluate + re-aggregate + re-plot"
echo "    bash run_local_pipeline.sh . --re-extract        # Re-extract tarballs + re-aggregate + re-plot"
echo "    bash run_local_pipeline.sh . --re-extract --force  # Redo everything from scratch"
echo ""
echo -e "${GREEN}  All done!${NC}"
echo ""
