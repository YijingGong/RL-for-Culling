#!/bin/bash

# Shell script to run DQN training for all scenarios with multiple runs each
# This script will automatically activate the conda environment and run the training
# for each scenario (2025, OG, OB, UG, UB) with the specified number of runs

# Configuration
PROJECT_DIR="/Users/yijinggong/Library/CloudStorage/Box-Box/phd/RL-for-Culling"
CONDA_ENV="rl_culling"
EPISODES=500000
NUM_RUNS=5
OUTPUT_DIR="outputs"

# Scenarios to run
SCENARIOS=("2025" "OG" "OB" "UG" "UB")

# Navigate to project directory
cd "$PROJECT_DIR" || { echo "Error: Could not navigate to $PROJECT_DIR"; exit 1; }

# Get the conda base directory
CONDA_BASE=$(conda info --base)

# Source conda.sh to enable conda activate in script
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Activate the conda environment
conda activate "$CONDA_ENV" || { echo "Error: Could not activate conda environment $CONDA_ENV"; exit 1; }

echo "=========================================="
echo "Starting DQN Training Automation - All Scenarios"
echo "Project Directory: $PROJECT_DIR"
echo "Conda Environment: $CONDA_ENV"
echo "Episodes per run: $EPISODES"
echo "Number of runs per scenario: $NUM_RUNS"
echo "Scenarios: ${SCENARIOS[@]}"
echo "=========================================="
echo ""

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Loop through each scenario
for scenario in "${SCENARIOS[@]}"; do
    echo ""
    echo "######################################"
    echo "# SCENARIO: $scenario"
    echo "######################################"
    echo ""
    
    # Loop through the runs for this scenario
    for i in $(seq 1 $NUM_RUNS); do
        echo "=========================================="
        echo "Scenario: $scenario | Run $i of $NUM_RUNS"
        echo "Timestamp: $(date)"
        echo "Output file: ${OUTPUT_DIR}/DQN_${scenario}_500k_run${i}.pkl"
        echo "=========================================="
        
        # Run the training with the scenario parameter
        python dqn_learning.py \
            --filename "${OUTPUT_DIR}/DQN_${scenario}_500k_run${i}.pkl" \
            --episodes $EPISODES \
            --scenario $scenario
        
        # Check if the training completed successfully
        if [ $? -eq 0 ]; then
            echo ""
            echo "✓ Scenario $scenario, Run $i completed successfully at $(date)"
            echo ""
        else
            echo ""
            echo "✗ Scenario $scenario, Run $i failed at $(date)"
            echo "Stopping automation."
            exit 1
        fi
    done
    
    echo ""
    echo "✓✓✓ All runs for scenario $scenario completed!"
    echo ""
done

echo ""
echo "=========================================="
echo "ALL SCENARIOS AND RUNS COMPLETED!"
echo "Total scenarios: ${#SCENARIOS[@]}"
echo "Runs per scenario: $NUM_RUNS"
echo "Total training runs: $((${#SCENARIOS[@]} * $NUM_RUNS))"
echo "Finished at: $(date)"
echo "=========================================="
