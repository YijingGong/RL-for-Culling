#!/bin/bash

# Shell script to run DQN training for a single scenario with multiple runs
# This script will automatically activate the conda environment and run the training

# Configuration
PROJECT_DIR="/Users/yijinggong/Library/CloudStorage/Box-Box/phd/RL-for-Culling"
CONDA_ENV="rl_culling"
EPISODES=500000
NUM_RUNS=5
OUTPUT_DIR="outputs"

# Scenario to run (change this to: 2025, OG, OB, UG, or UB)
SCENARIO="2025"

# Navigate to project directory
cd "$PROJECT_DIR" || { echo "Error: Could not navigate to $PROJECT_DIR"; exit 1; }

# Get the conda base directory
CONDA_BASE=$(conda info --base)

# Source conda.sh to enable conda activate in script
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Activate the conda environment
conda activate "$CONDA_ENV" || { echo "Error: Could not activate conda environment $CONDA_ENV"; exit 1; }

echo "=========================================="
echo "Starting DQN Training Automation"
echo "Project Directory: $PROJECT_DIR"
echo "Conda Environment: $CONDA_ENV"
echo "Scenario: $SCENARIO"
echo "Episodes per run: $EPISODES"
echo "Number of runs: $NUM_RUNS"
echo "=========================================="
echo ""

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Loop through the runs
for i in $(seq 1 $NUM_RUNS); do
    echo "=========================================="
    echo "Starting Run $i of $NUM_RUNS"
    echo "Timestamp: $(date)"
    echo "Output file: ${OUTPUT_DIR}/DQN_${SCENARIO}_500k_run${i}.pkl"
    echo "=========================================="
    
    # Run the training with the scenario parameter
    python dqn_learning.py \
        --filename "${OUTPUT_DIR}/DQN_${SCENARIO}_500k_run${i}.pkl" \
        --episodes $EPISODES \
        --scenario $SCENARIO
    
    # Check if the training completed successfully
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Run $i completed successfully at $(date)"
        echo ""
    else
        echo ""
        echo "✗ Run $i failed at $(date)"
        echo "Stopping automation."
        exit 1
    fi
done

echo "=========================================="
echo "All $NUM_RUNS training runs completed for scenario $SCENARIO!"
echo "Finished at: $(date)"
echo "=========================================="
