#!/bin/bash
# CHTC executable script for DQN training
# PRODUCTION MODEL: 5 state variables (parity, MAC, MIP, CM, production level)
# Outputs are prefixed "DQN_prod_" so they do NOT overwrite the base 4-variable models.
# Arguments: $1 = scenario, $2 = seed

SCENARIO=$1
SEED=$2
EPISODES=500000

echo "=========================================="
echo "DQN Training (production model, 5 state variables)"
echo "Scenario: $SCENARIO"
echo "Seed: $SEED"
echo "Episodes: $EPISODES"
echo "Timestamp: $(date)"
echo "=========================================="

# 1. Unpack packages
tar -xzf packages.tar.gz
export PYTHONPATH=$PWD/packages:$PYTHONPATH

# Set HOME so torch doesn't complain
export HOME=$PWD

# 2. Unpack project files
tar -xzf project.tar.gz

# 3. Navigate to project directory
cd project

# 4. Check Python version
echo "Python version: $(python3 --version)"
echo "Testing imports..."
python3 -c "import torch; import numpy; print('torch:', torch.__version__); print('numpy:', numpy.__version__)"

# 5. Create output directory
mkdir -p outputs

# Define output filename (production model)
OUTPUT_FILE="outputs/DQN_prod_${SCENARIO}_seed${SEED}.pkl"

echo ""
echo "--- Starting Training ---"
echo "Output file: $OUTPUT_FILE"
echo ""

# 6. Run DQN training
python3 dqn_learning.py \
    --filename "$OUTPUT_FILE" \
    --episodes $EPISODES \
    --scenario $SCENARIO \
    --seed $SEED \
    --restart

TRAIN_EXIT=$?

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "ERROR: Training failed with exit code $TRAIN_EXIT"
fi


echo ""
echo "--- Packaging Results ---"
echo ""

# 7. Package all outputs into a tar.gz for transfer back
#    (includes the .pkl gridded Q-table, the _model.pth network weights, and the run log)
cd outputs
tar -czf "DQN_prod_${SCENARIO}_seed${SEED}_results.tar.gz" \
    DQN_prod_${SCENARIO}_seed${SEED}*

# Move to top-level working directory (required by CHTC)
mv "DQN_prod_${SCENARIO}_seed${SEED}_results.tar.gz" ../../

cd ../..

echo ""
echo "=========================================="
echo "Job completed at $(date)"
echo "Scenario: $SCENARIO, Seed: $SEED"
echo "Training exit code: $TRAIN_EXIT"
echo "=========================================="