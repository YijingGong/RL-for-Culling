#!/bin/bash
# CHTC executable script for DQN training 
# Arguments: $1 = scenario, $2 = seed

SCENARIO=$1
SEED=$2
EPISODES=500000

echo "=========================================="
echo "DQN Training "
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

# Define output filename
OUTPUT_FILE="outputs/DQN_${SCENARIO}_seed${SEED}.pkl"

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
cd outputs
tar -czf "DQN_${SCENARIO}_seed${SEED}_results.tar.gz" \
    DQN_${SCENARIO}_seed${SEED}*

# Move to top-level working directory (required by CHTC)
mv "DQN_${SCENARIO}_seed${SEED}_results.tar.gz" ../../

cd ../..

echo ""
echo "=========================================="
echo "Job completed at $(date)"
echo "Scenario: $SCENARIO, Seed: $SEED"
echo "Training exit code: $TRAIN_EXIT"
echo "=========================================="
