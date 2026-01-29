#!/bin/bash
# Installation script for DQN dependencies

echo "Installing PyTorch for DQN..."
sudo pip3 install torch torchvision

echo ""
echo "Testing installation..."
python3.11 -c "import torch; print(f'✓ PyTorch {torch.__version__} installed successfully')"

echo ""
echo "✓ DQN is ready to use!"
echo ""
echo "Run DQN with:"
echo "  python3.11 dqn_learning.py --filename outputs/dqn_policy.pkl --episodes 1000000"
