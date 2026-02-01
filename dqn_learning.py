"""
Deep Q-Network (DQN) implementation for dairy cow replacement decisions.
Compatible interface with q_learning.py - can be run with same command-line arguments.

Usage: python dqn_learning.py --filename outputs/dqn_policy.pkl --episodes 1000000
"""

import argparse
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
import os
import sys
import time
import utility
import cow_environment2
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

parity_range = range(13)
mim_range = range(21)
mip_range = range(10)
disease_range = range(2)

# DQN Network Architecture
class DQN(nn.Module):
    """Deep Q-Network with 2 hidden layers"""
    def __init__(self, state_dim=4, hidden_dim=64, action_dim=2):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Experience Replay Buffer
class ReplayBuffer:
    """Store and sample experience tuples for training"""
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

def state_to_tensor(state):
    """Convert state tuple (parity, mac, mip, disease) to tensor"""
    return torch.FloatTensor(list(state))

def dqn_learning(env, policy_net, target_net, optimizer, replay_buffer, rewards_per_episode, 
                 num_episodes, max_steps, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, 
                 min_epsilon=0.01, batch_size=64, target_update_freq=1000, 
                 learning_start=1000):
    """
    DQN training loop - compatible interface with q_learning()
    
    Args:
        env: Environment instance
        policy_net: Policy network (DQN)
        target_net: Target network (DQN)
        optimizer: PyTorch optimizer
        replay_buffer: Experience replay buffer
        rewards_per_episode: List to store episode rewards
        num_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        gamma: Discount factor
        epsilon: Initial exploration rate
        epsilon_decay: Epsilon decay rate per episode
        min_epsilon: Minimum epsilon
        batch_size: Batch size for training
        target_update_freq: Frequency to update target network
        learning_start: Start learning after this many steps
    
    Returns:
        policy_net, rewards_per_episode, epsilon (compatible with q_learning)
    """
    
    action_map = {'keep': 0, 'replace': 1}
    reverse_action_map = {0: 'keep', 1: 'replace'}
    
    total_steps = 0
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while steps < max_steps:
            # Epsilon-greedy action selection
            if random.uniform(0, 1) < epsilon:
                action_idx = random.choice([0, 1])  # Explore
                action = reverse_action_map[action_idx]
            else:
                with torch.no_grad():
                    state_tensor = state_to_tensor(state).unsqueeze(0)
                    q_values = policy_net(state_tensor)
                    action_idx = q_values.argmax().item()  # Exploit
                    action = reverse_action_map[action_idx]
            
            next_state, reward = env.step(action)
            total_reward += reward
            
            # Check if episode should end
            if not (next_state[0] in parity_range and next_state[1] in mim_range and next_state[2] in mip_range):
                done = True
            else:
                done = False
            
            # Store transition in replay buffer
            replay_buffer.push(state, action_idx, reward, next_state, done)
            
            # Train the network if we have enough samples
            if len(replay_buffer) >= learning_start and total_steps % 4 == 0:
                train_dqn(policy_net, target_net, optimizer, replay_buffer, batch_size, gamma)
            
            # Update target network periodically
            if total_steps % target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())
            
            if done:
                break
            
            state = next_state
            steps += 1
            total_steps += 1
        
        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        rewards_per_episode.append(total_reward)
        
        # Print progress
        if (episode + 1) % 1000 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward:.0f}, Epsilon: {epsilon:.4f}, Buffer: {len(replay_buffer)}")
    
    return policy_net, rewards_per_episode, epsilon

def train_dqn(policy_net, target_net, optimizer, replay_buffer, batch_size, gamma):
    """Train DQN on a batch of experiences"""
    if len(replay_buffer) < batch_size:
        return
    
    # Sample batch
    batch = replay_buffer.sample(batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)
    
    # Convert to tensors
    states = torch.stack([state_to_tensor(s) for s in states])
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.stack([state_to_tensor(s) for s in next_states])
    dones = torch.FloatTensor(dones)
    
    # Compute current Q values
    current_q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    
    # Compute target Q values
    with torch.no_grad():
        next_q_values = target_net(next_states).max(1)[0]
        target_q_values = rewards + gamma * next_q_values * (1 - dones)
    
    # Compute loss and update
    loss = nn.MSELoss()(current_q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def extract_q_table_from_dqn(policy_net, env):
    """
    Extract Q-table from trained DQN for compatibility with analysis scripts.
    Creates a dictionary mapping states to action Q-values.
    """
    q_table = {}
    
    # Iterate through all possible states
    for parity in parity_range:
        for mim in mim_range:
            for mip in mip_range:
                for disease in disease_range:
                    state = (parity, mim, mip, disease)
                    if utility.possible_state2(state, parity_range, mim_range, mip_range, disease_range):
                        # Get Q-values from network
                        with torch.no_grad():
                            state_tensor = state_to_tensor(state).unsqueeze(0)
                            q_values = policy_net(state_tensor).squeeze(0)
                            q_keep = q_values[0].item()
                            q_replace = q_values[1].item()
                        
                        q_table[state] = {'keep': q_keep, 'replace': q_replace}
    
    return q_table

def ensure_directory(path):
    """Create the parent directory for a file path if it is missing."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


class StreamLogger:
    """Mirror stdout/stderr to a log file while keeping console output."""

    def __init__(self, log_path: str):
        self.log_path = log_path
        self._stdout = None
        self._stderr = None
        self._log_file = None

    def start(self):
        if self._log_file is not None:
            return
        ensure_directory(self.log_path)
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        self._log_file = open(self.log_path, "a", encoding="utf-8")
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self._log_file.write(f"\n--- Run started {timestamp} ---\n")
        self._log_file.flush()
        sys.stdout = self
        sys.stderr = self

    def stop(self):
        if self._log_file is None:
            return
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self._log_file.write(f"--- Run ended {timestamp} ---\n")
        self._log_file.flush()
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        self._log_file.close()
        self._log_file = None
        self._stdout = None
        self._stderr = None

    def write(self, data):
        if self._stdout is not None:
            self._stdout.write(data)
        if self._log_file is not None:
            self._log_file.write(data)

    def flush(self):
        if self._stdout is not None:
            self._stdout.flush()
        if self._log_file is not None:
            self._log_file.flush()

def save_dqn_model(policy_net, rewards_per_episode, epsilon, filename):
    """
    Save DQN model and extract Q-table for compatibility.
    Saves both the neural network and a Q-table dictionary.
    """
    ensure_directory(filename)
    
    # Save neural network state
    model_filename = filename.replace('.pkl', '_model.pth')
    torch.save({
        'policy_net_state_dict': policy_net.state_dict(),
        'rewards_per_episode': rewards_per_episode,
        'epsilon': epsilon
    }, model_filename)
    
    # Extract and save Q-table for compatibility with analysis scripts
    print("Extracting Q-table from DQN for compatibility...")
    env = cow_environment2.CowEnv(parity_range, mim_range, mip_range, disease_range)
    q_table = extract_q_table_from_dqn(policy_net, env)
    
    with open(filename, 'wb') as f:
        pickle.dump((q_table, rewards_per_episode, epsilon), f)
    
    print(f"Saved DQN model to {model_filename}")
    print(f"Saved Q-table (for compatibility) to {filename}")

def load_or_create_dqn(filename, env, force_restart=False):
    """
    Load existing DQN model or create new one.
    Returns policy_net, target_net, optimizer, replay_buffer, rewards_per_episode, epsilon
    
    Args:
        filename: Path to save/load model
        env: Environment instance
        force_restart: If True, ignore existing model and start fresh
    """
    model_filename = filename.replace('.pkl', '_model.pth')
    
    # Create networks
    policy_net = DQN(state_dim=4, hidden_dim=64, action_dim=2)
    target_net = DQN(state_dim=4, hidden_dim=64, action_dim=2)
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    replay_buffer = ReplayBuffer(capacity=100000)
    
    if os.path.exists(model_filename) and not force_restart:
        print(f"\n{'='*70}")
        print(f"LOADING EXISTING MODEL")
        print(f"{'='*70}")
        print(f"Model file: {model_filename}")
        checkpoint = torch.load(model_filename)
        policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        target_net.load_state_dict(checkpoint['policy_net_state_dict'])
        rewards_per_episode = checkpoint['rewards_per_episode']
        epsilon = checkpoint['epsilon']
        print(f"Loaded {len(rewards_per_episode)} previous episodes")
        print(f"Continuing from epsilon: {epsilon:.4f}")
        print(f"Previous final reward: {rewards_per_episode[-1]:.0f}")
        print(f"{'='*70}\n")
    else:
        if force_restart and os.path.exists(model_filename):
            print(f"\n{'='*70}")
            print(f"FORCE RESTART: Ignoring existing model")
            print(f"{'='*70}\n")
        else:
            print(f"\n{'='*70}")
            print(f"CREATING NEW MODEL")
            print(f"{'='*70}\n")
        target_net.load_state_dict(policy_net.state_dict())
        rewards_per_episode = []
        epsilon = 1.0
        print(f"Starting fresh with epsilon: {epsilon:.4f}")
        print(f"{'='*70}\n")
    
    return policy_net, target_net, optimizer, replay_buffer, rewards_per_episode, epsilon

def _run_dqn_pipeline(filename, num_episodes, force_restart):
    """Execute the DQN training workflow previously hosted in main."""
    # Check NumPy version
    import numpy as np
    numpy_version = np.__version__
    if numpy_version.startswith('2.'):
        print("\n" + "="*70)
        print("WARNING: NumPy 2.x detected")
        print("="*70)
        print(f"Current NumPy version: {numpy_version}")
        print("PyTorch may have compatibility issues with NumPy 2.x")
        print("Recommended: pip install 'numpy<2'")
        print("="*70 + "\n")
    
    # Initialize environment
    env = cow_environment2.CowEnv(parity_range, mim_range, mip_range, disease_range)
    
    # Load or create DQN
    policy_net, target_net, optimizer, replay_buffer, rewards_per_episode, epsilon = \
        load_or_create_dqn(filename, env, force_restart=force_restart)
    
    # Train
    print(f"Starting DQN training for {num_episodes} episodes...")
    start_time = time.time()
    
    policy_net, rewards_per_episode, epsilon = dqn_learning(
        env,
        policy_net=policy_net,
        target_net=target_net,
        optimizer=optimizer,
        replay_buffer=replay_buffer,
        rewards_per_episode=rewards_per_episode,
        num_episodes=num_episodes,
        max_steps=180,
        gamma=0.95,
        epsilon=epsilon,
        epsilon_decay=0.995,
        min_epsilon=0.01,
        batch_size=64,
        target_update_freq=1000,
        learning_start=1000
    )
    
    end_time = time.time()
    
    # Save model and extract Q-table
    save_dqn_model(policy_net, rewards_per_episode, epsilon, filename)
    
    # Load the extracted Q-table for printing (compatibility)
    with open(filename, 'rb') as f:
        q_table, _, _ = pickle.load(f)
    
    # Print sample Q-values
    print("\nLearned DQN Q-table (sample):")
    for i, (state, actions) in enumerate(q_table.items()):
        if i >= 10:  # Print fewer samples than q_learning.py
            break
        print(f"State: {state}")
        for action, value in actions.items():
            print(f"  Action: {action}, Q-value: {value:.2f}")
    print(f"Total states: {len(q_table)}")
    print(f"\nTime taken for training: {end_time - start_time:.2f} seconds")
    print(f"Average reward (last 1000 episodes): {np.mean(rewards_per_episode[-1000:]):.2f}")


def main(filename, num_episodes, force_restart=False):
    """Entry point that wraps training with a tee logger."""
    base, _ = os.path.splitext(filename)
    log_path = f"{base}_run.log"
    logger = StreamLogger(log_path)
    logger.start()
    try:
        _run_dqn_pipeline(filename, num_episodes, force_restart)
    finally:
        logger.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL for Culling - DQN runner")
    parser.add_argument(
        "--filename",
        default="outputs/dqn_policy.pkl",
        help="Path to save the DQN model and Q-table (default: outputs/dqn_policy.pkl)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1_000_000,
        help="Number of episodes for training (default: 1,000,000)",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Force restart training from scratch (ignore existing model)",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="2025",
        choices=["2025", "OG", "OB", "UG", "UB"],
        help="Scenario to use for animal constants (default: 2025)",
    )
    args = parser.parse_args()
    
    # Set the scenario before running main
    cow_environment2.set_scenario(args.scenario)
    
    main(args.filename, args.episodes, force_restart=args.restart)
