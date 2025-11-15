import sys 
import json 
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim 
from collections import deque
from SailingEnv import SailingEnv
from utility_funcs import test_agent, extract_moves


NUM_ACTIONS = 6
STATE_DIMENSIONS = 9

# Map file paths - use relative path from project root
import os
project_root = os.path.dirname(os.path.abspath(__file__))
map_dir = os.path.join(project_root, "Maps data", "map_exports")

# Load single map for training (Map 2 - Main)
file_path = os.path.join(map_dir, "map_1_Training.json")
with open(file_path, "r") as f:
    map_data = json.load(f)

print(f"Loaded Map 2 (Main) for training - overfitting to single map")

epsilon = 1.0 #exploration rate
epsilon_decay = 0.999 # Slower decay - explore more
epsilon_min = 0.05  # Higher minimum - always explore a bit
gamma = 0.995 #discount factor
learning_rate = 0.001
batch_size = 128 #change later
number_episodes = 3000 # Increased for more learning
max_steps = 450 #started at 900
target_update_freq = 200 # Update target network every N steps

class QNetwork(nn.Module):
    def __init__(self, input_dim=STATE_DIMENSIONS, output_dim=NUM_ACTIONS, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.fc_1 = nn.Linear(input_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = self.fc_3(x)
        return x

    
class ReplayBuffer():
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action_idx, reward, next_state, done):
        self.buffer.append((state, action_idx, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards), 
               np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)


network = QNetwork(input_dim=STATE_DIMENSIONS, output_dim=NUM_ACTIONS, hidden_dim=128)
target_network = QNetwork(input_dim=STATE_DIMENSIONS, output_dim=NUM_ACTIONS, hidden_dim=128)
target_network.load_state_dict(network.state_dict())  # Initialize with same weights
target_network.eval()  # Target network is always in eval mode

optimiser = optim.Adam(network.parameters(), lr=learning_rate)
buffer = ReplayBuffer(capacity=10000)
actions = np.array([0, 60, 120, 180, 240, 300])

total_steps = 0  # Track total steps for target network updates

# Track best performance during training
best_time = float('inf')
best_episode = -1
best_reward = -float('inf')
best_moves = []  # Track the actual move sequence

env = SailingEnv(map_data)
print('Starting DQN Train with Target Network')
print(f'Target network will update every {target_update_freq} steps')
print(f'Training on Map 2 (Main) to overfit')
for episodes in range(number_episodes):
    state = env.reset()
    total_reward = 0
    episode_moves = []  # Track moves for this episode
    episode_path = [(env.current_row + 1, env.current_col + 1)]  # 1-indexed

    for step in range(max_steps):
        if np.random.random() < epsilon:
            action_idx = np.random.randint(0, NUM_ACTIONS) #explore (0 to 5)
        else:
            #batch dimension add 9 -> 1,9
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = network(state_tensor)
            action_idx = torch.argmax(q_values).item() 

        # Record the move
        heading = env.actions[action_idx]
        old_pos = (env.current_row + 1, env.current_col + 1)

        next_state, reward, done, info = env.step(action_idx)
        buffer.push(state, action_idx, reward, next_state, done)

        # Track move if position changed (valid move)
        new_pos = (env.current_row + 1, env.current_col + 1)
        if new_pos != old_pos:
            episode_moves.append(heading)
            episode_path.append(new_pos)

        if len(buffer) > batch_size:
            states, action_idxs, rewards, next_states, dones = buffer.sample(batch_size)
            states_t = torch.FloatTensor(states)
            actions_t = torch.LongTensor(action_idxs)
            rewards_t = torch.FloatTensor(rewards)
            next_states_t = torch.FloatTensor(next_states)
            dones_t = torch.FloatTensor(dones) # 1.0 for done, 0.0 for not done

            # DQN update with target network
            # Current Q-values from main network
            q_values = network(states_t)
            current_q = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

            # Next Q-values from TARGET network (stable targets)
            with torch.no_grad():
                next_q_values = target_network(next_states_t)
                max_next_q = next_q_values.max(1)[0]

            target_q = rewards_t + gamma * max_next_q * (1-dones_t)
            loss = F.mse_loss(current_q, target_q)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            total_steps += 1

            # Update target network periodically
            if total_steps % target_update_freq == 0:
                target_network.load_state_dict(network.state_dict())
                # print(f"  [Step {total_steps}] Target network updated")

        state = next_state
        total_reward += reward
        if done:
            break 

    # Check if this episode achieved the best result
    episode_finished = (env.current_row, env.current_col) == env.finish_pos
    if episode_finished and env.current_time < best_time:
        # New best time!
        best_time = env.current_time
        best_episode = episodes
        best_reward = total_reward
        best_moves = episode_moves.copy()

        # Save the model immediately
        best_model_path = "sailing_dqn_best_main.pth"
        torch.save(network.state_dict(), best_model_path)

        # Save the actual moves immediately
        best_moves_file = "dqn_moves_best_during_training.txt"
        with open(best_moves_file, 'w') as f:
            f.write(f"# Best DQN Moves from Training Episode {episodes}\n")
            f.write(f"# Start: {env.start_pos[0]+1}, {env.start_pos[1]+1}\n")
            f.write(f"# Finish: {env.finish_pos[0]+1}, {env.finish_pos[1]+1}\n")
            f.write(f"# Time: {env.current_time:.2f}\n")
            f.write(f"# Reward: {total_reward:.2f}\n")
            f.write(f"# Moves: {len(episode_moves)}\n")
            f.write(f"# Path: {' -> '.join([f'({r},{c})' for r, c in episode_path])}\n")
            f.write("#\n")
            for move in episode_moves:
                f.write(f"{move}\n")

        print(f"\n🏆 NEW BEST! Episode {episodes}: time={env.current_time:.2f}, reward={total_reward:.2f}, moves={len(episode_moves)}")
        print(f"   Model saved to: {best_model_path}")
        print(f"   Moves saved to: {best_moves_file}\n")

    epsilon = max(epsilon_min, epsilon*epsilon_decay)
    if episodes % 10 == 0:
        print(f"Episode {episodes}: reward={total_reward:.2f}, epsilon={epsilon:.3f}, steps={step+1}, time={env.current_time:.2f}, finished={episode_finished}")


MODEL_SAVE_PATH = "sailing_dqn_model.pth"
torch.save(network.state_dict(), MODEL_SAVE_PATH)
print(f"Final model weights saved to: {MODEL_SAVE_PATH}")
print('Training Complete')

# Print best results found during training
print("\n" + "="*60)
print("BEST RESULT DURING TRAINING")
print("="*60)
if best_episode >= 0:
    print(f"Map 2 (Main):   Time={best_time:7.2f} (Episode {best_episode})")
    print(f"                Moves: {len(best_moves)}")
    print(f"                Model: sailing_dqn_best_main.pth")
    print(f"                Moves: dqn_moves_best_during_training.txt")
else:
    print(f"Map 2 (Main):   No successful completion")
print("="*60 + "\n")

# ========== EXTRACT MOVES FOR MAP 2 (MAIN) ==========
print("\n" + "="*60)
print("EXTRACTING MOVES FOR MAP 2 (MAIN)")
print("="*60)

output_file = "dqn_moves_main.txt"
test_env = SailingEnv(map_data)

# Load the best model if it exists
best_model_path = "sailing_dqn_best_main.pth"
extraction_network = QNetwork(input_dim=STATE_DIMENSIONS, output_dim=NUM_ACTIONS, hidden_dim=128)

if best_episode >= 0:
    print(f"Loading best model from episode {best_episode} (time: {best_time:.2f})")
    extraction_network.load_state_dict(torch.load(best_model_path))
else:
    print(f"No successful completion during training, using final model")
    extraction_network.load_state_dict(network.state_dict())

extraction_network.eval()

# Test agent on this map first
print(f"Testing agent on Main map...")
test_agent(test_env, extraction_network, num_episodes_to_test=5)


# Extract moves
result = extract_moves(test_env, extraction_network, output_file, num_attempts=20)

# Summary
print("\n" + "="*60)
print("EXTRACTION SUMMARY")
print("="*60)
status = "✓ FINISHED" if result['finished'] else "⚠ INCOMPLETE"
print(f"Map 2 (Main)    -> {output_file:30s}")
print(f"  {status}  |  Time: {result['time']:7.2f}  |  Moves: {result['steps']:3d}")
print("="*60)
print("\nTo test in MATLAB:")
print("  1. Open sailingGame.m")
print("  2. Select 'Main' map")
print("  3. Click 'Load Moves (auto)'")
print("  4. Select 'dqn_moves_main.txt'")
print("  5. Watch the DQN agent's path replay!")
print("="*60 + "\n")