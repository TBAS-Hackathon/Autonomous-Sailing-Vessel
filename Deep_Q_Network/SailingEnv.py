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

# AI generated code to set up agent testing and the gym environment
# -----------------------------------------------------------------
# ## 1. The Environment Class
# (Paste the full SailingEnv class here)
# -----------------------------------------------------------------
class SailingEnv():
    def __init__(self, map_data):
        # --- Load Map Data ---
        self.wind_speed = np.array(map_data['windSpeed'])
        self.wind_dir = np.array(map_data['windDir'])
        self.N_rows, self.N_cols = self.wind_speed.shape
        
        # Adjust for 0-indexing
        self.start_pos = (map_data['startPos'][0] - 1, map_data['startPos'][1] - 1)
        self.finish_pos = (map_data['finishPos'][0] - 1, map_data['finishPos'][1] - 1)
        
        # --- Game Rules (Ported from MATLAB) ---
        self.actions = [0, 60, 120, 180, 240, 300]
        self.BASE_TIME = 10
        self.NO_GO_ANGLE = 30
        
        # --- Game State ---
        self.current_row = 0
        self.current_col = 0
        self.current_heading = None # Use None for NaN
        self.current_time = 0
        self.max_steps = 1000 # Max moves per episode
        self.steps_taken = 0
        
        # State for reward calc
        self.last_pos = np.array(self.start_pos)


    def reset(self):
        """Resets the game to the start position."""
        self.current_row, self.current_col = self.start_pos
        self.current_heading = None
        self.current_time = 0
        self.steps_taken = 0
        
        self.last_pos = np.array(self.start_pos)
        
        # Return the initial state vector
        return self._get_state_vector()

    def step(self, action_idx):
        """
        This is your 'moveBoat' logic, re-written in Python.
        """
        boat_dir = self.actions[action_idx]
        
        # Store current position for reward calc
        self.last_pos = np.array([self.current_row, self.current_col])
        
        # --- Get wind at current cell ---
        w_to = self.wind_dir[self.current_row, self.current_col]
        w_from = (w_to + 180) % 360
        w_speed = self.wind_speed[self.current_row, self.current_col]

        # --- 1. Check No-Go Zone ---
        raw = abs(boat_dir - w_from)
        rel = min(raw, 360 - raw)
        if rel < self.NO_GO_ANGLE:
            self.steps_taken += 1
            done = (self.steps_taken >= self.max_steps)
            return self._get_state_vector(), -10.0, done, {} 

        # --- 2. Check Speed from Polar ---
        f = self._polar_factor(rel)
        if f <= 0:
            self.steps_taken += 1
            done = (self.steps_taken >= self.max_steps)
            return self._get_state_vector(), -10.0, done, {} 

        # --- 3. Get New Position ---
        dr, dc = self._get_move_vector(boat_dir)
        new_row = self.current_row + dr
        new_col = self.current_col + dc

        # --- 4. Check Map Bounds ---
        if not (0 <= new_row < self.N_rows and 0 <= new_col < self.N_cols):
            self.steps_taken += 1
            done = (self.steps_taken >= self.max_steps)
            return self._get_state_vector(), -10.0, done, {}

        # --- 5. Valid Move: Calculate Time & Update State ---
        boat_speed = w_speed * f if w_speed > 0 else 1e-6 # Avoid div by zero
        tp = self._turn_penalty(self.current_heading, boat_dir)
        move_time = (self.BASE_TIME / boat_speed) + tp

        self.current_row = new_row
        self.current_col = new_col
        self.current_heading = boat_dir
        self.current_time += move_time
        self.steps_taken += 1

        # --- 6. Calculate Reward & Done ---
        next_state = self._get_state_vector()
        reward, done = self._calculate_reward(move_time)

        return next_state, reward, done, {}

    def _get_state_vector(self):
        """Converts the current game state to the DQN's 9-element input vector."""
        
        # --- Current Cell Info ---
        row_norm = self.current_row / self.N_rows
        col_norm = self.current_col / self.N_cols
        
        # --- Wind Info ---
        w_speed = self.wind_speed[self.current_row, self.current_col] / 10.0 # Normalize
        w_dir_rad = np.deg2rad(self.wind_dir[self.current_row, self.current_col])
        w_cos = np.cos(w_dir_rad)
        w_sin = np.sin(w_dir_rad)

        # --- Goal Info ---
        delta_r = (self.finish_pos[0] - self.current_row) / self.N_rows
        delta_c = (self.finish_pos[1] - self.current_col) / self.N_cols

        # --- Boat Info ---
        if self.current_heading is None:
            heading_cos, heading_sin = 0.0, 0.0
        else:
            heading_rad = np.deg2rad(self.current_heading)
            heading_cos = np.cos(heading_rad)
            heading_sin = np.sin(heading_rad)

        state_vec = np.array([
            row_norm, w_speed, w_cos, w_sin,
            col_norm, delta_r, delta_c,
            heading_cos, heading_sin
        ], dtype=np.float32) # Use float32 for neural networks
        
        # Quick check for 9 elements
        if len(state_vec) != 9:
             print(f"STATE VECTOR WRONG SIZE: {len(state_vec)}")
             
        return state_vec
    
    def _calculate_reward(self, move_time):
        """
        Calculates reward based purely on time optimization.

        Reward = -move_time (includes turn penalty)

        This encourages the agent to:
        - Take faster routes (higher wind speed cells)
        - Minimize turns (turn penalty is in move_time)
        - Reach the goal quickly (cumulative reward is better with less time)

        Args:
            move_time: Time cost of the move (BASE_TIME/boat_speed + turn_penalty)

        Returns:
            (reward, done): Reward for this step and done flag
        """
        done = False

        # Check if reached finish
        if (self.current_row, self.current_col) == self.finish_pos:
            done = True
            # Give bonus for finishing to encourage completion
            return 100.0, done

        # Penalty for running out of moves (didn't finish)
        if self.steps_taken >= self.max_steps:
            done = True
            return -100.0, done

        # Reward is negative time cost
        # Lower time = higher reward (less negative)
        # Turn penalties are already included in move_time
        reward = -move_time

        return reward, done

    # --- (HELPER FUNCTIONS: Ported from MATLAB) ---
    def _polar_factor(self, rel_deg):
        if 30 <= rel_deg < 60:   return 1.0
        elif 60 <= rel_deg < 90:  return 0.95
        elif 90 <= rel_deg < 135: return 0.85
        elif 135 <= rel_deg <= 180: return 0.70
        else: return 0

    def _turn_penalty(self, prev_dir, new_dir):
        """Turn penalty - EXACTLY matches MATLAB sailingGame.m implementation"""
        if prev_dir is None: return 0
        d = abs(prev_dir - new_dir)
        d = min(d, 360 - d)

        if d == 0:
            return 0
        elif d <= 10:
            return 0.5
        elif d <= 20:
            return 1.0
        elif d <= 30:
            return 1.5
        elif d <= 40:
            return 2.0
        elif d <= 50:
            return 2.5
        elif d <= 60:
            return 3.0
        else:
            return 4.0

    def _get_move_vector(self, heading):
        if heading == 0:   return (-1, 0)
        elif heading == 60:  return (-1, 1)
        elif heading == 120: return (1, 1)
        elif heading == 180: return (1, 0)
        elif heading == 240: return (1, -1)
        elif heading == 300: return (-1, -1)
        else: return (0, 0)

# -----------------------------------------------------------------
# ## 2. The Test Code
# (This code now runs in the same file)
# -----------------------------------------------------------------
def create_mock_map():
    """
    Creates a tiny 5x5 map for testing.
    - Start: (4, 0) [Python index]
    - Finish: (0, 2) [Python index]
    - Wind: Blowing TO 90° (East).
    - Wind-FROM: 270° (West).
    - No-Go Zone: 240° (SW), 300° (NW) are invalid.
    """
    N_rows, N_cols = 5, 5
    
    wind_speed = np.full((N_rows, N_cols), 5.0)
    
    # --- CHANGED THIS LINE ---
    wind_dir = np.full((N_rows, N_cols), 90.0) # Wind blows TO East
    
    map_data = {
        'windSpeed': wind_speed,
        'windDir': wind_dir,
        'startPos': [5, 1],   # MATLAB 1-indexed (Python 4,0)
        'finishPos': [1, 3]   # MATLAB 1-indexed (Python 0,2)
    }
    return map_data

def run_tests():
    print("--- 1. Creating Mock Map & Environment ---")
    mock_map = create_mock_map()
    
    # --- Since the class is in this file, we just call it ---
    env = SailingEnv(mock_map)
    # ------------------------------------------------------
    
    print(f"Environment created.")
    print(f"Start (Python index): {env.start_pos} | Finish (Python index): {env.finish_pos}")
    print(f"Actions: {env.actions}")
    print("-" * 30)

    # --- Test 1: Reset ---
    print("--- 2. Testing env.reset() ---")
    state = env.reset()
    print(f"Reset successful.")
    assert state.shape == (9,), f"State shape is wrong! Expected (9,), got {state.shape}"
    print(f"State shape: {state.shape} (Correct)")
    print(f"Initial state vector:\n{np.round(state, 2)}")
    print("-" * 30)

    # --- Test 2: Valid Step ---
    print("--- 3. Testing env.step() - VALID Move ---")
    # Wind-FROM is 0°. A 180° (South) move is valid. (Action index 3)
    action_idx = 0
    print(f"Taking action: Index {action_idx} ({env.actions[action_idx]}°)")
    
    next_state, reward, done, _ = env.step(action_idx)
    
    print(f"  Next State:\n{np.round(next_state, 2)}")
    print(f"  Reward: {reward}")
    print(f"  Done: {done}")
    assert reward != -10.0, "Valid move was incorrectly penalized!"
    print("-" * 30)

    # --- Test 3: Invalid Step (No-Go) ---
    print("--- 4. Testing env.step() - INVALID Move (No-Go) ---")
    env.reset() 
    # Wind-FROM is 0°. A 0° (North) move is invalid. (Action index 0)
    action_idx = 4
    print(f"Taking action: Index {action_idx} ({env.actions[action_idx]}°)")
    
    next_state, reward, done, _ = env.step(action_idx)
    
    print(f"  Next State (should be same as initial):\n{np.round(next_state, 2)}")
    print(f"  Reward: {reward}")
    print(f"  Done: {done}")
    assert reward == -10.0, "Invalid move did not receive penalty!"
    print("  (Correctly received penalty)")
    print("-" * 30)

    # --- Test 4: Run a short random episode ---
    print("--- 5. Testing a full (short) random episode ---")
    state = env.reset()
    total_reward = 0
    
    for i in range(15):
        action_idx = np.random.randint(0, len(env.actions))
        next_state, reward, done, _ = env.step(action_idx)
        total_reward += reward
        print(f"  Step {i+1:02d} | Action: {env.actions[action_idx]:3d}° | Row: {env.current_row}, Col: {env.current_col} | Reward: {reward:6.2f} | Done: {done}")
        
        if done:
            print("Episode finished early!")
            break
        state = next_state
        
    print(f"Episode finished. Total Reward: {total_reward:.2f}")
    print("="*50)
    print("ALL TESTS COMPLETED!")
    print("="*50)

# --- This makes the file runnable ---
if __name__ == "__main__":
    run_tests()
       
# End of AI generated code 