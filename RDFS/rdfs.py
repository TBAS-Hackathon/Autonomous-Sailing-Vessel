#!/usr/bin/env python3
"""
Sailing Game Brute-Force (Recursive DFS) Solver

Finds the optimal path for the sailing game using a recursive
depth-first search. This is extremely inefficient, compared to the other algorithms used here, but demonstrates
the brute-force concept, fairly well, especially on 30x30 maps.
"""

import numpy as np
import json
import math
import sys
import os

BASE_TIME = 10.0
NO_GO_ANGLE = 30.0
HEADINGS = [0, 60, 120, 180, 240, 300]
NAN_HEADING = -1

def polar_factor(rel_deg):
    if 30 <= rel_deg < 60: return 1.0
    if 60 <= rel_deg < 90: return 0.95
    if 90 <= rel_deg < 135: return 0.85
    if 135 <= rel_deg <= 180: return 0.70
    return 0.0

def turn_penalty(prev_dir, new_dir):
    if prev_dir == NAN_HEADING:
        return 0.0
    d = abs(prev_dir - new_dir)
    d = min(d, 360 - d)
    if d <= 60:
        return d * 0.05
    return 4.0

def get_move_cost(r, c, h_prev, h_new, wind_speed_map, wind_dir_map):
    try:
        w_speed = wind_speed_map[r-1, c-1]
        w_to = wind_dir_map[r-1, c-1]
        w_from = (w_to + 180) % 360
    except IndexError:
        return (np.inf, -1, -1) 

    raw = abs(h_new - w_from)
    rel = min(raw, 360 - raw)
    if rel < NO_GO_ANGLE:
        return (np.inf, -1, -1)

    if h_new == 0:   dr, dc = -1,  0
    elif h_new == 60:  dr, dc = -1,  1
    elif h_new == 120: dr, dc =  1,  1
    elif h_new == 180: dr, dc =  1,  0
    elif h_new == 240: dr, dc =  1, -1
    elif h_new == 300: dr, dc = -1, -1
    else:
        return (np.inf, -1, -1)
    
    new_r, new_c = r + dr, c + dc

    nrows, ncols = wind_speed_map.shape
    if not (1 <= new_r <= nrows and 1 <= new_c <= ncols):
        return (np.inf, -1, -1)

    f = polar_factor(rel)
    boat_speed = w_speed * f
    if f <= 0 or boat_speed <= 0:
        return (np.inf, -1, -1)
        
    tp = turn_penalty(h_prev, h_new)
    move_time = (BASE_TIME / boat_speed) + tp
    
    return (move_time, new_r, new_c)

best_time_so_far = np.inf
best_path_so_far = []

visited_states = {}

# Store map data globally for the recursive function
g_wind_speed = None
g_wind_dir = None
g_finish_pos = None

def solve_bruteforce_recursive(r, c, h_prev, current_time, current_path):

    global best_time_so_far, best_path_so_far, visited_states
    global g_wind_speed, g_wind_dir, g_finish_pos

    # Pruning Rule 1: Branch and Bound
    # If this path is already worse than our best, stop.
    if current_time >= best_time_so_far:
        return

    # Pruning Rule 2: Visited States
    # If we've been to this exact state (r, c, h_prev) before
    # with a better or equal time, stop.
    state = (r, c, h_prev)
    if current_time >= visited_states.get(state, np.inf):
        return
    # This is a new best path to this state, record it.
    visited_states[state] = current_time

    # Goal Check
    if (r, c) == g_finish_pos:
        # New, better path scenario
        if current_time < best_time_so_far:
            best_time_so_far = current_time
            best_path_so_far = current_path
            # Print progress
            print(f"  ... New best time found: {current_time:.4f} (path len {len(current_path)})")
        return # Stop this path at the finish

    # Recursive Step
    for h_new in HEADINGS:
        (cost, new_r, new_c) = get_move_cost(
            r, c, h_prev, h_new, g_wind_speed, g_wind_dir
        )
        
        if np.isinf(cost): # Invalid move
            continue

        solve_bruteforce_recursive(
            new_r, new_c, 
            h_new, 
            current_time + cost, 
            current_path + [h_new]
        )

def main():
    global g_wind_speed, g_wind_dir, g_finish_pos, best_time_so_far, best_path_so_far, visited_states

    # Increase Python's recursion depth limit
    try:
        sys.setrecursionlimit(50000) 
    except Exception as e:
        print(f"Warning: Could not set recursion limit. {e}")

    # Hardcoded map metadata
    map_meta_data = {
        "map_1_Training": {"startPos": [30, 1], "finishPos": [1, 15], "name": "Training"},
        "map_2_Main": {"startPos": [30, 1], "finishPos": [1, 20], "name": "Main"},
        "map_3_Tiebreaker": {"startPos": [30, 1], "finishPos": [1, 25], "name": "Tiebreaker"},
        "map_100": {"startPos": [100, 1], "finishPos": [1, 50], "name": "Large Map Example"}
    }
    
    map_prefix = "map_100"
    
    try:
        ws_file = f"./prompts/{map_prefix}_windSpeed.csv"
        wd_file = f"./prompts/{map_prefix}_windDir.csv"
        
        g_wind_speed = np.loadtxt(ws_file, delimiter=",")
        g_wind_dir = np.loadtxt(wd_file, delimiter=",")
        
        meta = map_meta_data[map_prefix]
        start_pos = tuple(meta["startPos"])
        g_finish_pos = tuple(meta["finishPos"])
        map_name = meta["name"]
        
    except FileNotFoundError as e:
        print(f"Error: Missing map file. {e}", file=sys.stderr)
        print("Please ensure map_1_Training_*.csv are present.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading map files: {e}")
        return

    print(f"--- Running Brute-Force DFS on: {map_name} ---")
    print(f"  Start: {start_pos}, Finish: {g_finish_pos}")
    print("  This will be very slow and will print new best times as it finds them...")
    
    start_r, start_c = start_pos
    
    solve_bruteforce_recursive(start_r, start_c, NAN_HEADING, 0.0, [])
    
    # Results
    print("\n--- Brute-Force Search Complete ---")
    if np.isinf(best_time_so_far):
        print("  No path found to the finish.")
    else:
        print(f"  Optimal Time: {best_time_so_far:.4f}")
        print(f"  Move Count:   {len(best_path_so_far)}")
        path_str = ", ".join(map(str, best_path_so_far))
        print(f"  Path (headings):\n    [{path_str}]\n")
        
        path_filename = f"bruteforce_path_{map_prefix}.txt"
        with open(path_filename, "w") as f:
            for heading in best_path_so_far:
                f.write(f"{heading}\n")
        print(f"  (Saved optimal move list to: {path_filename})\n")

if __name__ == "__main__":
    main()