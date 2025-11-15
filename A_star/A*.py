import numpy as np
import json

# Map Loading (Change these filenames for different maps)
wind_speed = np.loadtxt("./prompts/map_3_Tiebreaker_windSpeed.csv", delimiter=",")
wind_dir = np.loadtxt("./prompts/map_3_Tiebreaker_windDir.csv", delimiter=",")

# Metadata Loading
with open("./prompts/map_3_Tiebreaker_meta.json", "r") as f:
    meta = json.load(f)

start_pos = tuple(meta["startPos"])
finish_pos = tuple(meta["finishPos"])
headings = [0, 60, 120, 180, 240, 300]
base_time = 10.0
no_go_angle = 30.0

def polar_factor(rel_deg):
    if 30 <= rel_deg < 60: return 1.0
    if 60 <= rel_deg < 90: return 0.95
    if 90 <= rel_deg < 135: return 0.85
    if 135 <= rel_deg <= 180: return 0.70
    return 0.0 # Should be blocked by no-go check anyway

def turn_penalty(prev_dir, new_dir):
    if np.isnan(prev_dir):
        return 0.0
    d = abs(prev_dir - new_dir)
    d = min(d, 360 - d)
    if d <= 60: return d * 0.05 ##
    return 4.0

def get_move_cost(r, c, h_prev, h_new):
    """
    Calculates the cost of moving from (r, c) with new heading h_new,
    having arrived with h_prev.
    Returns (cost, new_r, new_c) or (inf, -1, -1) if invalid.
    """
    # 1. Get wind at (r, c) [convert from 1-based to 0-based]
    try:
        w_speed = wind_speed[r-1, c-1]
        w_to = wind_dir[r-1, c-1]
        w_from = (w_to + 180) % 360
    except IndexError:
        return (np.inf, -1, -1) # Stay on map

    # 2. Check no-go angle
    raw = abs(h_new - w_from)
    rel = min(raw, 360 - raw)
    if rel < no_go_angle:
        return (np.inf, -1, -1) # Invalid move

    # 3. Get new position (dr, dc)
    if h_new == 0:   dr, dc = -1,  0
    elif h_new == 60:  dr, dc = -1,  1
    elif h_new == 120: dr, dc =  1,  1
    elif h_new == 180: dr, dc =  1,  0
    elif h_new == 240: dr, dc =  1, -1
    elif h_new == 300: dr, dc = -1, -1
    
    new_r, new_c = r + dr, c + dc

    # 4. Check map bounds
    nrows, ncols = wind_speed.shape
    if not (1 <= new_r <= nrows and 1 <= new_c <= ncols):
        return (np.inf, -1, -1)

    # 5. Calculate speed and time
    f = polar_factor(rel)
    if f <= 0:
        return (np.inf, -1, -1) # No speed
        
    boat_speed = w_speed * f
    tp = turn_penalty(h_prev, h_new)
    move_time = (base_time / boat_speed) + tp
    
    return (move_time, new_r, new_c)

import heapq
import math

# Heuristic 
max_wind_speed = np.max(wind_speed)
fr, fc = finish_pos
def heuristic(r, c):
    dist = math.sqrt((r - fr)**2 + (c - fc)**2)
    return (dist * base_time) / max_wind_speed

# A*
def solve_map():
    start_r, start_c = start_pos
    
    # Priority queue stores: (f_cost, g_cost, r, c, heading, path)
    # path is (heading, r, c, time)
    pq = [(heuristic(start_r, start_c), 0.0, start_r, start_c, np.nan, [])]
    
    # visited dict stores: min_time_to_reach[(r, c, h)]
    visited = {}

    while pq:
        f, g, r, c, h_prev, path = heapq.heappop(pq)
        
        # 1. Finished?
        if (r, c) == finish_pos:
            print(f"🎉 Finished! Optimal time: {g:.2f}")
            return path + [('FINISH', r, c, g)]

        # 2. Better path to current state?
        state = (r, c, h_prev)
        if state in visited and visited[state] <= g:
            continue
        visited[state] = g

        # 3. Explore 6 possible moves
        for h_new in headings:
            cost, new_r, new_c = get_move_cost(r, c, h_prev, h_new)
            
            if np.isinf(cost):
                continue

            new_g = g + cost
            new_state = (new_r, new_c, h_new)

            # Add if better than all known paths
            if new_state not in visited or visited[new_state] > new_g:
                new_f = new_g + heuristic(new_r, new_c)
                new_path = path + [(h_new, new_r, new_c, new_g)]
                heapq.heappush(pq, (new_f, new_g, new_r, new_c, h_new, new_path))
                
    return None # If no path found, should not be the case

optimal_path = solve_map()

if optimal_path:
    print("\n--- Optimal Move List ---")
    with open("optimal_moves_3.txt", "w") as f:
        for move in optimal_path:
            if move[0] != 'FINISH':
                print(f"Move {move[0]}° -> ({move[1]}, {move[2]}). Time: {move[3]:.2f}")
                f.write(f"{move[0]}\n")

