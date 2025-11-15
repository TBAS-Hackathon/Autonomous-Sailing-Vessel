import numpy as np
import json
import heapq
import math

# Load map data (Change these filenames for different maps)
wind_speed = np.loadtxt("./prompts/map_100_windSpeed.csv", delimiter=",")
wind_dir   = np.loadtxt("./prompts/map_100_windDir.csv", delimiter=",")

with open("./prompts/map_100_meta.json", "r") as f:
    meta = json.load(f)

start_pos = tuple(meta["startPos"])   # assumed 1-based (r,c)
finish_pos = tuple(meta["finishPos"])
headings = [0, 60, 120, 180, 240, 300]

base_time = 10.0
no_go_angle = 30.0

def polar_factor(rel_deg):
    # keep exactly your lookup
    if 30 <= rel_deg < 60: return 1.0
    if 60 <= rel_deg < 90: return 0.95
    if 90 <= rel_deg < 135: return 0.85
    if 135 <= rel_deg <= 180: return 0.70
    return 0.0 # Should be blocked by no-go check anyway

def turn_penalty(prev_dir, new_dir):
    # Accept None (start) as "no previous heading" -> zero penalty
    if prev_dir is None:
        return 0.0
    # keep same numeric behavior as before for floats
    # prev_dir should be numeric (0..360)
    d = abs(prev_dir - new_dir)
    d = min(d, 360 - d)
    if d <= 60:
        return d * 0.05
    return 4.0

def get_move_cost(r, c, h_prev, h_new):
    """
    r,c are 1-based integer grid coordinates.
    Returns (cost, new_r, new_c) or (np.inf, -1, -1) if invalid.
    """
    # 1. Get wind at (r, c) [convert to 0-based indices]
    nrows, ncols = wind_speed.shape
    if not (1 <= r <= nrows and 1 <= c <= ncols):
        return (np.inf, -1, -1)

    w_speed = wind_speed[r-1, c-1]
    w_to    = wind_dir[r-1, c-1]
    w_from  = (w_to + 180) % 360

    # 2. Check no-go angle
    raw = abs(h_new - w_from)
    rel = min(raw, 360 - raw)
    if rel < no_go_angle:
        return (np.inf, -1, -1)  # invalid move into no-go

    # 3. Heading -> delta (1-based grid)
    if h_new == 0:     dr, dc = -1,  0
    elif h_new == 60:  dr, dc = -1,  1
    elif h_new == 120: dr, dc =  1,  1
    elif h_new == 180: dr, dc =  1,  0
    elif h_new == 240: dr, dc =  1, -1
    elif h_new == 300: dr, dc = -1, -1
    else:
        return (np.inf, -1, -1)

    new_r, new_c = r + dr, c + dc

    # 4. Check bounds for destination
    if not (1 <= new_r <= nrows and 1 <= new_c <= ncols):
        return (np.inf, -1, -1)

    # 5. Compute speed/time
    f = polar_factor(rel)
    if f <= 0:
        return (np.inf, -1, -1)

    boat_speed = w_speed * f
    if boat_speed <= 0:
        return (np.inf, -1, -1)  # cannot move (zero speed)

    tp = turn_penalty(h_prev, h_new)
    move_time = (base_time / boat_speed) + tp

    return (move_time, new_r, new_c)


max_wind_speed = np.max(wind_speed)
fr, fc = finish_pos

def heuristic(r, c):
    # Straight-line Euclidean distance (in grid units) scaled to time lower bound.
    # Using max possible speed (max_wind_speed * max_polar_factor(=1.0)) gives admissible heuristic.
    dist = math.sqrt((r - fr)**2 + (c - fc)**2)
    # Lower bound time = (distance * base_time) / max_wind_speed
    # (this matches your earlier heuristic; keep it admissible by not adding turn penalties)
    if max_wind_speed <= 0:
        return 0.0
    return (dist * base_time) / max_wind_speed


def solve_map():
    start_r, start_c = start_pos
    goal_r, goal_c = finish_pos

    # Priority queue: (f_cost, g_cost, r, c, prev_heading, path)
    # prev_heading is None for starting state (no previous heading)
    start_state = (heuristic(start_r, start_c), 0.0, start_r, start_c, None, [])
    pq = [start_state]

    # visited: map state -> best g encountered. state key: (r, c, heading_or_None)
    visited = {}

    EPS = 1e-9

    while pq:
        f, g, r, c, h_prev, path = heapq.heappop(pq)

        # If we reached goal (any heading), return
        if (r, c) == (goal_r, goal_c):
            print(f"🎉 Finished! Optimal time: {g:.2f}")
            return path + [('FINISH', r, c, g)]

        state_key = (r, c, h_prev)
        # If we already have a better or equal arrival to this exact (r,c,heading), skip.
        if state_key in visited and visited[state_key] <= g + EPS:
            continue
        visited[state_key] = g

        # Explore neighbors (6 headings)
        for h_new in headings:
            cost, new_r, new_c = get_move_cost(r, c, h_prev, h_new)
            if np.isinf(cost):
                continue

            new_g = g + cost
            new_state_key = (new_r, new_c, h_new)

            # If better than known for that state, push
            if (new_state_key not in visited) or (visited[new_state_key] > new_g + EPS):
                new_f = new_g + heuristic(new_r, new_c)
                new_path = path + [(h_new, new_r, new_c, new_g)]
                heapq.heappush(pq, (new_f, new_g, new_r, new_c, h_new, new_path))

    # No path found
    print("⚠️ No path found")
    return None

if __name__ == "__main__":
    optimal_path = solve_map()

    if optimal_path:
        print("\n--- Optimal Move List ---")
        with open("optimal_heading_moves_100.txt", "w") as f:
            for move in optimal_path:
                if move[0] != 'FINISH':
                    deg, r, c, t = move
                    print(f"Move {deg}° -> ({r}, {c}). Time: {t:.2f}")
                    f.write(f"{deg}\n")
                else:
                    # FINISH line
                    print(f"{move[0]} at ({move[1]}, {move[2]}). Total time: {move[3]:.2f}")
                    f.write("FINISH\n")
