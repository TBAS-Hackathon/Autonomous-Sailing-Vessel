import numpy as np
import json
import heapq
import math

# -----------------------------
# Configuration / augmentation
# -----------------------------
# mode: 'last_k', 'run_length', or 'both'
AUGMENTATION = {
    'mode': 'last_k',
    'k': 1,
    'max_run_len': 30,
    'break_run_penalty': 0.0
}

# -----------------------------
# Map Loading (unchanged)
# -----------------------------
wind_speed = np.loadtxt("./prompts/map_100_windSpeed.csv", delimiter=",")
wind_dir = np.loadtxt("./prompts/map_100_windDir.csv", delimiter=",")

with open("./prompts/map_100_meta.json", "r") as f:
    meta = json.load(f)

start_pos = tuple(meta["startPos"])   # (r, c) 1-based
finish_pos = tuple(meta["finishPos"]) # (r, c) 1-based

# Move headings (degrees)
headings = [0, 60, 120, 180, 240, 300]

# Base time model
base_time = 10.0
no_go_angle = 30.0

# -----------------------------
# Helper cost functions (from your original code)
# -----------------------------

def polar_factor(rel_deg):
    if 30 <= rel_deg < 60: return 1.0
    if 60 <= rel_deg < 90: return 0.95
    if 90 <= rel_deg < 135: return 0.85
    if 135 <= rel_deg <= 180: return 0.70
    return 0.0 # blocked (should be filtered by no-go check)


def turn_penalty(prev_dir, new_dir):
    # prev_dir can be np.nan to indicate "no previous heading"
    if np.isnan(prev_dir):
        return 0.0
    d = abs(prev_dir - new_dir)
    d = min(d, 360 - d)
    if d <= 60: return d * 0.05
    return 4.0


# Original move cost function (keeps same semantics). It expects h_prev scalar.
def get_move_cost(r, c, h_prev, h_new):
    """
    Returns (move_time, new_r, new_c) or (np.inf, -1, -1) if invalid.
    Coordinates r,c are 1-based as in your original code.
    """
    try:
        w_speed = wind_speed[r-1, c-1]
        w_to = wind_dir[r-1, c-1]
        w_from = (w_to + 180) % 360
    except IndexError:
        return (np.inf, -1, -1)

    raw = abs(h_new - w_from)
    rel = min(raw, 360 - raw)
    if rel < no_go_angle:
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

    nrows, ncols = wind_speed.shape
    if not (1 <= new_r <= nrows and 1 <= new_c <= ncols):
        return (np.inf, -1, -1)

    f = polar_factor(rel)
    if f <= 0:
        return (np.inf, -1, -1)

    boat_speed = w_speed * f
    if boat_speed <= 0:
        return (np.inf, -1, -1)

    tp = turn_penalty(h_prev, h_new)
    move_time = (base_time / boat_speed) + tp

    return (move_time, new_r, new_c)


# -----------------------------
# Augmented-state wrappers
# -----------------------------

# Utility: create the initial history tuple of length k filled with np.nan
def make_initial_history(k):
    return tuple([float('nan')] * k)

# Update history tuple (keeps last k headings)
def update_history(history_tuple, new_heading):
    # history_tuple is a tuple length k
    lst = list(history_tuple)
    lst.append(new_heading)
    # Keep last k elements
    k = len(history_tuple)
    return tuple(lst[-k:])


# If run-length mode is used, update run length
def update_run_length(prev_run_len, last_heading, new_heading):
    if np.isnan(last_heading):
        return 1
    if last_heading == new_heading:
        return min(prev_run_len + 1, AUGMENTATION['max_run_len'])
    return 1


# Compute any extra history-dependent penalty. By default this calls your
# original turn_penalty (which looks only at the last heading) and then
# optionally adds a "break_run_penalty" if configured and a long straight run
# is broken.
def compute_full_move_cost(r, c, history_tuple, run_length, h_new):
    # last heading (most recent) for backward compatibility
    last_heading = history_tuple[-1] if history_tuple is not None else float('nan')

    # Use the original get_move_cost which already includes turn_penalty(last_heading, h_new)
    base_cost, new_r, new_c = get_move_cost(r, c, last_heading, h_new)
    if np.isinf(base_cost):
        return (np.inf, -1, -1, None)

    extra = 0.0
    if AUGMENTATION['mode'] in ('run_length', 'both'):
        # If breaking a long run, optionally add extra penalty
        if AUGMENTATION.get('break_run_penalty', 0.0) > 0.0:
            if not np.isnan(last_heading) and last_heading != h_new:
                if run_length is not None and run_length >= AUGMENTATION.get('max_run_len', 30):
                    extra += AUGMENTATION['break_run_penalty']

    total_cost = base_cost + extra
    return (total_cost, new_r, new_c, last_heading)


# -----------------------------
# Heuristic (admissible)
# -----------------------------
max_wind_speed = np.max(wind_speed)
fr, fc = finish_pos

def heuristic(r, c):
    dist = math.sqrt((r - fr)**2 + (c - fc)**2)
    # Lower bound time: travel distance times base_time divided by best possible boat speed
    return (dist * base_time) / max_wind_speed


# -----------------------------
# Augmented-state A* solver
# -----------------------------

def solve_map_augmented():
    start_r, start_c = start_pos

    mode = AUGMENTATION['mode']
    k = AUGMENTATION.get('k', 1)

    # initial history / run_length according to mode
    history0 = make_initial_history(k) if mode in ('last_k', 'both') else None
    runlen0 = 0 if mode in ('run_length', 'both') else None

    # Priority queue stores: (f_cost, g_cost, r, c, history_tuple, run_len, path)
    # path entries: (heading, r, c, cumulative_time)
    start_f = heuristic(start_r, start_c)
    pq = [(start_f, 0.0, start_r, start_c, history0, runlen0, [])]

    # visited dict stores best g for each augmented state: key -> (r,c,history_tuple,run_len)
    visited = {}

    while pq:
        f, g, r, c, history_t, run_len, path = heapq.heappop(pq)

        # If reached goal cell (regardless of history) -> return path
        if (r, c) == finish_pos:
            print(f"🎉 Finished! Optimal time: {g:.2f}")
            return path + [('FINISH', r, c, g)]

        # State key: pack history and run_len into the key
        key = (r, c, history_t, run_len)
        if key in visited and visited[key] <= g:
            continue
        visited[key] = g

        # Expand neighbors (6 headings)
        for h_new in headings:
            cost, new_r, new_c, last_heading = compute_full_move_cost(r, c, history_t, run_len, h_new)
            if np.isinf(cost):
                continue

            new_g = g + cost

            # Update history/run length for new state
            if history_t is not None:
                new_history = update_history(history_t, h_new)
            else:
                new_history = None

            if run_len is not None:
                # last_heading returned above is previous last heading
                new_run_len = update_run_length(run_len if run_len>0 else 0, last_heading, h_new)
            else:
                new_run_len = None

            new_key = (new_r, new_c, new_history, new_run_len)

            if new_key not in visited or visited[new_key] > new_g:
                new_f = new_g + heuristic(new_r, new_c)
                new_path = path + [(h_new, new_r, new_c, new_g)]
                heapq.heappush(pq, (new_f, new_g, new_r, new_c, new_history, new_run_len, new_path))

    return None


# -----------------------------
# Run and write output
# -----------------------------
if __name__ == '__main__':
    print("Augmentation mode:", AUGMENTATION)
    optimal_path = solve_map_augmented()

    if optimal_path:
        print("\n--- Optimal Move List ---")
        with open("./A_star/heading_A_star/optimal_moves_augmented_100.txt", "w") as f:
            for move in optimal_path:
                if move[0] != 'FINISH':
                    print(f"Move {move[0]}° -> ({move[1]}, {move[2]}). Time: {move[3]:.2f}")
                    f.write(f"{move[0]}\n")
            # write FINISH symbol
            f.write('FINISH\n')

    else:
        print("No path found.")
