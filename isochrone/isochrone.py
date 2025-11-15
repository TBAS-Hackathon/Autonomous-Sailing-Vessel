import numpy as np
import json
import heapq
import sys

BASE_TIME = 10.0
NO_GO_ANGLE = 30.0
HEADINGS = [0, 60, 120, 180, 240, 300]

def polar_factor(rel_deg):
    """Returns polar speed factor based on relative wind angle."""
    if 30 <= rel_deg < 60: return 1.0
    elif 60 <= rel_deg < 90: return 0.95
    elif 90 <= rel_deg < 135: return 0.85
    elif 135 <= rel_deg <= 180: return 0.70
    return 0.0  # Should be blocked by no-go check anyway

def turn_penalty(prev_dir, new_dir):
    """Calculates turn penalty based on previous and new heading."""
    if np.isnan(prev_dir):
        return 0.0
    d = abs(prev_dir - new_dir)
    d = min(d, 360 - d)
    if d <= 60: return d * 0.05
    return 4.0

def get_move_cost(r, c, h_prev, h_new, wind_speed_map, wind_dir_map):
    """
    Calculates the cost of moving from (r, c) with new heading h_new,
    having arrived with h_prev.
    Returns (cost, new_r, new_c) or (inf, -1, -1) if invalid.
    """
    # 1. Get wind at (r, c) [convert from 1-based to 0-based]
    try:
        w_speed = wind_speed_map[r-1, c-1]
        w_to = wind_dir_map[r-1, c-1]
        w_from = (w_to + 180) % 360
    except IndexError:
        return (np.inf, -1, -1)  # Stay on map

    # 2. Check no-go angle
    raw = abs(h_new - w_from)
    rel = min(raw, 360 - raw)
    if rel < NO_GO_ANGLE:
        return (np.inf, -1, -1)  # Invalid move

    # 3. Get new position (dr, dc)
    if h_new == 0:   dr, dc = -1,  0
    elif h_new == 60:  dr, dc = -1,  1
    elif h_new == 120: dr, dc =  1,  1
    elif h_new == 180: dr, dc =  1,  0
    elif h_new == 240: dr, dc =  1, -1
    elif h_new == 300: dr, dc = -1, -1
    else:
        return (np.inf, -1, -1)  # Invalid heading

    new_r, new_c = r + dr, c + dc

    # 4. Check map bounds
    nrows, ncols = wind_speed_map.shape
    if not (1 <= new_r <= nrows and 1 <= new_c <= ncols):
        return (np.inf, -1, -1)

    # 5. Calculate speed and time
    f = polar_factor(rel)
    if f <= 0:
        return (np.inf, -1, -1)  # No speed

    boat_speed = f * w_speed
    if boat_speed <= 0:
        return (np.inf, -1, -1)  # No movement possible

    penalty = turn_penalty(h_prev, h_new)
    move_time = (BASE_TIME / boat_speed) + penalty

    return (move_time, new_r, new_c)

def build_isochrome_map(wind_speed_map, wind_dir_map, start_pos):
    """
    Runs Dijkstra's algorithm to build an isochrone map from start_pos.
    Returns:
    - min_times (dict): {(r, c, h): time}
    - isochrone_grid (np.array): 30x30 grid of min times
    """

    pq = []
    min_times = {}
    start_r, start_c = start_pos
    start_state = (start_r, start_c, np.nan)  # (r, c, heading)
    heapq.heappush(pq, (0.0, start_r, start_c, np.nan))
    min_times[start_state] = 0.0

    print("Starting Dijkstra's algorithm for isochrone map...")

    while pq:
        # Pop the state with the lowest time
        (g, r, c, h_prev) = heapq.heappop(pq)

        # Skip if we have already found a better path
        current_state = (r, c, h_prev)
        if g > min_times.get(current_state, np.inf):
            continue

        # Explore all possible new headings
        for h_new in HEADINGS:
            (cost, new_r, new_c) = get_move_cost(
                r, c, h_prev, h_new, wind_speed_map, wind_dir_map
            )

            if np.isinf(cost):
                continue  # Invalid move

            # New state to reach
            new_state = (new_r, new_c, h_new)
            # New total time to reach new state
            new_g = g + cost

            # If this path to new_state is better, record it and push to queue
            if new_g < min_times.get(new_state, np.inf):
                min_times[new_state] = new_g
                heapq.heappush(pq, (new_g, new_r, new_c, h_new))
    
    print("Dijkstra's algorithm completed.")

    nrows, ncols = wind_speed_map.shape
    isochrone_grid = np.full((nrows, ncols), np.inf)

    for (r, c, h), time in min_times.items():
        isochrone_grid[r-1, c-1] = min(isochrone_grid[r-1, c-1], time)
    
    return min_times, isochrone_grid

def main():
    try:        
        # Load map data (Change these filenames for different maps)
        wind_speed = np.loadtxt("map_3_Tiebreaker_windSpeed.csv", delimiter=",")
        wind_dir = np.loadtxt("map_3_Tiebreaker_windDir.csv", delimiter=",")

        with open("map_3_Tiebreaker_meta.json", "r") as f:
            meta = json.load(f)

        start_pos = tuple(meta["startPos"])
        finish_pos = tuple(meta["finishPos"])

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    all_state_times, isochrone_grid = build_isochrome_map(
        wind_speed, wind_dir, start_pos
    )

    fr, fc = finish_pos
    finish_time = isochrone_grid[fr-1, fc-1]

    if np.isinf(finish_time):
        print("No valid path to finish position.")
    else:
        print(f"Minimum time to finish position {finish_pos}: {finish_time:.2f} units.")
    
    output_file = "isochrone_map_3.txt"
    np.savetxt(output_file, isochrone_grid, fmt="%.4f", delimiter=",")
    print(f"Isochrone map saved to {output_file}.")

if __name__ == "__main__":
    main()
