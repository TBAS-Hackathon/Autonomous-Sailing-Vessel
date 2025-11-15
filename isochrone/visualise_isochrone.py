import numpy as np
import json
import heapq
import math
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

BASE_TIME = 10.0
NO_GO_ANGLE = 30.0
HEADINGS = [0, 60, 120, 180, 240, 300]
NAN_HEADING = -1

def polar_factor(rel_deg):
    """Returns polar speed factor based on relative wind angle."""
    if 30 <= rel_deg < 60: return 1.0
    elif 60 <= rel_deg < 90: return 0.95
    elif 90 <= rel_deg < 135: return 0.85
    elif 135 <= rel_deg <= 180: return 0.70
    return 0.0  # Should be blocked by no-go check anyway

def turn_penalty(prev_dir, new_dir):
    """Calculates turn penalty based on previous and new heading."""
    if prev_dir == NAN_HEADING:
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

    tp = turn_penalty(h_prev, h_new)
    move_time = (BASE_TIME / boat_speed) + tp

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
    predecessors = {}
    start_r, start_c = start_pos
    start_state = (start_r, start_c, NAN_HEADING)  # (r, c, heading)

    heapq.heappush(pq, (0.0, start_r, start_c, NAN_HEADING))
    min_times[start_state] = 0.0
    predecessors[start_state] = None

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
                predecessors[new_state] = current_state
                heapq.heappush(pq, (new_g, new_r, new_c, h_new))
    
    print("Dijkstra's algorithm completed.")

    nrows, ncols = wind_speed_map.shape
    isochrone_grid = np.full((nrows, ncols), np.inf)

    for (r, c, h), time in min_times.items():
        isochrone_grid[r-1, c-1] = min(isochrone_grid[r-1, c-1], time)
    
    return min_times, isochrone_grid, predecessors

def reconstruct_path(min_times, predecessors, finish_pos):
    """
    Reconstructs the optimal path from start_pos to finish_pos
    using the predecessors dictionary.
    Returns a list of (heading, r, c, time) tuples.
    """
    fr, fc = finish_pos
    # Find the best heading at finish position
    best_time = np.inf
    best_finish_state = None

    for h in HEADINGS:
        state = (fr, fc, h)
        time = min_times.get(state, np.inf)
        if time < best_time:
            best_time = time
            best_finish_state = state

    if best_finish_state is None:
        return None, np.inf  # No path found

    path = []
    curr = best_finish_state
    while curr is not None:
        path.append(curr)
        curr = predecessors.get(curr)

    return list(reversed(path)), best_time

def make_quiver_components(windDir_deg):
    """
    U = sin(theta_deg)
    V = -cos(theta_deg)
    """
    rad = np.deg2rad(windDir_deg)
    U = np.sin(rad)  # x-component
    V = -np.cos(rad)  # y-component
    return U, V

def plot_isochrone_map(isochrone_grid, wind_dir_map, start_pos, finish_pos, path):
    nrows, ncols = isochrone_grid.shape
    
    plot_grid = isochrone_grid.copy()
    plot_grid[np.isinf(plot_grid)] = np.nan  # For better color mapping

    fig, ax = plt.subplots(figsize=(12, 10), dpi=100)
    fig.patch.set_facecolor('#2B2B2B')
    ax.set_facecolor('#1E1E1E')

    cmap = plt.cm.plasma
    cmap.set_bad(color='#333333')  # Color for NaN values

    vmax = np.nanmax(plot_grid)
    im = ax.imshow(
        plot_grid,
        cmap=cmap,
        origin='upper',
        interpolation='nearest',
        vmax=vmax * 1.1,
        zorder=1
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Minimum Time to Reach (units)', color='white', fontsize=12)
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    Y, X = np.meshgrid(np.arange(nrows), np.arange(ncols), indexing='ij')
    U, V = make_quiver_components(wind_dir_map)

    stride = 2
    ax.quiver(
        X[::stride, ::stride],
        Y[::stride, ::stride],
        U[::stride, ::stride],
        V[::stride, ::stride],
        color='white',
        alpha=0.4,
        scale=50,
        width=0.002,
        zorder=2
    )

    sr, sc = start_pos
    fr, fc = finish_pos
    ax.plot(sc-1, sr-1, 'o', markersize=12, markerfacecolor='#00FF00', markeredgecolor='white', label='Start (30, 1)', zorder=10)
    ax.plot(fc-1, fr-1, 'X', markersize=14, markerfacecolor='#FF0000', markeredgecolor='white', label=f'Finish ({fr}, {fc})', zorder=10)

    if path:
        print(f"\n--- DEBUG: Plotting path with {len(path)} points. ---")
        path_c = [s[1]-1 for s in path]
        path_r = [s[0]-1 for s in path]
        print(f"    First point (x,y): ({path_c[0]}, {path_r[0]})")
        print(f"    Last point (x,y): ({path_c[-1]}, {path_r[-1]})")
        
        ax.plot(path_c, path_r, 'o-', color='#FFFF00', lw=2, 
                markersize=4, alpha=0.8, label='Optimal Path', 
                zorder=9) # Plot path on top
    
    ax.set_title('Isochrone Map with Wind Directions and Optimal Path', color='white', fontsize=16)
    ax.set_xlabel('Column (X)', color='white', fontsize=12)
    ax.set_ylabel('Row (Y)', color='white', fontsize=12)

    ax.set_xticks(np.arange(0, ncols, 5))
    ax.set_yticks(np.arange(0, nrows, 5))
    ax.set_xticklabels(np.arange(1, ncols+1, 5))
    ax.set_yticklabels(np.arange(1, nrows+1, 5))
    ax.tick_params(colors='white')

    ax.set_xlim(-0.5, ncols-0.5)
    ax.set_ylim(nrows-0.5, -0.5)

    ax.grid(color='white', alpha=0.1)
    ax.legend(facecolor='#444444', labelcolor='white', loc='upper left')
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()

def main():
    try:        
        # Load map data (change filenames for different maps)
        wind_speed = np.loadtxt("./prompts/map_100_windSpeed.csv", delimiter=",")
        wind_dir = np.loadtxt("./prompts/map_100_windDir.csv", delimiter=",")

        with open("./prompts/map_100_meta.json", "r") as f:
            meta = json.load(f)

        start_pos = tuple(meta["startPos"])
        finish_pos = tuple(meta["finishPos"])

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    min_times, isochrone_grid, predecessors = build_isochrome_map(
        wind_speed, wind_dir, start_pos
    )

    path, best_time = reconstruct_path(min_times, predecessors, finish_pos)
    
    if path:
        print("Optimal path from start to finish:")
    else:
        print("No valid path found from start to finish.")
    
    plot_isochrone_map(isochrone_grid, wind_dir, start_pos, finish_pos, path)

if __name__ == "__main__":
    main()
