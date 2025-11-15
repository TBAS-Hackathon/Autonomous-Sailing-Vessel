#!/usr/bin/env python3
"""
PSO-then-GA pathfinder for 30x30 wind grid.

This script reads the same CSV/JSON files used by your A* implementation
(map_3_Tiebreaker_windSpeed.csv, map_3_Tiebreaker_windDir.csv, map_3_Tiebreaker_meta.json),
re-uses the polar_factor / turn_penalty / get_move_cost logic, and implements
A hybrid optimizer:
 1) PSO on continuous waypoint positions
 2) GA seeded from the best PSO particles

Output: best path printed and written to `pso_ga_best_moves.txt` as headings one-per-line.

Tune parameters in the `if __name__ == '__main__'` block.
"""

import numpy as np
import json
import math
import random
from copy import deepcopy

# --------------------------- Map and cost utilities ---------------------------
# Map Loading (change filenames for different maps)
wind_speed = np.loadtxt("./prompts/map_3_Tiebreaker_windSpeed.csv", delimiter=",")
wind_dir = np.loadtxt("./prompts/map_3_Tiebreaker_windDir.csv", delimiter=",")
with open("./prompts/map_3_Tiebreaker_meta.json", "r") as f:
    meta = json.load(f)

start_pos = tuple(meta["startPos"])  # (r, c) in 1-based coordinates
finish_pos = tuple(meta["finishPos"])  # (r, c)
headings = [0, 60, 120, 180, 240, 300]
base_time = 10.0
no_go_angle = 30.0

nrows, ncols = wind_speed.shape


def polar_factor(rel_deg):
    if 30 <= rel_deg < 60: return 1.0
    if 60 <= rel_deg < 90: return 0.95
    if 90 <= rel_deg < 135: return 0.85
    if 135 <= rel_deg <= 180: return 0.70
    return 0.0


def turn_penalty(prev_dir, new_dir):
    if np.isnan(prev_dir):
        return 0.0
    d = abs(prev_dir - new_dir)
    d = min(d, 360 - d)
    if d <= 60: return d * 0.05
    return 4.0


def get_move_cost(r, c, h_prev, h_new):
    """
    Same semantics as your A* get_move_cost: r,c are 1-based.
    Returns (move_time, new_r, new_c) or (np.inf, -1, -1)
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
    if not (1 <= new_r <= nrows and 1 <= new_c <= ncols):
        return (np.inf, -1, -1)

    f = polar_factor(rel)
    if f <= 0:
        return (np.inf, -1, -1)

    boat_speed = w_speed * f
    # protect against zero speed
    if boat_speed <= 1e-8:
        return (np.inf, -1, -1)

    tp = turn_penalty(h_prev, h_new)
    move_time = (base_time / boat_speed) + tp
    return (move_time, new_r, new_c)

# --------------------------- Decoder: waypoints -> discrete path ---------------------------

def decode_waypoints(waypoints, max_moves=1000, verbose=False):
    """
    waypoints: list of (r,c) floats (1-based coordinates inside grid)
    Decoding strategy (greedy steering):
      - Start at start_pos with prev heading NaN
      - For each waypoint target in order: repeatedly choose among the 6 heading moves
        the one that is valid (get_move_cost != inf) and minimizes distance to the waypoint.
      - When within 0.5 cell of waypoint, switch to next waypoint.
      - Stop when goal reached or max_moves exceeded.

    Returns: path_list, total_time, reached_goal
      path_list entries: (heading, r, c, cumulative_time)
    """
    r, c = start_pos
    h_prev = np.nan
    path = []
    total_time = 0.0

    waypoint_idx = 0
    moves = 0

    def dist(a, b):
        return math.hypot(a[0]-b[0], a[1]-b[1])

    while moves < max_moves:
        # If we've reached finish, stop
        if (r, c) == finish_pos:
            return path, total_time, True

        # If we have no remaining waypoints, target the finish directly
        if waypoint_idx >= len(waypoints):
            target = finish_pos
        else:
            target = waypoints[waypoint_idx]

        # If close enough to waypoint, advance to next
        if dist((r, c), target) <= 0.6:
            waypoint_idx += 1
            continue

        # Choose best heading for approaching the target
        best = None
        best_score = float('inf')
        best_choice = None
        for h in headings:
            cost, nr, nc = get_move_cost(r, c, h_prev, h)
            if np.isinf(cost):
                continue
            score = math.hypot(nr - target[0], nc - target[1])
            # tie-breaker: prefer lower move_time
            if score < best_score - 1e-9 or (abs(score - best_score) < 1e-9 and cost < best[0]):
                best = (cost, nr, nc, h)
                best_score = score
                best_choice = h

        if best is None:
            # no valid move -> dead end
            return path, total_time + 1e6 + dist((r,c), finish_pos)*100.0, False

        cost, r, c, chosen_h = best
        total_time += cost
        moves += 1
        path.append((chosen_h, r, c, total_time))
        h_prev = chosen_h

    # exhausted moves
    return path, total_time + 1e5, False

# --------------------------- Fitness -----------------------------------------

def fitness_from_waypoints(waypoints, penalty_for_not_reaching=1e5):
    """
    Lower fitness is better (total time). If decoder fails to reach goal, returns large penalty + remaining distance.
    waypoints: flat list or array length 2*K -> [(r1,c1),...]
    """
    # convert flat to list of tuples if necessary
    pts = [(float(waypoints[i]), float(waypoints[i+1])) for i in range(0, len(waypoints), 2)]
    path, tot_time, reached = decode_waypoints(pts)
    if reached:
        return tot_time, path
    else:
        # add final distance penalty
        last_pos = (path[-1][1], path[-1][2]) if path else start_pos
        rem = math.hypot(last_pos[0] - finish_pos[0], last_pos[1] - finish_pos[1])
        return tot_time + penalty_for_not_reaching + rem*100.0, path

# --------------------------- PSO implementation -------------------------------

def run_pso(K=4, pop_size=40, iters=200, inertia=0.7, c1=1.5, c2=1.5):
    """Runs PSO on K waypoints (each waypoint is 2 dims). Returns list of best particles (positions) and their fitnesses."""
    dim = 2 * K
    # bounds: row in [1, nrows], col in [1, ncols]
    lower = np.array([1]*(dim))
    upper = np.array([nrows if i%2==0 else ncols for i in range(dim)])

    # Initialize particles: random waypoints biased towards straight-line between start and finish
    start = np.array(start_pos)
    finish = np.array(finish_pos)
    particles = []
    for i in range(pop_size):
        pos = np.zeros(dim)
        for k in range(K):
            t = (k+1)/(K+1)
            # linear interpolation + gaussian jitter
            interp = start*(1-t) + finish*t
            pos[2*k] = np.clip(interp[0] + np.random.randn()*3.0, 1, nrows)
            pos[2*k+1] = np.clip(interp[1] + np.random.randn()*3.0, 1, ncols)
        vel = np.random.randn(dim) * 0.5
        particles.append({'pos': pos, 'vel': vel})

    pbest_pos = [p['pos'].copy() for p in particles]
    pbest_val = [float('inf')]*pop_size

    gbest_pos = None
    gbest_val = float('inf')

    # Evaluate initial
    for i, p in enumerate(particles):
        val, _ = fitness_from_waypoints(p['pos'])
        pbest_val[i] = val
        if val < gbest_val:
            gbest_val = val
            gbest_pos = p['pos'].copy()

    # Main loop
    for it in range(iters):
        for i, p in enumerate(particles):
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)
            p['vel'] = inertia*p['vel'] + c1*r1*(pbest_pos[i]-p['pos']) + c2*r2*(gbest_pos-p['pos'])
            # clamp velocity
            vmax = (upper - lower) * 0.2
            p['vel'] = np.clip(p['vel'], -vmax, vmax)
            p['pos'] = p['pos'] + p['vel']
            # clamp position
            p['pos'] = np.clip(p['pos'], lower, upper)

            val, _ = fitness_from_waypoints(p['pos'])
            if val < pbest_val[i]:
                pbest_val[i] = val
                pbest_pos[i] = p['pos'].copy()
                if val < gbest_val:
                    gbest_val = val
                    gbest_pos = p['pos'].copy()
        # optional: print progress
        if (it+1) % max(1, iters//10) == 0:
            print(f"PSO iter {it+1}/{iters} best={gbest_val:.2f}")

    # Return sorted particles by pbest_val
    idx = np.argsort(pbest_val)
    ranked = [{'pos': pbest_pos[i], 'val': pbest_val[i]} for i in idx]
    return ranked, gbest_pos, gbest_val

# --------------------------- GA implementation -------------------------------

def run_ga(initial_population, fitness_func, pop_size=60, generations=200, crossover_p=0.9, mutation_p=0.2, mutation_scale=1.0, elite_fraction=0.1):
    """
    initial_population: list of flat arrays (positions) used to seed GA (will be resized to pop_size)
    fitness_func: function returning (fitness, path)
    """
    dim = initial_population[0].shape[0]
    # build initial pop
    pop = [ind.copy() for ind in initial_population]
    # if needed, fill with jittered copies
    while len(pop) < pop_size:
        base = random.choice(initial_population).copy()
        jitter = np.random.randn(dim) * 1.5
        new = np.clip(base + jitter, 1, np.array([nrows if i%2==0 else ncols for i in range(dim)]))
        pop.append(new)

    # Evaluate
    fitnesses = []
    paths = []
    for ind in pop:
        val, path = fitness_func(ind)
        fitnesses.append(val)
        paths.append(path)

    for gen in range(generations):
        # Elitism
        elite_count = max(1, int(elite_fraction * pop_size))
        idx = np.argsort(fitnesses)
        new_pop = [pop[i].copy() for i in idx[:elite_count]]

        # Generate rest
        while len(new_pop) < pop_size:
            # tournament selection
            def tournament_select():
                a,b = random.randrange(pop_size), random.randrange(pop_size)
                return pop[a] if fitnesses[a] < fitnesses[b] else pop[b]
            p1 = tournament_select()
            p2 = tournament_select()
            if random.random() < crossover_p:
                # arithmetic crossover (blend)
                alpha = random.random()
                child = alpha*p1 + (1-alpha)*p2
            else:
                child = p1.copy()
            # mutation: gaussian noise
            if random.random() < mutation_p:
                child = child + np.random.randn(dim) * mutation_scale
            # clip
            child = np.clip(child, 1, np.array([nrows if i%2==0 else ncols for i in range(dim)]))
            new_pop.append(child)

        # evaluate new_pop
        pop = new_pop
        fitnesses = []
        paths = []
        for ind in pop:
            val, path = fitness_func(ind)
            fitnesses.append(val)
            paths.append(path)

        if (gen+1) % max(1, generations//10) == 0:
            best = np.min(fitnesses)
            print(f"GA gen {gen+1}/{generations} best={best:.2f}")

    # return best individual and its path
    best_idx = int(np.argmin(fitnesses))
    return pop[best_idx], fitnesses[best_idx], paths[best_idx]

# --------------------------- Utilities --------------------------------------

def waypoints_from_flat(flat):
    return [(flat[i], flat[i+1]) for i in range(0, len(flat), 2)]

# --------------------------- Putting it together ---------------------------

def pso_then_ga(K=4, pso_pop=40, pso_iters=200, ga_pop=80, ga_gens=200):
    print("Running PSO phase...")
    ranked, gbest_pos, gbest_val = run_pso(K=K, pop_size=pso_pop, iters=pso_iters)
    print(f"PSO done. Best={gbest_val:.2f}")

    # Seed GA initial population with top PSO particles
    top = [r['pos'] for r in ranked[:max(5, pso_pop//5)]]
    # ensure they are numpy arrays
    initial_pop = [np.array(x) for x in top]

    # Fill initial_pop to at least ga_pop/4 distinct seeds by jittering
    while len(initial_pop) < max(4, ga_pop//4):
        base = random.choice(initial_pop)
        jitter = np.random.randn(base.shape[0]) * 1.0
        candidate = np.clip(base + jitter, 1, np.array([nrows if i%2==0 else ncols for i in range(base.shape[0])]))
        initial_pop.append(candidate)

    print("Running GA phase...")
    best_ind, best_val, best_path = run_ga(initial_pop, fitness_from_waypoints,
                                           pop_size=ga_pop, generations=ga_gens,
                                           crossover_p=0.9, mutation_p=0.25, mutation_scale=1.2,
                                           elite_fraction=0.12)
    print(f"GA done. Best={best_val:.2f}")

    # decode best into headings list
    _, reached = fitness_from_waypoints(best_ind)
    best_waypoints = waypoints_from_flat(best_ind)
    decoded_path, total_time, reached = decode_waypoints(best_waypoints)

    # Write out move headings like your A* output
    with open("pso_ga_best_moves_3.txt", "w") as f:
        for move in decoded_path:
            f.write(f"{move[0]}\n")

    print(f"Final best time: {total_time:.2f} Reached goal: {reached}")
    print(f"Wrote headings to pso_ga_best_moves_n.txt (one heading per line)")
    return decoded_path, total_time, reached

# --------------------------- Run as script ----------------------------------

if __name__ == '__main__':
    # Parameters you can tune
    K = 5                # number of waypoints
    PSO_POP = 60
    PSO_ITERS = 250
    GA_POP = 120
    GA_GENS = 300

    best_path, best_time, reached = pso_then_ga(K=K, pso_pop=PSO_POP, pso_iters=PSO_ITERS, ga_pop=GA_POP, ga_gens=GA_GENS)

    print('\n--- Best path summary (first 50 moves) ---')
    for i, move in enumerate(best_path[:50]):
        print(f"Move {i+1}: {move[0]}° -> ({move[1]}, {move[2]}) Time={move[3]:.2f}")

    if not reached:
        print("Note: best candidate did not reach the finish exactly. Consider increasing waypoints or tuning parameters.")

