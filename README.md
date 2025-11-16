# Sailing Route Optimisation - TBAS Hackathon 2025

## Task

Simulate an autonomous sailing vessel and optimise its route to reach the checkpoint on a given map in the shortest possible time.

Built as part of the Team Bath Autonomous Sailing (TBAS) 24 Hour Hackathon 2025.

## Leaderboard (Final Times)

These are the fastest-found optimal times for each map using the specified turn_penalty function.

- Map 1 (Training): 91.05 (Found by A*, Heading-Augmented A* & RDFS, DQN)
- Map 2 (Main): 89.05 (Found by A*, Heading-Augmented A* & RDFS, DQN)
- Map 3 (Tiebreaker): 123.32 (Found by A*, Heading-Augmented A* & RDFS, DQN)

## Approaches Explored

We explored several algorithms to solve this shortest path problem, each with different trade-offs in complexity, speed, and optimality.

### 1. Heading-Augmented A* (Optimal Solver)

This was our primary and most successful solver. A* is a graph-search algorithm that finds the shortest path by balancing the actual cost from the start (g) and an estimated cost to the finish (h).

**The Problem:** A standard A* on a (row, col) grid is insufficient. The game's turnPenalty function means the cost of a move (an "edge") depends on the previous move.

**The Solution:** We augmented the state to (row, col, heading). This expands the graph size (from 900 nodes to 5,400) but accurately models the game's rules.

**Heuristic:** To be optimal, the heuristic h must never overestimate the true cost (i.e., be "admissible"). We used the Chebyshev distance (minimum number of grid hops, max(dx, dy)) multiplied by the fastest possible time for a single hop (max wind speed on map, no turn penalty). This combination is admissible and guided the search efficiently.

This approach finds the fastest possible time for all maps.

### 2. Isochrone Map (Dijkstra's Visualiser)

Instead of racing to a single goal, Dijkstra's algorithm explores outwards from the start, finding the optimal time to reach every reachable state (row, col, heading). By collapsing this 6D data into a 2D grid (by taking the minimum time to arrive at each (row, col) regardless of heading), we created an "isochrone" map.

While this algorithm also finds the optimal path (Dijkstra's is essentially A* with a heuristic of 0), its true value was in visualisation.

The resulting heatmap gave us immediate, powerful strategic insights:

- It clearly visualised "corridors" of fast-moving, favourable wind and "dead zones" of slow, unfavourable wind.
- It helped us understand why the optimal path behaves as it does, often "tacking" along these fast-wind corridors rather than taking a direct line.
- For autonomous sailing, this map is invaluable for high-level strategy and provides a perfect "reward map" to guide a Machine Learning approach.

### 3. Recursive Depth-First Search (RDFS) (Verification)

This is a "brute-force" algorithm that recursively tries every possible path from the start. To be feasible, it uses two optimisations:

- **Visited States:** Remembers the fastest time to a state (row, col, heading) and prunes any path that reaches it slower.
- **Branch and Bound:** Prunes any path whose current_time is already worse than the best_time_so_far.

While exceptionally slow, this algorithm is guaranteed to find the optimal solution. We used it as an independent verifier to prove that our Heading-Augmented A* solver was also producing the correct, optimal times.

### 4. Basic A* (Initial Attempt)

Our first attempt used a naive A* implementation with a (row, col) state and a simple Euclidean distance heuristic. This approach was fundamentally flawed as it could not account for the turnPenalty (which is state-dependent) or the grid's hexagonal movement. It produced highly sub-optimal paths but was a necessary first step.

### 5. Deep Q-Network (DQN) (Incomplete)

A reinforcement learning approach was explored to navigate the sailing environment through trial and error. The DQN implementation used experience replay and a target network architecture following the original DeepMind paper specifications.

**Implementation Details:**
- **State Representation:** 9-dimensional vector encoding position, wind conditions, goal direction, and current heading
- **Action Space:** 6 discrete headings (0°, 60°, 120°, 180°, 240°, 300°)
- **Reward Function:** Negative time cost per move, with bonuses for reaching the finish and penalties for invalid moves
- **Architecture:** 3-layer neural network (128 hidden units per layer)
- **Training:** Experience replay buffer (10,000 transitions), target network updates every 200 steps, epsilon-greedy exploration with decay

While the DQN showed promising learning behaviour and successfully completed training runs, it did not consistently match the optimal solutions found by A* and RDFS within the hackathon timeframe. The approach demonstrated the viability of reinforcement learning for autonomous sailing navigation, though further hyperparameter tuning and training time would be needed to achieve optimal performance. The visual isochrone maps proved valuable for debugging and understanding the agent's learned strategies.

### 6. PSO/GA (Incomplete)

An incomplete attempt at applying a Particle Swarm Optimisation or Genetic Algorithm to the problem.

A compact hybrid optimization script that finds sailing routes on the same wind maps used by the A* solver. Instead of searching discretely over (r,c,heading) states, this tool optimizes a small set of continuous waypoints and decodes them to a concrete heading sequence using the same sailing physics (polar curve, no-go angle, turn penalty).

## Map Data Format

### Files

```
sailing_maps.mat              -> struct array "maps"
map_<i>_<Name>.json           -> full map (name, windSpeed, windDir, startPos, finishPos)
map_<i>_<Name>_windSpeed.csv  -> numeric grid
map_<i>_<Name>_windDir.csv    -> numeric grid (degrees, 0=up/N) **TO direction**
map_<i>_<Name>_meta.json      -> name/start/finish/size
```

### Conventions

- windDir is the direction the wind BLOWS TO (same as arrows on the map).
- wind-from (for upwind calculations) is (windDir + 180) % 360.
- Angles: 0° = up (North), 90° = right (East), 180° = down (South).
- Positions: [row, col], 1-based indexing.
- No-Go Zone: 30° from the wind-from direction.

## Contributors

- Alan Gruszkiewicz
- Amy Tidd
- Jamie Andrews
- Hulusi Erin Ozyahyalar
- Marina Kewnark

