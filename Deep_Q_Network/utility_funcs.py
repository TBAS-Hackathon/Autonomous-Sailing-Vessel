import torch

def test_agent(env, network, num_episodes_to_test=10):
    """
    Evaluates the trained agent and returns the path of the BEST episode.
    """
    print("\n" + "="*30)
    print(f"--- STARTING EVALUATION ---")
    print(f"Running {num_episodes_to_test} test episodes...")
    print("="*30)

    network.eval()

    # Keep track of the best run
    best_reward = -float('inf')
    best_path = []
    best_steps = 0

    for episode in range(num_episodes_to_test):
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        path = [(env.current_row, env.current_col)]

        with torch.no_grad():
            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = network(state_tensor)
                action_idx = torch.argmax(q_values).item()

                next_state, reward, done, info = env.step(action_idx)

                path.append((env.current_row, env.current_col))
                episode_reward += reward
                episode_steps += 1
                state = next_state

                if episode_steps >= env.max_steps:
                    break

        print(f"Test Episode {episode+1}: Reward={episode_reward:.2f}, Steps={episode_steps}")

        # Check if this episode is the new best
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_path = path
            best_steps = episode_steps

    print("="*30)
    print(f"--- EVALUATION COMPLETE ---")
    print(f"Best Reward Found: {best_reward:.2f} in {best_steps} steps.")
    print("="*30)

    return best_path, best_reward, best_steps


def extract_moves(env, network, output_file, num_attempts=10):
    """
    Runs the trained DQN agent on the environment and extracts moves to a file.
    Compatible with MATLAB sailingGame.m "Load Moves (auto)" feature.

    Args:
        env: SailingEnv instance
        network: Trained QNetwork
        output_file: Path to save moves (e.g., "moves_training.txt")
        num_attempts: Number of episodes to try (saves the best one)

    Returns:
        Dictionary with run statistics
    """
    print("\n" + "="*50)
    print(f"EXTRACTING MOVES TO: {output_file}")
    print("="*50)

    network.eval()

    best_result = {
        'moves': [],
        'path': [],
        'time': float('inf'),
        'reward': -float('inf'),
        'finished': False,
        'steps': 0
    }

    for attempt in range(num_attempts):
        state = env.reset()
        moves = []
        path = [(env.current_row + 1, env.current_col + 1)]  # Convert to 1-indexed
        done = False
        episode_reward = 0
        steps = 0

        with torch.no_grad():
            while not done and steps < env.max_steps:
                # Get action from network
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = network(state_tensor)
                action_idx = torch.argmax(q_values).item()

                # Convert action index to heading
                heading = env.actions[action_idx]

                # Take step
                next_state, reward, done, info = env.step(action_idx)

                # Only record move if it was valid (position changed)
                new_pos = (env.current_row + 1, env.current_col + 1)
                if new_pos != path[-1]:
                    moves.append(heading)
                    path.append(new_pos)

                episode_reward += reward
                steps += 1
                state = next_state

        finished = (env.current_row, env.current_col) == env.finish_pos

        print(f"Attempt {attempt+1}/{num_attempts}: "
              f"Steps={len(moves)}, Time={env.current_time:.2f}, "
              f"Reward={episode_reward:.2f}, Finished={finished}")

        # Update best if this run is better
        # Priority: 1) Finished, 2) Lower time, 3) Higher reward
        is_better = False
        if finished and not best_result['finished']:
            is_better = True
        elif finished and best_result['finished'] and env.current_time < best_result['time']:
            is_better = True
        elif not finished and not best_result['finished'] and episode_reward > best_result['reward']:
            is_better = True

        if is_better:
            best_result = {
                'moves': moves.copy(),
                'path': path.copy(),
                'time': env.current_time,
                'reward': episode_reward,
                'finished': finished,
                'steps': len(moves)
            }

    # Save moves to file
    with open(output_file, 'w') as f:
        # Write header with metadata
        f.write(f"# DQN Moves for {env.__class__.__name__}\n")
        f.write(f"# Start: {env.start_pos[0]+1}, {env.start_pos[1]+1}\n")
        f.write(f"# Finish: {env.finish_pos[0]+1}, {env.finish_pos[1]+1}\n")
        f.write(f"# Finished: {best_result['finished']}\n")
        f.write(f"# Time: {best_result['time']:.2f}\n")
        f.write(f"# Reward: {best_result['reward']:.2f}\n")
        f.write(f"# Steps: {best_result['steps']}\n")
        f.write(f"# Path (1-indexed): {' -> '.join([f'({r},{c})' for r, c in best_result['path']])}\n")
        f.write("#\n")
        f.write("# Format: One move per line (0, 60, 120, 180, 240, 300)\n")
        f.write("#\n")

        # Write moves (one per line)
        for move in best_result['moves']:
            f.write(f"{move}\n")

    print("\n" + "="*50)
    print("EXTRACTION COMPLETE!")
    print("="*50)
    print(f"File saved: {output_file}")
    print(f"Moves: {len(best_result['moves'])}")
    print(f"Finished: {best_result['finished']}")
    print(f"Time: {best_result['time']:.2f}")
    print(f"Reward: {best_result['reward']:.2f}")
    if best_result['finished']:
        print(f"✓ Agent reached the finish!")
    else:
        final_pos = best_result['path'][-1]
        print(f"⚠ Agent stopped at position {final_pos}")
    print("="*50 + "\n")

    return best_result
