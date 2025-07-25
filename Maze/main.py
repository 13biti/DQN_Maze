import numpy as np
from agent import EpsilonPolicyType, General_DQN_Agent, EpsilonPolicy
from MDP_maze import maze
import time
import pickle
import os


def save_maze_and_weights(game, agent, maze_filename, weights_filename):
    maze_pattern = game.generate_visual_pattern()
    maze_filename += ".weights.h5"
    with open(maze_filename, "wb") as f:
        pickle.dump(maze_pattern, f)
    agent.model.save_weights(weights_filename)
    print(f"saved maze pattern to {maze_filename} and weights to {weights_filename}")


def load_maze_and_weights(game, agent, maze_filename, weights_filename):
    if not (os.path.exists(maze_filename) and os.path.exists(weights_filename)):
        print("one or both files not found. Creating new maze.")
        return False

    try:
        with open(maze_filename, "rb") as f:
            maze_pattern = pickle.load(f)
        game.generate_playGround_from_pattern(maze_pattern)
        agent.model.load_weights(weights_filename)
        print(
            f"loaded maze pattern from {maze_filename} and weights from {weights_filename}"
        )
        return True
    except Exception as e:
        print(f"error loading files: {e}")
        return False


def main():
    game_size = 4
    max_frame = 0
    extra_paths = 10
    episodes = 10
    max_steps = 100
    render = False

    game = maze(game_size=game_size, max_frame=max_frame, extra_paths=extra_paths)
    state_size = 2
    epsilon_min = 0.005
    epsilon_decay = 0.9995
    action_size = len(game.get_actions())
    ep_policy = EpsilonPolicy(
        epsilon_min,
        epsilon_decay,
        progress_bonus=0.05,
        exploration_bonus=0.1,
        policy=EpsilonPolicyType.ERM,
    )
    agent = General_DQN_Agent(
        action_size=action_size,
        state_size=state_size,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        batch_size=8,
        buffer_size=100,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        epsilon_policy=ep_policy,
    )

    game.print_maze(game.generate_visual_pattern())
    for episode in range(episodes):
        state = game.reset(hard_reset=False)
        total_reward = 0
        steps = 0
        done = False

        while not done and steps < max_steps:
            action = agent.compute_action(state)

            is_success, info, reward, huristic, next_state, done = game.act(action)

            agent.store_experience(state, next_state, reward, action, done, huristic)

            loss = agent.train()
            state = next_state
            total_reward += reward
            steps += 1

            if render:
                game.print_maze(game.generate_visual_pattern())
                time.sleep(0.1)

        print(
            f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward:.2f}, "
            f"Steps: {steps}, Epsilon: {agent.epsilon:.3f}, "
            f"loss : {loss} "
            f"{'Goal Reached' if done else 'Not Reached'}"
        )

    print("\ntesting trained agent...")
    state = game.reset(hard_reset=False)
    steps = 0
    done = False
    total_reward = 0
    game.print_maze(game.generate_visual_pattern())
    while not done and steps < max_steps:
        action = agent.compute_action(state)
        is_success, info, reward, huristic, next_state, done = game.act(action)
        state = next_state
        total_reward += reward
        steps += 1
        if render:
            game.print_maze(game.generate_visual_pattern())
            time.sleep(0.5)
    print(
        f"Test Episode: Total Reward: {total_reward:.2f}, Steps: {steps}, "
        f"{'Goal Reached' if done else 'Not Reached'}"
    )


if __name__ == "__main__":
    main()
