import numpy as np
from agent import General_DQN_Agent
from MDP_maze import maze
import time


def main():
    game_size = 4
    max_frame = 0
    extra_paths = 10
    episodes = 1000
    max_steps = 100
    render = False

    game = maze(game_size=game_size, max_frame=max_frame, extra_paths=extra_paths)
    state_size = 2
    action_size = len(game.get_actions())
    agent = General_DQN_Agent(
        action_size=action_size,
        state_size=state_size,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        batch_size=32,
        buffer_size=2000,
    )

    game.print_maze(game.generate_visual_pattern())
    for episode in range(episodes):
        state = game.reset(hard_reset=False)
        total_reward = 0
        steps = 0
        done = False

        while not done and steps < max_steps:
            action = agent.compute_action(state)

            is_success, info, reward, next_state, done = game.act(action)

            agent.store_experience(state, next_state, reward, action, done)

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
        is_success, info, reward, next_state, done = game.act(action)
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
