import time
from MDP_maze import Maze
import random


def main():
    maze_size = 10  # Must be an odd number for proper path generation
    env = Maze(
        game_size=maze_size, max_steps=100, extra_paths=10, fixed_maze_episodes=0
    )

    state = env.reset()
    done = False
    total_reward = 0

    print("Initial Maze:")
    env.render()
    time.sleep(1)

    while not done:
        valid_actions = env.get_actions(only_valid=True)
        action = random.choice(valid_actions)
        next_state, reward, done, info = env.step(action)
        total_reward += reward

        env.render()
        print(
            f"Step: {env.current_step}, Action: {env.get_action_name(action)}, Reward: {reward:.2f}"
        )
        print(f"Player Position: {env.player_location}")
        time.sleep(0.2)  # Slow down the output so it's easier to watch

    print("Episode finished!")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Reached goal: {env.player_location == env.goal_location}")


if __name__ == "__main__":
    main()
