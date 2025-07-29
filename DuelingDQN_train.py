import numpy as np
from DQN.DQN_Agent import (
    EpsilonPolicyType,
    DuelingDQNAgent,
    EpsilonPolicy,
    UpdateTargetNetworkType,
    RewardPolicyType,
)
from MDP_maze import maze
import time
import pickle
import os


def main():
    game_size = 8
    max_frame = 0
    extra_paths = 10
    episodes = 50
    max_steps = 100
    render = False

    game = maze(game_size=game_size, max_frame=max_frame, extra_paths=extra_paths)

    state_size = 2
    epsilon_min = 0.1
    epsilon_decay = 0.95
    action_size = len(game.get_actions())
    ep_policy = EpsilonPolicy(
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        policy=EpsilonPolicyType.DECAY,
    )
    agent = DuelingDQNAgent(
        action_size=action_size,
        state_size=state_size,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        batch_size=50,
        buffer_size=2000,
        max_episodes=100,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        epsilon_policy=ep_policy,
        reward_policy=RewardPolicyType.NONE,
        prefer_lower_heuristic=True,
        progress_bonus=0.05,
        exploration_bonus=0.1,
        update_target_network_method=UpdateTargetNetworkType.SOFT,
        target_update_frequency=5,
        reward_range=(-10, 20),
        use_normalization=False,
        update_factor=0.8,
    )

    game.print_maze(game.generate_visual_pattern())
    success_count = 0
    for episode in range(episodes):
        state = game.reset(hard_reset=False)
        total_reward = 0
        steps = 0
        done = False

        while not done and steps < max_steps:
            action = agent.select_action(state)
            is_success, info, reward, huristic, next_state, done = game.act(action)
            agent.buffer_helper.store_experience(
                state, next_state, reward, action, done, huristic
            )
            loss = agent.train(episode)
            state = next_state
            total_reward += reward
            steps += 1
            if render:
                game.print_maze(game.generate_visual_pattern())
                time.sleep(0.1)

        if done:
            success_count += 1
        print(
            f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward:.2f}, "
            f"Steps: {steps}, Epsilon: {agent.epsilon:.3f}, "
            f"Loss: {loss}, Success Rate: {success_count / (episode + 1):.2%}, "
            f"{'Goal Reached' if done else 'Not Reached'}"
        )


if __name__ == "__main__":
    main()
