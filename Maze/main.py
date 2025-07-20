import os
import numpy as np
from MDP_maze import Maze
from agent import General_DQN_Agent
import matplotlib.pyplot as plt


def main():
    # Environment parameters
    game_size = 8  # Changed to even number for consistent maze generation
    max_steps = game_size * 20
    extra_paths = 10
    fixed_maze_episodes = 50

    # Agent parameters
    state_size = 2
    action_size = 4
    learning_rate = 0.001
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    batch_size = 32
    buffer_size = 2000

    # Initialize environment and agent
    env = Maze(
        game_size=game_size,
        max_steps=max_steps,
        extra_paths=extra_paths,
        fixed_maze_episodes=fixed_maze_episodes,
    )
    agent = General_DQN_Agent(
        action_size=action_size,
        state_size=state_size,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        batch_size=batch_size,
        buffer_size=buffer_size,
    )

    num_episodes = 1000
    max_steps_per_episode = max_steps
    render_frequency = 10

    rewards = []
    epsilons = []
    losses = []
    action_counts = np.zeros(action_size)
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        step = 0
        episode_action_counts = np.zeros(action_size)  # Per-episode action counts

        while not done and step < max_steps_per_episode:
            action = agent.compute_action(state)
            next_state, reward, done, info = env.step(action)
            agent.store_experience(state, next_state, reward, action, done)
            loss = (
                agent.train()
            )  # Assuming train returns loss (modify agent.py if needed)
            state = next_state
            total_reward += reward
            step += 1
            episode_action_counts[action] += 1  # Count actions per episode

            env.render()
            print(
                f"Step {step}: State={state[0]}, Action={env.get_action_name(action)}, Reward={reward:.2f}"
            )

        rewards.append(total_reward)
        epsilons.append(agent.epsilon)
        losses.append(loss if loss is not None else 0)  # Append loss
        action_counts += episode_action_counts  # Aggregate action counts
        print(
            f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}, Loss: {losses[-1]:.4f}"
        )

    # Save the trained model
    agent.model.save("dqn_maze_model.h5")
    print("Model saved to 'dqn_maze_model.h5'")

    # Plot training progress
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(range(1, num_episodes + 1), rewards, label="Total Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Progress: Total Reward per Episode")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(range(1, num_episodes + 1), epsilons, label="Epsilon", color="orange")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title("Epsilon Decay over Episodes")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(range(1, num_episodes + 1), losses, label="Loss", color="green")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Training Loss over Episodes")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_progress.png")
    plt.close()  # Close to avoid display issues in non-GUI environments
    print("Training progress plot saved to 'training_progress.png'")

    # Plot action distribution
    plt.figure(figsize=(8, 5))
    actions = ["Up", "Down", "Left", "Right"]
    plt.bar(actions, action_counts / np.sum(action_counts) * 100)
    plt.xlabel("Actions")
    plt.ylabel("Percentage (%)")
    plt.title("Action Distribution Over Training")
    plt.savefig("action_distribution.png")
    plt.close()
    print("Action distribution plot saved to 'action_distribution.png'")


if __name__ == "__main__":
    main()
