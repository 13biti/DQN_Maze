## Project Source

# 🧩 Maze Solver with Deep Q-Network (DQN)

A reinforcement learning agent trained to solve a maze using Deep Q-Networks. This project demonstrates how an agent learns to navigate environments through trial and error and neural Q-value estimation.

---

## 📦 Project Structure

- `DQN_Agent.py`:  
  The core submodule implementing the DQN agent.

  - Originally written within this repository.
  - Later refactored into a standalone submodule named `agent`.
  - The majority of commit history reflects updates, bug fixes, and improvements to this file.
  - Check commit messages for a timeline of development, experiments, and agent evolution.

- `MDP_maze.py`:  
  Custom maze environment used for training and testing the agent.

- `train.py`:  
  Training pipeline, including hyperparameters, environment interaction, and performance tracking.

- `test.py`:  
  Evaluation script to test the trained agent’s generalization and maze-solving skills.

---

## 📋 Features

- Deep Q-Network implementation with:
  - Experience replay buffer
  - Epsilon-greedy exploration strategy
- Modular design for easy experimentation

---

```

```
