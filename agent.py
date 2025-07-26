from os import stat
import numpy as np
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
import random
from tensorflow.keras.layers import Dense, Input
from collections import deque
from enum import Enum


class EpsilonPolicyType(Enum):
    NONE = 0
    DECAY = 1
    SOFTLINEAR = 2


class RewardPolicyType(Enum):
    NONE = 0
    ERM = 1


INFINIT = float("inf")


class Buffering:
    def __init__(
        self,
        rewarder: "RewardHelper",
        buffer_size=2000,
        lowerHuristicBetter=True,
    ) -> None:
        self.buffer_mem = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.rewarding = rewarder
        self.lowerHuristicBetter = lowerHuristicBetter

    def store_experience(
        self, current_state, next_state, imm_reward, action, done, heuristic=0
    ):
        imm_reward = self.rewarding.findReward(
            current_state, imm_reward, heuristic, self.lowerHuristicBetter
        )
        self.buffer_mem.append(
            {
                "current_state": current_state,
                "action": action,
                "reward": imm_reward,
                "next_state": next_state,
                "heuristic": heuristic,
                "done": done,
            }
        )

    def BulkData(self, count):
        if len(self.buffer_mem) < count:
            return None
        batch = random.sample(self.buffer_mem, count)
        states = np.vstack([item["current_state"] for item in batch])
        next_states = np.vstack([item["next_state"] for item in batch])
        rewards = np.array([item["reward"] for item in batch])
        actions = np.array([item["action"] for item in batch])
        dones = np.array([item["done"] for item in batch])
        heuristics = np.array([item["heuristic"] for item in batch])
        return states, next_states, rewards, actions, dones, heuristics


class QNetwork:
    def __init__(self, state_size, action_size, learning_rate=0.001) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self._model = self._initiate_model()
        self._model.compile(loss="mse", optimizer=Adam(learning_rate=learning_rate))

    def _initiate_model(self):
        return keras.Sequential(
            [
                Input(shape=(self.state_size,)),
                Dense(units=64, activation="relu"),
                Dense(units=34, activation="relu"),
                Dense(units=self.action_size, activation="linear"),
            ]
        )

    def predict(self, states, verbose=0):
        return self._model.predict(states, verbose=verbose)

    def fit(self, states, q_targets, epochs=1, verbose=0):
        history = self._model.fit(states, q_targets, epochs=epochs, verbose=verbose)
        return history.history["loss"][0]


# we learnd that agent should balance between exploration and exploitation
# without this section , it seems that agent just try to explor new path
#
class EpsilonPolicy:
    def __init__(
        self,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        policy: EpsilonPolicyType = EpsilonPolicyType.DECAY,
        update_per_episod: bool = True,
    ):
        self.policy = policy
        self.visited_states = {}
        self.epsilon: float = INFINIT
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.update_per_episod = update_per_episod
        self.old_episod = 0

    def updateEpsilon(
        self,
        epsilon,
        episode_count,
        max_episodes=100,
    ):
        self.epsilon = epsilon
        if self.update_per_episod:
            if self.old_episod == episode_count:
                return self.epsilon
            else:
                self.old_episod = episode_count
        if self.policy == EpsilonPolicyType.DECAY:
            return self.updateEpsilon_linear(episode_count, max_episodes)
        elif self.policy == EpsilonPolicyType.SOFTLINEAR:
            return self.updateEpsilon_SoftLinear(episode_count, max_episodes)
        else:
            return self.epsilon

    def updateEpsilon_linear(self, episode_count, max_episodes):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_min)
        return self.epsilon

    def updateEpsilon_SoftLinear(self, episode_count, max_episodes=200):
        if self.epsilon > self.epsilon_min:
            target_epsilon = max(
                self.epsilon_min,
                1.0 - (1.0 - self.epsilon_min) * (episode_count / max_episodes),
            )
            self.epsilon = self.epsilon * self.epsilon_decay + target_epsilon * (
                1 - self.epsilon_decay
            )
        return self.epsilon


class RewardHelper:
    def __init__(
        self,
        progress_bonus: float = 0.05,
        exploration_bonus: float = 0.1,
        policy: RewardPolicyType = RewardPolicyType.ERM,
    ):
        self.progress_bonus = progress_bonus
        self.exploration_bonus = exploration_bonus
        self.policy = policy
        self.old_huristic = INFINIT
        self.visited_states = {}

    def findReward(self, state, reward, heuristic=None, lowerHuristicBetter=True):
        if self.policy == RewardPolicyType.ERM:
            return self._ERM(state, reward, heuristic, lowerHuristicBetter)
        elif self.policy == RewardPolicyType.NONE:
            return self._none(reward)
        else:
            return reward

    def _none(self, reward):
        return reward

    def _ERM(self, state, reward, heuristic=None, lowerHuristicBetter=True):
        state_key = tuple(state.flatten()) if state is not None else None
        is_new_state = state_key is not None and state_key not in self.visited_states
        if is_new_state:
            if is_new_state and state_key is not None:
                self.visited_states[state_key] = (
                    heuristic if heuristic is not None else INFINIT
                )

        progress = 0.0
        if heuristic is not None and state_key is not None:
            old_heuristic = self.visited_states[state_key]
            if (
                old_heuristic != INFINIT
                and (heuristic < old_heuristic and lowerHuristicBetter)
                or (heuristic > old_heuristic and not lowerHuristicBetter)
            ):
                progress = self.progress_bonus
                self.visited_states[state_key] = heuristic

        new_reward = reward
        if is_new_state:
            new_reward += self.exploration_bonus
        if progress > 0:
            new_reward += self.progress_bonus
        return new_reward


class General_DQN_Agent:
    def __init__(
        self,
        action_size,
        state_size,
        learning_rate=0.001,
        buffer_size=2000,
        batch_size=32,
        gamma=0.99,
        max_episodes=200,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        epsilon_policy: EpsilonPolicy = None,
        reward_policy: RewardPolicyType = RewardPolicyType.NONE,
        lowerHuristicBetter=True,
        progress_bonus: float = 0.05,
        exploration_bonus: float = 0.1,
    ) -> None:
        self.action_size = action_size
        self.state_size = state_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.episode_count = 0
        self.max_episodes = max_episodes

        self.model = QNetwork(state_size, action_size, learning_rate)
        self.epsilon_policy = epsilon_policy or EpsilonPolicy(
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            policy=EpsilonPolicyType.DECAY,
        )
        self.buffer_helper = Buffering(
            RewardHelper(progress_bonus, exploration_bonus, reward_policy),
            buffer_size=buffer_size,
            lowerHuristicBetter=lowerHuristicBetter,
        )

    def train(self, episode):
        data = self.buffer_helper.BulkData(self.batch_size)
        if data is None:
            return None
        states, next_states, rewards, actions, dones, heuristics = data
        q_current = self.model.predict(states, verbose=0)
        q_next = self.model.predict(next_states, verbose=0)
        q_targets = q_current.copy()
        for i in range(self.batch_size):
            if not dones[i]:
                q_targets[i, actions[i]] = rewards[i] + self.gamma * np.max(q_next[i])
            else:
                q_targets[i, actions[i]] = rewards[i]
            q_targets[i, actions[i]] = np.clip(q_targets[i, actions[i]], -10, 10)

        self._handle_epsilon(episode_count=episode)
        loss = self.model.fit(states, q_targets, epochs=1, verbose=0)
        return loss

    def _handle_epsilon(self, episode_count):
        self.epsilon = self.epsilon_policy.updateEpsilon(
            self.epsilon, episode_count=episode_count, max_episodes=self.max_episodes
        )

    def compute_action(self, current_state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            q_value = self.model.predict(current_state, verbose=0)[0]
            return np.argmax(q_value)
