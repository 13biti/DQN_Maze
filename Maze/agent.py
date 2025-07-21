import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import random
from tensorflow.keras.layers import Dense, Input
from collections import deque

INFINIT = float("inf")


class General_DQN_Agent:
    def __init__(
        self,
        action_size,
        state_size,
        learning_rate=0.001,
        gamma=0.99,
        batch_size=32,
        buffer_size=2000,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        epsilon_policy=None,
    ) -> None:
        self.action_size = action_size
        self.state_size = state_size
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        # suggested by ai , seems its have O(1) pop timecomplexity
        self.buffer_mem = deque(maxlen=buffer_size)
        self.model = self._initiate_model()
        self.model.compile(loss="mse", optimizer=Adam(learning_rate=self.lr))
        self.buffer_size = buffer_size
        self.epsilon_policy = (
            EpsilonPolicy("None", self.epsilon, epsilon_min, epsilon_decay)
            if epsilon_policy is None
            else epsilon_policy
        )

    def _initiate_model(self):
        return keras.Sequential(
            [
                Input(shape=(self.state_size,)),
                Dense(units=24, activation="relu"),
                Dense(units=24, activation="relu"),
                Dense(units=self.action_size, activation="linear"),
            ]
        )

    def store_experience(
        self, current_state, next_state, imm_reward, action, done, heuristic=0
    ):
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

    def train(self):
        if len(self.buffer_mem) < self.batch_size:
            return None
        batch = random.sample(self.buffer_mem, self.batch_size)
        states = np.vstack([item["current_state"] for item in batch])
        next_states = np.vstack([item["next_state"] for item in batch])
        rewards = np.array([item["reward"] for item in batch])
        actions = np.array([item["action"] for item in batch])
        dones = np.array([item["done"] for item in batch])
        q_current = self.model.predict(states, verbose=0)
        q_next = self.model.predict(next_states, verbose=0)
        q_targets = q_current.copy()
        for i in range(self.batch_size):
            if not dones[i]:
                q_targets[i, actions[i]] = rewards[i] + self.gamma * np.max(q_next[i])
            else:
                q_targets[i, actions[i]] = rewards[i]
        history = self.model.fit(states, q_targets, epochs=1, verbose=0)
        loss = history.history["loss"][0]
        return loss

    def compute_action(self, current_state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            q_value = self.model.predict(current_state, verbose=0)[0]
            return np.argmax(q_value)


# we learnd that agent should balance between exploration and exploitation
# without this section , it seems that agent just try to explor new path
#
class EpsilonPolicy:
    def __init__(
        self,
        policy="None",
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        progress_bonus: float = 0.05,
        exploration_bonus: float = 0.1,
    ):
        self.policy = policy
        self.visited_states = {}
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.old_huristic = INFINIT
        self.progress_bonus = progress_bonus
        self.exploration_bonus = exploration_bonus

    def updateEpsilon(self, state=None, heuristic=None):
        if self.policy.lower() == "erm":
            return self.updateEpsilon_EMR(state, heuristic)
        else:
            return self.updateEpsilon_Nonlinear()

    def updateEpsilon_Nonlinear(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return self.epsilon

    # this method is not complited,  for now , it only store visited states ,
    # and give extera bonus for the states that are not visited yet , this way , model will encoraged to learn more about new enviroment
    # i wish i can use it to improve rewards , but it need lots of changes , and also reward is not the problem
    def updateEpsilon_EMR(self, state, heuristic=None):
        is_new_state = state not in self.visited_states
        if is_new_state:
            self.visited_states[state] = heuristic if heuristic is not None else 0

        progress = 0.0
        if heuristic is not None:
            old_heuristic = self.visited_states[state]
            if heuristic > old_heuristic:  # assuming higher heuristic = better
                progress = self.progress_bonus
                self.visited_states[state] = heuristic  # Update best heuristic

        if self.epsilon > self.epsilon_min:
            decay_factor = self.epsilon_decay
            if is_new_state:
                decay_factor = min(1.0, decay_factor + self.exploration_bonus)
            if progress > 0:
                decay_factor = min(1.0, decay_factor + progress)
            self.epsilon *= decay_factor
        return self.epsilon
