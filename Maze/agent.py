import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import random
from tensorflow.keras.layers import Dense, Input
from collections import deque


class General_DQN_Agent:
    def __init__(
        self,
        action_size,
        state_size,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        batch_size=32,
        buffer_size=2000,
    ) -> None:
        self.action_size = action_size
        self.state_size = state_size
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        # suggested by ai , seems its have O(1) pop timecomplexity
        self.buffer_mem = deque(maxlen=buffer_size)
        self.model = self._initiate_model()
        self.model.compile(loss="mse", optimizer=Adam(learning_rate=self.lr))
        self.buffer_size = buffer_size

    def _initiate_model(self):
        return keras.Sequential(
            [
                Input(shape=(self.state_size,)),
                Dense(units=24, activation="relu"),
                Dense(units=24, activation="relu"),
                Dense(units=self.action_size, activation="linear"),
            ]
        )

    def store_experience(self, current_state, next_state, imm_reward, action, done):
        self.buffer_mem.append(
            {
                "current_state": current_state,
                "action": action,
                "reward": imm_reward,
                "next_state": next_state,
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
        # need for more knowledge for this method
        # self.updateEpsilon_BaseOnLoss(loss)
        self.updateEpsilon_Nonlinear()
        return loss

    # we learnd that agent should balance between exploration and exploitation
    # without this section , it seems that agent just try to explor the new pathes
    #
    def updateEpsilon_Nonlinear(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def updateEpsilon_BaseOnLoss(self, loss):
        if loss < 0.1:
            self.epsilon = 0.1
        else:
            self.epsilon -= 0.05

    def compute_action(self, current_state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            q_value = self.model.predict(current_state, verbose=0)[0]
            return np.argmax(q_value)
