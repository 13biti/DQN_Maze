import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import random


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
        self.buffer_mem = []
        self.model = self._initiate_model()
        self.model.compile(loss="mse", optimizer=Adam(learning_rate=self.lr))
        self.buffer_size = buffer_size

    def _initiate_model(self):
        return keras.Sequential(
            [
                Dense(units=24, input_dim=self.state_size, activation="relu"),
                Dense(units=24, activation="relu"),
                Dense(units=self.action_size, activation="linear"),
            ]
        )

    def store_exprience(self, current_state, next_state, imm_reward, action, done):
        if len(self.buffer_mem) > self.buffer_size:
            self.buffer_mem.pop(0)
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
        # if there is not enough state in buffer then :
        if len(self.buffer_mem) < self.batch_size:
            return

        batch = random.sample(self.buffer_mem, self.batch_size)
        states = np.vstack([item["current_state"] for item in batch])
        next_states = np.vstack([item["next_state"] for item in batch])
        rewards = np.array([item["reward"] for item in batch])
        actions = np.array([item["action"] for item in batch])
        dones = np.array([item["done"] for item in batch])

        # this Q(s,a)
        q_current = self.model.predict(states, verbose=0)
        q_next = self.model.predict(next_states, verbose=0)
        q_targets = q_current.copy()
        for i in range(self.batch_size):
            if not dones[i]:
                # r + gamma*argmax(Q(s',a))
                q_targets[i, actions[i]] = rewards[i] + self.gamma * np.max(q_next[i])
            else:
                q_targets[i, actions[i]] = rewards[i]
        # trian the model
        self.model.fit(states, q_targets, epochs=1, verbose=0)
        """ this part is not complited yet 
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            """

    def compute_action(self, current_state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(range(self.action_size))
        else:
            q_value = self.model.predict(current_state)[0]
            return np.argmax(q_value)
