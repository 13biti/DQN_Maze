from ast import mod
import numpy as np
import gym
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import mean_squared_error
import copy
from matplotlib import pyplot as plt


class General_DQN_Agent:
    def __init__(
        self,
        action_size,
        state_size,
        learning_rate=0.01,
        epsilon_policy=0.1,
        batch_size=25,
        buffer_size=2000,
    ) -> None:
        self.action_size = action_size
        self.state_size = state_size
        self.lr = learning_rate
        self.epsilon = epsilon_policy
        self.batch_size = batch_size
        self.buffer_mem = []
        self.buffer_size = buffer_size
        self.model = self._initiate_model()
        self.model.compile(loss="adam", optimizer=Adam(lr=self.lr))

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
        temp_buffer = copy.deepcopy(self.buffer_mem)
        np.random.shuffle(temp_buffer)
        batch_items = temp_buffer[0 : self.batch_size]
        for item in batch_items:
            # this Q(s,a)
            Q_current_state_predict = self.model.predict(item["current_state"])
            if not item["done"]:
                # r + gamma*argmax(Q(s',a))
                Q_target = item["reward"] + self.epsilon * np.max(
                    self.model.predict(item["next_state"])[0]
                )
                Q_current_state_predict[0][item["action"]] = Q_target
                self.model.fit(
                    item["current_state"], Q_current_state_predict, verbose=1
                )

    def compute_action(self, current_state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(range(self.action_size))
        else:
            q_value = self.model.predict(current_state)[0]
            return np.argmax(q_value)
