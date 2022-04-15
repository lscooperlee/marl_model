from collections import deque
import random

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

np.set_printoptions(precision=3)


class RDQNModel:

    def __init__(self,
                 input_size,
                 output_size,
                 input_channel=4,
                 model_path=None) -> None:

        self.kernel_size = (3, 3)
        assert len(self.kernel_size) == len(input_size)
        assert all(inp % ker == 0 for inp, ker in zip(input_size, self.kernel_size))

        self.input_size = input_size
        self.output_size = output_size
        self.input_channel = input_channel

        REPLAY_MEMORY_SIZE = 100000
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.learning_rate = 0.001
        self.gamma = 0.95
        self.mini_batch_size = 32
        self.epsilon = 1
        self.min_epsilon = 0.05
        self.epsilon_decay = 0.99

        self.model = self.create_q_model(self.kernel_size[0] * self.kernel_size[1]*self.input_channel,
                                         self.output_size, model_path)
        self.target_model = self.create_q_model(
            self.kernel_size[0] * self.kernel_size[1]*self.input_channel, self.output_size, model_path)

        self.training_level = 0
        self.max_level = 2

    def create_q_model(self, state_size, action_size, model_path=None):
        if model_path is None:
            observation = layers.Input(shape=state_size, name='input')
            # layer1 = layers.Dense(512, activation="relu")(observation)
            # layer2 = layers.Dense(64, activation="relu")(layer1)
            # layer3 = layers.Dense(64, activation="relu")(layer2)
            # action = layers.Dense(action_size, activation="linear")(layer3)
            layer1 = layers.Dense(64, activation="relu")(observation)
            layer2 = layers.Dense(64, activation="relu")(layer1)
            action = layers.Dense(action_size, activation="linear")(layer2)
            # action = layers.Dense(action_space, activation="relu")(layer2)

            model = keras.Model(inputs=observation, outputs=action)
        else:
            model = keras.models.load_model(model_path)

        #optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
        #model.compile(loss="mse", optimizer=optimizer)
        model.compile(
            loss="mse",
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            metrics=['accuracy'])

        return model

    def scale_state(self, state):
        return np.max(state.reshape(9, -1, self.input_channel), axis=1).reshape(-1)

    def scale_state_batch(self, state):
        return np.max(state.reshape(self.mini_batch_size, 9, -1, self.input_channel), axis=2).reshape(self.mini_batch_size, -1)

    def get_next_action(self, state, with_random=False):
        repeat = self.input_size[0] // np.power(self.kernel_size[0], self.training_level) 
        scaled_state = self.scale_state(np.array(state))
        return self.get_next_kernel_action(scaled_state, with_random), repeat

    def get_next_kernel_action(self, state, with_random=False):

        # With probability epsilon select a random action
        if with_random and np.random.uniform(0, 1) < self.epsilon:
            return random.randrange(self.output_size)
        else:  # otherwise select from model (at = maxa Q∗ (φ(st ), a; θ))
            qv = self.model.predict(state.reshape(1, -1))[0]
            return np.argmax(qv)

    def memorize(self, state_t, action_t, reward_t, state_t_1, done):
        self.replay_memory.append(
            [state_t, action_t, reward_t, state_t_1, done])

    def recall(self):
        return random.sample(self.replay_memory, self.mini_batch_size)

    def train(self):
        if len(self.replay_memory) < self.mini_batch_size:
            return

        NUM_OF_EPISODE_M = 1

        for _ in range(NUM_OF_EPISODE_M):  # for episode = 1, M do

            data = self.recall()
            states, actions, rewards, next_states, are_done = list(zip(*data))
            scaled_states = self.scale_state_batch(np.array(states))
            scaled_next_states = self.scale_state_batch(np.array(next_states))

            self.train_kernel(scaled_states, actions, rewards, scaled_next_states, are_done)

            # before model trained well, use random
            if self.epsilon > self.min_epsilon:
                # print(self.epsilon)
                self.epsilon *= self.epsilon_decay
            
            q_v = np.max(self.target_model.predict(scaled_states), axis=1)
            # print(q_v)

        self.target_model.set_weights(self.model.get_weights())

    def train_kernel(self, states, actions, rewards, next_states, are_done):

        expected_q_value_next_state = np.max(self.target_model.predict(next_states), axis=1)
        q_value_for_update = self.gamma * expected_q_value_next_state * np.logical_not(are_done) + rewards
        q_value_from_model = self.model.predict(states)
        q_value_from_model[np.arange(len(q_value_from_model)), actions] = q_value_for_update

        self.model.fit(np.array(states),
                       q_value_from_model,
                       batch_size=self.mini_batch_size,
                       verbose=0)

    def save(self, model_path):
        self.model.save(model_path)