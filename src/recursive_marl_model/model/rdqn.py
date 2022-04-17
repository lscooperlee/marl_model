from collections import deque
import os
import random

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

np.set_printoptions(precision=3)


class RDQNModel:

    def __init__(self, input_size, output_size, kernel_size=(3, 3), input_channel=4, model_path=None) -> None:
        self.truck_channel = 0
        self.kernel_size = kernel_size
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

        self.model_path = model_path
        if model_path and os.path.exists(f'{model_path}/model_file'):
            self.model = keras.models.load_model(f'{model_path}/model_file')
            self.target_model = keras.models.load_model(f'{model_path}/model_file')
        else:
            model_size = (self.kernel_size[0] * self.kernel_size[1]) * self.input_channel
            self.model = self.create_q_model(model_size*2, self.output_size)
            self.target_model = self.create_q_model(model_size*2, self.output_size)

    def create_q_model(self, state_size, action_size):

        observation = layers.Input(shape=state_size, name='input')
        layer1 = layers.Dense(64, activation="relu")(observation)
        layer2 = layers.Dense(64, activation="relu")(layer1)
        action = layers.Dense(action_size, activation="linear")(layer2)

        model = keras.Model(inputs=observation, outputs=action)

        model.compile(loss="mse",
                      optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                      metrics=['accuracy'])

        return model

    def scale_state_level1(self, state):
        reshape_3x3 = state.reshape(
            (self.input_size[0] // self.kernel_size[0]) * (self.input_size[1] // self.kernel_size[1]), -1,
            self.input_channel)
        reduce_each_input_channel = np.max(reshape_3x3, axis=1)
        flatten = reduce_each_input_channel.reshape(-1)
        return flatten

    def scale_state_batch_level1(self, state):
        reshape_to_batch_size = state.reshape(self.mini_batch_size, (self.input_size[0] // self.kernel_size[0]) *
                                              (self.input_size[1] // self.kernel_size[1]), -1, self.input_channel)
        reduce_each_input_channel = np.max(reshape_to_batch_size, axis=2)
        reshape_flatten_last = reduce_each_input_channel.reshape(self.mini_batch_size, -1)
        return reshape_flatten_last

    def scale_state_level2(self, state):
        reshape_kernel = state.reshape(-1, self.kernel_size[0] * self.kernel_size[1], self.input_channel)
        reduce_each_input_channel = np.max(reshape_kernel, axis=1)
        index = np.argmax(reduce_each_input_channel[:, 0] == 1)
        return reshape_kernel[index].reshape(-1)

    def scale_state_batch_level2(self, state):
        reshape_kernel = state.reshape(self.mini_batch_size, -1, self.kernel_size[0] * self.kernel_size[1],
                                       self.input_channel)
        reduce_each_input_channel = np.max(reshape_kernel, axis=2)
        
        index = np.argmax(reduce_each_input_channel[:, :, 0] == 1, axis=1)
        index_ = np.arange(self.mini_batch_size)  # [0, 1, 2, ..., batch_size]
        reshape_flatten_last = reshape_kernel[index_, index]
        return reshape_flatten_last.reshape(self.mini_batch_size, -1)

    def get_next_action(self, state, with_random=False):
        repeat_n = self.input_size[1] // self.kernel_size[1]
        repeat = np.random.choice(range(1, repeat_n + 1))

        if with_random and np.random.uniform(0, 1) < self.epsilon:
            return random.randrange(self.output_size), repeat
        else:
            scaled_state1 = self.scale_state_level1(np.array(state))
            scaled_state2 = self.scale_state_level2(np.array(state))
            scaled_state = np.array([scaled_state1, scaled_state2])
            qv = self.model.predict(scaled_state.reshape(1, -1))[0]
            action = np.argmax(qv)

            return action, repeat

    def memorize(self, state_t, action_t, reward_t, state_t_1, done):
        self.replay_memory.append([state_t, action_t, reward_t, state_t_1, done])

    def recall(self):
        return random.sample(self.replay_memory, self.mini_batch_size)

    def train(self):
        if len(self.replay_memory) < self.mini_batch_size:
            return

        data = self.recall()
        states, actions, rewards, next_states, are_done = list(zip(*data))

        scaled_states1 = self.scale_state_batch_level1(np.array(states))
        scaled_next_states1 = self.scale_state_batch_level1(np.array(next_states))

        scaled_states_stack1 = np.hstack((scaled_states1, np.zeros(scaled_states1.shape)))
        scaled_next_states_stack1 = np.hstack((scaled_next_states1, np.zeros(scaled_next_states1.shape)))
        up_qvalue = self.train_kernel_level1(scaled_states_stack1, actions, rewards, scaled_next_states_stack1, are_done)

        scaled_states2 = self.scale_state_batch_level2(np.array(states))
        scaled_next_states2 = self.scale_state_batch_level2(np.array(next_states))

        scaled_states_stack2 = np.hstack((scaled_states1, scaled_states2))
        scaled_next_states_stack2 = np.hstack((scaled_next_states1, scaled_next_states2))

        self.train_kernel_level2(scaled_states_stack2, actions, rewards, scaled_next_states_stack2, are_done, up_qvalue)

        # before model trained well, use random
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

        self.target_model.set_weights(self.model.get_weights())

    def train_kernel_level1(self, states, actions, rewards, next_states, are_done):

        expected_q_value_next_state = np.max(self.target_model.predict(next_states), axis=1)
        q_value_for_update = self.gamma * expected_q_value_next_state * np.logical_not(are_done) + rewards
        q_value_from_model = self.model.predict(states)
        index = np.arange(len(q_value_from_model))  # [0, 1, 2, ..., batch_size]
        q_value_from_model[index, actions] = q_value_for_update

        self.model.fit(np.array(states), q_value_from_model, batch_size=self.mini_batch_size, verbose=0)

        return q_value_for_update

    def train_kernel_level2(self, states, actions, rewards, next_states, are_done, uplevel_rewards):

        expected_q_value_next_state = np.max(self.target_model.predict(next_states), axis=1)
        q_value_for_update = self.gamma * expected_q_value_next_state * np.logical_not(are_done) + rewards

        # strategy 1: max
        q_value_for_update = np.max(np.array([q_value_for_update, uplevel_rewards]), axis=0)

        q_value_from_model = self.model.predict(states)
        index = np.arange(len(q_value_from_model))  # [0, 1, 2, ..., batch_size]
        q_value_from_model[index, actions] = q_value_for_update

        self.model.fit(np.array(states), q_value_from_model, batch_size=self.mini_batch_size, verbose=0)

    def save(self):
        if self.model_path:
            self.model.save(f'{self.model_path}/model_file')