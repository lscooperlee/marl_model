from collections import deque
import os
import random

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

np.set_printoptions(precision=3)


class DDQNModel:

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
            self.model_level1 = keras.models.load_model(f'{model_path}/model_file')
            self.target_model_level1 = keras.models.load_model(f'{model_path}/model_file')
        else:
            model_size_level1 = (self.kernel_size[0] * self.kernel_size[1]) * self.input_channel
            self.model_level1 = self.create_q_model_level1(model_size_level1, self.output_size)
            self.target_model_level1 = self.create_q_model_level1(model_size_level1, self.output_size)
            self.model_level2 = self.create_q_model_level1(model_size_level1 * 2, self.output_size)
            self.target_model_level2 = self.create_q_model_level1(model_size_level1 * 2, self.output_size)

        self.up_qvalue_ratio_decay = 0.99
        self.up_qvalue_ratio = 1
        self.replay_memory1 = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.replay_memory2 = deque(maxlen=REPLAY_MEMORY_SIZE)

    def create_q_model_level1(self, state_size, action_size):

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

    def scale_state_batch_level1(self, state, level=0, action=0):
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

    def scale_state_batch_level2(self, state, level=0, action=0):
        reshape_kernel = state.reshape(self.mini_batch_size, -1, self.kernel_size[0] * self.kernel_size[1],
                                       self.input_channel)
        reduce_each_input_channel = np.max(reshape_kernel, axis=2)

        index = np.argmax(reduce_each_input_channel[:, :, 0] == 1, axis=1)
        index_ = np.arange(self.mini_batch_size)  # [0, 1, 2, ..., batch_size]
        reshape_flatten_last = reshape_kernel[index_, index]
        return reshape_flatten_last.reshape(self.mini_batch_size, -1)

    def get_next_action(self, state, with_random=False, level=2):
        repeat_n = self.input_size[1] // self.kernel_size[1]
        repeat = np.random.choice(range(1, repeat_n + 1))
        #repeat = repeat + repeat_n - 1
        repeat = 3

        if with_random and np.random.uniform(0, 1) < self.epsilon:
            # if np.random.uniform(0, 1) < 0.5:
            if level == 1:
                return random.randrange(self.output_size), repeat
            else:
                return random.randrange(self.output_size), 1
        else:
            #if with_random and np.random.uniform(0, 1) < self.epsilon:
            # if with_random and np.random.uniform(0, 1) < 0.5:
            if level == 1:
                scaled_state1 = self.scale_state_level1(np.array(state))
                qv = self.model_level1.predict(scaled_state1.reshape(1, -1))[0]
                action = np.argmax(qv)
                return action, repeat
            else:
                scaled_state1 = self.scale_state_level1(np.array(state))
                scaled_state2 = self.scale_state_level2(np.array(state))
                scaled_state = np.array([scaled_state1, scaled_state2])
                qv = self.model_level2.predict(scaled_state.reshape(1, -1))[0]
                action = np.argmax(qv)

                return action, 1

    def memorize(self, state_t, action_t, reward_t, state_t_1, done, level):
        #        self.replay_memory.append([state_t, action_t, reward_t, state_t_1, done])
        if level == 1:
            self.replay_memory1.append([state_t, action_t, reward_t, state_t_1, done])
        elif level == 2:
            self.replay_memory2.append([state_t, action_t, reward_t, state_t_1, done])

    def recall(self, level):
        #        return random.sample(self.replay_memory, self.mini_batch_size)
        if level == 1:
            return random.sample(self.replay_memory1, self.mini_batch_size)
        elif level == 2:
            return random.sample(self.replay_memory2, self.mini_batch_size)

    def train(self):
        # if len(self.replay_memory) < self.mini_batch_size:
        #     return
        if len(self.replay_memory1) < self.mini_batch_size:
            return
        if len(self.replay_memory2) < self.mini_batch_size:
            return

        data = self.recall(level=1)
        states, actions, rewards, next_states, are_done = list(zip(*data))

        scaled_states = self.scale_state_batch_level1(np.array(states))
        scaled_next_states = self.scale_state_batch_level1(np.array(next_states))
        up_qvalue = self.train_kernel_level1(scaled_states, actions, rewards, scaled_next_states, are_done)

        if self.up_qvalue_ratio < 0.9:
            data = self.recall(level=2)
            states, actions, rewards, next_states, are_done = list(zip(*data))

            scaled_states = self.scale_state_batch_level1(np.array(states))
            scaled_next_states = self.scale_state_batch_level1(np.array(next_states))
            scaled_states2 = self.scale_state_batch_level2(np.array(states))
            scaled_next_states2 = self.scale_state_batch_level2(np.array(next_states))
            scaled_states = np.hstack((scaled_states, scaled_states2))
            scaled_next_states = np.hstack((scaled_next_states, scaled_next_states2))

            print(states[0].reshape(-1, 3).T)
            print(next_states[0].reshape(-1, 3).T)
            #up_qvalue = np.max(self.target_model_level1.predict(self.scale_state_batch_level1(np.array(next_states))), axis=1)
            # up_action = np.argmax(self.target_model_level1.predict(self.scale_state_batch_level1(np.array(states))), axis=1)
            up_qvalue = self.target_model_level1.predict(self.scale_state_batch_level1(np.array(next_states)))
            self.train_kernel_level2(scaled_states, actions, rewards, scaled_next_states, are_done, up_qvalue)

        # before model trained well, use random
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

        self.up_qvalue_ratio *= self.up_qvalue_ratio_decay

        self.target_model_level1.set_weights(self.model_level1.get_weights())
        self.target_model_level2.set_weights(self.model_level2.get_weights())

    def train_kernel_level1(self, states, actions, rewards, next_states, are_done):

        #print('........')
        ttt = self.target_model_level1.predict(next_states)
        #print('ttt: ', ttt[:1])
        expected_q_value_next_state = np.max(self.target_model_level1.predict(next_states), axis=1)
        #print('ddd: ', expected_q_value_next_state[:1])
        q_value_for_update = self.gamma * expected_q_value_next_state * np.logical_not(are_done) + rewards
        q_value_from_model = self.model_level1.predict(states)
        #print('original qvalue: ', q_value_from_model[:1])
        index = np.arange(len(q_value_from_model))  # [0, 1, 2, ..., batch_size]
        q_value_from_model[index, actions] = q_value_for_update
        #print('action: ', actions[:1])
        #print('qvalue for update: ', q_value_for_update[:1])
        #print('final qvalue: ', q_value_from_model[:1])
        #print('original state: ', states[:1])
        #print('next state: ', next_states[:1])


        self.model_level1.fit(np.array(states), q_value_from_model, batch_size=self.mini_batch_size, verbose=0)

        return q_value_for_update

    def train_kernel_level2(self, states, actions, rewards, next_states, are_done, uplevel_rewards):

        expected_q_value_next_state = np.max(self.target_model_level2.predict(next_states), axis=1)
        print('level1 expected q value: ', expected_q_value_next_state[0])
        # strategy 4: 
        # index = np.arange(len(expected_q_value_next_state))  # [0, 1, 2, ..., batch_size]
        # expected_q_value_next_state += 0.4*uplevel_rewards[index, actions]

        # strategy 5:
        xx = np.argmax(uplevel_rewards, axis=1)

        print('level1 expected q value: ', expected_q_value_next_state[0])

        q_value_for_update = self.gamma * expected_q_value_next_state * np.logical_not(are_done) + rewards
        zz = q_value_for_update * (np.array(actions == xx) - 0.5)
        q_value_for_update += zz
        print('llllll q_value_for_update: ', q_value_for_update[0])
        print('llllll uplevel reward: ', uplevel_rewards[0])
        print('llllllllll action: ', actions[0])
        print('llllllllll state: ', states[0])
        print('llllllllll next state: ', next_states[0])

        # strategy 1: max
        #q_value_for_update = np.max(np.array([q_value_for_update, uplevel_rewards]), axis=0)

        # strategy 2: with ratio
        # q_value_for_update = q_value_for_update + uplevel_rewards * self.up_qvalue_ratio

        # strategy 3: with prob
        # if random.random() < self.up_qvalue_ratio:
        #     q_value_for_update = np.max(np.array([q_value_for_update, uplevel_rewards]), axis=0)

        q_value_from_model = self.model_level2.predict(states)
        print('llllllll qvalue for update: ', q_value_from_model[0])
        index = np.arange(len(q_value_from_model))  # [0, 1, 2, ..., batch_size]
        q_value_from_model[index, actions] = q_value_for_update
        print('final llllllll qvalue for update: ', q_value_from_model[0])


        self.model_level2.fit(np.array(states), q_value_from_model, batch_size=self.mini_batch_size, verbose=0)

    def save(self):
        if self.model_path:
            self.model_level1.save(f'{self.model_path}/model_file')