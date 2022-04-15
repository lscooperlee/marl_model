from collections import deque
from gc import callbacks
import random

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

#np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
np.set_printoptions(precision=3)


class DQNModel:

    def __init__(self, input_size, output_size, model_path=None) -> None:
        self.input_size = input_size
        self.output_size = output_size

        REPLAY_MEMORY_SIZE = 100000
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.learning_rate = 0.001
        self.gamma = 0.95
        self.mini_batch_size = 32
        self.epsilon = 1
        self.min_epsilon = 0.05
        self.epsilon_decay = 0.99

        # GAMMA_DISCOUNT = 0.99
        # MINI_BATCH_SIZE = 64
        # REPLAY_MEMORY_SIZE = 10000
        # NUM_OF_EPISODE_M = 100000

        self.model_path = model_path

        self.model = self.create_q_model(self.input_size, self.output_size)
        self.target_model = self.create_q_model(self.input_size, self.output_size)

    def create_q_model(self, state_size, action_size):
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
        # else:
        #     model = keras.models.load_model('my_model')

        #optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
        #model.compile(loss="mse", optimizer=optimizer)
        model.compile(
            loss="mse",
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            metrics=['accuracy'],
        )

        return model

    def get_next_action(self, state, with_random=False):

        # With probability epsilon select a random action
        if with_random and np.random.uniform(0, 1) < self.epsilon:
            return random.randrange(self.output_size), 1
        else:  # otherwise select from model (at = maxa Q∗ (φ(st ), a; θ))
            qv = self.model.predict(state.reshape(1, -1))[0]
            return np.argmax(qv), 1

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
            expected_q_value_next_state = np.max(self.target_model.predict(
                np.array(next_states)),
                                                 axis=1)
            #print(expected_q_value_next_state)
            q_value_for_update = self.gamma * expected_q_value_next_state * np.logical_not(
                are_done) + rewards
            #print(q_value_for_update)
            q_value_from_model = self.model.predict(np.array(states))
            #print(q_value_from_model)
            q_value_from_model[np.arange(len(q_value_from_model)),
                               actions] = q_value_for_update
            #print(states)
            #print(q_value_from_model)

            self.model.fit(np.array(states),
                           q_value_from_model,
                           batch_size=self.mini_batch_size,
                           callbacks=[keras.callbacks.TensorBoard(f'{self.model_path}')] if self.model_path else None,
                           verbose=0)

            # before model trained well, use random
            if self.epsilon > self.min_epsilon:
                # print(self.epsilon)
                self.epsilon *= self.epsilon_decay

        self.target_model.set_weights(self.model.get_weights())

    def save(self):
        if self.model_path:
            self.model.save(f'{self.model_path}/model_file')