from collections import deque
import random

import numpy as np


class BaseModel:

    def __init__(self, input_size, output_size, input_channel=4, **kwargs) -> None:
        self.input_size = input_size
        self.output_size = output_size

        self.replay_memory = deque(maxlen=self.replay_memory_size)
        self.last_training_frame_counter = 0
        self.last_updating_frame_counter = 0
        self.last_epsilon_frame_counter = 0
        self.frame_counter = 0
        self.epsilon = 1

        self.epsilon_update_step = (self.epsilon - self.min_epsilon) / self.epsilon_end_frame

        self.model = self.create_q_model(input_size, output_size, input_channel)
        self.target_model = self.create_q_model(input_size, output_size, input_channel)

    def get_next_action(self, state, with_random=False):
        return np.argmax(self.get_qvalues(state, with_random))

    def get_qvalues(self, state, with_random=False):
        # With probability epsilon select a random action
        if with_random and np.random.uniform(0, 1) < self.epsilon:
            qv = np.random.rand(self.output_size)
        else:  # otherwise select from model (at = maxa Q∗ (φ(st ), a; θ))
            qv = self.model.predict(np.expand_dims(state, axis=0), verbose=0)[0]
        return qv

    def memorize(self, state_t, action_t, reward_t, state_t_1, done):
        self.frame_counter += 1
        self.replay_memory.append([state_t, action_t, reward_t, state_t_1, done])

    def recall(self):
        return random.sample(self.replay_memory, self.batch_size)

    def train(self):

        if len(self.replay_memory) < self.train_frame_rate + self.batch_size:
            return

        train_history = []
        for _ in range(1 + (self.frame_counter - self.last_training_frame_counter) // self.train_frame_rate):
            data = self.recall()
            states, actions, rewards, next_states, are_done = list(zip(*data))
            history = self.train_kernel(states, actions, rewards, next_states, are_done)
            train_history.append(history)
        self.last_training_frame_counter = self.frame_counter

        if self.epsilon > self.min_epsilon and self.frame_counter - self.last_epsilon_frame_counter > 0:
            self.epsilon -= (self.frame_counter - self.last_epsilon_frame_counter) * self.epsilon_update_step
            self.last_epsilon_frame_counter = self.frame_counter

        if (self.frame_counter - self.last_updating_frame_counter) > self.target_update_frame_rate:
            self.target_model.set_weights(self.model.get_weights())
            self.last_updating_frame_counter = self.frame_counter
            print(self.epsilon, self.frame_counter)

        hist = np.array(train_history).mean(axis=0)  # [loss, accuracy]
        return hist

    def train_kernel(self, states, actions, rewards, next_states, are_done):

        expected_q_value_next_state = np.max(self.target_model.predict(np.array(next_states), verbose=0), axis=1)
        q_value_for_update = self.gamma * expected_q_value_next_state * np.logical_not(are_done) + rewards
        q_value_from_model = self.model.predict(np.array(states), verbose=0)
        q_value_from_model[np.arange(len(q_value_from_model)), actions] = q_value_for_update

        history = self.model.fit(np.array(states),
                                 q_value_from_model,
                                 batch_size=self.batch_size,
                                 shuffle=False,
                                 epochs=self.epochs,
                                 callbacks=[self.callback] if hasattr(self, 'callback') else None,
                                 verbose=0)

        return [history.history['loss'][0], history.history['accuracy'][0]]

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        import keras
        self.model = keras.models.load_model(path)
        self.model.summary()
        #self.target_model.set_weights(self.model.get_weights())