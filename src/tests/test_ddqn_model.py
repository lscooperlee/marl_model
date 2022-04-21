import unittest

import numpy as np

from marl_model.model.ddqn import DDQNModel
from marl_model.env.simple_env import RobotTaskAllocationEnv


class TestDDQNModel(unittest.TestCase):

    def test_scale_state_level1(self):
        model = DDQNModel((1, 9), 4, kernel_size=(1, 3), input_channel=3)
        state = np.zeros((1, 9, 3))
        state[0, 0, 1] = 1
        state[0, 2, 2] = 1
        state[0, 8, 0] = 1
        state = state.reshape(-1)

        expected = np.array([0, 1, 1, 0, 0, 0, 1, 0, 0])
        scaled = model.scale_state_level1(state)

        self.assertTrue(np.all(scaled == expected))

        model = DDQNModel((9, 9), 4, kernel_size=(3, 3), input_channel=3)
        state = np.zeros((3, 3, 3, 3, 3))
        state[0, 0, 1, 1, 1] = 1
        state[2, 2, 1, 1, 2] = 1
        state[0, 2, 1, 1, 0] = 1
        state = state.reshape(-1)

        expected = np.array([0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        scaled = model.scale_state_level1(state)

        self.assertTrue(np.all(scaled == expected))

    def test_get_scale_batch_level1(self):

        model = DDQNModel((9, 9), 4, input_channel=3)
        model.mini_batch_size = 2

        state1 = np.zeros((3, 3, 3, 3, 3))
        state1[1, 0, 0, 1, 0] = 1
        state1[1, 1, 0, 1, 1] = 1
        state1[1, 2, 2, 2, 2] = 1
        state1.reshape(-1)

        state2 = np.zeros((3, 3, 3, 3, 3))
        state2[0, 0, 1, 1, 1] = 1
        state2[2, 2, 1, 1, 2] = 1
        state2[0, 2, 1, 1, 0] = 1
        state2.reshape(-1)

        state = np.array([state1, state2])
        state = state.reshape(2, -1)

        expected = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ])
        scaled = model.scale_state_batch_level1(state)

        self.assertTrue(np.all(scaled == expected))

    def test_scale_state_level2(self):
        model = DDQNModel((1, 9), 4, kernel_size=(1, 3), input_channel=3)
        state = np.zeros((1, 9, 3))
        state[0, 0, 1] = 1
        state[0, 2, 2] = 1
        state[0, 8, 0] = 1
        state = state.reshape(-1)

        expected = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0])
        scaled = model.scale_state_level2(state)

        self.assertTrue(np.all(scaled == expected))

        model = DDQNModel((9, 9), 4, kernel_size=(3, 3), input_channel=3)
        state = np.zeros((3, 3, 3, 3, 3))
        state[0, 0, 1, 1, 1] = 1
        state[2, 2, 1, 1, 2] = 1
        state[0, 2, 2, 2, 0] = 1
        state = state.reshape(-1)

        expected = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        scaled = model.scale_state_level2(state)

        self.assertTrue(np.all(scaled == expected))

    def test_get_scale_batch_level2(self):

        model = DDQNModel((1, 9), 4, kernel_size=(1, 3), input_channel=3)
        model.mini_batch_size = 2

        state1 = np.zeros((1, 9, 3))
        state1[0, 0, 1] = 1
        state1[0, 2, 2] = 1
        state1[0, 8, 0] = 1
        state1 = state1.reshape(-1)

        state2 = np.zeros((1, 9, 3))
        state2[0, 8, 1] = 1
        state2[0, 2, 2] = 1
        state2[0, 0, 0] = 1
        state2 = state2.reshape(-1)

        state = np.array([state1, state2])
        state = state.reshape(2, -1)

        expected = np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 1]])
        scaled = model.scale_state_batch_level2(state)
        self.assertTrue(np.all(scaled == expected))

        #################3

        model = DDQNModel((9, 9), 4, input_channel=3)
        model.mini_batch_size = 2

        state1 = np.zeros((3, 3, 3, 3, 3))
        state1[1, 0, 0, 1, 0] = 1
        state1[1, 1, 0, 1, 1] = 1
        state1[1, 2, 2, 2, 2] = 1
        state1.reshape(-1)

        state2 = np.zeros((3, 3, 3, 3, 3))
        state2[0, 0, 1, 1, 1] = 1
        state2[2, 2, 1, 1, 2] = 1
        state2[0, 2, 1, 1, 0] = 1
        state2.reshape(-1)

        state = np.array([state1, state2])
        state = state.reshape(2, -1)

        expected = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        scaled = model.scale_state_batch_level2(state)

        self.assertTrue(np.all(scaled == expected))

    def test_train(self):
        env = RobotTaskAllocationEnv((1, 9))
        state = env.reset()
        model = DDQNModel(env.observation_space, env.action_space, kernel_size=(1, 3), input_channel=env.TOTAL_CHANNEL)
        for _ in range(model.mini_batch_size * 2):
            action, _ = model.get_next_action(state)
            next_state, reward, done = env.step(action)
            model.memorize(state, action, reward, next_state, done)
            state = next_state
        model.train()

        env = RobotTaskAllocationEnv((9, 9))
        state = env.reset()
        model = DDQNModel(env.observation_space, env.action_space, kernel_size=(3, 3), input_channel=env.TOTAL_CHANNEL)
        for _ in range(model.mini_batch_size * 2):
            action, _ = model.get_next_action(state)
            next_state, reward, done = env.step(action)
            model.memorize(state, action, reward, next_state, done)
            state = next_state
        model.train()

if __name__ == '__main__':
    unittest.main()