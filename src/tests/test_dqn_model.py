import unittest
import random

import numpy as np

from recursive_marl_model.model.dqn import DQNModel
from recursive_marl_model.env.simple_env import RobotTaskAllocationEnv


class TestModel(unittest.TestCase):

    def test_train(self):
        env = RobotTaskAllocationEnv((1, 9))
        state = env.reset()
        model = DQNModel(env.observation_space, env.action_space, kernel_size=(1, 3), input_channel=env.TOTAL_CHANNEL)
        for _ in range(model.mini_batch_size * 2):
            action, _ = model.get_next_action(state)
            next_state, reward, done = env.step(action)
            model.memorize(state, action, reward, next_state, done)
            state = next_state
        model.train()

        env = RobotTaskAllocationEnv((9, 9))
        state = env.reset()
        model = DQNModel(env.observation_space, env.action_space, kernel_size=(3, 3), input_channel=env.TOTAL_CHANNEL)
        for _ in range(model.mini_batch_size * 2):
            action, _ = model.get_next_action(state)
            next_state, reward, done = env.step(action)
            model.memorize(state, action, reward, next_state, done)
            state = next_state
        model.train()


if __name__ == '__main__':
    unittest.main()