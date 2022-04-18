import random

import unittest

import numpy as np

from recursive_marl_model.env.simple_env import RobotTaskAllocationEnv


class TestSimpleEnv(unittest.TestCase):

    def test_move(self):
        env = RobotTaskAllocationEnv((5, 5), start_loc=(0, 0), end_loc=(4, 4), fixed_task=True)
        env.reset()

        # move left
        env.truck_loc = (0, 1)
        ret = env._move(0)
        self.assertTrue(env.truck_loc == (0, 0))
        self.assertTrue(ret == 0)
        ret = env._move(0)
        self.assertTrue(env.truck_loc == (0, 0))
        self.assertTrue(ret == 1)

        # move right
        env.truck_loc = (0, 3)
        ret = env._move(1)
        self.assertTrue(env.truck_loc == (0, 4))
        self.assertTrue(ret == 0)
        ret = env._move(1)
        self.assertTrue(env.truck_loc == (0, 4))
        self.assertTrue(ret == 1)

    def test_reset(self):
        env = RobotTaskAllocationEnv((1, 9))
        env.reset()

    def test_step(self):
        env = RobotTaskAllocationEnv((1, 5), start_loc=(0, 0), end_loc=(0, 4), fixed_task=True)
        env.reset()
        env.map[env.truck_loc][env.TRUCK_CHANNEL] = 0
        env.truck_loc = (0, 3)
        env.map[env.truck_loc][env.TRUCK_CHANNEL] = 1

        # print(env.map)
        # env.step(0)
        # print(env.map)

