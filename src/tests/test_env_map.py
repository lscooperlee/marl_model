import random

import unittest

import numpy as np

from marl_model.env.simple_env import RobotTaskAllocationEnv
from marl_model.env.env_map import SimpleEnvMap, BaseEnvMap


class TestEnvMap(unittest.TestCase):

    def test_construct(self):
        s = SimpleEnvMap()
        c = SimpleEnvMap((10, 10))
        e = BaseEnvMap(data=[[1, 0, 1], [1, 1, 1], [1, 1, 1]])

    def test_map_in_env(self):
        simple_map = SimpleEnvMap()
        env = RobotTaskAllocationEnv(occupancy_grid_map=simple_map)
        self.assertEqual(simple_map[4][1], 1.0)


if __name__ == '__main__':
    unittest.main()