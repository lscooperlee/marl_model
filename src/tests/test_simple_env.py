import unittest

from marl_model.env.simple_env import BaseTaskAllocationEnv
from marl_model.env.simple_env import RobotTaskAllocationMapStateEnv
from marl_model.env.env_map import SimpleEnvMap


class TestSimpleEnv(unittest.TestCase):

    def test_move(self):
        env = BaseTaskAllocationEnv((5, 5), start_loc=(0, 0), end_loc=(4, 4), fixed_task=True)
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
        env = BaseTaskAllocationEnv((1, 9))
        env.reset()

    def test_step(self):
        env = BaseTaskAllocationEnv((1, 5), start_loc=(0, 0), end_loc=(0, 4), fixed_task=True)
        env.reset()
        env.map[env.truck_loc][env.TRUCK_CHANNEL] = 0
        env.truck_loc = (0, 3)
        env.map[env.truck_loc][env.TRUCK_CHANNEL] = 1

        # print(env.map)
        # env.step(0)
        # print(env.map)


class TestMapStateEnv(unittest.TestCase):

    def test_state(self):
        env = RobotTaskAllocationMapStateEnv((3, 3),
                                             start_loc=(0, 0),
                                             end_loc=(1, 1),
                                             fixed_task=True,
                                             is_1d=False,
                                             occupancy_grid_map=SimpleEnvMap(shape=(3, 3)))
        c = env.reset(truck=(2, 2))


if __name__ == '__main__':
    unittest.main()