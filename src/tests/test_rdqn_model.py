import unittest

import numpy as np

from recursive_marl_model.model.rdqn import RDQNModel


class TestModel(unittest.TestCase):

    def test_get_next_actions(self):
        model = RDQNModel((3, 3), 4, input_channel=4)
        # state = np.zeros(3*3*4)
        # actions = model.get_next_actions(state)
        # print(actions)

    def test_scale_state(self):
        model = RDQNModel((3, 3), 4, input_channel=3)
        state = np.zeros((3, 3, 3, 3, 3))
        state[0, 0, 1, 1, 1] = 1
        state[2, 2, 1, 1, 2] = 1
        state[0, 2, 1, 1, 0] = 1
        state = state.reshape(-1)

        expected = np.array([
            0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 1
        ])

        self.assertTrue(np.all(model.scale_state(state) == expected))

    def test_scale_state_level(self):
        model = RDQNModel((3, 3), 4, input_channel=3)
        state = np.zeros((3, 3, 3, 3, 3))
        state[0, 0, 1, 1, 1] = 1
        state[2, 2, 1, 1, 2] = 1
        state[0, 2, 1, 1, 0] = 1
        state = state.reshape(-1)

        expected = np.array([
            0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 0, 0
        ])

        self.assertTrue(np.all(model.scale_state_level(state) == expected))

    def test_get_scale_batch(self):
        model = RDQNModel((3, 3), 4, input_channel=3)
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
            [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
                0, 0, 0, 0, 0, 0
            ],
            [
                0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 1
            ],
        ])

        self.assertTrue(np.all(model.scale_state_batch(state) == expected))


if __name__ == '__main__':
    unittest.main()