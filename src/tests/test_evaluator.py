import unittest
import tempfile

from marl_model.evaluation.evaluator import RewardEvaluator
from marl_model.env.simple_env import BaseTaskAllocationEnv
from marl_model.model.dqn import DQNModel


class TestEvaluator(unittest.TestCase):

    def test_construct(self):
        v = RewardEvaluator()

    def test_construct(self):
        model = DQNModel((3, 3), 2, 1)
        env = BaseTaskAllocationEnv()
        v = RewardEvaluator()
        with tempfile.TemporaryDirectory() as tf:
            v.save(1, 10.0, 2.0, model, env, tf)
            c = v.load(tf)

    def test_plot(self):
        model = DQNModel((3, 3), 2, 1)
        env = BaseTaskAllocationEnv()
        v = RewardEvaluator()


if __name__ == '__main__':
    unittest.main()