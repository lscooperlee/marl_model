import os
import numpy as np
import time

from ..env.simple_env import RobotTaskAllocationEnv
from ..model.dqn import DQNModel
from ..evaluation.evaluator import RewardEvaluator


def run_env_model(env, model, train=True, with_random=True, with_render=False):
    state = env.reset()
    if with_render:
        env.render()

    episode_reward = 0
    while True:

        action, repeat = model.get_next_action(state, with_random)
        for _ in range(repeat):
            next_state, reward, done, _ = env.step(action)
            if done:
                break

        if train:
            model.memorize(state, action, reward, next_state, done)
        
        if with_render:
            env.render()

        state = next_state

        episode_reward += reward
        if done:
            break

    return episode_reward


def train(min_episodes=500000, model_path='/tmp/model_path'):
    if os.path.exists(model_path):
        ans = input(f'Warning: {model_path} exists, files will be rewrite if continue. [y/N]?')
        if ans not in ['y', 'Y']:
            return
    else:
        os.mkdir(model_path)

    env = RobotTaskAllocationEnv(map_shape=(9, 9))
    model = DQNModel(env.observation_space, env.action_space, model_path)
    evaluator = RewardEvaluator(model_path)

    episode_rewards = []

    start_time = time.perf_counter()

    for i in range(1, min_episodes):
        episode_reward = run_env_model(env, model)
        episode_rewards.append(episode_reward)

        model.train()

        if i % 10 == 0:
            ret = run_env_model(env, model, train=False, with_random=False, with_render=True)
            mean_reward = np.array(episode_rewards).mean()
            evaluator.save_reward(i, mean_reward)
            model.save()

            passed_time = time.perf_counter() - start_time
            print(f'{i}: passed time={passed_time}, mean_reward={mean_reward}')


def evaluate(model_path='/tmp/model_path'):
    evaluator = RewardEvaluator(model_path)
    evaluator.plot()