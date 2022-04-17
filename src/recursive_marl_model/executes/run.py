import os
import numpy as np
import time

from ..env.simple_env import RobotTaskAllocationEnv
from ..model.dqn import DQNModel
from ..model.rdqn import RDQNModel
from ..model.ddqn import DDQNModel
from ..evaluation.evaluator import RewardEvaluator


def run_env_model(env, model, train=True, with_random=True, with_render=False, render_clear=True, delay=0):
    state = env.reset()
    if with_render:
        env.render(render_clear)

    episode_reward = 0

    while True:
        action, repeat = model.get_next_action(state, with_random)
        if train:
            for _ in range(repeat):
                next_state, reward, done = env.step(action)
                if done:
                    break
        else:
            next_state, reward, done = env.step(action)

        if train:
            #model.memorize(state, action, reward, next_state, done)
            model.memorize(state, action, reward * repeat, next_state, done)

        state = next_state
        episode_reward += reward
        # print(episode_reward, reward, action, ii)

        if with_render:
            #print(action, repeat)
            env.render(render_clear)

        if delay:
            time.sleep(delay)

        if done:
            break

    return episode_reward


def train(min_episodes=500000, model_path='/tmp/model_path', env_shape=(1, 9), qmodel='rdqn'):
    if os.path.exists(model_path):
        ans = input(f'Warning: {model_path} exists, files will be rewrite if continue. [y/N]?')
        if ans not in ['y', 'Y']:
            return
    else:
        os.mkdir(model_path)

    env = RobotTaskAllocationEnv(env_shape)
    if env_shape == (1, 9):
        kernel_size = (1, 3)
    elif env_shape == (9, 9):
        kernel_size = (3, 3)
    elif env_shape == (3, 3):
        kernel_size = None
    else:
        raise RuntimeError(f'{env_shape} map not supported')

    if qmodel == 'rdqn':
        if kernel_size:
            model = RDQNModel(env.observation_space,
                              env.action_space,
                              model_path=model_path,
                              kernel_size=kernel_size,
                              input_channel=env.TOTAL_CHANNEL)
        else:
            raise RuntimeError('3x3 map not supported for RDQN')
    elif qmodel == 'dqn':
        model = DQNModel(env.observation_space,
                         env.action_space,
                         model_path=model_path,
                         kernel_size=kernel_size,
                         input_channel=env.TOTAL_CHANNEL)
    else:
        raise RuntimeError(f'{model} map not supported')


    evaluator = RewardEvaluator(model_path)

    episode_rewards = []

    start_time = time.perf_counter()

    for i in range(1, min_episodes):
        episode_reward = run_env_model(env, model)
        episode_rewards.append(episode_reward)

        model.train()

        if i % 100 == 0:
            ret = run_env_model(env,
                                model,
                                train=False,
                                with_random=False,
                                with_render=True,
                                delay=0,
                                render_clear=True)

            mean_reward = np.array(episode_rewards).mean()
            evaluator.save_reward(i, mean_reward)
            model.save()

            passed_time = time.perf_counter() - start_time
            print(f'{i}: passed time={passed_time}, mean_reward={mean_reward}')


def replay(model_path='/tmp/model_path', env_shape=(1, 9), model='rdqn'):
    env = RobotTaskAllocationEnv(env_shape)
    if env_shape == (1, 9):
        kernel_size = (1, 3)
    elif env_shape == (9, 9):
        kernel_size = (3, 3)
    elif env_shape == (3, 3):
        kernel_size = None

    if model == 'rdqn':
        if kernel_size:
            model = RDQNModel(env.observation_space, env.action_space, model_path=model_path, kernel_size=kernel_size)
        else:
            raise RuntimeError('3x3 map not supported for RDQN')
    elif model == 'dqn':
        model = DQNModel(env.observation_space, env.action_space, model_path=model_path, kernel_size=kernel_size)

    run_env_model(env, model, train=False, with_random=False, with_render=True, delay=0, render_clear=False)


def evaluate(model_path='/tmp/model_path'):
    evaluator = RewardEvaluator(model_path)
    evaluator.plot()