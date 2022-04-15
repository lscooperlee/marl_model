import os
import numpy as np
import time

#from ..env.simple_env import RobotTaskAllocationEnv
from ..env.simple_env1 import RobotTaskAllocationEnv
from ..model.dqn import DQNModel
from ..model.rdqn import RDQNModel
from ..model.ddqn import RDQNModel as DDQNModel
from ..evaluation.evaluator import RewardEvaluator


def run_env_model(env, model, train=True, with_random=True, with_render=False, render_clear=True, delay=0):
    state = env.reset()
    if with_render:
        env.render(render_clear)

    episode_reward = 0
    for ii in range(1000000):

        #action, repeat = model.get_next_action_level1(state, with_random)
        action, repeat = model.get_next_action(state, with_random)
        if train:
            for _ in range(repeat):
                next_state, reward, done, _ = env.step(action)
                if done:
                    break
        else:
            next_state, reward, done, _ = env.step(action)

        # if reward > 0:
        #     reward = reward * repeat * repeat * repeat * repeat

        if train:
            model.memorize(state, action, reward, next_state, done)

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


def run_env_model_back(env, model, train=True, with_random=True, with_render=False, render_clear=True, delay=0):
    state = env.reset()
    if with_render:
        env.render(render_clear)

    episode_reward = 0
    while True:

        action, repeat = model.get_next_action(state, with_random)
        next_state, reward, done, _ = env.step(action)

        # for _ in range(repeat):
        #     next_state, reward, done, _ = env.step(action)

        #     if done:
        #         break

        # if reward > 0:
        #     reward = reward * repeat * repeat * repeat * repeat

        if train:
            model.memorize(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

        if with_render:
            print(action)
            env.render(render_clear)

        if delay:
            time.sleep(delay)

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

    # simple test
    # env = RobotTaskAllocationEnv(map_shape=(3, 3))
    # model = DQNModel(env.observation_space, env.action_space, model_path)

    # for ddqn
    # env = RobotTaskAllocationEnv(map_shape=(1, 9))
    # model = DDQNModel((1, 9), env.action_space, kernel_size=(1, 3))

    # for ddqn 2D
    # env = RobotTaskAllocationEnv(map_shape=(9, 9))
    # model = DDQNModel((9, 9), env.action_space, kernel_size=(3, 3))

    evaluator = RewardEvaluator(model_path)

    episode_rewards = []

    start_time = time.perf_counter()

    for i in range(1, min_episodes):
        #episode_reward = run_env_model(env, model)#, with_render=True, render_clear=False)
        # for ddqn
        episode_reward = run_env_model(env, model)  #, with_render=True, render_clear=False)
        episode_rewards.append(episode_reward)

        model.train()

        if i % 100 == 0:
            # ret = run_env_model(env, model, train=False, with_random=False, with_render=True, delay=0, render_clear=False)
            # for ddqn
            ret = run_env_model(env,
                                model,
                                train=False,
                                with_random=False,
                                with_render=True,
                                delay=0,
                                render_clear=True)
            #print(ret)
            mean_reward = np.array(episode_rewards).mean()
            evaluator.save_reward(i, mean_reward)
            model.save()

            passed_time = time.perf_counter() - start_time
            print(f'{i}: passed time={passed_time}, mean_reward={mean_reward}')


def replay(model_path='/tmp/model_path'):
    env = RobotTaskAllocationEnv(map_shape=(9, 9))
    model = DDQNModel((3, 3), env.action_space, model_path=model_path)
    run_env_model(env, model, train=False, with_random=False, with_render=True, delay=0, render_clear=False)


def evaluate(model_path='/tmp/model_path'):
    evaluator = RewardEvaluator(model_path)
    evaluator.plot()