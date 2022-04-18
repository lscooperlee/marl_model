import os
import numpy as np
import time

from ..env.simple_env import RobotTaskAllocationEnv
from ..model.dqn import DQNModel
from ..model.rdqn import RDQNModel
from ..model.ddqn import DDQNModel
from ..evaluation.evaluator import RewardEvaluator

is_train_level2 = True

def run_env_model(env, model, train=True, with_random=True, with_render=False, render_clear=True, delay=0, truck=None):
    state = env.reset(truck)
    if with_render:
        env.render(render_clear)

    episode_reward = 0

    global is_train_level2
    if train:
        if is_train_level2:
            is_train_level2 = False
        else:
            is_train_level2 = True

    while True:
        if train:
            
            if is_train_level2:
                # is_train_level2 = False
                action, repeat = model.get_next_action(state, with_random, level=2)
                next_state, reward, done = env.step(action)
                model.memorize(state, action, reward, next_state, done, 2)
                state = next_state
                if done:
                    break
            else:
                # is_train_level2 = True
                action, repeat = model.get_next_action(state, with_random, level=1)
                next_state, reward, done = env.step(action)
                action, repeat = model.get_next_action(state, with_random, level=1)
                for actual_repeat in range(repeat):
                    next_state, reward, done = env.step(action)
                    if done:
                        break

                model.memorize(state, action, reward, next_state, done, 1)

            # action, repeat = model.get_next_action(state, with_random)
            # for actual_repeat in range(repeat):
            #     next_state, reward, done = env.step(action)
            #     if done:
            #         break

            # if repeat > 1:
            #     model.memorize(state, action, reward, next_state, done, 1)
            # else:
            #     model.memorize(state, action, reward, next_state, done, 2)

        else:
            action, repeat = model.get_next_action(state, with_random)
            next_state, reward, done = env.step(action)

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


def setup_env_and_model(model_path, env_shape, qmodel):

    # fix start, end location for simplicity
    if env_shape == (1, 9):
        kernel_size = (1, 3)
        #start_loc, end_loc = (0, 1), (0, 7) not quite work for repeat
        start_loc, end_loc = (0, 2), (0, 6)
    elif env_shape == (9, 9):
        kernel_size = (3, 3)
        start_loc, end_loc = (1, 1), (4, 7)
    elif env_shape == (3, 3):
        kernel_size = None
        start_loc, end_loc = (0, 0), (2, 1)
    else:
        raise RuntimeError(f'{env_shape} map not supported')

    env = RobotTaskAllocationEnv(env_shape, start_loc=start_loc, end_loc=end_loc)

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
    elif qmodel == 'ddqn':
        model = DDQNModel(env.observation_space,
                          env.action_space,
                          model_path=model_path,
                          kernel_size=kernel_size,
                          input_channel=env.TOTAL_CHANNEL)

    else:
        raise RuntimeError(f'{model} map not supported')

    return env, model


def train(min_episodes=500000, model_path='/tmp/model_path', env_shape=(1, 9), qmodel='ddqn'):
    if os.path.exists(model_path):
        ans = input(f'Warning: {model_path} exists, files will be rewrite if continue. [y/N]?')
        if ans not in ['y', 'Y']:
            return
        else:
            os.rmdir(model_path)
    else:
        os.mkdir(model_path)

    env, model = setup_env_and_model(model_path, env_shape, qmodel)

    evaluator = RewardEvaluator(model_path)

    episode_rewards = []

    start_time = time.perf_counter()

    for i in range(1, min_episodes):
        episode_reward = run_env_model(env, model)  #, with_render=True, render_clear=False)
        episode_rewards.append(episode_reward)

        model.train()

        min_ret = -1000000
        if i % 10 == 0:
            ret = run_env_model(env,
                                model,
                                train=False,
                                with_random=False,
                                with_render=True,
                                delay=0,
                                truck=(0, 5),
                                render_clear=False)
            

            mean_reward = np.array(episode_rewards).mean()
            evaluator.save_reward(i, mean_reward)
            model.save()

            passed_time = time.perf_counter() - start_time
            print(f'{i}: passed time={passed_time}, mean_reward={mean_reward}')

            if ret > 0:
                return


def replay(model_path='/tmp/model_path', env_shape=(1, 9), qmodel='rdqn'):

    env, model = setup_env_and_model(model_path, env_shape, qmodel)

    run_env_model(env, model, train=False, with_random=False, with_render=True, delay=0, render_clear=False)


def evaluate(model_path='/tmp/model_path'):
    evaluator = RewardEvaluator(model_path)
    evaluator.plot()