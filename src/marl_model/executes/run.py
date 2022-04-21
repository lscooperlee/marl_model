import os
import numpy as np
import time

from ..env.env_map import SimpleEnvMap, SimpleEnvMap1, ProbEnvMap, ProbEnvMap1
from ..env.simple_env import RobotTaskAllocationEnvMTNS
from ..env.simple_env import RobotTaskAllocationEnvMTR
from ..env.simple_env import RobotTaskAllocationMapStateEnv
from ..model.dqn import DQNModel
from ..model.rdqn import RDQNModel
from ..model.ddqn import DDQNModel
from ..model.cdqn import CDQNModel
from ..evaluation.evaluator import RewardEvaluator


def run_env_model(env,
                  model,
                  train=True,
                  with_random=True,
                  with_render=False,
                  render_clear=True,
                  delay=0,
                  truck_loc=None):

    state = env.reset(truck_loc)
    if with_render:
        env.render(render_clear)

    episode_reward = 0
    save_state = []

    while True:
        if train:
            action_repeat = model.get_next_action(state, with_random)
            try:
                action, repeat = action_repeat
            except TypeError:
                action, repeat = action_repeat, 1

            for _ in range(repeat):
                next_state, reward, done = env.step(action)
                if done:
                    break

            model.memorize(state, action, reward * repeat, next_state, done)
        else:
            while True:
                action_repeat = model.get_next_action(state, with_random)
                try:
                    action, repeat = action_repeat
                except TypeError:
                    action = action_repeat
                    break
            next_state, reward, done = env.step(action)

        save_state.append((action, state))

        state = next_state
        episode_reward += reward

        if with_render:
            env.render(render_clear)

        if done:
            break

        if delay:
            time.sleep(delay)

    return episode_reward, save_state


def setup_env_and_model(model_path, qmodel, env_shape, env_map=None, env_name=None, resume_mode=None, fix_task=True):

    model = None
    env_cls = None

    if resume_mode == 'all':
        e = RewardEvaluator.load(model_path)
        return e.env, e.model, e.data[-1][1]
    elif resume_mode == 'model':
        e = RewardEvaluator.load(model_path)
        model = e.model
        model.replay_memory.clear()
        env_cls = type(e.env)
        env_shape = e.env.map_shape

    # fix start, end location for simplicity
    if env_shape == (1, 9):
        kernel_size = (1, 3)
        start_loc, end_loc = (0, 2), (0, 6)
    elif env_shape == (9, 9):
        kernel_size = (3, 3)
        start_loc, end_loc = (3, 2), (7, 4)
    elif env_shape == (3, 3):
        kernel_size = None
        start_loc, end_loc = (0, 0), (2, 1)
    elif env_shape == (5, 5):
        kernel_size = None
        start_loc, end_loc = (0, 0), (2, 1)
    elif env_shape == (10, 10):
        kernel_size = None
        start_loc, end_loc = (3, 2), (7, 4)
    else:
        raise RuntimeError(f'{env_shape} map not supported')

    if env_map == 'simple':
        ogm = SimpleEnvMap(env_shape)
    if env_map == 'prob':
        ogm = ProbEnvMap(env_shape)
    elif env_map == 'simple_update':
        ogm = SimpleEnvMap1(env_shape)
    elif env_map == 'prob_update':
        ogm = ProbEnvMap1(env_shape)
    elif env_map == 'none':
        ogm = None
    
    if env_cls is None:
        if env_name == 'mts':
            env_cls = RobotTaskAllocationMapStateEnv
        elif env_name == 'mtns':
            env_cls = RobotTaskAllocationEnvMTNS
        elif env_name == 'mtr':
            env_cls = RobotTaskAllocationEnvMTR
        else:
            raise RuntimeError(f'{env} not supported')

    if fix_task:
        env = env_cls(env_shape, start_loc=start_loc, end_loc=end_loc, occupancy_grid_map=ogm)
    else:
        env = env_cls(env_shape, occupancy_grid_map=ogm)

    if model is None:
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
        elif qmodel == 'cdqn':
            if env_shape[0] > 9 and env_shape[1] > 9:
                model = CDQNModel(env.observation_space,
                                env.action_space,
                                model_path=model_path,
                                kernel_size=kernel_size,
                                input_channel=env.TOTAL_CHANNEL)
                env.is_1d = False
            else:
                raise RuntimeError('CDQN requires map size larger than 9x9.')
        else:
            raise RuntimeError(f'{model} map not supported')
    else:
        if str(model).startswith('cdqn'):
            env.is_1d = False

    return env, model, 0


def train(n_episodes=100000,
          model_path='/tmp/',
          qmodel='rdqn',
          env_shape=(1, 9),
          env_map=None,
          env_name=None,
          resume_mode=None,
          fix_task=True):

    if os.path.exists(model_path):
        ans = input(f'Warning: {model_path} exists, files may be rewrite. y/N')
        if ans not in ['y', 'Y']:
            return
    else:
        os.mkdir(model_path)

    env, model, count = setup_env_and_model(model_path, qmodel, env_shape, env_map, env_name, resume_mode, fix_task)

    evaluator = RewardEvaluator()

    episode_rewards = []

    for i in range(count + 1, n_episodes + count):
        episode_reward, _ = run_env_model(env, model)
        episode_rewards.append(episode_reward)

        model.train()

        if i % 100 == 0:
            truck_loc = env.map.shape[0] // 2, env.map.shape[1] // 2
            eval_reward, _ = run_env_model(
                env,
                model,
                train=False,
                with_random=False,
                with_render=False,
                # with_render=True,
                delay=0,
                render_clear=False,
                truck_loc=truck_loc)

            mean_reward = np.array(episode_rewards).mean()
            evaluator.save(i, mean_reward, eval_reward, model, env, model_path)
            print(i, mean_reward, eval_reward)


def replay(model_path, truck_loc, env_map):

    if env_map:
        env, model, count = setup_env_and_model(model_path, None, None, resume_mode='model', env_map=env_map)
    else:
        env, model, count = setup_env_and_model(model_path, None, None, resume_mode='all')


    if truck_loc is None:
        truck_loc = env.map.shape[0] // 2, env.map.shape[1] // 2

    ret, actions = run_env_model(env,
                                 model,
                                 train=False,
                                 with_random=False,
                                 with_render=True,
                                 delay=0.5,
                                 render_clear=False,
                                 truck_loc=truck_loc)


def evaluate(model_path='/tmp/model_path'):
    evaluator = RewardEvaluator()
    evaluator.plot(model_path)
