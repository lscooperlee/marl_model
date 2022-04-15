import numpy as np

from ..env.gym_simple import RobotTaskAllocationEnv
from ..model.rdqn import RDQNModel
from .run import run_env_model


# def run(env, model, train=True, with_random=True, with_render=False):
#     state = env.reset()
#     if with_render:
#         env.render()

#     episode_reward = 0
#     while True:

#         action, repeat = model.get_next_action(state, with_random)
#         for _ in range(repeat):
#             next_state, reward, done, _ = env.step(action)
#             if done:
#                 break

#         if train:
#             model.memorize(state, action, reward, next_state, done)

#         if with_render:
#             env.render()

#         state = next_state

#         episode_reward += reward
#         # env.render()
#         if done:
#             # print('done')
#             break

#     return episode_reward


def train(min_episodes=5000):

    env = RobotTaskAllocationEnv(map_shape=(9, 9))
    model = RDQNModel((3, 3), env.action_space)

    episode_rewards = []

    for i in range(min_episodes):
        episode_reward = run(env, model)
        episode_rewards.append(episode_reward)

        model.train()

        if (i + 1) % 100 == 0:
            ret = run(env, model, train=False, with_random=False, with_render=False)
            print(i, np.array(episode_rewards).mean())

            model.save("saved_model.model")
