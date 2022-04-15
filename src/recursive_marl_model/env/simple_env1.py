import random
from typing import Optional

import numpy as np


class RobotTaskAllocationEnv:
    """
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}
    TRUCK_CHANNEL = 0
    START_CHANNEL = 1
    END_CHANNEL = 2
    ACTION_CHANNEL = 3
    TOTAL_CHANNEL = 4

    def __init__(self, map_shape=(3, 3)):

        self.map = np.zeros((map_shape[0], map_shape[1], self.TOTAL_CHANNEL), dtype=np.float32)

        map_size = map_shape[0] * map_shape[1]
        # num_states = map_size * (map_size - 1)*(map_size - 2) # if use number as state
        # num_states = map_size # if use array as state
        num_states = map_size  # For simple fixed start, end situation, 2 robot state (picked up and not picked up)
        num_actions = 5

        self.start = None
        self.end = None
        self.truck = None
        self.map_shape = map_shape
        self.action_space = num_actions
        self.observation_space = num_states * self.TOTAL_CHANNEL

        # self.action_space.shape = (num_actions, )
        # self.observation_space.shape = map_shape

        self.is_picked = False

        self.step_count = 0

    def step(self, action):
        prob = 1

        self.step_count += 1
        done = False
        reward = -1

        if self.is_picked and self.truck == self.end:
            done = True
            reward = 50
            return self.map.copy().reshape(-1), reward, done, {"prob": prob}

        self.map[self.truck][self.TRUCK_CHANNEL] = 0
        is_hit_wall = self._move(action)

        if is_hit_wall:
            done = True
            reward = -20
            return self.map.copy().reshape(-1), reward, done, {"prob": prob}

        if self.truck == self.start:
            self.is_picked = True

        if self.is_picked:
            self.map[self.truck][self.TRUCK_CHANNEL] = 2
        else:
            self.map[self.truck][self.TRUCK_CHANNEL] = 1

        if self.is_picked and self.truck == self.end:
            done = True
            reward = 100
            return self.map.copy().reshape(-1), reward, done, {"prob": prob}

        if self.step_count == 150:  # timeout in 100 steps
            done = True
            reward = -10
            return self.map.copy().reshape(-1), reward, done, {"prob": prob}

        return self.map.copy().reshape(-1), reward, done, {"prob": prob}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        # without overlap
        # self.start, self.end, self.truck = random.sample([(x, y)
        #                                       for x in range(self.map_shape[0])
        #                                       for y in range(self.map_shape[1])],
        #                                      k=3)

        # without overlap
        # self.start, self.end, self.truck = random.choices(
        #     [(x, y) for x in range(self.map_shape[0])
        #      for y in range(self.map_shape[1])],
        #     k=3)

        # fix with start/end for simplicity
        # if self.map_shape[0] == 3:
        #     self.start = (0, 1)
        #     self.end = (self.map_shape[0]-1, self.map_shape[1]-1)
        #     self.truck = random.choices([(x, y) for x in range(self.map_shape[0]) for y in range(self.map_shape[1])])[0]

        # fix with start/end for simplicity for level 1
        if self.map_shape[1] == 9 and self.map_shape[0] == 1:
            self.start = random.choices([(0, y) for x in range(3) for y in range(3)])[0]
            self.end = random.choices([(0, y) for x in range(6, 9) for y in range(3, 6)])[0]
            self.truck = random.choices([(0, y) for x in range(self.map_shape[0]) for y in range(self.map_shape[1])])[0]
        elif self.map_shape[1] == 9:
             self.start = random.choices([(x, y) for x in range(3) for y in range(3)])[0]
             self.end = random.choices([(x, y) for x in range(6, 9) for y in range(3, 6)])[0]
             self.truck = random.choices([(x, y) for x in range(self.map_shape[0]) for y in range(self.map_shape[1])])[0]


        self.step_count = 0
        if self.truck == self.start:
            self.is_picked = True
        else:
            self.is_picked = False

        self.map[:] = 0
        self.map[self.truck][self.TRUCK_CHANNEL] = 1
        self.map[self.start][self.START_CHANNEL] = 1
        self.map[self.end][self.END_CHANNEL] = 1

        return self.map.copy().reshape(-1)

    def render(self, clear=True):
        render_map = np.full(self.map_shape, '0', dtype='c')
        render_map[self.start] = 'S'
        render_map[self.end] = 'E'
        if self.is_picked:
            render_map[self.truck] = 'P'
        else:
            render_map[self.truck] = 'T'

        for row in render_map:
            print(row.tobytes().decode())

        if clear:
            for row in render_map:
                print("\033[F\033[J", end="")
        else:
            print('-' * self.map_shape[0])

    def _move(self, action):  # 0=left, 1=right, 2=up, 3=down, 4=stay
        new_truck = list(self.truck)
        is_hit_wall = False
        if action == 0:
            if self.truck[1] != 0:     
                new_truck[1] += -1
            else:
                new_truck[1] += 0
                is_hit_wall = True
        elif action == 1:
            if self.truck[1] != self.map_shape[1] - 1: 
                new_truck[1] += 1
            else:
                new_truck[1] += 0
                is_hit_wall = True
        elif action == 2:
            if self.truck[0] != 0:
                new_truck[0] += -1
            else:
                new_truck[0] += 0
                is_hit_wall = True
        elif action == 3:
            if self.truck[0] != self.map_shape[0] - 1:
                new_truck[0] += 1
            else:
                new_truck[0] += 0
                is_hit_wall = True
        elif action == 4:
            pass

        self.truck = tuple(new_truck)

        return is_hit_wall


if __name__ == "__main__":
    env = RobotTaskAllocationEnv()
    env.reset()
    #env.truck = (0, 2)
    #print(env.map)
    print(env.step(0))
    env.render()
    print(env.step(0))
    env.render()