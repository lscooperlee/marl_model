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
        num_states = map_size # For simple fixed start, end situation, 2 robot state (picked up and not picked up)
        num_actions = 4

        self.start = None
        self.end = None
        self.truck = None
        self.map_shape = map_shape
        self.action_space = num_actions
        self.observation_space = num_states*self.TOTAL_CHANNEL

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
        is_moved = self._move(action)

        if not is_moved:
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
            reward = 50
            return self.map.copy().reshape(-1), reward, done, {"prob": prob}

        if self.step_count == 50: # timeout in 50 steps
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
        # super().reset(seed=seed)
        # self.start, self.end, self.truck = random.sample([(x, y)
        #                                       for x in range(self.map_shape[0])
        #                                       for y in range(self.map_shape[1])],
        #                                      k=3)
        self.start, self.end, self.truck = random.choices([(x, y)
                                              for x in range(self.map_shape[0])
                                              for y in range(self.map_shape[1])],
                                             k=3)

        self.step_count = 0
        if self.truck == self.start:
            self.is_picked = True
        else:
            self.is_picked = False


        self.map[:] = 0
        self.map[self.truck][self.TRUCK_CHANNEL] = 1
        self.map[self.start][self.START_CHANNEL] = 1
        self.map[self.end][self.END_CHANNEL] = 1
#        print(self.map)
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
        
    
    def _move(self, action): # 0=left, 1=right, 2=up, 3=down
        new_truck = list(self.truck)
        if action == 0:
                new_truck[1] += -1 if self.truck[1] != 0 else 0
        elif action == 1:
                new_truck[1] += 1 if self.truck[1] != self.map_shape[1]-1 else 0
        elif action == 2:
                new_truck[0] += -1 if self.truck[0] != 0 else 0
        elif action == 3:
                new_truck[0] += 1 if self.truck[0] != self.map_shape[0]-1 else 0

        is_moved = tuple(new_truck) != self.truck

        self.truck = tuple(new_truck)

        return is_moved

if __name__ == "__main__":
    env = RobotTaskAllocationEnv()
    env.reset()
    #env.truck = (0, 2)
    #print(env.map)
    print(env.step(0))
    env.render()
    print(env.step(0))
    env.render()