import random

import numpy as np

a = 0


class RobotTaskAllocationEnv:
    """
    """

    TRUCK_CHANNEL = 0
    START_CHANNEL = 1
    END_CHANNEL = 2
    TOTAL_CHANNEL = 3

    ACTION = ["left", "right", "up", "down", "stay"]

    REWARDS = {"finished": 100, "hit": -20, "timeout": -40, "move": -1}

    TIMEOUT = 100

    def __init__(self, map_shape=(3, 3), occupancy_grid_map=None, start_loc=None, end_loc=None, fixed_task=True):

        self.map = np.zeros((map_shape[0], map_shape[1], self.TOTAL_CHANNEL), dtype=np.float32)
        self.ogm = np.zeros(map_shape) if occupancy_grid_map is None else occupancy_grid_map

        map_size = map_shape[0] * map_shape[1]
        num_actions = len(self.ACTION)

        self.start_loc = start_loc
        self.end_loc = end_loc
        self.truck_loc = None

        assert (self.ogm[self.start_loc] == 0 if self.start_loc else True)
        assert (self.ogm[self.end_loc] == 0 if self.end_loc else True)
        assert (self.ogm[self.truck_loc] == 0 if self.truck_loc else True)

        self.map_shape = map_shape

        self.action_space = num_actions
        self.observation_space = map_shape

        self.is_picked = False

        self.step_count = 0

        self.fixed_task = fixed_task

    def step(self, action):

        self.step_count += 1

        # finished
        if self.is_picked and self.truck_loc == self.end_loc:
            return self.map.copy().reshape(-1), self.REWARDS['finished'], True

        # move
        self.map[self.truck_loc][self.TRUCK_CHANNEL] = 0
        hit_prob = self._move(action)
        # hit
        # not punish hit for now, meaning cound still action after hit
        # if random.random() < hit_prob:
        #     return self.map.copy().reshape(-1), self.REWARDS['hit'], True

        if self.truck_loc == self.start_loc:
            self.is_picked = True

        if self.is_picked:
            self.map[self.truck_loc][self.TRUCK_CHANNEL] = 2
        else:
            self.map[self.truck_loc][self.TRUCK_CHANNEL] = 1

        if self.step_count == self.TIMEOUT:
            return self.map.copy().reshape(-1), self.REWARDS['timeout'], True

        # finished
        if self.is_picked and self.truck_loc == self.end_loc:
            return self.map.copy().reshape(-1), self.REWARDS['finished'], True

        return self.map.copy().reshape(-1), self.REWARDS['move'], False

    def reset(self, truck=None):

        idx_space = [(x, y) for x in range(self.map_shape[0]) for y in range(self.map_shape[1])]
        idx_weight = (self.ogm == 0).astype(int).reshape(-1).tolist()
        if not self.fixed_task:
            self.start_loc, self.end_loc = random.choices(idx_space, k=2, weights=idx_weight)
        else:
            self.start_loc = self.start_loc if self.start_loc else random.choices(idx_space, k=1, weights=idx_weight)[0]
            self.end_loc = self.end_loc if self.end_loc else random.choices(idx_space, k=1, weights=idx_weight)[0]

        if truck is None:
            self.truck_loc = random.choices(idx_space, k=1, weights=idx_weight)[0]
        else:
            self.truck_loc = truck

        self.step_count = 0

        if self.truck_loc == self.start_loc:
            self.is_picked = True
        else:
            self.is_picked = False

        self.map[:] = 0
        self.map[self.truck_loc][self.TRUCK_CHANNEL] = 1
        self.map[self.start_loc][self.START_CHANNEL] = 1
        self.map[self.end_loc][self.END_CHANNEL] = 1

        return self.map.copy().reshape(-1)

    def render(self, clear=True):
        render_map = np.full(self.map_shape, '0', dtype='c')
        render_map[self.start_loc] = 'S'
        render_map[self.end_loc] = 'E'
        if self.is_picked:
            render_map[self.truck_loc] = 'P'
        else:
            render_map[self.truck_loc] = 'T'

        output = '-' * self.map_shape[1] + '\n'
        for row in render_map:
            output += row.tobytes().decode() + '\n'
        output += '+' * self.map_shape[1]

        print(output)

        if clear:
            for _ in range(len(render_map) + 2):
                print("\033[F\033[J", end="")

        return output

    def _move(self, action):  # 0=left, 1=right, 2=up, 3=down, 4=stay
        new_truck = list(self.truck_loc)
        hit_prob = 0

        if action == 0:
            if self.truck_loc[1] != 0:
                new_truck[1] += -1
            else:
                new_truck[1] += 0
                hit_prob = 1
        elif action == 1:
            if self.truck_loc[1] != self.map_shape[1] - 1:
                new_truck[1] += 1
            else:
                new_truck[1] += 0
                hit_prob = 1
        elif action == 2:
            if self.truck_loc[0] != 0:
                new_truck[0] += -1
            else:
                new_truck[0] += 0
                hit_prob = 1
        elif action == 3:
            if self.truck_loc[0] != self.map_shape[0] - 1:
                new_truck[0] += 1
            else:
                new_truck[0] += 0
                hit_prob = 1
        elif action == 4:
            pass
        else:
            raise RuntimeError(f'unknown action {action}')

        self.truck_loc = tuple(new_truck)
        hit_prob = hit_prob if hit_prob else self.ogm[self.truck_loc]

        return hit_prob