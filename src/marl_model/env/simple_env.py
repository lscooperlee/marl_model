import random

import numpy as np


class BaseTaskAllocationEnv:
    """
    """

    TRUCK_CHANNEL = 0
    START_CHANNEL = 1
    END_CHANNEL = 2
    TOTAL_CHANNEL = 3

    ACTION = ["left", "right", "up", "down", "stay"]

    REWARDS = {"finished": 100, "hit": -20, "timeout": -40, "move": -1}

    TIMEOUT = 100

    def __init__(self,
                 map_shape=(3, 3),
                 occupancy_grid_map=None,
                 start_loc=None,
                 end_loc=None,
                 fixed_task=True,
                 is_1d=True):

        self.map = np.zeros((map_shape[0], map_shape[1], self.TOTAL_CHANNEL), dtype=np.float32)
        self.ogm = np.zeros(map_shape) if occupancy_grid_map is None else occupancy_grid_map

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

        self.is_1d = is_1d

    def _format_map(self, map):
        if self.is_1d:
            return map.reshape(-1)
        return map

    def step(self, action):

        self.step_count += 1

        # finished
        if self.is_picked and self.truck_loc == self.end_loc:
            return self._format_map(self.map.copy()), self.REWARDS['finished'], True

        # move
        self.map[self.truck_loc][self.TRUCK_CHANNEL] = 0
        is_stopped = self._move(action)
        # hit
        # if random.random() < hit_prob:
        #     return self._format_map(self.map.copy()), self.REWARDS['hit'], True
        if is_stopped:
            return self._format_map(self.map.copy()), self.REWARDS['hit'], True

        if self.truck_loc == self.start_loc:
            self.is_picked = True

        if self.is_picked:
            self.map[self.truck_loc][self.TRUCK_CHANNEL] = 2
        else:
            self.map[self.truck_loc][self.TRUCK_CHANNEL] = 1

        if self.step_count == self.TIMEOUT:
            return self._format_map(self.map.copy()), self.REWARDS['timeout'], True

        # # finished
        # if self.is_picked and self.truck_loc == self.end_loc:
        #     return self._format_map(self.map.copy()), self.REWARDS['finished'], True

        return self._format_map(self.map.copy()), self.REWARDS['move'], False

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
            self.truck_loc = tuple(truck)
            assert (self.ogm[self.truck_loc] == 0)

        self.step_count = 0

        if self.truck_loc == self.start_loc:
            self.is_picked = True
        else:
            self.is_picked = False

        self.map[:] = 0
        self.map[self.truck_loc][self.TRUCK_CHANNEL] = 1
        self.map[self.start_loc][self.START_CHANNEL] = 1
        self.map[self.end_loc][self.END_CHANNEL] = 1

        return self._format_map(self.map.copy())

    def render(self, clear=True):
        render_map = (self.ogm * 10).astype(int).astype(str)
        render_map[render_map=='10'] = 'A'
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

        is_stopped = False
        hit_prob = hit_prob if hit_prob else self.ogm[tuple(new_truck)]
        if random.random() > hit_prob:
            self.truck_loc = tuple(new_truck)
        else:
            is_stopped = True

        return is_stopped


class RobotTaskAllocationEnvMTNS(BaseTaskAllocationEnv):

    def step(self, action):

        self.step_count += 1

        # finished
        if self.is_picked and self.truck_loc == self.end_loc:
            return self._format_map(self.map.copy()), self.REWARDS['finished'], True

        # move
        self.map[self.truck_loc][self.TRUCK_CHANNEL] = 0
        is_stopped = self._move(action)
        # hit
        # if random.random() < hit_prob:
        #     return self._format_map(self.map.copy()), self.REWARDS['hit'], True
        if is_stopped:
            return self._format_map(self.map.copy()), self.REWARDS['hit'], True

        if self.truck_loc == self.start_loc:
            self.is_picked = True

        if self.is_picked:
            self.map[self.truck_loc][self.TRUCK_CHANNEL] = 2
        else:
            self.map[self.truck_loc][self.TRUCK_CHANNEL] = 1

        if self.step_count == self.TIMEOUT:
            return self._format_map(self.map.copy()), self.REWARDS['timeout'], True

        # # finished
        # if self.is_picked and self.truck_loc == self.end_loc:
        #     return self._format_map(self.map.copy()), self.REWARDS['finished'], True

        return self._format_map(self.map.copy()), self.REWARDS['move'], False


class RobotTaskAllocationEnvMTR(BaseTaskAllocationEnv):

    def step(self, action):

        self.step_count += 1
        reward = self.REWARDS['move']

        # finished
        if self.is_picked and self.truck_loc == self.end_loc:
            return self._format_map(self.map.copy()), self.REWARDS['finished'], True

        # move
        self.map[self.truck_loc][self.TRUCK_CHANNEL] = 0
        hit_prob = self._move(action)
        reward = reward + hit_prob*self.REWARDS['hit']
        # hit
        # if random.random() < hit_prob:
            # return self._format_map(self.map.copy()), self.REWARDS['hit'], False
        # if is_stopped:
        #     return self._format_map(self.map.copy()), self.REWARDS['hit'], False

        if self.truck_loc == self.start_loc:
            self.is_picked = True

        if self.is_picked:
            self.map[self.truck_loc][self.TRUCK_CHANNEL] = 2
        else:
            self.map[self.truck_loc][self.TRUCK_CHANNEL] = 1

        if self.step_count == self.TIMEOUT:
            return self._format_map(self.map.copy()), self.REWARDS['timeout'], True

        #return self._format_map(self.map.copy()), self.REWARDS['move'], False
        return self._format_map(self.map.copy()), reward, False

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

        hit_prob = hit_prob if hit_prob else self.ogm[tuple(new_truck)]
        if hit_prob != 1:
            self.truck_loc = tuple(new_truck)

        return hit_prob

class RobotTaskAllocationMapStateEnv(BaseTaskAllocationEnv):
    TRUCK_CHANNEL = 0
    START_CHANNEL = 1
    END_CHANNEL = 2
    MAP_CHANNEL = 3
    TOTAL_CHANNEL = 4

    def _format_map(self, map):
        return super()._format_map(np.dstack((map[:,:,:-1], self.ogm)))