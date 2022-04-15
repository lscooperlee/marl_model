import random
from simple_env import RobotTaskAllocationEnv


r = RobotTaskAllocationEnv((2, 10))
r.reset()
#print(r.start, r.end, r.truck)

for _ in range(10):
    a = random.choice(range(r.action_space))
    print(a)
    c = r.step(a)
    print(c)
    r.render()