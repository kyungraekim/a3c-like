import concurrent.futures
import threading
import time

import gym
import numpy as np
from gym.spaces import Box


class MockEnv(gym.Env):
    def __init__(self):
        self.observation_space = Box(1, 100, shape=(4, 3), dtype=np.float32)
        self.action_space = Box(1, 100, shape=(4,), dtype=np.int)
        self.max_steps = 0
        self.steps = 0
        self.identity = 1

    def make(self, conf):
        self.identity = conf.get('identity', 1)
        self.max_steps = conf.get('steps', 10)

    def step(self, action):
        self.steps += 1
        done = False
        if self.steps >= self.max_steps:
            done = True

        time.sleep(1)
        return np.ones(shape=(4,)) * self.identity, 1, done, {}

    def reset(self):
        self.steps = 0
        return np.ones(shape=(4,))

    def render(self, mode='human'):
        pass


def concurrent_run():
    envs = []
    for i in range(10):
        e = MockEnv()
        e.make({
            'identity': i,
            'steps': 1
        })
        envs.append(e)

    def step(iter):
        return envs[iter].step(envs[iter].action_space.sample())[0]

    def tt():
        return int(time.time() * 1000)

    a = tt()
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(step, i) for i in range(len(envs))]
        data = [future.result() for future in futures]
        print(data)
    print(tt() - a)

    a = tt()
    threads = []
    for i in range(len(envs)):
        threads.append(threading.Thread(target=step, args=(i,), daemon=True))
    for th in threads:
        th.start()

    for th in threads:
        th.join()
    print(data)
    print(tt() - a)


concurrent_run()
