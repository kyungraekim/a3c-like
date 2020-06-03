import functools
import time

import gym

from env_util.concurrent_handler import ThreadController
from util import config


def make(env_name, conf):
    def _init(name, env_config):
        env = gym.make(name)
        env.make(env_config)
        return env

    return functools.partial(_init, name=env_name, env_config=conf)


def tt():
    return int(time.time() * 1000)


def main():
    config_list = config.read_actors('actor-list.json')
    create_env_fns = [make('mmmock-v1', {
        'identity': i,
        'steps': config_list[i].total_steps
    }) for i in range(len(config_list))]
    env = ThreadController(create_env_fns, start_method='forkserver')
    obs = env.reset()
    done = False
    cnt = 0
    while not done:
        a = tt()
        action = env.action_space.sample()
        print('step', cnt)
        obs, rew, dones, info = env.step(action)
        done = dones.all()
        print(obs)
        cnt += 1
        print('elapsed:', tt() - a)
    print('finished')


if __name__ == '__main__':
    main()
