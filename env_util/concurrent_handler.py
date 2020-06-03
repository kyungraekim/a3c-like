import concurrent.futures
import multiprocessing
import pickle
from collections import OrderedDict

import cloudpickle
import gym
import numpy as np


class CloudPickleWrapper(object):
    def __init__(self, var):
        """
        Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)

        :param var: (Any) the variable you wish to wrap for pickling with cloudpickle
        """
        self.var = var

    def __getstate__(self):
        return cloudpickle.dumps(self.var)

    def __setstate__(self, obs):
        self.var = pickle.loads(obs)


def _worker(remote, parent_remote, *env_fn_wrappers):
    parent_remote.close()
    env_list = [fn.var() for fn in env_fn_wrappers]

    def step(index, action):
        return env_list[index].step(action)

    def reset(index):
        return env_list[index].reset()

    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [executor.submit(step, i, data[i]) for i in range(len(env_list))]
                    data_list = [future.result() for future in futures]
                obses, rewards, dones, infos = zip(*data_list)
                obses = _flatten_obs(obses, env_list[0].observation_space)
                rewards = np.stack(rewards)
                dones = np.stack(dones)
                remote.send((obses, rewards, dones, infos))
                print('send step')
            elif cmd == 'seed':
                remote.send(env_list[0].seed(data))
            elif cmd == 'reset':
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [executor.submit(reset, i) for i in range(len(env_list))]
                    observations = [future.result() for future in futures]
                observations = _flatten_obs(observations, env_list[0].observation_space)
                remote.send(observations)
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces':
                print(env_list[0].observation_space, env_list[0].action_space)
                remote.send((env_list[0].observation_space, env_list[0].action_space))
            else:
                raise NotImplementedError
        except EOFError:
            break


class ThreadController(object):
    def __init__(self, env_fns, start_method=None):
        self.waiting = False
        self.closed = False
        self.n_envs = len(env_fns)

        if start_method is None:
            forkserver_available = 'forkserver' in multiprocessing.get_all_start_methods()
            start_method = 'forkserver' if forkserver_available else 'spawn'
        ctx = multiprocessing.get_context(start_method)

        self.remote, self.work_remotes = ctx.Pipe(duplex=True)

        env_fn_list = [CloudPickleWrapper(env_fn) for env_fn in env_fns]
        args = [self.work_remotes, self.remote]
        args.extend(env_fn_list)
        process = ctx.Process(target=_worker, args=args, daemon=True)
        process.start()
        self.process = process
        self.work_remotes.close()

        self.remote.send(('get_spaces', None))
        observation_space, action_space = self.remote.recv()

        self.observation_space = self._extend_dim(observation_space)
        self.action_space = self._extend_dim(action_space)

    def _extend_dim(self, box):
        if isinstance(box, gym.spaces.Box):
            low = np.min(box.low.astype(int))
            high = np.max(box.high.astype(int))
            shape = (self.n_envs,) + box.shape
            return gym.spaces.Box(low=low, high=high, shape=shape, dtype=box.dtype)
        raise NotImplementedError('Only Box-type space is supported')

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions):
        self.remote.send(('step', actions))
        print('req')
        self.waiting = True

    def step_wait(self):
        results = self.remote.recv()
        self.waiting = False
        print('wait')
        return results

    def reset(self):
        self.remote.send(('reset', None))
        obs = self.remote.recv()
        return obs

    def close(self):
        if self.closed:
            return
        if self.waiting:
            self.remote.recv()
        self.remote.send(('close', None))
        self.process.join()
        self.closed = True


def _flatten_obs(obs, space):
    if isinstance(space, gym.spaces.Dict):
        assert isinstance(space.spaces, OrderedDict), "Dict space must have ordered subspaces"
        assert isinstance(obs[0], dict), \
            "non-dict observation for environment with Dict observation space"
        return OrderedDict([(k, np.stack([o[k] for o in obs])) for k in space.spaces.keys()])
    elif isinstance(space, gym.spaces.Tuple):
        assert isinstance(obs[0], tuple), \
            "non-tuple observation for environment with Tuple observation space"
        obs_len = len(space.spaces)
        return tuple((np.stack([o[i] for o in obs]) for i in range(obs_len)))
    else:
        return np.stack(obs)
