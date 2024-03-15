from abc import ABC, abstractmethod
from collections import OrderedDict

import gymnasium as gym
import numpy as np

from rl_utils.gym import space_utils


class ResetOptionsWrapper(gym.Wrapper):
    def __init__(self, env, *, options=None):
        super().__init__(env)
        if options is None:
            options = {}
        self.options = options

    def reset(self, *, seed=None, options=None):
        if options is None:
            options = {}
        options = self.options | options
        return self.env.reset(seed=seed, options=options)


class RecurseObsWrapper(gym.ObservationWrapper, ABC):
    def __init__(self, env: gym.Env):
        super().__init__(env)

        self.obs_proc = self._build_fn(self.observation_space)
        self.observation_space = self._build_space(self.observation_space)

    def _build_fn(self, space):
        if isinstance(space, gym.spaces.Tuple):
            return tuple(map(self._build_fn, space.spaces))
        elif isinstance(space, gym.spaces.Dict):
            return {k: self._build_fn(s) for k, s in space.spaces.items()}
        else:
            return self._proc_space(space)[1]

    def _build_space(self, space):
        if isinstance(space, gym.spaces.Tuple):
            _spaces = tuple(self._build_space(s) for s in space.spaces)
            return gym.spaces.Tuple(_spaces)
        elif isinstance(space, gym.spaces.Dict):
            _spaces = {k: self._build_space(s) for k, s in space.spaces.items()}
            return gym.spaces.Dict(OrderedDict(_spaces))
        else:
            return self._proc_space(space)[0]

    @abstractmethod
    def _proc_space(self, space):
        raise NotImplementedError

    def observation(self, obs):
        def apply(obs, func):
            if isinstance(obs, tuple):
                return tuple(map(apply, obs, func))
            elif isinstance(obs, dict):
                return {k: apply(obs[k], func[k]) for k in obs.keys()}
            else:
                return func(obs)

        return apply(obs, self.obs_proc)


class RecurseOneHotWrapper(RecurseObsWrapper):
    def __init__(self, env: gym.Env, redundant: bool = False):
        self.redundant = redundant
        super().__init__(env)

    def _proc_space(self, space):
        if not isinstance(space, gym.spaces.MultiDiscrete):
            return space, lambda x: x  # TODO: super call?

        _n_e = space.nvec.flatten()
        n_enc = sum(_n_e)
        if not self.redundant:
            n_enc -= len(_n_e)
        space = gym.spaces.MultiBinary(n_enc)

        def _proc_func(obs):
            slices = []
            for n, obs_i in zip(_n_e, obs.flatten()):
                s = np.zeros(n, dtype=space.dtype)
                s[obs_i] = 1
                if not self.redundant:
                    s = s[1:]
                slices.append(s)
            return np.concatenate(slices)

        return space, _proc_func


class RecurseMoveAxisWrapper(RecurseObsWrapper):
    def __init__(self, env, source, destination):
        self.source, self.destination = source, destination
        super().__init__(env)

    def _proc_space(self, space):
        space = space_utils.moveaxis(space, self.source, self.destination)

        def _proc_func(obs):
            return np.moveaxis(obs, self.source, self.destination)

        return space, _proc_func


class MoveAxisWrapper(gym.ObservationWrapper):
    def __init__(self, env, source, destination):
        super().__init__(env)
        self.source, self.destination = source, destination

        self.observation_space = space_utils.moveaxis(
            self.observation_space, source, destination
        )

    def observation(self, obs):
        return np.moveaxis(obs, self.source, self.destination)


class OneHotWrapper(gym.ObservationWrapper):
    """One-hot encodes the observation."""

    def __init__(self, env: gym.Env, redundant: bool = False):
        super().__init__(env)
        if not isinstance(self.observation_space, gym.spaces.MultiDiscrete):
            raise TypeError

        self.redundant = redundant

        self._n_e = self.observation_space.nvec.flatten()
        n_enc = sum(self._n_e)
        if not redundant:
            n_enc -= len(self._n_e)
        self.observation_space = gym.spaces.MultiBinary(n_enc)

    def observation(self, obs):
        """Perform one-hot encoding on the observation."""
        slices = []
        for n, obs_i in zip(self._n_e, obs.flatten()):
            s = np.zeros(n, dtype=self.observation_space.dtype)
            s[obs_i] = 1
            if not self.redundant:
                s = s[1:]
            slices.append(s)
        return np.concatenate(slices)


class StepPenalty(gym.RewardWrapper):
    def __init__(self, env: gym.Env, penalty: float = 0.0):
        super().__init__(env)
        self.penalty = penalty

    def reward(self, reward):
        return reward - self.penalty


# class DiscountedReward(RewardWrapper):
#     def __init__(self, env: Env, gamma: float = 1.0):
#         super().__init__(env)
#         self.gamma = gamma
#         self.i_step = 0

#     def reset(self, *args, **kwargs):
#         super().reset(*args, **kwargs)
#         self.i_step = 0

#     def reward(self, reward):
#         reward *= self.gamma**self.i_step
#         self.i_step += 1
#         return reward


# def check_wraps(env, *, req_wrappers=(), invalid_wrappers=()):
#     req_flags = dict(zip(req_wrappers, [False] * len(req_wrappers)))
#     while isinstance(env, gym.Wrapper):
#         for wrapper in req_wrappers:
#             if isinstance(env, wrapper):
#                 req_flags[wrapper] = True
#         if isinstance(env, tuple(invalid_wrappers)):
#             return False
#             # raise ValueError(f"Environment is wrapped by invalid {env.__class__}.")
#         env = env.env
#     # if not all(req_flags.values()):
#     #     missing_keys = {k for k, v in req_flags.items() if not v}
#     #     raise ValueError(f"Missing required wrappers: {', '.join(missing_keys)}")
#     return all(req_flags.values())
