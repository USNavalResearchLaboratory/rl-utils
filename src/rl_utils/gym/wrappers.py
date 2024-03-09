import gymnasium as gym
import numpy as np


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


class MoveAxisWrapper(gym.ObservationWrapper):
    # TODO: refactor w/ single dispatch

    def __init__(self, env, source, destination):
        super().__init__(env)
        self.s, self.d = source, destination

        space = self.observation_space
        if isinstance(space, gym.spaces.Box):
            space = gym.spaces.Box(
                low=np.moveaxis(space.low, self.s, self.d),
                high=np.moveaxis(space.high, self.s, self.d),
                dtype=space.dtype,
            )
        elif isinstance(space, gym.spaces.MultiDiscrete):
            space = gym.spaces.MultiDiscrete(
                nvec=np.moveaxis(space.nvec, self.s, self.d), dtype=space.dtype
            )
        elif isinstance(space, gym.spaces.MultiBinary):
            n = np.moveaxis(np.ones(space.shape), self.s, self.d).shape  # TODO: better?
            space = gym.spaces.MultiBinary(n=n)
        self.observation_space = space

    def observation(self, obs):
        return np.moveaxis(obs, self.s, self.d)


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
