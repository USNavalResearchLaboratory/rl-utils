import gymnasium as gym
import numpy as np


def check_wraps(env, *, req_wrappers=(), invalid_wrappers=()):
    req_flags = dict(zip(req_wrappers, [False] * len(req_wrappers)))
    while isinstance(env, gym.Wrapper):
        for wrapper in req_wrappers:
            if isinstance(env, wrapper):
                req_flags[wrapper] = True
        if isinstance(env, tuple(invalid_wrappers)):
            return False
            # raise ValueError(f"Environment is wrapped by invalid {env.__class__}.")
        env = env.env
    # if not all(req_flags.values()):
    #     missing_keys = {k for k, v in req_flags.items() if not v}
    #     raise ValueError(f"Missing required wrappers: {', '.join(missing_keys)}")
    return all(req_flags.values())


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

    def __init__(self, env: gym.Env):
        super().__init__(env)
        if isinstance(self.observation_space, gym.spaces.MultiDiscrete):
            n_e = self.observation_space.nvec.max()  # TODO: disallow diff `nvec` vals?
        else:
            # TODO: allow `gym.spaces.Box` via n_encoding arg. `high`, `dtype`...
            raise TypeError
        self.observation_space = gym.spaces.MultiBinary(
            (*self.observation_space.shape, n_e)
        )
        self._eye = np.eye(n_e, dtype=self.observation_space.dtype)

    def observation(self, obs):
        """Perform one-hot encoding on the observation."""
        # out = np.zeros(
        #     self.observation_space.shape,
        #     dtype=self.observation_space.dtype,
        # )
        # idx = np.ix_(*map(np.arange, out.shape[:-1])) + (obs,)
        # out[idx] = 1
        # return out
        return self._eye[obs]


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
