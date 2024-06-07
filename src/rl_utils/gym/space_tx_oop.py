from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable, Sequence
from copy import copy
from dataclasses import dataclass
from functools import partial
from operator import itemgetter

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl_utils.gym import space_utils

# TODO: deprecate recurse wrappers?
# TODO: stack tx
# TODO: docs


class SpaceTransform(ABC):
    @abstractmethod
    def __call__(self, space): ...

    # def __call__(self, space: spaces.Space) -> tuple[spaces.Space, Callable]: ...


@dataclass(init=False)
class ChainedTransform(SpaceTransform):
    transforms: tuple[SpaceTransform]

    def __init__(self, *transforms):
        self.transforms = tuple(transforms)

    def __call__(self, space):
        fns = []
        for transform in self.transforms:
            space, fn = transform(space)
            fns.append(fn)

        def fn(x):
            for _fn in fns:
                x = _fn(x)
            return x

        return space, fn


@dataclass
class ItemTransform(SpaceTransform):
    transform: SpaceTransform
    key: int | str

    def __call__(self, space: spaces.Tuple | spaces.Dict):
        _spaces = space.spaces
        _spaces[self.key], sub_fn = self.transform(_spaces[self.key])
        space = space.__class__(_spaces)

        def fn(x):
            x = copy(x)
            x[self.key] = sub_fn(x[self.key])
            return x

        return space, fn


@dataclass
class RecursiveTransform(SpaceTransform):
    transform: SpaceTransform

    def __call__(self, space):
        def _build(space):
            if isinstance(space, spaces.Tuple):
                _spaces, funcs = zip(*map(_build, space.spaces))
                space = spaces.Tuple(_spaces)
                return space, funcs
            elif isinstance(space, spaces.Dict):
                keys, _spaces = zip(*space.spaces.items())
                _spaces, _funcs = zip(*map(_build, _spaces))
                space = spaces.Dict(OrderedDict(zip(keys, _spaces)))
                funcs = dict(zip(keys, _funcs))
                return space, funcs
            else:
                return self.transform(space)

        space, fns = _build(space)

        def fn(x):
            def _apply(x, fns):
                if isinstance(x, tuple):
                    return tuple(map(_apply, x, fns))
                elif isinstance(x, dict):
                    return {k: _apply(x[k], fns[k]) for k in x.keys()}
                else:
                    return fns(x)

            return _apply(x, fns)

        return space, fn


#
@dataclass
class GetItem(SpaceTransform):
    key: int | str

    def __call__(self, space: spaces.Tuple | spaces.Dict):
        space = space[self.key]
        fn = itemgetter(self.key)
        return space, fn


@dataclass
class Unnest(SpaceTransform):
    def __call__(self, space):
        space = space_utils.unnest(space)

        def fn(x):
            def unnest(x):
                if isinstance(x, tuple):
                    for x_i in x:
                        yield from unnest(x_i)
                elif isinstance(x, dict):
                    for x_i in x.values():
                        yield from unnest(x_i)
                else:
                    yield x

            return tuple(unnest(x))

        return space, fn


@dataclass
class Concat(SpaceTransform):
    axis: int | Sequence[int]

    def __call__(self, space: spaces.Tuple | spaces.Dict):
        if isinstance(space, spaces.Dict):
            _spaces = tuple(space.spaces.values())
        elif isinstance(space, spaces.Tuple):
            _spaces = space.spaces
        else:
            raise TypeError
        space = space_utils.concatenate(_spaces, axis=self.axis)

        def fn(x):
            if isinstance(x, dict):
                _arrs = tuple(x.values())
            else:
                _arrs = x
            return np.concatenate(_arrs, self.axis)

        return space, fn


@dataclass
class MoveAxis(SpaceTransform):
    source: int | Sequence[int]
    destination: int | Sequence[int]

    def __call__(self, space):
        space = space_utils.moveaxis(space, self.source, self.destination)
        fn = partial(np.moveaxis, source=self.source, destination=self.destination)
        return space, fn


@dataclass
class OneHot(SpaceTransform):
    redundant: bool = False

    def __call__(self, space):
        n_e = space.nvec.flatten()
        n_enc = sum(n_e)
        if not self.redundant:
            n_enc -= len(n_e)
        space = spaces.MultiBinary(n_enc)

        def fn(x):
            slices = []
            for n, x_i in zip(n_e, x.flatten()):
                s = np.zeros(n, dtype=space.dtype)
                s[x_i] = 1
                if not self.redundant:
                    s = s[1:]
                slices.append(s)
            return np.concatenate(slices)

        return space, fn


#
@dataclass
class OneHotGrid(SpaceTransform):
    # FIXME: move to `gridworld`

    redundant: bool = False

    def __call__(self, space):
        if not isinstance(space, spaces.MultiDiscrete):
            raise TypeError("Space must be `MultiDiscrete`.")

        nvec = space.nvec
        if nvec.ndim == 2:
            nvec = nvec[..., np.newaxis]
        elif not nvec.ndim == 3:
            raise ValueError("Space must be 2- or 3-dimensional.")

        n_e = []
        for i in range(nvec.shape[-1]):  # assumes channel-last array
            n_e.append(np.unique(nvec[..., i]).item())
        n_enc = sum(n_e)
        if not self.redundant:
            n_enc -= len(n_e)
        space = spaces.MultiBinary((*nvec.shape[:-1], n_enc))

        def fn(x):
            if x.ndim == 2:
                x = x[..., np.newaxis]

            def _encode(a, n_e):
                arr = np.zeros((*a.shape, n_e), dtype=space.dtype)
                idx = np.ix_(*map(np.arange, arr.shape[:-1])) + (a,)
                arr[idx] = 1
                start = int(not self.redundant)
                return arr[..., start:]

            slices = []
            for i, n_e_i in enumerate(n_e):
                s = _encode(x[..., i], n_e_i)
                slices.append(s)
            return np.concatenate(slices, axis=-1)

        return space, fn


#
class ObservationWrapper(gym.ObservationWrapper):
    # FIXME: move to separate module?
    # TODO: *args and chain?

    def __init__(self, env: gym.Env, transform: SpaceTransform):
        super().__init__(env)
        self._transform = transform
        self.observation_space, self._fn = transform(self.observation_space)

    def observation(self, observation):
        return self._fn(observation)
