import operator
from collections import OrderedDict
from collections.abc import Callable, Sequence
from copy import copy
from functools import partial
from typing import TypeVar, overload

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl_utils.gym.spaces import utils as space_utils

# TODO: deprecate recurse wrappers
# TODO: stack tx
# TODO: docs

# TODO: apply condition for `recursive_tx`!?


T_si = TypeVar("T_si")
T_so = TypeVar("T_so")
T_tx_io = Callable[[T_si], tuple[T_so, Callable]]
T_tx_i = T_tx_io[T_si, spaces.Space]
T_tx = T_tx_i[spaces.Space]


#
def chain(*transforms) -> T_tx:
    def tx(space):
        fns = []
        for transform in transforms:
            space, fn = transform(space)
            fns.append(fn)

        def fn(x):
            for _fn in fns:
                x = _fn(x)
            return x

        return space, fn

    return tx


# @overload  # TODO
# def item_apply(transform, key: int) -> T_tx_io[spaces.Tuple, spaces.Tuple]: ...


# @overload
# def item_apply(transform, key: str) -> T_tx_io[spaces.Dict, spaces.Dict]: ...


@overload
def item_apply(transform, key: int) -> T_tx_i[spaces.Tuple]: ...


@overload
def item_apply(transform, key: str) -> T_tx_i[spaces.Dict]: ...


def item_apply(transform, key: int | str) -> T_tx:
    def tx(space):
        _spaces = space.spaces
        _spaces[key], sub_fn = transform(_spaces[key])
        space = space.__class__(_spaces)

        def fn(x):
            x = copy(x)
            x[key] = sub_fn(x[key])
            return x

        return space, fn

    return tx


def recursive_apply(transform) -> T_tx_i[spaces.Tuple | spaces.Dict]:
    def tx(space):
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
                return transform(space)

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

    return tx


#
@overload
def getitem(key: int) -> T_tx_i[spaces.Tuple]: ...


@overload
def getitem(key: str) -> T_tx_i[spaces.Dict]: ...


def getitem(key: int | str) -> T_tx:
    def tx(space):
        space = space[key]
        fn = operator.itemgetter(key)
        return space, fn

    return tx


def unnest() -> T_tx_io[spaces.Space, spaces.Tuple]:  # TODO
    def tx(space: spaces.Space):
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

    return tx


def stack(
    axis: int | Sequence[int] | None,
) -> T_tx_io[spaces.Tuple | spaces.Dict, spaces.Tuple]:
    def tx(space):
        if isinstance(space, spaces.Dict):
            _spaces = tuple(space.spaces.values())
        elif isinstance(space, spaces.Tuple):
            _spaces = space.spaces
        else:
            raise TypeError
        space = space_utils.stack(_spaces, axis=axis)

        def fn(x):
            if isinstance(x, dict):
                _arrs = tuple(x.values())
            else:
                _arrs = x
            return np.stack(_arrs, axis)

        return space, fn

    return tx


def concat(
    axis: int | Sequence[int] | None,
) -> T_tx_io[spaces.Tuple | spaces.Dict, spaces.Tuple]:
    def tx(space):
        if isinstance(space, spaces.Dict):
            _spaces = tuple(space.spaces.values())
        elif isinstance(space, spaces.Tuple):
            _spaces = space.spaces
        else:
            raise TypeError
        space = space_utils.concatenate(_spaces, axis=axis)

        def fn(x):
            if isinstance(x, dict):
                _arrs = tuple(x.values())
            else:
                _arrs = x
            return np.concatenate(_arrs, axis)

        return space, fn

    return tx


T_move = spaces.Box | spaces.MultiDiscrete | spaces.MultiBinary


def move_axis(
    source: int | Sequence[int], destination: int | Sequence[int]
) -> T_tx_io[T_move, T_move]:
    def tx(space):
        space = space_utils.moveaxis(space, source, destination)
        fn = partial(np.moveaxis, source=source, destination=destination)
        return space, fn

    return tx


def one_hot(
    redundant: bool = False,
) -> T_tx_io[spaces.MultiDiscrete, spaces.MultiBinary]:
    def tx(space):
        n_e = space.nvec.flatten()
        n_enc = sum(n_e)
        if not redundant:
            n_enc -= len(n_e)
        space = spaces.MultiBinary(n_enc)

        def fn(x):
            slices = []
            for n, x_i in zip(n_e, x.flatten()):
                s = np.zeros(n, dtype=space.dtype)
                s[x_i] = 1
                if not redundant:
                    s = s[1:]
                slices.append(s)
            return np.concatenate(slices)

        return space, fn

    return tx


#
def one_hot_grid(  # noqa: C901
    redundant: bool = False,
) -> T_tx_io[spaces.MultiDiscrete, spaces.MultiBinary]:
    def tx(space):
        nvec = space.nvec
        if nvec.ndim == 2:
            nvec = nvec[..., np.newaxis]
        elif not nvec.ndim == 3:
            raise ValueError("Space must be 2- or 3-dimensional.")

        n_e = []
        for i in range(nvec.shape[-1]):  # assumes channel-last array
            n_e.append(np.unique(nvec[..., i]).item())
        n_enc = sum(n_e)
        if not redundant:
            n_enc -= len(n_e)
        space = spaces.MultiBinary((*nvec.shape[:-1], n_enc))

        def fn(x):
            if x.ndim == 2:
                x = x[..., np.newaxis]

            def _encode(a, n_e):
                arr = np.zeros((*a.shape, n_e), dtype=space.dtype)
                idx = np.ix_(*map(np.arange, arr.shape[:-1])) + (a,)
                arr[idx] = 1
                start = int(not redundant)
                return arr[..., start:]

            slices = []
            for i, n_e_i in enumerate(n_e):
                s = _encode(x[..., i], n_e_i)
                slices.append(s)
            return np.concatenate(slices, axis=-1)

        return space, fn

    return tx


#
class ObservationWrapper(gym.ObservationWrapper):
    # FIXME: move to separate module

    def __init__(self, env: gym.Env, transform):
        super().__init__(env)
        self.observation_space, self._fn = transform(self.observation_space)

    def observation(self, observation):
        return self._fn(observation)
