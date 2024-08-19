"""Additional `gymnasium` spaces and utilities."""

from collections import OrderedDict
from functools import singledispatch, wraps

import numpy as np
from gymnasium.spaces import (
    Box,
    Dict,
    Discrete,
    MultiBinary,
    MultiDiscrete,
    Space,
    Tuple,
)


def recurse(fn):
    @wraps(fn)
    def _recursive_fn(space, *args, **kwargs):
        if isinstance(space, Tuple):
            _spaces = tuple(_recursive_fn(s, *args, **kwargs) for s in space.spaces)
            return Tuple(_spaces)
        elif isinstance(space, Dict):
            _spaces = {
                k: _recursive_fn(s, *args, **kwargs) for k, s in space.spaces.items()
            }
            return Dict(OrderedDict(_spaces))
        else:
            return fn(space, *args, **kwargs)

    return _recursive_fn


# Transforms
@singledispatch
def reshape(space: Space, shape):
    """Reshape space."""
    raise NotImplementedError


@reshape.register
def _(space: Box, shape):
    low, high = space.low.reshape(shape), space.high.reshape(shape)
    return Box(low, high, dtype=space.dtype)  # type: ignore[arg-type]


@reshape.register
def _(space: MultiDiscrete, shape):
    return MultiDiscrete(space.nvec.reshape(shape))


@reshape.register
def _(space: Discrete, shape):
    return reshape(MultiDiscrete(np.array(space.n)), shape)


@reshape.register
def _(space: MultiBinary, shape):
    _tmp = np.empty(space.shape, space.dtype)
    _tmp = _tmp.reshape(shape)  # raises exception if shapes not compatible
    return MultiBinary(_tmp.shape)


@singledispatch
def broadcast_to(space: Space, shape):
    """Broadcast space to new shape."""
    raise NotImplementedError


@broadcast_to.register
def _(space: Box, shape):
    low, high = (
        np.broadcast_to(space.low, shape),
        np.broadcast_to(space.high, shape),
    )
    return Box(low, high, dtype=space.dtype)  # type: ignore[arg-type]


@broadcast_to.register
def _(space: MultiDiscrete, shape):
    return MultiDiscrete(np.broadcast_to(space.nvec, shape))


@broadcast_to.register
def _(space: Discrete, shape):
    # if shape == ():
    #     return space
    return broadcast_to(MultiDiscrete(np.array(space.n)), shape)


@broadcast_to.register
def _(space: MultiBinary, shape):
    return MultiBinary(np.broadcast_shapes(space.shape, shape))


def broadcast_prepend(space: Box | MultiDiscrete | Discrete | MultiBinary, shape):
    return broadcast_to(space, shape + space.shape)


@singledispatch
def tile(space: Space, reps):
    raise NotImplementedError


@tile.register
def _(space: Box, reps):
    low, high = np.tile(space.low, reps), np.tile(space.high, reps)
    return Box(low, high, dtype=np.float32)


@tile.register
def _(space: MultiDiscrete, reps):
    return MultiDiscrete(np.tile(space.nvec, reps))


@tile.register
def _(space: Discrete, reps):
    return tile(MultiDiscrete(np.array(space.n)), reps)


@tile.register
def _(space: MultiBinary, reps):
    _tmp = np.empty(space.shape, space.dtype)
    _tmp = np.tile(_tmp, reps)
    return MultiBinary(_tmp.shape)


@singledispatch
def moveaxis(space: Space, source, destination):
    raise NotImplementedError


@moveaxis.register
def _(space: Box, source, destination):
    return Box(
        low=np.moveaxis(space.low, source, destination),
        high=np.moveaxis(space.high, source, destination),
        dtype=space.dtype,  # type: ignore[arg-type]
    )


@moveaxis.register
def _(space: MultiDiscrete, source, destination):
    return MultiDiscrete(
        nvec=np.moveaxis(space.nvec, source, destination), dtype=space.dtype  # type: ignore[arg-type]
    )


@moveaxis.register
def _(space: MultiBinary, source, destination):
    n = np.moveaxis(np.ones(space.shape), source, destination).shape
    # TODO: improve
    return MultiBinary(n=n)


@singledispatch
def _get_space_lims(space: Space):
    """Get minimum and maximum values of a space."""
    raise NotImplementedError


@_get_space_lims.register
def _(space: Box):
    return np.stack((space.low, space.high))


@_get_space_lims.register
def _(space: Discrete):
    return np.array([0, space.n - 1])


@_get_space_lims.register
def _(space: MultiDiscrete):
    return np.stack((np.zeros(space.shape), space.nvec - 1))


@_get_space_lims.register
def _(space: MultiBinary):
    return np.stack((np.zeros(space.shape), np.ones(space.shape)))


def stack(spaces, axis=None):
    """Join a sequence of spaces along a new axis.

    'Upcasts' to superset space when required.
    """
    if len(spaces) == 1:
        return spaces[0]

    if all(isinstance(space, Discrete) for space in spaces):
        nvecs = [space.n for space in spaces]
        return MultiDiscrete(np.stack(nvecs, axis=axis))
    elif all(isinstance(space, MultiDiscrete) for space in spaces):
        nvecs = [space.nvec for space in spaces]
        return MultiDiscrete(np.stack(nvecs, axis=axis))
    elif all(isinstance(space, MultiBinary) for space in spaces):
        tmp = tuple(np.empty(space.n) for space in spaces)
        n = np.stack(tmp, axis=axis).shape
        return MultiBinary(n)
    else:
        lows, highs = zip(*(_get_space_lims(space) for space in spaces))
        low, high = np.stack(lows, axis=axis), np.stack(highs, axis=axis)
        return Box(low, high, dtype=float)


def concatenate(spaces, axis=None):
    """Join a sequence of spaces along an existing axis.

    'Upcasts' to superset space when required.
    """
    if len(spaces) == 1:
        return spaces[0]

    if all(isinstance(space, MultiDiscrete) for space in spaces):
        nvecs = [space.nvec for space in spaces]
        return MultiDiscrete(np.concatenate(nvecs, axis=axis))
    elif all(isinstance(space, MultiBinary) for space in spaces):
        tmp = tuple(np.empty(space.n) for space in spaces)
        n = np.concatenate(tmp, axis=axis).shape
        return MultiBinary(n)
    else:
        lows, highs = zip(*(_get_space_lims(space) for space in spaces))
        low, high = (np.concatenate(a, axis=axis) for a in (lows, highs))
        return Box(low, high, dtype=float)


def unnest(space):
    def _fn(space):
        if isinstance(space, Tuple):
            for s_i in space:
                yield from unnest(s_i)
        elif isinstance(space, Dict):
            for s in space.values():
                yield from unnest(s)
        else:
            yield space

    return Tuple(_fn(space))


def unnest_dict(space: Dict):
    spaces = {}
    for k, v in space.spaces.items():
        if isinstance(v, Dict):
            for vk, vv in unnest_dict(v).items():
                spaces[f"{k}_{vk}"] = vv
        else:
            spaces[k] = v
    return Dict(spaces)


# Space classes
class DiscreteMasked(Discrete):
    r"""A Discrete space with masked elements for sampling and membership testing.

    Args:
        n (int): Space assumes values in :math:`\{ 0, 1, \\dots, n-1 \}`.
        mask (Sequence of bool): Length `n` array where `True` elements indicate invalid
            actions.

    """

    def __init__(self, n, mask=np.ma.nomask):
        super().__init__(n)
        self.mask = mask
        self._rng = np.random.default_rng()

    @property
    def n(self):
        return self._ma.size

    @n.setter
    def n(self, value):
        self._ma = np.ma.masked_array(range(int(value)))

    @property
    def mask(self):
        return self._ma.mask

    @mask.setter
    def mask(self, value):
        self._ma.mask = np.ma.nomask
        self._ma[np.array(value, dtype=bool)] = np.ma.masked

    @property
    def valid_entries(self):
        return self._ma.compressed()

    def sample(self):
        return self._rng.choice(self.valid_entries)

    def contains(self, x):
        return x in self.valid_entries

    def __str__(self):
        return f"DiscreteMasked({self.n}, mask={self.mask})"

    def __eq__(self, other):
        return isinstance(other, DiscreteMasked) and (self._ma == other._ma).all()
