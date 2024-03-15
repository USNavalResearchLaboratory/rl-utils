"""Additional `gymnasium` spaces and utilities."""

from collections import OrderedDict
from functools import wraps

import numpy as np
from gymnasium.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete, Tuple

# TODO: refactor transforms using `singledispatch`


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
def reshape(space, shape):
    """Reshape space."""
    if isinstance(space, Box):
        low, high = space.low.reshape(shape), space.high.reshape(shape)
        return Box(low, high, dtype=float)
    elif isinstance(space, MultiDiscrete):
        return MultiDiscrete(space.nvec.reshape(shape))
    elif isinstance(space, Discrete):
        # if shape == ():
        #     return space
        return reshape(MultiDiscrete(space.n), shape)
    elif isinstance(space, MultiBinary):
        _tmp = np.empty(space.shape, space.dtype)
        _tmp = _tmp.reshape(shape)  # raises exception if shapes not compatible
        return MultiBinary(_tmp.shape)
    else:
        raise NotImplementedError


def broadcast_to(space, shape):
    """Broadcast space to new shape."""
    if isinstance(space, Box):
        low, high = (
            np.broadcast_to(space.low, shape),
            np.broadcast_to(space.high, shape),
        )
        return Box(low, high, dtype=space.dtype)
    elif isinstance(space, MultiDiscrete):
        return MultiDiscrete(np.broadcast_to(space.nvec, shape))
    elif isinstance(space, Discrete):
        # if shape == ():
        #     return space
        return broadcast_to(MultiDiscrete(space.n), shape)
    elif isinstance(space, MultiBinary):
        return MultiBinary(np.broadcast_shapes(space.shape, shape))
    else:
        raise NotImplementedError


def broadcast_prepend(space, shape):
    if isinstance(space, Box | MultiDiscrete | Discrete | MultiBinary):
        return broadcast_to(space, shape + space.shape)
    else:
        raise NotImplementedError


def tile(space, reps):
    if isinstance(space, Box):
        low, high = np.tile(space.low, reps), np.tile(space.high, reps)
        return Box(low, high, dtype=float)
    elif isinstance(space, MultiDiscrete):
        return MultiDiscrete(np.tile(space.nvec, reps))
    elif isinstance(space, Discrete):
        return tile(MultiDiscrete(space.n), reps)
    elif isinstance(space, MultiBinary):
        _tmp = np.empty(space.shape, space.dtype)
        _tmp = np.tile(_tmp, reps)
        return MultiBinary(_tmp.shape)
    else:
        raise NotImplementedError


def moveaxis(space, source, destination):
    if isinstance(space, Box):
        return Box(
            low=np.moveaxis(space.low, source, destination),
            high=np.moveaxis(space.high, source, destination),
            dtype=space.dtype,
        )
    elif isinstance(space, MultiDiscrete):
        return MultiDiscrete(
            nvec=np.moveaxis(space.nvec, source, destination), dtype=space.dtype
        )
    elif isinstance(space, MultiBinary):
        n = np.moveaxis(np.ones(space.shape), source, destination).shape
        # TODO: improve
        return MultiBinary(n=n)
    else:
        raise NotImplementedError


def _get_space_lims(space):
    """Get minimum and maximum values of a space."""
    if isinstance(space, Box):
        return np.stack((space.low, space.high))
    elif isinstance(space, Discrete):
        return np.array([0, space.n - 1])
    elif isinstance(space, MultiDiscrete):
        return np.stack((np.zeros(space.shape), space.nvec - 1))
    else:
        raise NotImplementedError


def stack(spaces, axis=0):
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
    else:
        lows, highs = zip(*(_get_space_lims(space) for space in spaces))
        low, high = np.stack(lows, axis=axis), np.stack(highs, axis=axis)
        return Box(low, high, dtype=float)


def concatenate(spaces, axis=0):  # FIXME
    """Join a sequence of spaces along an existing axis.

    'Upcasts' to superset space when required.
    """
    if len(spaces) == 1:
        return spaces[0]

    if all(isinstance(space, MultiDiscrete) for space in spaces):
        nvecs = [space.nvec for space in spaces]
        return MultiDiscrete(np.concatenate(nvecs, axis=axis))
    else:
        lows, highs = zip(*(_get_space_lims(space) for space in spaces))
        low, high = np.concatenate(lows, axis=axis), np.concatenate(highs, axis=axis)
        return Box(low, high, dtype=float)


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
