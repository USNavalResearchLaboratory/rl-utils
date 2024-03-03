"""Additional `gymnasium` spaces and utilities."""

import numpy as np
from gymnasium.spaces import Box, Discrete, MultiDiscrete

# TODO: refactor transforms using `singledispatch`


# Transforms
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
        return MultiDiscrete(np.broadcast_to(space.n, shape))
    else:
        raise NotImplementedError("Only supported for Box and MultiDiscrete spaces.")


def get_space_lims(space):
    """Get minimum and maximum values of a space."""
    if isinstance(space, Box):
        return np.stack((space.low, space.high))
    elif isinstance(space, Discrete):
        return np.array([0, space.n - 1])
    elif isinstance(space, MultiDiscrete):
        return np.stack((np.zeros(space.shape), space.nvec - 1))
    else:
        raise NotImplementedError(
            "Only supported for Box, Discrete, or DiscreteSet spaces."
        )


def stack(spaces, axis=0):
    """Join a sequence of spaces along a new axis.

    Does 'upcasting' to superset spaces when required.
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
        lows, highs = zip(*(get_space_lims(space) for space in spaces))
        low, high = np.stack(lows, axis=axis), np.stack(highs, axis=axis)
        return Box(low, high, dtype=float)


def concatenate(spaces, axis=0):
    """Join a sequence of spaces along an existing axis.

    'Upcasts' to superset spaces when required.
    """
    if len(spaces) == 1:
        return spaces[0]

    if all(isinstance(space, MultiDiscrete) for space in spaces):
        nvecs = [space.nvec for space in spaces]
        return MultiDiscrete(np.concatenate(nvecs, axis=axis))
    else:
        lows, highs = zip(*(get_space_lims(space) for space in spaces))
        low, high = np.concatenate(lows, axis=axis), np.concatenate(highs, axis=axis)
        return Box(low, high, dtype=float)


def reshape(space, shape):
    """Reshape space."""
    if isinstance(space, Box):
        low, high = space.low.reshape(shape), space.high.reshape(shape)
        return Box(low, high, dtype=float)
    elif isinstance(space, MultiDiscrete):
        return MultiDiscrete(space.nvec.reshape(shape))
    else:
        raise NotImplementedError


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
