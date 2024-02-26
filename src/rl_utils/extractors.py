import math
from collections.abc import Callable, Sequence
from typing import Any

import torch as th
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


def build_mlp(
    layer_sizes: Sequence[int],
    activation: Callable[[], nn.Module] = nn.ReLU,
    last_act: bool = False,
):
    """PyTorch sequential MLP.

    Args:
        layer_sizes: Hidden layer sizes.
        activation: Non-linear activation function.
        last_act: Include final activation function.

    Returns:
        nn.Sequential

    """
    layers: list[nn.Module] = []
    for i, (in_, out_) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layers.append(nn.Linear(in_, out_))
        if last_act or i < len(layer_sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


def build_cnn_2d(
    layer_sizes: Sequence[int],
    kernel_sizes: int | tuple[int, ...] | Sequence[tuple[int, ...]],
    conv2d_kwargs: dict[str, Any] | None = None,
    pooling_layers: nn.Module | Sequence[nn.Module | None] | None = None,
    activation: Callable[[], nn.Module] = nn.ReLU,
    last_act: bool = False,
):
    """PyTorch sequential CNN.

    Args:
        layer_sizes: Hidden layer sizes.
        kernel_sizes: Kernel sizes for convolutional layers.
            If only one value is provided, the same is used for all convolutional
            layers.
        conv2d_kwargs: Keyword arguments `torch.nn.Conv2d` layers.
        pooling_layers: Pooling modules.
            If only one value is provided, the same is used after each convolutional
            layer.
        activation: Non-linear activation function.
        last_act (bool): Include final activation function.

    Returns:
        nn.Sequential

    """
    if isinstance(kernel_sizes, int):
        kernel_sizes = (kernel_sizes,) * (len(layer_sizes) - 1)
    if conv2d_kwargs is None:
        conv2d_kwargs = {}
    if pooling_layers is None or isinstance(pooling_layers, nn.Module):
        pooling_layers = [pooling_layers] * (len(layer_sizes) - 1)

    layers: list[nn.Module] = []
    for i, (in_, out_, kernel_size, pooling) in enumerate(
        zip(layer_sizes[:-1], layer_sizes[1:], kernel_sizes, pooling_layers)
    ):
        layers.append(nn.Conv2d(in_, out_, kernel_size=kernel_size, **conv2d_kwargs))  # type: ignore
        if last_act or i < len(layer_sizes) - 2:
            layers.append(activation())
        if pooling is not None:
            layers.append(pooling)
    return nn.Sequential(*layers)


class MLPExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space,
        hidden_sizes=(),
        build_kwargs=None,
    ):
        n_in = math.prod(observation_space.shape)
        layer_sizes = [n_in, *hidden_sizes]
        super().__init__(observation_space, features_dim=layer_sizes[-1])

        if build_kwargs is None:
            build_kwargs = {}
        self.mlp = nn.Sequential(
            nn.Flatten(),
            *build_mlp(layer_sizes, last_act=True, **build_kwargs),
        )

    def forward(self, observations):
        return self.mlp(observations)


class ConvExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space,
        hidden_sizes=(),
        kernel_sizes=3,
        conv2d_kwargs=None,
        build_kwargs=None,
    ):
        if len(observation_space.shape) != 3:
            raise ValueError("Observation space must be 3D (CxHxW)")

        n_channels = observation_space.shape[0]
        layer_sizes = [n_channels, *hidden_sizes]
        if build_kwargs is None:
            build_kwargs = {}
        cnn = nn.Sequential(
            *build_cnn_2d(
                layer_sizes, kernel_sizes, conv2d_kwargs, last_act=True, **build_kwargs
            ),
            nn.Flatten(),
        )

        with th.no_grad():
            sample = observation_space.sample()
            sample = th.as_tensor(sample[None]).float()
            sample = preprocess_obs(sample, observation_space, normalize_images=False)
            features_dim = cnn(sample).shape[1]

        super().__init__(observation_space, features_dim=features_dim)
        self.cnn = cnn

    def forward(self, observations):
        return self.cnn(observations)
