# RL Utils

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/charliermarsh/ruff)
[![types - Mypy](https://img.shields.io/badge/types-Mypy-blue.svg)](https://github.com/python/mypy)
[![Hatch project](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg)](https://github.com/pypa/hatch)

> **Note**: This project is under active development. :construction:

A collection of utilities for reinforcement learning.

## Installation
This package is developed for [Python](https://www.python.org/downloads/) 3.11+. Best practice is to first create a [virtual environment](https://docs.python.org/3/tutorial/venv.html). The package can be installed locally using `pip install <path>`, where `<path>` is the top-level directory containing `pyproject.toml`. Note that the [editable option](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs) can be included to track any package modifications. To install optional packages, specify [extras](https://peps.python.org/pep-0508/#extras), as exemplified [here](https://pip.pypa.io/en/stable/cli/pip_install/#examples). Developers should install the package with `pip install -e <path>[dev]`. The additional commands below should be run on new environments, activating formatting/linting [hooks](https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks) and [static typing](https://mypy.readthedocs.io/en/stable/index.html) support:
```
pip install --upgrade pip
pre-commit install
mypy --install-types --non-interactive .
```

The project management tool [hatch](https://hatch.pypa.io/) is recommended to simplify local installation. Simply execute `hatch -e dev shell` to create and enter a virtual environment with the package installed in development mode (editable install plus extras). Post-install commands (for `pre-commit`, `mypy`, etc.) are executed automatically.


## Quickstart

### Training with Stable-Baselines3

The Stable-Baselines3 training utility `rl_utils.sb3.train` can be invoked from the command line, specifying arguments for a variety of environment and learning algorithm parameters. For example,
```
python <path>/src/rl_utils/sb3/train.py --env=CartPole-v1 --algo=PPO --policy=MlpPolicy --timesteps=30000
```
Alternatively, with the package installed the utilities can be invoked from any directory using `python -m rl_utils.sb3.train [<args>]`. Execute either command with the `--help` option for a complete list of supported CLI options.

To support easier maintenance and reproduction of environment/model setups, JSON configuration files can be used instead of CLI options. This fully exposes the interface of `rl_utils.sb3.train` and enables more detailed setup than the CLI allows. Example configurations are provided in `examples/` and can be invoked with syntax such as
```
python -m rl_utils.sb3.train --env-config=env_cfg.json --model-config=model_cfg.json
```
Note that the environment configuration conforms with Gymnasium [`EnvSpec`](https://gymnasium.farama.org/api/registry/#gymnasium.envs.registration.EnvSpec) and allows environment/wrapper entry points to be defined with strings of the form `"<module>:<attr>"`, enabling imports of user-defined objects. Similarly, SB3 model/training configuration allows the same string syntax to be used to specify custom algorithms, policies, extractors, callbacks, etc.


### Playing environments

To interactively play an environment, the functions `play` and `play_live` are provided in `rl_utils.gym.play`. Note that the environment render mode must be `"rgb_array"` and the action space must be `gym.spaces.Discrete`. Users can invoke the module from the command line with commands such as `python -m rl_utils.gym.play --env CartPole-v1 --keys a d`; the specified keys will correspond to the elements of the action space, in order. Default behavior uses `play`, which waits for user input for each action. The `--continuous` option uses `play_live`, which uses other optional arguments `--fps` and `--noop` to execute the environment in real time (as CartPole would, for example).


### Space and observation transforms

The subpackage `rl_utils.gym.spaces` aims to provide the user with a helpful suite of functions for the transformation of spaces and arrays. The `utils` submodule provides functions that operate on `gym.Space` and return a modified space. Many functions intend to mirror NumPy functions that operate on array instances of these spaces (e.g. reshape, stack, etc.)

The `tx` submodule extends the utility of the commonly used `gym.ObservationWrapper`. Unfortunately, the functionality of `ObservationWrapper` cannot be naturally applied to subspaces of `Tuple`, `Dict`, etc. The provided functions in `tx` generate callable transforms that operate on spaces. These transforms return both a modified space (using `utils`) and a callable for transforming arrays from the original space to the new space. The provided class `rl_utils.gym.wrappers.TxObsWrapper` thinly wraps these transforms, overriding `observation_space` and modifying tensors in `ObservationWrapper.observation`.
