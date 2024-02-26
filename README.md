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

The Stable-Baselines3 training utility `rl_utils.sb3_utils.train` can be invoked from the command line, specifying arguments for a variety of environment and learning algorithm parameters. For example,
```
python <path>/src/rl_utils/sb3_utils.py --env=CartPole-v1 --algo=PPO --policy=MlpPolicy --timesteps=30000
```
Alternatively, with the package installed the utilities can be invoked from any directory using `python -m rl_utils.sb3_utils [<args>]`. Execute either command with the `--help` option for a complete list of supported CLI options.

To support easier maintenance and reproduction of environment/model setups, configuration files can be used instead of CLI options. This fully exposes the interface of `rl_utils.sb3_utils.train` and enables more detailed setup than the CLI allows. Configurations can be formatted in YAML or in Python; the latter allows the user to define/import custom objects (e.g. environment wrappers, custom policies, etc.) Example configurations are provided in `examples/` and can be invoked with syntax such as
```
python -m rl_utils.sb3_utils --env-config=env_cfg.yml --model-config=model_cfg.yml
```
