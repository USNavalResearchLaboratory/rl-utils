import argparse
import importlib
import json
import os
import pickle
import sys
from collections.abc import Callable, Sequence
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import optuna
import pandas as pd
import stable_baselines3
import yaml
from gymnasium.envs.registration import EnvSpec
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from torch import nn

from rl_utils.sb3.callbacks import TrialEvalCallback

# TODO: better separate model cfg, algo vs learning
# TODO: rework CLI parsing w/ defaults and `Namespace`?

PathOrStr = str | os.PathLike


def _load_attr(entry_point: str) -> Any:
    """Load module attribute from string specification.

    Assumes `entry_point` is of the form "<module>:<attr>", where <module> can
    be a subpackage (e.g. "foo.bar") and <attr> is the attribute to load "module".

    Note:
        Derived from `gymnasium.envs.registration.load_env_creator`.

    Args:
        entry_point: The string specification "<module>:<attr>"

    Returns:
        The desired attribute, as if `from <module> import <attr>` had been executed.
    """
    # _prefix = "import::"
    # if entry_point.startswith(_prefix):
    #     entry_point = entry_point.removeprefix(_prefix)
    # else:
    #     raise ValueError
    if ":" not in entry_point:
        raise ValueError
    mod_name, attr_name = entry_point.split(":")
    mod = importlib.import_module(mod_name)
    return getattr(mod, attr_name)


def load_attributes(obj: Any):
    """Recursively load attributes from string specification.

    Note:
        See `_load_attr` for a description of the spec syntax.
    """
    if isinstance(obj, dict):
        return {k: load_attributes(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return list(map(load_attributes, obj))
    elif isinstance(obj, str):
        try:
            return _load_attr(obj)
        except ValueError:
            return obj
    else:
        return obj


def _make_from_spec(spec):
    """Instantiate object from constructor/args specification.

    Input `spec` is a constructor with an optional kwarg dictionary. If a the
    constructor is a `str`, it is loaded using `_load_attr`.
    """
    if isinstance(spec, Sequence):
        entry_point, kwargs = spec
    else:
        entry_point, kwargs = spec, {}
    if isinstance(entry_point, str):
        entry_point = _load_attr(entry_point)
    return entry_point(**kwargs)


def get_now():
    """Get current date/time in ISO format."""
    str_ = datetime.now().isoformat(timespec="seconds")
    if sys.platform.startswith("win32"):
        str_ = str_.replace(":", "_")
    return str_


def _make_vec_env(
    env_id: str | EnvSpec,
    env_kwargs: dict[str, Any] | None = None,
    max_episode_steps: int | None = None,
    n_envs: int = 1,
    multiproc: bool = False,
    seed: int | None = None,
) -> VecEnv:
    """Make SB3 vectorized enviornment.

    Args:
        env_id: Name of the environment. Optionally, a module to import can be included,
            eg. 'module:Env-v0'
        env_kwargs: Additional arguments to pass to the environment constructor.
        max_episode_steps: Maximum length of an episode (TimeLimit wrapper).
        n_envs: The number of environments you wish to have in parallel.
        multiproc: Activates use of `SubprocVecEnv` instead of `DummyVecEnv`.
        seed: The initial seed for the random number generator.

    Returns:
        Vectorized wrapped environment.

    """
    if env_kwargs is None:
        env_kwargs = {}
    func = partial(gym.make, env_id, max_episode_steps, **env_kwargs)
    vec_env_cls = SubprocVecEnv if multiproc and n_envs > 1 else None
    env = make_vec_env(func, n_envs=n_envs, seed=seed, vec_env_cls=vec_env_cls)
    return env


def record_vid_vec(
    model: BaseAlgorithm,
    env: GymEnv,
    video_length: int,
    deterministic: bool = True,
    video_folder: PathOrStr = ".",
    video_type: str = "mp4",
    seed: int | None = None,
) -> list[np.ndarray]:
    """Save MP4 vid of agent rollouts.

    Args:
        model: Model object with a `predict` method.
        env: Environment object.
        deterministic: Whether to use deterministic or stochastic actions.
        video_folder: The folder where the recordings will be stored.
        video_length: The number of recording steps.
        video_type: Either "mp4" or "gif"
        seed: The initial seed for the random number generator.

    Returns:
        List of rendered RGB arrays.

    """
    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]
    env.seed(seed)

    frames: list = []
    obs = env.reset()
    frames.append(env.render())
    while len(frames) < video_length:
        action, _state = model.predict(obs, deterministic=deterministic)  # type: ignore
        obs, _reward, _dones, _info = env.step(action)
        frames.append(env.render())
    env.close()

    clip = ImageSequenceClip(frames, fps=env.metadata["render_fps"])
    if video_type == "mp4":
        clip.write_videofile(str(Path(video_folder) / "render.mp4"))
    elif video_type == "gif":
        clip.write_gif(str(Path(video_folder) / "render.gif"))
    else:
        raise ValueError(f"Video type '{video_type}' not supported.")

    # model.logger.record(
    #     "render/eps",
    #     Video(th.ByteTensor(np.stack(frames)), fps=env.metadata["render_fps"]),
    #     exclude=("stdout", "log", "json", "csv"),
    # )

    return frames


def record_vid(
    model: BaseAlgorithm,
    env: gym.Env,
    video_length: int,
    deterministic: bool = True,
    video_folder: PathOrStr = ".",
    video_type: str = "mp4",
    seed: int | None = None,
) -> list[np.ndarray]:
    """Save MP4 vid of agent rollouts.

    Args:
        model: Model object with a `predict` method.
        env: Environment object.
        deterministic: Whether to use deterministic or stochastic actions.
        video_folder: The folder where the recordings will be stored.
        video_length: The number of recording steps.
        video_type: Either "mp4" or "gif"
        seed: The initial seed for the random number generator.

    Returns:
        List of rendered RGB arrays.

    """
    env.np_random = np.random.default_rng(seed)

    frames: list = []
    while len(frames) < video_length:
        obs, _info = env.reset()
        frames.append(env.render())
        terminated, truncated = False, False
        while not (terminated or truncated):
            if len(frames) == video_length:
                break
            action, _state = model.predict(obs, deterministic=deterministic)
            obs, _reward, terminated, truncated, _info = env.step(action)
            frames.append(env.render())
    env.close()

    clip = ImageSequenceClip(frames, fps=env.metadata["render_fps"])
    if video_type == "mp4":
        clip.write_videofile(str(Path(video_folder) / "render.mp4"))
    elif video_type == "gif":
        clip.write_gif(str(Path(video_folder) / "render.gif"))
    else:
        raise ValueError(f"Video type '{video_type}' not supported.")

    # model.logger.record(
    #     "render/eps",
    #     Video(th.ByteTensor(np.stack(frames)), fps=env.metadata["render_fps"]),
    #     exclude=("stdout", "log", "json", "csv"),
    # )

    return frames


def _update_algo_kwargs(
    kwargs: dict[str, Any] | None,
    n_envs: int = 1,
    log_path: PathOrStr = ".",
    seed: int | None = None,
    verbose: int = 0,
) -> dict[str, Any]:
    """Update algorithm kwargs.

    Sets Tensorboard log path and converts `str` activation policy kwarg.
    Handles custom kwarg `n_steps_total` for multiprocessing; note that SB3 kwarg
    `n_steps` actually specifies the number of steps per environment.

    Args:
        kwargs: Additional arguments to pass to the algorithm constructor.
        n_envs: The number of environments you wish to have in parallel.
        log_path: Path for saving results, trained model, Tensorboard logs, videos, etc.
        seed: The initial seed for the random number generator.
        verbose: Enables verbosity for algorithm and callbacks.

    Returns:
        Updated `algo_kwargs`.

    """
    if kwargs is None:
        algo_kwargs = {}
    else:
        algo_kwargs = kwargs.copy()

    algo_kwargs["seed"] = seed
    algo_kwargs["tensorboard_log"] = str(log_path)
    algo_kwargs["verbose"] = verbose
    if "n_steps_total" in algo_kwargs:
        algo_kwargs["n_steps"] = algo_kwargs.pop("n_steps_total") // n_envs
    if "policy_kwargs" in algo_kwargs:
        policy_kwargs = algo_kwargs["policy_kwargs"]
        if "activation_fn" in policy_kwargs:
            activation_fn = policy_kwargs["activation_fn"]
            if isinstance(activation_fn, str):
                policy_kwargs["activation_fn"] = getattr(nn, activation_fn)

    return algo_kwargs


def _make_model(
    env: GymEnv,
    algo: str | BaseAlgorithm,
    policy: str | BasePolicy,
    algo_kwargs: dict[str, Any] | None = None,
    params_path: PathOrStr | None = None,
    n_envs: int = 1,
    log_path: PathOrStr = ".",
    seed: int | None = None,
    verbose: int = 0,
) -> BaseAlgorithm:
    """Make and setup SB3 model.

    Args:
        env: Environment object.
        algo: Stable-Baselines3 algorithm.
        policy: Stable-Baselines3 policy.
        algo_kwargs: Additional arguments to pass to the algorithm constructor.
        params_path: Path to saved initial model parameters.
        n_envs: The number of environments you wish to have in parallel.
        log_path: Path for saving results, trained model, Tensorboard logs, videos, etc.
        seed: The initial seed for the random number generator.
        verbose: Enables verbosity for algorithm and callbacks.

    Returns:
        SB3 model with parameters/logger/etc. set.
    """
    if isinstance(algo, str):
        algo_cls = getattr(stable_baselines3, algo)
    else:
        algo_cls = algo
    algo_kwargs = _update_algo_kwargs(algo_kwargs, n_envs, log_path, seed, verbose)

    model = algo_cls(policy, env, **algo_kwargs)
    if params_path is not None:
        model.set_parameters(str(params_path))

    _format_strings = []
    if verbose >= 1:
        _format_strings.append("stdout")
    if importlib.util.find_spec("tensorboard") is not None:
        _format_strings.append("tensorboard")
    _logger = configure(str(log_path), _format_strings)
    model.set_logger(_logger)

    return model


def _update_eval_callback_kwargs(
    kwargs: dict[str, Any] | None,
    deterministic: bool = True,
    n_envs: int = 1,
    log_path: PathOrStr = ".",
    verbose: int = 0,
) -> dict[str, Any]:
    """Update `EvalCallback` kwargs.

    Handles custom kwarg `eval_freq_total` for multiprocessing; note that SB3 kwarg
    `eval_freq` actually specifies the number of steps per environment. Assumes
    "callback_*" kwargs are constructors and are used to instantiate `BaseCallback`
    objects.

    Args:
        kwargs: Additional arguments to pass to the `EvalCallback` constructor.
        deterministic: Whether to use deterministic or stochastic actions.
        n_envs: The number of environments you wish to have in parallel.
        log_path: Path for saving results, trained model, Tensorboard logs, videos, etc.
        verbose: Enables verbosity for algorithm and callbacks.

    Returns:
        Updated `eval_callback_kwargs`.

    """
    if kwargs is None:
        eval_callback_kwargs = {}
    else:
        eval_callback_kwargs = kwargs.copy()

    eval_callback_kwargs["deterministic"] = deterministic
    eval_callback_kwargs["log_path"] = str(log_path)
    eval_callback_kwargs["best_model_save_path"] = str(log_path)
    eval_callback_kwargs["verbose"] = verbose
    if "eval_freq_total" in eval_callback_kwargs:
        eval_freq_total = eval_callback_kwargs.pop("eval_freq_total")
        eval_callback_kwargs["eval_freq"] = max(eval_freq_total // n_envs, 1)

    for k in ("callback_on_new_best", "callback_after_eval"):
        if k in eval_callback_kwargs:
            eval_callback_kwargs[k] = _make_from_spec(eval_callback_kwargs.pop(k))

    if "callback_on_new_best" in eval_callback_kwargs:
        eval_callback_kwargs["callback_on_new_best"].verbose = verbose
    if "callback_after_eval" in eval_callback_kwargs:
        eval_callback_kwargs["callback_after_eval"].verbose = verbose

    return eval_callback_kwargs


def _load_best_model(model: BaseAlgorithm, best_model_save_path: PathOrStr):
    """Load best model weights from `EvalCallback`.

    Args:
        model: SB3 model.
        best_model_save_path: Directory of saved parameters from `EvalCallback`.

    """
    best_model_save_path = Path(best_model_save_path)
    best_model_path = best_model_save_path / "best_model.zip"
    if best_model_path.exists():
        model.set_parameters(str(best_model_path), device=model.device)
        best_model_path.unlink()
    model.save(best_model_save_path / "model.zip")


def train(
    env_id: str | EnvSpec,
    algo: str | BaseAlgorithm,
    policy: str | BasePolicy,
    env_kwargs: dict[str, Any] | None = None,
    max_episode_steps: int | None = None,
    algo_kwargs: dict[str, Any] | None = None,
    params_path: PathOrStr | None = None,
    total_timesteps: int = 0,
    callbacks: list[BaseCallback | list] | None = None,
    eval_callback_kwargs: dict[str, Any] | None = None,
    deterministic: bool = True,
    n_envs: int = 1,
    multiproc: bool = False,
    log_path: PathOrStr = ".",
    video_length: int = 0,
    seed: int | None = None,
    verbose: int = 0,
) -> float:
    """Train and evaluate an agent.

    Args:
        env_id: Name of the environment. Optionally, a module to import can be included,
            eg. 'module:Env-v0'
        algo: Stable-Baselines3 algorithm.
        policy: Stable-Baselines3 policy.
        env_kwargs: Additional arguments to pass to the environment constructor.
        max_episode_steps: Maximum length of an episode (TimeLimit wrapper).
        algo_kwargs: Additional arguments to pass to the algorithm constructor.
        params_path: Path to saved initial model parameters.
        total_timesteps: The total number of samples (env steps) to train on.
        callbacks: Constructor specs for SB3 callback objects.
        eval_callback_kwargs: Additional arguments to pass to the `EvalCallback`
            constructor.
        deterministic: Whether to use deterministic or stochastic actions.
        n_envs: The number of environments you wish to have in parallel.
        multiproc: Activates use of `SubprocVecEnv` instead of `DummyVecEnv`.
        log_path: Path for saving results, trained model, Tensorboard logs, videos, etc.
        video_length: The number of recording steps.
        seed: The initial seed for the random number generator.
        verbose: Enables verbosity for algorithm and callbacks.

    """
    log_path = Path(log_path)
    log_path.mkdir(parents=True, exist_ok=True)
    if callbacks is None:
        callback_list = []
    else:
        callback_list = list(map(_make_from_spec, callbacks))

    env = _make_vec_env(env_id, env_kwargs, max_episode_steps, n_envs, multiproc)
    eval_env = _make_vec_env(
        env_id, env_kwargs, max_episode_steps, n_envs, multiproc, seed
    )

    model = _make_model(
        env, algo, policy, algo_kwargs, params_path, n_envs, log_path, seed, verbose
    )

    eval_callback_kwargs = _update_eval_callback_kwargs(
        eval_callback_kwargs, deterministic, n_envs, log_path, verbose
    )
    eval_callback = EvalCallback(eval_env, **eval_callback_kwargs)

    model.learn(total_timesteps, callback=callback_list + [eval_callback])

    _load_best_model(model, best_model_save_path=log_path)

    episode_rewards, _ = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=eval_callback.n_eval_episodes,
        deterministic=deterministic,
        return_episode_rewards=True,
    )
    mean_reward = np.mean(episode_rewards).item()
    std_reward = np.std(episode_rewards).item()

    if verbose >= 1:
        print(f"reward = {mean_reward:.2f} +/- {std_reward}")
    with open(log_path / "eval.yml", "w") as f:
        yaml.dump(dict(mean_reward=mean_reward, std_reward=std_reward), f)

    if video_length > 0:
        _kwargs = env_kwargs.copy() if env_kwargs is not None else {}
        _kwargs["render_mode"] = "rgb_array"
        render_env = gym.make(env_id, max_episode_steps, **_kwargs)
        record_vid(
            model, render_env, video_length, deterministic, str(log_path), seed=seed
        )
        # record_vid_vec(
        #     model, eval_env, video_length, deterministic, str(log_path), seed=seed
        # )

    return mean_reward


def hyperopt(
    env_id: str | EnvSpec,
    algo: str | BaseAlgorithm,
    policy: str | BasePolicy,
    get_trial_params: Callable[[optuna.Trial], dict],  # TODO
    study_kwargs: dict[str, Any] | None = None,
    optimize_kwargs: dict[str, Any] | None = None,
    env_kwargs: dict[str, Any] | None = None,
    max_episode_steps: int | None = None,
    algo_kwargs: dict[str, Any] | None = None,
    params_path: PathOrStr | None = None,
    total_timesteps: int = 0,
    callbacks: list[BaseCallback | list] | None = None,
    eval_callback_kwargs: dict[str, Any] | None = None,
    deterministic: bool = True,
    n_envs: int = 1,
    multiproc: bool = False,
    log_path: PathOrStr = ".",
    video_length: int = 0,
    seed: int | None = None,
    verbose: int = 0,
) -> tuple[dict[str, Any], float | None]:
    """Optimize hyperparameters, train, and evaluate an agent.

    Args:
        env_id: Name of the environment. Optionally, a module to import can be included,
            eg. 'module:Env-v0'
        algo: Stable-Baselines3 algorithm class.
        policy : Stable-Baselines3 policy class.
        get_trial_params: Function of `trial`, returns `dict` of algorithm kwargs.
        study_kwargs: Additional arguments to pass to `create_study`.
        optimize_kwargs: Additional arguments to pass to `study.optimize`.
        env_kwargs: Additional arguments to pass to the environment constructor.
        max_episode_steps: Maximum length of an episode (TimeLimit wrapper).
        algo_kwargs: Additional arguments to pass to the algorithm constructor.
        params_path: Path to saved initial model parameters.
        total_timesteps: The total number of samples (env steps) to train on.
        callbacks: Constructor specs for SB3 callback objects.
        eval_callback_kwargs: Additional arguments to pass to the `EvalCallback`
            constructor.
        deterministic: Whether to use deterministic or stochastic actions.
        n_envs: The number of environments you wish to have in parallel.
        multiproc: Activates use of `SubprocVecEnv` instead of `DummyVecEnv`.
        log_path: Path for saving results, trained model, Tensorboard logs, videos, etc.
        video_length: The number of recording steps.
        seed: The initial seed for the random number generator.
        verbose: Enables verbosity for algorithm and callbacks.

    """
    log_path = Path(log_path)
    log_path.mkdir(parents=True, exist_ok=True)
    if algo_kwargs is None:
        algo_kwargs = {}
    if callbacks is None:
        callback_list = []
    else:
        callback_list = list(map(_make_from_spec, callbacks))

    def objective(trial):
        trial_path = log_path / f"T{trial.number}/"
        trial_path.mkdir(parents=True, exist_ok=True)
        algo_kwargs_trial = algo_kwargs | get_trial_params(trial)
        with open(trial_path / "params.yml", "w") as f:
            yaml.dump(trial.params, f)

        env = _make_vec_env(env_id, env_kwargs, max_episode_steps, n_envs, multiproc)
        eval_env = _make_vec_env(
            env_id, env_kwargs, max_episode_steps, n_envs, multiproc, seed
        )

        model = _make_model(
            env,
            algo,
            policy,
            algo_kwargs_trial,
            params_path,
            n_envs,
            trial_path,
            seed,
            verbose,
        )

        eval_callback_kwargs_trial = _update_eval_callback_kwargs(
            eval_callback_kwargs, deterministic, n_envs, trial_path, verbose
        )
        eval_callback = TrialEvalCallback(eval_env, trial, **eval_callback_kwargs_trial)

        model.learn(total_timesteps, callback=callback_list + [eval_callback])

        model.env.close()
        eval_env.close()

        _load_best_model(model, best_model_save_path=trial_path)

        # episode_rewards, _ = evaluate_policy(
        #     model,
        #     eval_env,
        #     n_eval_episodes=eval_callback.n_eval_episodes,
        #     deterministic=deterministic,
        #     return_episode_rewards=True,
        # )
        # mean_reward = np.mean(episode_rewards).item()
        # std_reward = np.std(episode_rewards).item()

        # if verbose >= 2:
        #     print(f"reward = {mean_reward:.2f} +/- {std_reward}")
        # with open(trial_path / "eval.yml", "w") as f:
        #     yaml.dump(dict(mean_reward=mean_reward, std_reward=std_reward), f)

        # if video_length > 0:
        #     _kwargs = env_kwargs.copy() if env_kwargs is not None else {}
        #     _kwargs["render_mode"] = "rgb_array"
        #     render_env = gym.make(env_id, max_episode_steps, **_kwargs)
        #     record_vid(
        #         model,
        #         render_env,
        #         video_length,
        #         deterministic,
        #         str(trial_path),
        #         seed=seed,
        #     )

        del model.env, eval_env
        del model

        if eval_callback.is_pruned:
            raise optuna.exceptions.TrialPruned()

        # return eval_callback.last_mean_reward
        return eval_callback.best_mean_reward
        # return mean_reward

    if study_kwargs is None:
        study_kwargs = {}
    if optimize_kwargs is None:
        optimize_kwargs = {}
    optimize_kwargs["show_progress_bar"] = bool(verbose)
    study = optuna.create_study(**study_kwargs)
    study.optimize(objective, **optimize_kwargs)

    df = study.trials_dataframe()
    df.to_csv(log_path / "study.csv")

    best_trial = study.best_trial
    with open(log_path / "best_trial.yml", "w") as f:
        yaml.dump(
            dict(
                number=best_trial.number,
                mean_reward=best_trial.value,
                params=best_trial.params,
            ),
            f,
        )

    with open(log_path / "study.pkl", "wb") as fb:
        pickle.dump(study, fb)

    if verbose >= 1:
        print(f"Number of finished trials: {len(study.trials)}")
        print("Best trial:")
        print(f"  Value: {best_trial.value}")
        print("  Params: ")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")

    return best_trial.params, best_trial.value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an agent.")

    # Core arguments
    parser.add_argument("--env-config", help="Path to env config file.")
    parser.add_argument("--model-config", help="Path to model config file.")
    parser.add_argument("--env", help="Environment ID.")
    parser.add_argument("--algo", help="Algorithm class.")
    parser.add_argument("--policy", help="Policy class.")

    # Configurable argument overrides
    parser.add_argument(
        "--max-ep-steps", type=int, help="Maximum episode length in steps."
    )
    parser.add_argument("--lr", type=float, help="Learning rate.")
    parser.add_argument("--gamma", type=float, help="Discount factor.")
    parser.add_argument("--params-path", help="Path to saved model parameters.")
    parser.add_argument("--timesteps", type=int, help="Total # of learning steps.")
    parser.add_argument(
        "--n-eval-eps", type=int, help="Number of agent evaluation episodes."
    )
    parser.add_argument(
        "--eval-freq", type=int, help="Agent evaluation frequency (in steps)."
    )

    # Additional arguments
    parser.add_argument(
        "--deterministic",
        type=bool,
        default=True,
        help="Use deterministic policy for action evaluation.",
    )
    parser.add_argument(
        "--n-envs", type=int, default=1, help="Number of vectorized environments."
    )
    parser.add_argument(
        "--multiproc",
        action="store_true",
        help="Use multiprocessing for vectorized environments.",
    )
    parser.add_argument("--device", help="Device for training PyTorch policy.")
    parser.add_argument("--log-path", default=".", help="Logging path.")
    parser.add_argument(
        "--timestamp",
        action="store_true",
        help="Extend log path with datetime subdirectory.",
    )
    parser.add_argument("--video-length", type=int, default=0, help="Video length.")
    parser.add_argument("--seed", type=int, help="RNG seed.")
    parser.add_argument("--verbose", type=int, default=0, help="Verbosity level.")

    args = parser.parse_args()

    # Load config
    if args.env_config is not None:
        with open(args.env_config) as f:
            env_id = EnvSpec.from_json(f.read())

    kwargs: dict[str, Any] = dict(algo_kwargs={}, eval_callback_kwargs={})
    hyperopt_cfg = None  # FIXME
    if args.model_config is not None:
        with open(args.model_config) as f:
            _cfg = json.loads(f.read())
        model_cfg = load_attributes(_cfg)

        algo = model_cfg.pop("algo")
        policy = model_cfg.pop("policy")
        kwargs |= model_cfg

    # CLI overrides
    if args.env is not None:  # TODO: clear related variables?
        env_id = args.env
    if args.algo is not None:
        algo = args.algo
    if args.policy is not None:
        policy = args.policy

    if args.max_ep_steps is not None:
        kwargs["max_episode_steps"] = args.max_ep_steps
    if args.lr is not None:
        kwargs["algo_kwargs"]["learning_rate"] = args.lr
    if args.gamma is not None:
        kwargs["algo_kwargs"]["gamma"] = args.gamma
    if args.params_path is not None:
        kwargs["params_path"] = args.params_path
    if args.timesteps is not None:
        kwargs["total_timesteps"] = args.timesteps
    if args.n_eval_eps is not None:
        kwargs["eval_callback_kwargs"]["n_eval_episodes"] = args.n_eval_eps
    if args.eval_freq is not None:
        kwargs["eval_callback_kwargs"]["eval_freq_total"] = args.eval_freq
    if args.device is not None:
        kwargs["algo_kwargs"]["device"] = args.device

    log_path = Path(args.log_path)
    if args.timestamp:
        log_path = log_path / get_now()

    if hyperopt_cfg is not None:
        get_trial_params = hyperopt_cfg.pop("get_trial_params")
        kwargs |= hyperopt_cfg
        hyperopt(
            env_id,
            algo,
            policy,
            get_trial_params,
            deterministic=args.deterministic,
            n_envs=args.n_envs,
            multiproc=args.multiproc,
            log_path=log_path,
            video_length=args.video_length,
            seed=args.seed,
            verbose=args.verbose,
            **kwargs,
        )

        df = pd.read_csv(log_path / "study.csv")
        for key in df.columns:
            if not key.startswith("params_"):
                continue
            df.plot(
                x=key,
                y="value",
                c="number",
                cmap="viridis",
                xlabel=key[7:],
                ylabel="reward",
                kind="scatter",
            )

    else:
        train(
            env_id,
            algo,
            policy,
            deterministic=args.deterministic,
            n_envs=args.n_envs,
            multiproc=args.multiproc,
            log_path=log_path,
            video_length=args.video_length,
            seed=args.seed,
            verbose=args.verbose,
            **kwargs,
        )
