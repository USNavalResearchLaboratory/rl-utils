import argparse
from pathlib import Path
from typing import Any

import pygame
import yaml

from rl_utils.sb3_utils import EnvSpec, WrapFactorySeq, make_env


def play(  # noqa: C901
    env_id: str | EnvSpec,
    key_to_action: dict[str, Any],
    env_kwargs: dict[str, Any] | None = None,
    max_episode_steps: int | None = None,
    wrappers: WrapFactorySeq | None = None,
    verbose: bool = False,
):
    if env_kwargs is None:
        env_kwargs = {}
    env_kwargs["render_mode"] = "human"
    env = make_env(env_id, env_kwargs, max_episode_steps, wrappers)

    def _log(msg: str):
        if verbose:
            print(msg)

    def _reset():
        obs, _info = env.reset()
        _log(f"Obs: \n{obs}")
        return 0.0

    play = True
    return_ = _reset()
    while play:
        env.render()
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                key = pygame.key.name(int(event.key))
                if key == "escape":
                    play = False
                    env.close()
                    break
                elif key == "tab":
                    print("Manual reset.\n")
                    return_ = _reset()
                    break
                else:
                    action = key_to_action[key]
                    _log(f"Action: {action}")

                    obs, reward, terminated, truncated, _info = env.step(action)
                    return_ += float(reward)
                    _log(f"Obs: \n{obs}")
                    print(f"Reward: {reward}, Return: {return_}")

                    if terminated:
                        print("Episode terminated.\n")
                        return_ = _reset()
                    if truncated:
                        print("Episode truncated.\n")
                        return_ = _reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play an environment.")

    parser.add_argument("--env-config", help="Path to env config file.")

    parser.add_argument("--env", help="Environment ID.")
    parser.add_argument(
        "--keys", nargs="*", help="Sequence of keys that map to `Discrete` actions."
    )
    parser.add_argument(
        "--max-ep-steps", type=int, help="Maximum episode length in steps."
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbosity.")

    args = parser.parse_args()

    # Load config
    kwargs: dict[str, Any] = {}

    if args.env_config is not None:
        path = args.env_config
        if path.endswith((".yml", ".yaml")):
            with open(path) as f:
                env_cfg = yaml.safe_load(f)
        elif path.endswith(".py"):
            _env_globals: dict = {}
            exec(Path(path).read_text(), _env_globals)
            env_cfg = _env_globals["env_cfg"]
        else:
            raise ValueError
        env_id = env_cfg.pop("id")
        kwargs |= env_cfg

    # CLI overrides
    if args.env is not None:
        env_id = args.env
    if args.keys is not None:
        key_to_action = dict(zip(args.keys, range(len(args.keys))))
    if args.max_ep_steps is not None:
        kwargs["max_episode_steps"] = args.max_ep_steps

    play(env_id, key_to_action, verbose=args.verbose, **kwargs)
