import argparse
from typing import Any

import gymnasium as gym
import pygame
from gymnasium.envs.registration import EnvSpec


def play(  # noqa: C901
    env_id: str | EnvSpec,
    key_to_action: dict[str, Any],
    env_kwargs: dict[str, Any] | None = None,
    verbose: bool = False,
):
    if env_kwargs is None:
        env_kwargs = {}
    env_kwargs["render_mode"] = "human"
    env = gym.make(env_id, **env_kwargs)

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

    parser.add_argument("--config", help="Path to env config file.")
    parser.add_argument("--env", help="Environment ID.")
    parser.add_argument(
        "--keys", nargs="*", help="Sequence of keys that map to `Discrete` actions."
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbosity.")

    args = parser.parse_args()

    # Load config
    if args.config is not None:
        with open(args.config) as f:
            env_id = EnvSpec.from_json(f.read())

    # CLI overrides
    if args.env is not None:
        env_id = args.env
    if args.keys is not None:
        key_to_action = dict(zip(args.keys, range(len(args.keys))))

    play(env_id, key_to_action, verbose=args.verbose)
