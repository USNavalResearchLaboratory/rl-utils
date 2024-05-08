import argparse
from typing import Any, cast

import gymnasium as gym
import numpy as np
import pygame
from gymnasium.envs.registration import EnvSpec


def _display_arr(
    screen: pygame.Surface,
    arr: np.ndarray,
    video_size: tuple[int, int],
    scale: bool = False,
):
    """Display rendered image.

    Minor modification of `gymnasium.utils.play.display_arr`

    """
    arr = np.moveaxis(arr, 1, 0)

    if scale:
        arr_min, arr_max = np.min(arr), np.max(arr)
        arr = 255.0 * (arr - arr_min) / (arr_max - arr_min)

    surface = pygame.surfarray.make_surface(arr)
    surface = pygame.transform.scale(surface, video_size)

    surface_size = screen.get_size()
    width_offset = (surface_size[0] - video_size[0]) / 2
    height_offset = (surface_size[1] - video_size[1]) / 2
    screen.fill((0, 0, 0))
    screen.blit(surface, (width_offset, height_offset))


def play(env: gym.Env, key_to_action: dict[str, Any]):  # noqa: C901
    """Play a gym environment.

    Args:
        env: The environment.
        key_to_action: Mapping of keys to actions.

    """
    if env.render_mode != "rgb_array":
        raise ValueError("Render mode must be 'rgb_array'.")
    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise TypeError("Action space must be `Discrete`")

    obs, _info = env.reset()
    terminated, truncated = False, False
    return_ = 0.0
    print(f"Obs: \n{obs}")

    render_img = cast(np.ndarray, env.render())
    window_size = cast(tuple[int, int], render_img.shape[:2])

    pygame.init()
    pygame.display.init()
    pygame.display.set_caption(env.unwrapped.__class__.__name__)
    screen = pygame.display.set_mode(window_size, pygame.RESIZABLE)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                key = pygame.key.name(int(event.key))
                if key == "escape":
                    running = False
                    break
                elif key == "tab":
                    print("Manual reset.\n")
                    obs, _info = env.reset()
                    terminated, truncated = False, False
                    return_ = 0.0
                    print(f"Obs: \n{obs}")
                    break
                elif terminated or truncated:
                    obs, _info = env.reset()
                    terminated, truncated = False, False
                    return_ = 0.0
                    print(f"Obs: \n{obs}")
                else:
                    action = key_to_action[key]
                    print(f"Action: {action}")

                    obs, reward, terminated, truncated, _info = env.step(action)
                    return_ += float(reward)
                    print(f"Obs: \n{obs}")
                    print(f"Reward: {reward}, Return: {return_}")
                    if terminated:
                        print("Episode terminated.\n")
                    elif truncated:
                        print("Episode truncated.\n")

            elif event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.WINDOWRESIZED:
                scale_width = event.x / window_size[0]
                scale_height = event.y / window_size[1]
                scale = min(scale_height, scale_width)
                window_size = (scale * window_size[0], scale * window_size[1])

            render_img = cast(np.ndarray, env.render())
            _display_arr(screen, render_img, video_size=window_size)

            pygame.display.flip()

    env.close()
    pygame.quit()


def play_live(  # noqa: C901
    env: gym.Env,
    key_to_action: dict[str, Any],
    fps: float | None = None,
    noop: str | None = None,
):
    """Play a gym environment in real-time.

    Args:
        env: The environment.
        key_to_action: Mapping of keys to actions.
        fps: Rendered frames per second. Defaults to `env.render_fps` or 1.0 otherwise.
        noop: The default key when no keyboard operation is recorded.

    """
    if env.render_mode != "rgb_array":
        raise ValueError("Render mode must be 'rgb_array'.")
    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise TypeError("Action space must be `Discrete`")

    obs, _info = env.reset()
    terminated, truncated = False, False
    return_ = 0.0
    print(f"Obs: \n{obs}")

    render_img = cast(np.ndarray, env.render())
    window_size = cast(tuple[int, int], render_img.shape[:2])

    pygame.init()
    pygame.display.init()
    pygame.display.set_caption(env.unwrapped.__class__.__name__)
    screen = pygame.display.set_mode(window_size, pygame.RESIZABLE)
    clock = pygame.time.Clock()

    if fps is None:
        fps = env.metadata.get("render_fps", 1.0)

    key = "escape"
    running = True
    while running:
        if noop is not None:
            key = noop
        elif key not in key_to_action.keys():
            key = list(key_to_action.keys())[0]

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                key = pygame.key.name(int(event.key))
            elif event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.WINDOWRESIZED:
                scale_width = event.x / window_size[0]
                scale_height = event.y / window_size[1]
                scale = min(scale_height, scale_width)
                window_size = (scale * window_size[0], scale * window_size[1])

        if key == "escape":
            running = False
        elif key == "tab":
            print("Manual reset.\n")
            obs, _info = env.reset()
            terminated, truncated = False, False
            return_ = 0.0
            print(f"Obs: \n{obs}")
        elif terminated or truncated:
            obs, _info = env.reset()
            terminated, truncated = False, False
            return_ = 0.0
            print(f"Obs: \n{obs}")
        else:
            action = key_to_action[key]
            print(f"Action: {action}")

            obs, reward, terminated, truncated, _info = env.step(action)
            return_ += float(reward)
            print(f"Obs: \n{obs}")
            print(f"Reward: {reward}, Return: {return_}")
            if terminated:
                print("Episode terminated.\n")
            elif truncated:
                print("Episode truncated.\n")

        render_img = cast(np.ndarray, env.render())
        _display_arr(screen, render_img, video_size=window_size)

        pygame.display.flip()
        clock.tick(fps)

    env.close()
    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play an environment.")

    parser.add_argument("--config", help="Path to env config file.")
    parser.add_argument("--env", help="Environment ID.")
    parser.add_argument(
        "--keys", nargs="*", help="Sequence of keys that map to `Discrete` actions."
    )
    parser.add_argument(
        "--continuous", action="store_true", help="Activate continuous play."
    )
    parser.add_argument("--fps", type=float, help="Frames per second")
    parser.add_argument("--noop", help="NOOP key w/o keyboard input")

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

    env = gym.make(env_id, render_mode="rgb_array")

    if args.continuous:
        play_live(env, key_to_action, fps=args.fps, noop=args.noop)
    else:
        play(env, key_to_action)
