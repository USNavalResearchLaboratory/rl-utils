from collections import deque

import optuna
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.type_aliases import GymEnv


class LogTruncationCallback(BaseCallback):
    """Callback for logging episode truncations.

    Buffers truncation values in the same fashion as `ep_info_buffer`. Converts to `int`
    and averages buffered values, providing an estimate of the probability of
    truncation.

    """

    # TODO: move to a separate module

    def _on_training_start(self) -> None:
        self.buffer: deque = deque(maxlen=self.model._stats_window_size)

    def _on_step(self) -> bool:
        dones = self.locals["dones"]
        infos = self.locals["infos"]
        for done, info in zip(dones, infos):
            if done:
                truncated = info.get("TimeLimit.truncated", False)
                self.buffer.append(truncated)

        return True

    def _on_rollout_end(self) -> None:
        ep_trunc_mean = sum(self.buffer) / len(self.buffer)
        self.logger.record("rollout/ep_trunc_mean", ep_trunc_mean)


class TrialEvalCallback(EvalCallback):
    """Callback for evaluating an agent and reporting a trial.

    Note:
        Modded from https://github.com/DLR-RM/rl-baselines3-zoo

    Args:
        eval_env: The environment used for initialization.
        trial: Trial to report eval rewards to.
        callback_on_new_best: Callback to trigger when there is a new best model
            according to the ``mean_reward``.
        callback_after_eval: Callback to trigger after every evaluation.
        n_eval_episodes: The number of episodes to test the agent.
        eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
        log_path: Path to a folder where the evaluations (``evaluations.npz``) will be
            saved. It will be updated at each evaluation.
        best_model_save_path: Path to a folder where the best model according to
            performance on the eval env will be saved.
        deterministic: Whether the evaluation should use a stochastic or deterministic
            actions.
        render: Whether to render or not the environment during evaluation.
        verbose: Verbosity level: 0 for no output, 1 for indicating information about
            evaluation results.
        warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been wrapped
            with a Monitor wrapper).

    """

    def __init__(
        self,
        eval_env: GymEnv,
        trial: optuna.Trial,
        callback_on_new_best: BaseCallback | None = None,
        callback_after_eval: BaseCallback | None = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: str | None = None,
        best_model_save_path: str | None = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
    ):
        super().__init__(
            eval_env=eval_env,
            callback_on_new_best=callback_on_new_best,
            callback_after_eval=callback_after_eval,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            log_path=log_path,
            best_model_save_path=best_model_save_path,
            deterministic=deterministic,
            render=render,
            verbose=verbose,
            warn=warn,
        )
        self._trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self):
        continue_training = super()._on_step()
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.eval_idx += 1
            # report best or report current ?
            # report num_timesteps or elasped time ?
            self._trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if need
            if self._trial.should_prune():
                self.is_pruned = True
                continue_training = False
        return continue_training
