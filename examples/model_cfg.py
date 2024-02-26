from functools import partial

from stable_baselines3.common.callbacks import (
    StopTrainingOnNoModelImprovement,
    StopTrainingOnRewardThreshold,
)
from torch import nn

from rl_utils.extractors import MLPExtractor

model_cfg = dict(
    algo="PPO",
    policy="MlpPolicy",
    algo_kwargs=dict(
        policy_kwargs=dict(
            features_extractor_class=MLPExtractor,
            features_extractor_kwargs=dict(
                hidden_sizes=[1024],
            ),
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
            activation_fn=nn.ReLU,
            normalize_images=False,
        ),
        learning_rate=3e-4,
        n_steps_total=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
    ),
    params_path=None,
    total_timesteps=50000,
    callbacks=[],
    eval_callback_kwargs=dict(
        callback_on_new_best=partial(
            StopTrainingOnRewardThreshold, reward_threshold=1000
        ),
        callback_after_eval=partial(
            StopTrainingOnNoModelImprovement, max_no_improvement_evals=3, min_evals=2
        ),
        n_eval_episodes=10,
        eval_freq_total=10000,
    ),
)
