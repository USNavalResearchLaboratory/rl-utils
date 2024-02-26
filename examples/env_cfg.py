from gymnasium.wrappers import RecordEpisodeStatistics

env_cfg = dict(
    id="CartPole-v1",
    max_episode_steps=1000,
    wrappers=[RecordEpisodeStatistics],
)
