from gymnasium.envs.registration import register
from src.environments.base_bimanual_env import BasePandaBimanualEnv

register(
    id="BasePandaBimanualEnv-v0",
    entry_point="src.environments:BasePandaBimanualEnv",
    max_episode_steps=1000,
)