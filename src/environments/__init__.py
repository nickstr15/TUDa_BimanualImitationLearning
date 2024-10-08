from gymnasium.envs.registration import register
from src.environments.base_bimanual_env import BasePandaBimanualEnv
from src.environments.bimanual_handover_env import PandaBimanualHandoverEnv

register(
    id="BasePandaBimanualEnv-v0",
    entry_point="src.environments:BasePandaBimanualEnv",
    max_episode_steps=1000,
)

register(
    id="PandaBimanualHandoverEnv-v0",
    entry_point="src.environments:PandaBimanualHandoverEnv",
    max_episode_steps=1000,
)