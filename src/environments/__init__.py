from gymnasium.envs.registration import register
from src.environments.empty_panda_env import EmptyPandaEnv
from src.environments.panda_handover_env import PandaHandoverEnv

register(
    id="EmptyPandaEnv-v0",
    entry_point="src.environments:EmptyPandaEnv",
    max_episode_steps=1000,
)

register(
    id="PandaHandoverEnv-v0",
    entry_point="src.environments:PandaHandoverEnv",
    max_episode_steps=1000,
)