from gymnasium.envs.registration import register
from src.environments.empty_panda_env import EmptyPandaEnv
from src.environments.panda_handover_env import PandaHandoverEnv
from src.environments.panda_quad_insert_env import PandaQuadInsertEnv

register(
    id="EmptyPandaEnv",
    entry_point="src.environments:EmptyPandaEnv",
    max_episode_steps=1000,
)

register(
    id="PandaHandoverEnv",
    entry_point="src.environments:PandaHandoverEnv",
    max_episode_steps=1000,
)

register(
    id="PandaQuadInsertEnv",
    entry_point="src.environments:PandaQuadInsertEnv",
    max_episode_steps=1000,
)