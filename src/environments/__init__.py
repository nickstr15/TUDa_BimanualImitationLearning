from robosuite.environments.base import register_env

from src.environments.manipulation.two_arm_empty import TwoArmEmpty
from src.environments.manipulation.two_arm_pick_place import TwoArmPickPlace

custom_envs = [
    TwoArmEmpty,
    TwoArmPickPlace
]

#for env in custom_envs:
#    register_env(env)