import gymnasium as gym
from src.environments import *

env = gym.make(id="BasePandaBimanualEnv-v0", render_mode='human')
env.visualize()


