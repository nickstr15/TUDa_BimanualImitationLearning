import numpy as np

MUJOCO_TIME_STEP = 0.001
MUJOCO_RENDER_FPS = 30
MUJOCO_FRAME_SKIP = int(1.0 / (MUJOCO_TIME_STEP * MUJOCO_RENDER_FPS))

DEFAULT_WIDTH = 480
DEFAULT_HEIGHT = 360

MAX_DELTA_TRANSLATION = 0.15
MAX_DELTA_ROTATION = np.pi / 4    #45°