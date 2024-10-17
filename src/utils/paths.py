import os

PROJECT_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))

CONTROL_CONFIGS_DIR = os.path.join(PROJECT_ROOT_DIR, "src", "control", "control_configs")
DEMOS_DIR = os.path.join(PROJECT_ROOT_DIR, "src", "data", "demos")
ENVIRONMENTS_DIR = os.path.join(PROJECT_ROOT_DIR, "src", "environments")
RECORDING_DIR = os.path.join(PROJECT_ROOT_DIR, "recordings")
SCENES_DIR = os.path.join(PROJECT_ROOT_DIR, "src", "environments", "scenes")
WAYPOINTS_DIR = os.path.join(PROJECT_ROOT_DIR, "src", "data", "waypoints", "files")

