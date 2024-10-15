import os

PROJECT_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))
SOURCE_DIR = os.path.join(PROJECT_ROOT_DIR, "src")
ENVIRONMENTS_DIR = os.path.join(PROJECT_ROOT_DIR, "src", "environments")
SCENES_DIR = os.path.join(PROJECT_ROOT_DIR, "src", "environments", "scenes")
CONTROL_CONFIGS_DIR = os.path.join(SOURCE_DIR, "control", "control_configs")

RECORDING_DIR = os.path.join(PROJECT_ROOT_DIR, "recordings")

WAYPOINTS_DIR = os.path.join(PROJECT_ROOT_DIR, "src", "data", "waypoints", "files")