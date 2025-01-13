import os

PROJECT_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

# DATA
DEMOS_DIR = os.path.join(PROJECT_ROOT_DIR, "data", "demonstrations")
DATASET_DIR = os.path.join(PROJECT_ROOT_DIR, "data", "datasets")
RECORDING_DIR = os.path.join(PROJECT_ROOT_DIR, "data", "recordings")
TRAINED_MODELS_DIR = os.path.join(PROJECT_ROOT_DIR, "data", "trained_models")
LOG_DIR = os.path.join(PROJECT_ROOT_DIR, "data", "logs")

# ROBOSUITE_EXT
RS_CONTROL_CONFIGS_DIR = os.path.join(PROJECT_ROOT_DIR, "robosuite_ext", "control", "control_configs")
RS_ENVIRONMENTS_DIR = os.path.join(PROJECT_ROOT_DIR, "robosuite_ext", "environments")
RS_SCENES_DIR = os.path.join(PROJECT_ROOT_DIR, "robosuite_ext", "environments", "scenes")
RS_WAYPOINTS_DIR = os.path.join(PROJECT_ROOT_DIR, "robosuite_ext", "demonstration", "waypoints", "files")
RS_MODELS_DIR = os.path.join(PROJECT_ROOT_DIR, "robosuite_ext", "models")

# ROBOMIMIC_EXT
...


def path_completion(path: str, root: str = None) -> str:
    """
        Takes in a local asset path and returns a full path.
            if @path is absolute, do nothing
            if @path is not absolute, load xml that is shipped by the package

        Args:
            path (str): local path
            root (str): root folder for asset path.

        Returns:
            str: Full (absolute) xml path
        """
    if path.startswith("/") or root is None:
        full_path = path
    else:
        full_path = os.path.join(root, path)
    return full_path

