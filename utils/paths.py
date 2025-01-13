import os

PROJECT_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

# DATA
DEMOS_DIR = os.path.join(PROJECT_ROOT_DIR, "data", "demonstrations")
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


def path_completion(asset_path: str, root: str = None) -> str:
    """
        Takes in a local asset path and returns a full path.
            if @asset_path is absolute, do nothing
            if @asset_path is not absolute, load xml that is shipped by the package

        Args:
            asset_path (str): local asset path
            root (str): root folder for asset path. If not specified defaults to MODELS_DIR/assets

        Returns:
            str: Full (absolute) xml path
        """
    if asset_path.startswith("/"):
        full_path = asset_path
    else:
        if root is None:
            root = os.path.join(RS_MODELS_DIR, "assets")
        full_path = os.path.join(root, asset_path)
    return full_path

