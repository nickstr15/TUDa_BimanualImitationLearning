import os

PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

RS_CONTROL_CONFIGS_DIR = os.path.join(PROJECT_ROOT_DIR, "robosuite", "control", "control_configs")
RS_ENVIRONMENTS_DIR = os.path.join(PROJECT_ROOT_DIR, "robosuite", "environments")
RS_SCENES_DIR = os.path.join(PROJECT_ROOT_DIR, "robosuite", "environments", "scenes")
RS_WAYPOINTS_DIR = os.path.join(PROJECT_ROOT_DIR, "robosuite", "demonstration", "waypoints", "files")
RS_MODELS_DIR = os.path.join(PROJECT_ROOT_DIR, "robosuite", "models")

DEMOS_DIR = os.path.join(PROJECT_ROOT_DIR, "data", "demonstrations")
RECORDING_DIR = os.path.join(PROJECT_ROOT_DIR, "data", "recordings")
TRAINED_MODELS_DIR = os.path.join(PROJECT_ROOT_DIR, "data", "trained_models")
LOG_DIR = os.path.join(PROJECT_ROOT_DIR, "data", "logs")

IL_PATH = os.path.join(PROJECT_ROOT_DIR, "robosuite", "imitation_learning")

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

