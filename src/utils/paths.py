import os

PROJECT_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))

CONTROL_CONFIGS_DIR = os.path.join(PROJECT_ROOT_DIR, "src", "control", "control_configs")
DEMOS_DIR = os.path.join(PROJECT_ROOT_DIR, "demonstrations")
ENVIRONMENTS_DIR = os.path.join(PROJECT_ROOT_DIR, "src", "environments")
RECORDING_DIR = os.path.join(PROJECT_ROOT_DIR, "recordings")
SCENES_DIR = os.path.join(PROJECT_ROOT_DIR, "src", "environments", "scenes")
WAYPOINTS_DIR = os.path.join(PROJECT_ROOT_DIR, "src", "demonstration", "waypoints", "files")
MODELS_DIR = os.path.join(PROJECT_ROOT_DIR, "src", "models")

PSMOVEAPI_LIBRARY_PATH = os.path.join(os.path.expanduser("~"), "psmoveapi", "build")

def asset_path_completion(asset_path: str, root: str = None) -> str:
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
            root = os.path.join(MODELS_DIR, "assets")
        full_path = os.path.join(root, asset_path)
    return full_path

