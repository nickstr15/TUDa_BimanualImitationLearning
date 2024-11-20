import os

PROJECT_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))

CONTROL_CONFIGS_DIR = os.path.join(PROJECT_ROOT_DIR, "src", "control", "control_configs")
DEMOS_DIR = os.path.join(PROJECT_ROOT_DIR, "demonstrations")
ENVIRONMENTS_DIR = os.path.join(PROJECT_ROOT_DIR, "src", "environments")
RECORDING_DIR = os.path.join(PROJECT_ROOT_DIR, "recordings")
SCENES_DIR = os.path.join(PROJECT_ROOT_DIR, "src", "environments", "scenes")
WAYPOINTS_DIR = os.path.join(PROJECT_ROOT_DIR, "src", "demonstration", "waypoints", "files")
MODELS_DIR = os.path.join(PROJECT_ROOT_DIR, "src", "models")

def xml_path_completion(xml_path: str, root: str = None) -> str:
    """
        Takes in a local xml path and returns a full path.
            if @xml_path is absolute, do nothing
            if @xml_path is not absolute, load xml that is shipped by the package

        Args:
            xml_path (str): local xml path
            root (str): root folder for xml path. If not specified defaults to robosuite.models.assets_root

        Returns:
            str: Full (absolute) xml path
        """
    if xml_path.startswith("/"):
        full_path = xml_path
    else:
        if root is None:
            root = os.path.join(MODELS_DIR, "assets")
        full_path = os.path.join(root, xml_path)
    return full_path

