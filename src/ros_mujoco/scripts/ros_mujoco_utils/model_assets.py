"""Locate the unchanged Crimson MJCF in source and Catkin install spaces."""

from pathlib import Path

MODEL_PACKAGE = "crimson_stand_legInit_forSimOnly"
MODEL_RELATIVE_PATH = Path("mjcf/crimson_scene.xml")


def find_crimson_scene():
    source = (
        Path(__file__).resolve().parents[3] / "models/crimson" / MODEL_RELATIVE_PATH
    )
    if source.is_file():
        return source
    try:
        import rospkg
    except ImportError as exc:
        raise FileNotFoundError(
            "Crimson MJCF not found in the checkout. For an installed ROS workspace, "
            "install rospkg and the %s model package." % MODEL_PACKAGE
        ) from exc
    try:
        installed = Path(rospkg.RosPack().get_path(MODEL_PACKAGE)) / MODEL_RELATIVE_PATH
    except rospkg.ResourceNotFound as exc:
        raise FileNotFoundError(
            "Missing ROS model package: %s" % MODEL_PACKAGE
        ) from exc
    # rospkg resolves the source URDF package in a devel workspace.
    for candidate in (installed, installed.parent.parent.parent / MODEL_RELATIVE_PATH):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Crimson MJCF not found under %s" % installed.parent.parent)
