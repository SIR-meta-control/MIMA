import numpy as np
import pytest

from mima_vr.baseline_features import (
    extract_depth_features,
    extract_point_cloud_features,
    extract_rgb_features,
)


def test_feature_extractors_are_finite_and_sensor_only():
    rgb = np.full((4, 4, 3), 128, dtype=np.uint8)
    depth = np.array([[1.0, 1.1], [1.2, 1.3]], dtype=np.float32)
    points = np.array(
        [
            [-0.5, 1.0, -0.5],
            [0.5, 1.0, -0.5],
            [-0.5, 2.0, -0.5],
            [0.5, 2.0, -0.5],
        ],
        dtype=np.float32,
    )
    values = {
        **extract_rgb_features(rgb),
        **extract_depth_features(depth),
        **extract_point_cloud_features(points),
    }
    assert all(np.isfinite(value) for value in values.values())
    assert values["pc_corridor_width_m"] == pytest.approx(1.0)


def test_depth_rejects_nonpositive_or_nonfinite_only_array():
    with pytest.raises(ValueError, match="no positive finite"):
        extract_depth_features(np.array([[0.0, np.nan], [-1.0, 0.0]]))
