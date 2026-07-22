from __future__ import annotations

import numpy as np
import torch

from visual_policy import (
    CameraCalibration,
    TinyUNet,
    extract_instances,
    relational_color_distance,
)


def test_tiny_unet_preserves_image_shape() -> None:
    model = TinyUNet(base_channels=4)
    output = model(torch.zeros((2, 3, 64, 64)))
    assert output.shape == (2, 3, 64, 64)


def test_center_pixel_projects_to_camera_xy() -> None:
    calibration = CameraCalibration(width=240, height=240)
    world = calibration.pixel_to_world((119.5, 119.5))
    np.testing.assert_allclose(world, [0.56, 0.0, 0.52], atol=1e-6)


def test_relational_color_distance_ignores_brightness_scale() -> None:
    base = np.array([80, 40, 20], dtype=np.float32)
    darker_same_color = np.array([40, 20, 10], dtype=np.float32)
    different_color = np.array([20, 40, 80], dtype=np.float32)
    assert relational_color_distance(base, darker_same_color) < 1e-6
    assert relational_color_distance(base, different_color) > 0.2


def test_bridged_bin_mask_is_split_into_two_instances() -> None:
    mask = np.zeros((120, 120), dtype=bool)
    mask[10:50, 35:85] = True
    mask[70:110, 35:85] = True
    mask[49:71, 58:61] = True
    rgb = np.zeros((120, 120, 3), dtype=np.uint8)
    rgb[10:50, 35:85] = [30, 180, 220]
    rgb[70:110, 35:85] = [230, 180, 30]
    instances = extract_instances(
        rgb,
        mask,
        np.ones((120, 120), dtype=np.float32),
        "bin",
        CameraCalibration(width=120, height=120),
        minimum_area=250,
    )
    assert len(instances) == 2
    assert abs(instances[0].centroid_xy[1] - instances[1].centroid_xy[1]) > 40
