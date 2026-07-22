from __future__ import annotations

import numpy as np
import torch

from visual_policy import CameraCalibration, TinyUNet, relational_color_distance


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

