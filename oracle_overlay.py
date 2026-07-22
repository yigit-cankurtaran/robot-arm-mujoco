from __future__ import annotations

import cv2
import mujoco
import numpy as np

from env import FactoryFloorEnv


def _geom_mask(segmentation: np.ndarray, geom_ids: list[int]) -> np.ndarray:
    is_geom = segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
    return np.isin(segmentation[..., 0], geom_ids) & is_geom


def _draw_instance(
    canvas: np.ndarray,
    mask: np.ndarray,
    label: str,
    color: tuple[int, int, int],
) -> tuple[int, int] | None:
    if not np.any(mask):
        return None
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(canvas, contours, -1, color, 2)
    y_pixels, x_pixels = np.nonzero(mask)
    center = (int(np.mean(x_pixels)), int(np.mean(y_pixels)))
    cv2.putText(
        canvas,
        label,
        center,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        color,
        2,
    )
    return center


def draw_oracle_task_overlay(
    env: FactoryFloorEnv, rgb: np.ndarray
) -> np.ndarray:
    """Annotate a copy for display without modifying the policy observation."""
    segmentation = env.render_segmentation()
    canvas = rgb.copy()
    bin_centers: dict[str, tuple[int, int]] = {}
    for index, bin_name in enumerate(env.bin_order):
        center = _draw_instance(
            canvas,
            _geom_mask(segmentation, env.bin_geom_ids[bin_name]),
            f"B{index}",
            (70, 130, 255),
        )
        if center is not None:
            bin_centers[bin_name] = center

    for index, part_name in enumerate(env.active_part_order):
        if part_name in env.completed_parts:
            continue
        center = _draw_instance(
            canvas,
            _geom_mask(segmentation, [env.part_geom_ids[part_name]]),
            f"P{index}",
            (60, 220, 80),
        )
        target_bin = str(env.part_specs[part_name]["target_bin"])
        if center is not None and target_bin in bin_centers:
            cv2.arrowedLine(
                canvas,
                center,
                bin_centers[target_bin],
                (240, 60, 50),
                1,
                tipLength=0.08,
            )
    return canvas
