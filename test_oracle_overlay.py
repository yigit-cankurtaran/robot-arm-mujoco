from __future__ import annotations

import numpy as np

from env import FactoryFloorEnv
from oracle_overlay import draw_oracle_task_overlay


def test_oracle_overlay_is_display_only_and_contains_annotations() -> None:
    env = FactoryFloorEnv(min_active_parts=3, max_active_parts=3)
    env.random_state = np.random.default_rng(1)
    observation, _ = env.reset()
    rgb = observation["rgb"]
    original = rgb.copy()
    try:
        overlay = draw_oracle_task_overlay(env, rgb)
        assert overlay.shape == rgb.shape
        assert np.array_equal(rgb, original)
        assert np.count_nonzero(overlay != rgb) > 100
    finally:
        env.close()
