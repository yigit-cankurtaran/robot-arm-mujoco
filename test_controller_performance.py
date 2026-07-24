from __future__ import annotations

import numpy as np

from env import FactoryFloorEnv


def test_active_trajectory_reuses_latched_ik_goal() -> None:
    env = FactoryFloorEnv(
        enable_rgb_observation=False,
        min_active_parts=1,
        max_active_parts=1,
    )
    env.random_state = np.random.default_rng(4)
    env.reset()
    original_solver = env._solve_inverse_kinematics
    calls = 0

    def counted_solver(target: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return original_solver(target)

    env._solve_inverse_kinematics = counted_solver
    try:
        for _ in range(20):
            env.scripted_step()
        assert calls == 1
        assert env.trajectory_phase == "move_to_pick_hover"
    finally:
        env.close()


def test_low_crowding_uses_faster_loaded_motion_scales() -> None:
    env = FactoryFloorEnv(enable_rgb_observation=False)
    try:
        assert (
            env.single_part_transfer_duration_scale
            < env.transfer_duration_scale
        )
        assert (
            env.single_part_bin_hover_duration_scale
            < env.bin_hover_duration_scale
        )
    finally:
        env.close()
