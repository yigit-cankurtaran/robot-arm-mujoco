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


def test_loaded_motion_scale_adapts_to_shape_stability() -> None:
    env = FactoryFloorEnv(enable_rgb_observation=False)
    try:
        env.active_part_order = ["part_0", "part_3"]

        env.holding_part = "part_0"
        assert env._loaded_motion_scale("transfer") == (
            env.stable_part_transfer_duration_scale
        )
        assert env._loaded_motion_scale("bin_hover") == (
            env.stable_part_bin_hover_duration_scale
        )

        env.holding_part = "part_3"
        assert env._loaded_motion_scale("transfer") == (
            env.transfer_duration_scale
        )
        assert env._loaded_motion_scale("bin_hover") == (
            env.bin_hover_duration_scale
        )
    finally:
        env.close()


def test_next_pick_branches_before_exact_home_only_after_safe_fraction() -> None:
    env = FactoryFloorEnv(enable_rgb_observation=False)
    try:
        env.active_part = env.active_part_order[0]
        env.controller_phase = "return_home"
        env.trajectory_phase = "return_home"
        env.trajectory_duration = 2.0

        env.trajectory_elapsed = 1.79
        env._advance_controller_state()
        assert env.controller_phase == "return_home"

        env.trajectory_elapsed = 1.8
        env._advance_controller_state()
        assert env.controller_phase == "move_to_pick_hover"
    finally:
        env.close()
