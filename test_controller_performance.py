from __future__ import annotations

import numpy as np

from audit_motion import _unsafe_contact_pairs
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


def test_single_unloaded_motion_uses_higher_bounded_derivative_limits() -> None:
    env = FactoryFloorEnv(
        enable_rgb_observation=False,
        min_active_parts=1,
        max_active_parts=1,
    )
    try:
        env.controller_phase = "move_to_pick_hover"
        env.holding_part = None
        velocity, acceleration, jerk = env._trajectory_derivative_limits()
        np.testing.assert_allclose(
            velocity,
            env.single_unloaded_limit_scale * env.max_joint_velocity,
        )
        np.testing.assert_allclose(
            acceleration,
            env.single_unloaded_limit_scale * env.max_joint_acceleration,
        )
        np.testing.assert_allclose(
            jerk,
            env.single_unloaded_limit_scale * env.max_joint_jerk,
        )

        env.holding_part = env.active_part_order[0]
        velocity, acceleration, jerk = env._trajectory_derivative_limits()
        np.testing.assert_allclose(velocity, env.max_joint_velocity)
        np.testing.assert_allclose(acceleration, env.max_joint_acceleration)
        np.testing.assert_allclose(jerk, env.max_joint_jerk)

        env.holding_part = None
        env.active_part = None
        env.controller_phase = "return_home"
        velocity, acceleration, jerk = env._trajectory_derivative_limits()
        np.testing.assert_allclose(
            velocity,
            np.minimum(
                env.shutdown_unloaded_limit_scale * env.max_joint_velocity,
                env.hardware_joint_velocity_limit,
            ),
        )
        np.testing.assert_allclose(
            acceleration,
            env.shutdown_unloaded_limit_scale
            * env.max_joint_acceleration,
        )
        np.testing.assert_allclose(
            jerk,
            env.shutdown_unloaded_limit_scale * env.max_joint_jerk,
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

        env.trajectory_elapsed = 1.59
        env._advance_controller_state()
        assert env.controller_phase == "return_home"

        env.trajectory_elapsed = 1.6
        env._advance_controller_state()
        assert env.controller_phase == "move_to_pick_hover"
    finally:
        env.close()


def test_mid_trajectory_replan_preserves_derivative_state_and_limits() -> None:
    env = FactoryFloorEnv(enable_rgb_observation=False)
    try:
        first_goal = np.clip(
            env.home_ctrl + np.array([0.5, -0.2, 0.3, 0.2, -0.3, 0.25]),
            env.ctrl_low,
            env.ctrl_high,
        )
        env.controller_phase = "synthetic_first_segment"
        for _ in range(20):
            command = env._trajectory_command(first_goal)
            env.data.ctrl[: env.arm_dofs] = command

        branch_position = command.copy()
        branch_velocity = env.trajectory_velocity.copy()
        branch_acceleration = env.trajectory_acceleration.copy()
        assert np.max(np.abs(branch_velocity)) > 0.05

        second_goal = np.clip(
            env.home_ctrl + np.array([-0.3, 0.25, -0.2, 0.1, 0.2, -0.15]),
            env.ctrl_low,
            env.ctrl_high,
        )
        env.controller_phase = "synthetic_second_segment"
        env._trajectory_command(second_goal)

        start_state = env._quintic_state(
            env.trajectory_coefficients,
            env.trajectory_duration,
            0.0,
        )
        np.testing.assert_allclose(start_state[0], branch_position, atol=1e-12)
        np.testing.assert_allclose(start_state[1], branch_velocity, atol=1e-12)
        np.testing.assert_allclose(
            start_state[2], branch_acceleration, atol=1e-12
        )

        elapsed = np.linspace(0.0, env.trajectory_duration, 1001)
        _, velocity, acceleration, jerk = env._quintic_state(
            env.trajectory_coefficients,
            env.trajectory_duration,
            elapsed,
        )
        assert np.all(
            np.max(np.abs(velocity), axis=0)
            <= env.max_joint_velocity + 1e-9
        )
        assert np.all(
            np.max(np.abs(acceleration), axis=0)
            <= env.max_joint_acceleration + 1e-9
        )
        assert np.all(
            np.max(np.abs(jerk), axis=0)
            <= env.trajectory_jerk_margin * env.max_joint_jerk + 1e-9
        )
        np.testing.assert_allclose(
            env._quintic_state(
                env.trajectory_coefficients,
                env.trajectory_duration,
                env.trajectory_duration,
            )[0],
            second_goal,
            atol=1e-10,
        )
    finally:
        env.close()


def test_route_preview_has_clearance_without_changing_live_contacts() -> None:
    env = FactoryFloorEnv(enable_rgb_observation=False)
    try:
        table_geom_id = env.model.geom("table_top").id
        assert env.model.geom_margin[table_geom_id] == 0.0
        assert (
            env.collision_model.geom_margin[table_geom_id]
            == env.route_collision_margin
        )
        for name in env.active_part_order:
            geom_id = env.part_geom_ids[name]
            assert env.model.geom_contype[geom_id] != 0
        for geom_id in env.part_geom_ids.values():
            assert env.collision_model.geom_contype[geom_id] == 0
    finally:
        env.close()


def test_swept_route_avoids_randomized_bin_graze_regression() -> None:
    env = FactoryFloorEnv(enable_rgb_observation=False)
    env.random_state = np.random.default_rng(2)
    env.reset()
    unsafe_contacts: set[str] = set()
    try:
        # The old fixed pick attitude touched the randomized orange bin near
        # t=21.3 s in this rollout.
        for _ in range(1_300):
            env.scripted_step()
            unsafe_contacts.update(_unsafe_contact_pairs(env))

        assert env.route_preflight_rejections > 0
        assert env.controller_phase != "route_failed"
        assert unsafe_contacts == set()
    finally:
        env.close()
