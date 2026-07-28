from __future__ import annotations

import mujoco
import numpy as np

from audit_motion import _unsafe_contact_pairs
from demo_runtime import sort_has_failed
from env import FactoryFloorEnv


def test_robotiq_gripper_is_part_of_the_compiled_model() -> None:
    env = FactoryFloorEnv(enable_rgb_observation=False)
    try:
        assert env.model.nu == env.arm_dofs + 3
        assert env.model.nmocap == 0
        assert env.gripper_actuator_id >= env.arm_dofs
        assert all(actuator_id >= env.arm_dofs for actuator_id in env.gripper_adhesion_actuator_ids)
        assert all(env.gripper_pad_geom_ids.values())
        visual_spacer_id = mujoco.mj_name2id(
            env.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "gripper/mount_spacer_visual",
        )
        collision_spacer_id = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_GEOM, "gripper/mount_spacer"
        )
        assert visual_spacer_id >= 0 and collision_spacer_id >= 0
        assert env.model.geom_group[visual_spacer_id] == 2
        assert env.model.geom_contype[visual_spacer_id] == 0
        assert env.model.geom_conaffinity[visual_spacer_id] == 0
        assert mujoco.mj_id2name(
            env.model, mujoco.mjtObj.mjOBJ_SITE, env.ee_site_id
        ) == "gripper/pinch"
    finally:
        env.close()


def test_physical_gripper_sorts_rounded_part_without_unsafe_contact() -> None:
    env = FactoryFloorEnv(
        enable_rgb_observation=False,
        min_active_parts=1,
        max_active_parts=1,
    )
    env.random_state = np.random.default_rng(4)  # Ellipsoid regression case.
    env.reset()
    unsafe_contacts: set[str] = set()
    try:
        for _ in range(1_500):
            env.scripted_step()
            unsafe_contacts.update(_unsafe_contact_pairs(env))
            if env.controller_phase == "idle" or sort_has_failed(env):
                break

        assert env.controller_phase == "idle"
        assert env.completed_parts == set(env.active_part_order)
        assert unsafe_contacts == set()
    finally:
        env.close()


def test_contact_width_hold_prevents_randomized_brick_ejection() -> None:
    """Regression for the marginal full-closure grasp reproduced by seed 3."""
    env = FactoryFloorEnv(enable_rgb_observation=False)
    env.random_state = np.random.default_rng(3)
    env.reset()
    saw_contact_hold = False
    try:
        for _ in range(5_000):
            env.scripted_step()
            if env.holding_part == "part_7":
                saw_contact_hold = True
                assert env.gripper_hold_ctrl is not None
                assert env.gripper_open_ctrl < env.gripper_hold_ctrl
                assert env.gripper_hold_ctrl < env.gripper_close_ctrl
            if "part_7" in env.completed_parts or sort_has_failed(env):
                break

        assert saw_contact_hold
        assert "part_7" in env.completed_parts
        assert env.object_drop_event is None
    finally:
        env.close()


def test_lost_held_object_enters_stable_terminal_failure() -> None:
    env = FactoryFloorEnv(enable_rgb_observation=False)
    env.random_state = np.random.default_rng(0)
    env.reset()
    try:
        part_name = env.active_part_order[0]
        env.active_part = part_name
        env.holding_part = part_name
        env.controller_phase = "move_to_transfer"
        env.grasp_lost_elapsed = env.grasp_loss_timeout - env.control_dt

        # The reset layout is well separated from the home pinch site, which
        # deterministically simulates an object that has left the gripper.
        assert not env._grasp_is_retained(part_name)
        env._update_task_progress()

        assert env.controller_phase == "object_dropped"
        assert env.holding_part is None
        assert env.gripper_hold_ctrl is None
        assert sort_has_failed(env)
        event = env.object_drop_event
        assert event is not None
        assert event.part_name == part_name
        assert event.controller_phase == "move_to_transfer"
        assert event.detected_time == env.data.time
        assert event.separation >= env.max_grasp_separation

        env._advance_controller_state()
        assert env.controller_phase == "object_dropped"
    finally:
        env.close()
