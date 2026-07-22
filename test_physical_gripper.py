from __future__ import annotations

import mujoco
import numpy as np

from audit_motion import _unsafe_contact_pairs
from env import FactoryFloorEnv


def test_robotiq_gripper_is_part_of_the_compiled_model() -> None:
    env = FactoryFloorEnv(enable_rgb_observation=False)
    try:
        assert env.model.nu == env.arm_dofs + 3
        assert env.model.nmocap == 0
        assert env.gripper_actuator_id >= env.arm_dofs
        assert all(actuator_id >= env.arm_dofs for actuator_id in env.gripper_adhesion_actuator_ids)
        assert all(env.gripper_pad_geom_ids.values())
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
            if env.controller_phase in {"idle", "drop_failed"}:
                break

        assert env.controller_phase == "idle"
        assert env.completed_parts == set(env.active_part_order)
        assert unsafe_contacts == set()
    finally:
        env.close()
