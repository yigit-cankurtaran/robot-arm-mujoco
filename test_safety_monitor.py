from __future__ import annotations

import json

import mujoco

from env import ContactEvent, FactoryFloorEnv
from safety_monitor import (
    detect_safety_failure,
    safety_failure_message,
    write_safety_failure,
)


def first_geom_on_body(env: FactoryFloorEnv, body_name: str) -> int:
    body_id = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_BODY, body_name
    )
    return int(env.model.body_geomadr[body_id])


def test_elbow_bin_contact_stops_and_writes_diagnostic_log(tmp_path) -> None:
    env = FactoryFloorEnv(enable_rgb_observation=False)
    try:
        elbow_geom = first_geom_on_body(env, "upper_arm_link")
        bin_geom = env.bin_geom_ids["bin_0"][0]
        env.contact_events_this_step = [
            ContactEvent(
                geom1=elbow_geom,
                geom2=bin_geom,
                distance=-0.0012,
                position=(0.41, 0.28, 0.63),
                simulated_time=12.34,
                controller_phase="move_to_bin_hover",
                expected_part_geom_id=None,
            )
        ]

        failure = detect_safety_failure(env, wall_seconds=13.5)
        assert failure is not None
        assert failure.simulated_time == 12.34
        assert failure.contacts[0].position == (0.41, 0.28, 0.63)
        assert "upper_arm_link" in failure.contacts[0].pair
        assert "blue_bin" in failure.contacts[0].pair

        log_path = write_safety_failure(failure, tmp_path)
        record = json.loads(log_path.read_text())
        assert record["controller_phase"] == "move_to_bin_hover"
        assert record["contacts"][0]["distance"] == -0.0012
        message = safety_failure_message(failure, log_path)
        assert "Safety stop:" in message
        assert "t=12.340s simulated" in message
        assert str(log_path) in message
    finally:
        env.close()


def test_expected_fingertip_part_contact_is_not_a_failure() -> None:
    env = FactoryFloorEnv(enable_rgb_observation=False)
    try:
        part_name = env.active_part_order[0]
        part_geom = env.part_geom_ids[part_name]
        pad_geom = next(iter(env.gripper_pad_geom_ids["left"]))
        env.contact_events_this_step = [
            ContactEvent(
                geom1=pad_geom,
                geom2=part_geom,
                distance=-0.0005,
                position=(0.4, -0.2, 0.52),
                simulated_time=1.0,
                controller_phase="close_gripper",
                expected_part_geom_id=part_geom,
            )
        ]
        assert detect_safety_failure(env, wall_seconds=1.1) is None
    finally:
        env.close()


def test_gripper_table_top_contact_is_not_a_failure() -> None:
    env = FactoryFloorEnv(enable_rgb_observation=False)
    try:
        table_geom = env.model.geom("table_top").id
        pad_geom = next(iter(env.gripper_pad_geom_ids["right"]))
        env.contact_events_this_step = [
            ContactEvent(
                # Reverse ordering verifies that classification is symmetric.
                geom1=table_geom,
                geom2=pad_geom,
                distance=-0.0008,
                position=(0.4, -0.2, env.table_surface_z),
                simulated_time=2.0,
                controller_phase="close_gripper",
                expected_part_geom_id=None,
            )
        ]
        assert detect_safety_failure(env, wall_seconds=2.1) is None
    finally:
        env.close()


def test_arm_table_top_contact_remains_a_failure() -> None:
    env = FactoryFloorEnv(enable_rgb_observation=False)
    try:
        table_geom = env.model.geom("table_top").id
        arm_geom = first_geom_on_body(env, "upper_arm_link")
        env.contact_events_this_step = [
            ContactEvent(
                geom1=arm_geom,
                geom2=table_geom,
                distance=-0.001,
                position=(0.3, 0.0, env.table_surface_z),
                simulated_time=3.0,
                controller_phase="move_to_pick",
                expected_part_geom_id=None,
            )
        ]
        failure = detect_safety_failure(env, wall_seconds=3.1)
        assert failure is not None
        assert "upper_arm_link" in failure.contacts[0].pair
        assert "table_top" in failure.contacts[0].pair
    finally:
        env.close()


def test_gripper_bin_contact_remains_a_failure() -> None:
    env = FactoryFloorEnv(enable_rgb_observation=False)
    try:
        pad_geom = next(iter(env.gripper_pad_geom_ids["left"]))
        bin_geom = env.bin_geom_ids["bin_1"][0]
        env.contact_events_this_step = [
            ContactEvent(
                geom1=pad_geom,
                geom2=bin_geom,
                distance=-0.001,
                position=(0.62, 0.08, 0.62),
                simulated_time=4.0,
                controller_phase="move_to_drop",
                expected_part_geom_id=None,
            )
        ]
        failure = detect_safety_failure(env, wall_seconds=4.1)
        assert failure is not None
        assert "gripper/" in failure.contacts[0].pair
        assert "orange_bin" in failure.contacts[0].pair
    finally:
        env.close()
