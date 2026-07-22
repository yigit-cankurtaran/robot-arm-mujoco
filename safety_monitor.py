from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import mujoco

from env import ContactEvent, FactoryFloorEnv


ROOT = Path(__file__).resolve().parent
DEFAULT_FAILURE_DIRECTORY = ROOT / "runs" / "safety_failures"


@dataclass(frozen=True)
class UnsafeContact:
    pair: str
    geom1: str
    geom2: str
    body1: str
    body2: str
    position: tuple[float, float, float]
    distance: float
    simulated_time: float
    controller_phase: str


@dataclass(frozen=True)
class SafetyFailure:
    recorded_at_utc: str
    simulated_time: float
    wall_seconds: float
    controller_phase: str
    active_part: str | None
    holding_part: str | None
    released_part: str | None
    completed_parts: list[str]
    active_parts: list[str]
    ee_position: list[float]
    arm_qpos: list[float]
    arm_ctrl: list[float]
    contacts: list[UnsafeContact]


def _body_descends_from(
    model: mujoco.MjModel, body_id: int, root_id: int
) -> bool:
    while body_id > 0:
        if body_id == root_id:
            return True
        body_id = int(model.body_parentid[body_id])
    return False


def _geom_label(env: FactoryFloorEnv, geom_id: int) -> str:
    geom_name = env.model.geom(geom_id).name
    if geom_name:
        return geom_name
    body_id = int(env.model.geom_bodyid[geom_id])
    return f"{env.model.body(body_id).name}:collision"


def _event_is_unsafe(env: FactoryFloorEnv, event: ContactEvent) -> bool:
    model = env.model
    robot_root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    gripper_root = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "gripper/base_mount"
    )
    body1 = int(model.geom_bodyid[event.geom1])
    body2 = int(model.geom_bodyid[event.geom2])
    robot1 = _body_descends_from(model, body1, robot_root)
    robot2 = _body_descends_from(model, body2, robot_root)
    unsafe = robot1 != robot2

    if robot1 != robot2 and event.expected_part_geom_id is not None:
        robot_geom = event.geom1 if robot1 else event.geom2
        external_geom = event.geom2 if robot1 else event.geom1
        robot_body = int(model.geom_bodyid[robot_geom])
        if (
            _body_descends_from(model, robot_body, gripper_root)
            and external_geom == event.expected_part_geom_id
        ):
            unsafe = False

    if robot1 and robot2 and body1 != body2:
        unsafe = int(model.body_parentid[body1]) != body2 and int(
            model.body_parentid[body2]
        ) != body1
        if _body_descends_from(
            model, body1, gripper_root
        ) and _body_descends_from(model, body2, gripper_root):
            unsafe = False
    return unsafe


def unsafe_contact_events(env: FactoryFloorEnv) -> list[UnsafeContact]:
    """Return one diagnostic record per unsafe geom pair in the latest tick."""
    unique: dict[tuple[int, int], UnsafeContact] = {}
    for event in env.contact_events_this_step:
        if not _event_is_unsafe(env, event):
            continue
        key = tuple(sorted((event.geom1, event.geom2)))
        if key in unique:
            continue
        body1_id = int(env.model.geom_bodyid[event.geom1])
        body2_id = int(env.model.geom_bodyid[event.geom2])
        geom1 = _geom_label(env, event.geom1)
        geom2 = _geom_label(env, event.geom2)
        unique[key] = UnsafeContact(
            pair=f"{geom1} <-> {geom2}",
            geom1=geom1,
            geom2=geom2,
            body1=env.model.body(body1_id).name,
            body2=env.model.body(body2_id).name,
            position=event.position,
            distance=event.distance,
            simulated_time=event.simulated_time,
            controller_phase=event.controller_phase,
        )
    return list(unique.values())


def detect_safety_failure(
    env: FactoryFloorEnv, wall_seconds: float
) -> SafetyFailure | None:
    contacts = unsafe_contact_events(env)
    if not contacts:
        return None
    first = min(contacts, key=lambda contact: contact.simulated_time)
    return SafetyFailure(
        recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        simulated_time=first.simulated_time,
        wall_seconds=wall_seconds,
        controller_phase=first.controller_phase,
        active_part=env.active_part,
        holding_part=env.holding_part,
        released_part=env.released_part,
        completed_parts=sorted(env.completed_parts),
        active_parts=env.active_part_order.copy(),
        ee_position=env.data.site_xpos[env.ee_site_id].round(6).tolist(),
        arm_qpos=env.data.qpos[: env.arm_dofs].round(6).tolist(),
        arm_ctrl=env.data.ctrl[: env.arm_dofs].round(6).tolist(),
        contacts=contacts,
    )


def write_safety_failure(
    failure: SafetyFailure,
    directory: Path = DEFAULT_FAILURE_DIRECTORY,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    path = directory / f"contact_failure_{stamp}.json"
    path.write_text(json.dumps(asdict(failure), indent=2) + "\n")
    return path


def safety_failure_message(failure: SafetyFailure, log_path: Path) -> str:
    contact = failure.contacts[0]
    position = ", ".join(f"{value:.3f}" for value in contact.position)
    return (
        f"Safety stop: {contact.pair} at world [{position}] during "
        f"{failure.controller_phase}, t={failure.simulated_time:.3f}s simulated "
        f"({failure.wall_seconds:.2f}s wall). Log: {log_path}"
    )
