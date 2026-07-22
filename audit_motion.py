from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from env import FactoryFloorEnv


@dataclass
class TrialMetrics:
    seed: int
    completed: bool
    cycle_time: float
    steps: int
    unsafe_contacts: int
    contact_pairs: list[str]
    peak_command_velocity: list[float]
    peak_command_acceleration: list[float]
    peak_command_jerk: list[float]
    peak_measured_velocity: list[float]
    peak_tracking_error: list[float]


def _body_descends_from(model: mujoco.MjModel, body_id: int, root_id: int) -> bool:
    while body_id > 0:
        if body_id == root_id:
            return True
        body_id = int(model.body_parentid[body_id])
    return False


def _contact_label(env: FactoryFloorEnv, geom_id: int) -> str:
    geom_name = env.model.geom(geom_id).name
    if geom_name:
        return geom_name
    body_id = int(env.model.geom_bodyid[geom_id])
    return f"{env.model.body(body_id).name}:collision"


def _unsafe_contact_pairs(env: FactoryFloorEnv) -> list[str]:
    robot_root = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "base")
    pairs: list[str] = []
    for geom1, geom2 in env.contact_pairs_this_step:
        body1 = int(env.model.geom_bodyid[geom1])
        body2 = int(env.model.geom_bodyid[geom2])
        robot1 = _body_descends_from(env.model, body1, robot_root)
        robot2 = _body_descends_from(env.model, body2, robot_root)

        # Part/table, part/bin, and part/part contacts are expected task physics.
        # Any robot contact with the workcell or a loose part is unsafe.  Robot
        # self-contact between non-adjacent collision bodies is unsafe as well.
        unsafe = robot1 != robot2
        if robot1 and robot2 and body1 != body2:
            unsafe = int(env.model.body_parentid[body1]) != body2 and int(
                env.model.body_parentid[body2]
            ) != body1
        if unsafe:
            pairs.append(
                f"{_contact_label(env, geom1)} <-> {_contact_label(env, geom2)}"
            )
    return pairs


def _peak_abs(values: np.ndarray, width: int) -> np.ndarray:
    if values.size == 0:
        return np.zeros(width, dtype=float)
    return np.max(np.abs(values), axis=0)


def run_trial(seed: int, max_seconds: float) -> TrialMetrics:
    env = FactoryFloorEnv(enable_rgb_observation=False)
    env.random_state = np.random.default_rng(seed)
    env.reset()
    max_steps = int(np.ceil(max_seconds / env.control_dt))

    commands: list[np.ndarray] = []
    measured_velocity: list[np.ndarray] = []
    tracking_error: list[np.ndarray] = []
    unsafe_contacts: list[str] = []

    completed = False
    for step_index in range(max_steps):
        env.scripted_step()
        commands.append(env.data.ctrl.copy())
        measured_velocity.append(env.data.qvel[: env.arm_dofs].copy())
        tracking_error.append(
            env.data.ctrl.copy() - env.data.qpos[: env.arm_dofs].copy()
        )
        unsafe_contacts.extend(_unsafe_contact_pairs(env))
        if (
            len(env.completed_parts) == len(env.active_part_order)
            and env.controller_phase == "idle"
        ):
            completed = True
            break

    command_array = np.asarray(commands)
    velocity = np.diff(command_array, axis=0) / env.control_dt
    acceleration = np.diff(velocity, axis=0) / env.control_dt
    jerk = np.diff(acceleration, axis=0) / env.control_dt
    steps = step_index + 1
    return TrialMetrics(
        seed=seed,
        completed=completed,
        cycle_time=steps * env.control_dt,
        steps=steps,
        unsafe_contacts=len(unsafe_contacts),
        contact_pairs=sorted(set(unsafe_contacts)),
        peak_command_velocity=_peak_abs(velocity, env.arm_dofs).round(4).tolist(),
        peak_command_acceleration=_peak_abs(acceleration, env.arm_dofs)
        .round(4)
        .tolist(),
        peak_command_jerk=_peak_abs(jerk, env.arm_dofs).round(4).tolist(),
        peak_measured_velocity=_peak_abs(
            np.asarray(measured_velocity), env.arm_dofs
        )
        .round(4)
        .tolist(),
        peak_tracking_error=_peak_abs(np.asarray(tracking_error), env.arm_dofs)
        .round(4)
        .tolist(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Headless speed, smoothness, completion, and collision audit."
    )
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--max-cycle-time", type=float, default=15.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    trials = [run_trial(seed, args.max_cycle_time) for seed in range(args.seeds)]
    limits = FactoryFloorEnv(enable_rgb_observation=False)

    def exceeds(values: list[float], bounds: np.ndarray) -> bool:
        # Small allowance covers finite-difference sampling and rounding.
        return bool(np.any(np.asarray(values) > bounds * 1.02 + 1e-3))

    failures = [
        trial
        for trial in trials
        if not trial.completed
        or trial.unsafe_contacts > 0
        or trial.cycle_time > args.max_cycle_time
        or exceeds(trial.peak_command_velocity, limits.max_joint_velocity)
        or exceeds(
            trial.peak_command_acceleration, limits.max_joint_acceleration
        )
        or exceeds(trial.peak_command_jerk, limits.max_joint_jerk)
    ]
    summary = {
        "passed": not failures,
        "seeds": args.seeds,
        "completed_trials": sum(trial.completed for trial in trials),
        "collision_free_trials": sum(
            trial.unsafe_contacts == 0 for trial in trials
        ),
        "cycle_time_seconds": {
            "min": round(min(trial.cycle_time for trial in trials), 4),
            "mean": round(float(np.mean([trial.cycle_time for trial in trials])), 4),
            "max": round(max(trial.cycle_time for trial in trials), 4),
        },
        "worst_peak_command_velocity": np.max(
            [trial.peak_command_velocity for trial in trials], axis=0
        ).round(4).tolist(),
        "worst_peak_command_acceleration": np.max(
            [trial.peak_command_acceleration for trial in trials], axis=0
        ).round(4).tolist(),
        "worst_peak_command_jerk": np.max(
            [trial.peak_command_jerk for trial in trials], axis=0
        ).round(4).tolist(),
        "worst_peak_measured_velocity": np.max(
            [trial.peak_measured_velocity for trial in trials], axis=0
        ).round(4).tolist(),
        "worst_peak_tracking_error": np.max(
            [trial.peak_tracking_error for trial in trials], axis=0
        ).round(4).tolist(),
        "failures": [asdict(trial) for trial in failures],
    }

    if args.json:
        print(json.dumps({"summary": summary, "trials": [asdict(t) for t in trials]}, indent=2))
    else:
        print(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
