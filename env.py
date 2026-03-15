from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parent
SCENE_PATH = (
    ROOT / "third_party" / "menagerie" / "universal_robots_ur5e" / "workcell_scene.xml"
)


@dataclass
class StepResult:
    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict


class FactoryFloorEnv:
    def __init__(self, xml_path: str | Path = SCENE_PATH):
        self.xml_path = Path(xml_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)
        self.ik_data = mujoco.MjData(self.model)

        self.arm_dofs = 6
        self.ctrl_low = self.model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_high = self.model.actuator_ctrlrange[:, 1].copy()
        self.control_substeps = 10
        self.control_dt = self.model.opt.timestep * self.control_substeps
        self.home_ctrl = np.array([-3.25, -1.72, 1.48, -1.72, -3.05, 0.0], dtype=float)

        self.ik_iterations = 32
        self.ik_damping = 0.08
        self.ik_step_scale = 0.8
        self.ik_rest_gain = 0.08
        self.ik_tolerance = 0.002
        self.ik_max_update = 0.35

        self.pick_offset = np.array([0.0, 0.0, 0.09], dtype=float)
        self.pick_hover_offset = np.array([0.0, 0.0, 0.19], dtype=float)
        self.transfer_target = np.array([0.45, 0.0, 0.90], dtype=float)
        self.home_position_tol = 0.06
        self.phase_position_tol = 0.05
        self.drop_release_xy_tol = 0.05
        self.drop_release_z_tol = 0.05

        self.table_surface_z = 0.49
        self.spawn_anchors = [
            np.array([0.34, 0.20], dtype=float),
            np.array([0.42, 0.23], dtype=float),
            np.array([0.48, 0.20], dtype=float),
            np.array([0.34, -0.20], dtype=float),
            np.array([0.42, -0.23], dtype=float),
            np.array([0.48, -0.20], dtype=float),
        ]
        self.spawn_jitter = np.array([0.02, 0.015], dtype=float)
        self.spawn_clearance = 0.08
        self.random_state = np.random.default_rng()

        self.part_specs = {
            "part_blue_1": {
                "color": "blue",
                "bin": "blue",
                "body": "part_blue_1",
                "joint": "part_blue_1_free",
                "geom": "part_blue_1_geom",
                "support_height": 0.025,
            },
            "part_orange_1": {
                "color": "orange",
                "bin": "orange",
                "body": "part_orange_1",
                "joint": "part_orange_1_free",
                "geom": "part_orange_1_geom",
                "support_height": 0.03,
            },
            "part_blue_2": {
                "color": "blue",
                "bin": "blue",
                "body": "part_blue_2",
                "joint": "part_blue_2_free",
                "geom": "part_blue_2_geom",
                "support_height": 0.03,
            },
        }
        self.color_encoding = {
            "blue": np.array([1.0, 0.0], dtype=float),
            "orange": np.array([0.0, 1.0], dtype=float),
        }
        self.part_order = list(self.part_specs)

        self.ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site"
        )
        self.bin_site_ids = {
            "blue": mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, "bin_blue_target"
            ),
            "orange": mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, "bin_orange_target"
            ),
        }
        self.bin_approach_site_ids = {
            "blue": mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, "bin_blue_hover"
            ),
            "orange": mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, "bin_orange_hover"
            ),
        }
        self.part_body_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, spec["body"])
            for name, spec in self.part_specs.items()
        }
        self.part_joint_ids = {
            name: mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, spec["joint"]
            )
            for name, spec in self.part_specs.items()
        }
        self.part_geom_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, spec["geom"])
            for name, spec in self.part_specs.items()
        }
        self.part_qpos_adr = {
            name: int(self.model.jnt_qposadr[joint_id])
            for name, joint_id in self.part_joint_ids.items()
        }
        self.part_qvel_adr = {
            name: int(self.model.jnt_dofadr[joint_id])
            for name, joint_id in self.part_joint_ids.items()
        }
        self.part_collision_masks = {
            name: (
                int(self.model.geom_contype[geom_id]),
                int(self.model.geom_conaffinity[geom_id]),
            )
            for name, geom_id in self.part_geom_ids.items()
        }

        mujoco.mj_resetData(self.model, self.data)
        self._set_robot_configuration(self.home_ctrl)
        self.home_ee_target = self.data.site_xpos[self.ee_site_id].copy()

        self.holding_part: str | None = None
        self.active_part: str | None = None
        self.completed_parts: set[str] = set()
        self.controller_phase = "idle"
        self.last_pick_hover_target = self.home_ee_target.copy()
        self.spawn_layout: dict[str, np.ndarray] = {}

        self.reset()

    def reset(self) -> tuple[np.ndarray, dict]:
        mujoco.mj_resetData(self.model, self.data)
        self._set_robot_configuration(self.home_ctrl)
        for name in self.part_order:
            self._set_part_collision_enabled(name, enabled=True)
        self._randomize_part_layout()
        mujoco.mj_forward(self.model, self.data)
        self.holding_part = None
        self.active_part = None
        self.completed_parts = set()
        self.controller_phase = "select_part"
        self.last_pick_hover_target = self.home_ee_target.copy()
        return self._get_observation(), self._get_info()

    def step(self, action: np.ndarray) -> StepResult:
        self.set_arm_configuration(action)
        self._advance_controller_state()
        return StepResult(
            observation=self._get_observation(),
            reward=self._task_reward(),
            terminated=False,
            truncated=False,
            info=self._get_info(),
        )

    def scripted_step(self, t: float | None = None) -> StepResult:
        del t
        self._advance_controller_state()
        target_pos = self._controller_target_position()
        target_q = self._solve_inverse_kinematics(target_pos)
        self.set_arm_configuration(target_q)
        self._advance_controller_state()
        return StepResult(
            observation=self._get_observation(),
            reward=self._task_reward(),
            terminated=False,
            truncated=False,
            info=self._get_info(),
        )

    def set_arm_configuration(self, qpos: np.ndarray) -> None:
        target = np.clip(qpos, self.ctrl_low, self.ctrl_high)
        self.data.ctrl[:] = target
        for _ in range(self.control_substeps):
            if self.holding_part is not None:
                self._update_attachment()
            mujoco.mj_step(self.model, self.data)
            if self.holding_part is not None:
                self._update_attachment()
        self._update_task_progress()

    def describe_task(self) -> dict:
        return {
            "scene": str(self.xml_path),
            "control_dt": self.control_dt,
            "controller_phase": self.controller_phase,
            "active_part": self.active_part,
            "parts": {
                name: self._part_state(name)
                for name in self.part_order
            },
            "bins": {
                name: self.data.site_xpos[site_id].round(4).tolist()
                for name, site_id in self.bin_site_ids.items()
            },
        }

    def _advance_controller_state(self) -> None:
        if self.controller_phase == "idle":
            if self._choose_next_part() is not None:
                self.controller_phase = "select_part"
            return

        if self.controller_phase == "select_part":
            next_part = self._choose_next_part()
            if next_part is None:
                self.active_part = None
                self.last_pick_hover_target = self.home_ee_target.copy()
                if self._ee_close_to_position(self.home_ee_target, self.home_position_tol):
                    self.controller_phase = "idle"
                else:
                    self.controller_phase = "return_home"
                return
            self.active_part = next_part
            self.last_pick_hover_target = self._pick_hover_target(next_part)
            self.controller_phase = "move_to_pick_hover"
            return

        if self.controller_phase == "move_to_pick_hover":
            if self.active_part is None or self.active_part in self.completed_parts:
                self.controller_phase = "select_part"
                return
            self.last_pick_hover_target = self._pick_hover_target(self.active_part)
            if self._ee_close_to_position(
                self.last_pick_hover_target, self.phase_position_tol
            ):
                self.controller_phase = "move_to_pick"
            return

        if self.controller_phase == "move_to_pick":
            if self.active_part is None or self.active_part in self.completed_parts:
                self.controller_phase = "select_part"
                return
            if self.holding_part == self.active_part:
                self.controller_phase = "lift_with_part"
            return

        if self.controller_phase == "lift_with_part":
            if self.holding_part is None:
                self.controller_phase = "select_part"
                return
            if self._ee_close_to_position(
                self.last_pick_hover_target, self.phase_position_tol
            ):
                self.controller_phase = "move_to_transfer"
            return

        if self.controller_phase == "move_to_transfer":
            if self.holding_part is None:
                self.controller_phase = "select_part"
                return
            if self._ee_close_to_position(self.transfer_target, self.phase_position_tol):
                self.controller_phase = "move_to_bin_hover"
            return

        if self.controller_phase == "move_to_bin_hover":
            if self.holding_part is None:
                self.controller_phase = "select_part"
                return
            hover_target = self._bin_hover_target(self.part_specs[self.holding_part]["bin"])
            if self._ee_close_to_position(hover_target, self.phase_position_tol):
                self.controller_phase = "move_to_drop"
            return

        if self.controller_phase == "move_to_drop":
            if self.holding_part is None:
                self.active_part = None
                self.controller_phase = "return_home"
            return

        if self.controller_phase == "return_home":
            if self._ee_close_to_position(self.home_ee_target, self.home_position_tol):
                self.controller_phase = (
                    "select_part" if self._choose_next_part() is not None else "idle"
                )

    def _controller_target_position(self) -> np.ndarray:
        if self.controller_phase in {"idle", "return_home"}:
            return self.home_ee_target

        if self.controller_phase == "move_to_pick_hover" and self.active_part is not None:
            return self._pick_hover_target(self.active_part)

        if self.controller_phase == "move_to_pick" and self.active_part is not None:
            return self._pick_target(self.active_part)

        if self.controller_phase == "lift_with_part":
            return self.last_pick_hover_target

        if self.controller_phase == "move_to_transfer":
            return self.transfer_target

        part_name = self.holding_part or self.active_part
        if part_name is None:
            return self.home_ee_target

        target_bin = self.part_specs[part_name]["bin"]
        if self.controller_phase == "move_to_bin_hover":
            return self._bin_hover_target(target_bin)
        if self.controller_phase == "move_to_drop":
            return self._drop_release_target(target_bin)
        return self.home_ee_target

    def _solve_inverse_kinematics(self, target_pos: np.ndarray) -> np.ndarray:
        q = self.data.qpos[: self.arm_dofs].copy()
        jacp = np.zeros((3, self.model.nv), dtype=float)
        jacr = np.zeros((3, self.model.nv), dtype=float)

        for _ in range(self.ik_iterations):
            self.ik_data.qpos[:] = self.data.qpos
            self.ik_data.qvel[:] = 0.0
            self.ik_data.qpos[: self.arm_dofs] = q
            mujoco.mj_forward(self.model, self.ik_data)

            ee_pos = self.ik_data.site_xpos[self.ee_site_id].copy()
            error = target_pos - ee_pos
            if float(np.linalg.norm(error)) < self.ik_tolerance:
                break

            mujoco.mj_jacSite(self.model, self.ik_data, jacp, jacr, self.ee_site_id)
            arm_jac = jacp[:, : self.arm_dofs]
            regularized = arm_jac @ arm_jac.T + (
                self.ik_damping**2
            ) * np.eye(error.shape[0])
            dq = arm_jac.T @ np.linalg.solve(regularized, error)
            dq += self.ik_rest_gain * (self.home_ctrl - q)

            step_norm = float(np.linalg.norm(dq))
            if step_norm > self.ik_max_update:
                dq *= self.ik_max_update / step_norm
            q = np.clip(q + self.ik_step_scale * dq, self.ctrl_low, self.ctrl_high)

        return q

    def _randomize_part_layout(self) -> None:
        occupied_xy: list[np.ndarray] = []
        self.spawn_layout = {}

        for name in self.part_order:
            xy = self._sample_spawn_xy(occupied_xy)
            occupied_xy.append(xy)
            yaw = float(self.random_state.uniform(-np.pi, np.pi))
            quat = np.array(
                [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)], dtype=float
            )
            qpos = np.array(
                [
                    xy[0],
                    xy[1],
                    self.table_surface_z + self.part_specs[name]["support_height"],
                    quat[0],
                    quat[1],
                    quat[2],
                    quat[3],
                ],
                dtype=float,
            )
            self._write_freejoint_qpos(name, qpos)
            self.spawn_layout[name] = qpos.copy()

    def _sample_spawn_xy(self, occupied_xy: list[np.ndarray]) -> np.ndarray:
        for _ in range(200):
            candidate = self._sample_spawn_candidate()
            if all(
                float(np.linalg.norm(candidate - existing)) >= self.spawn_clearance
                for existing in occupied_xy
            ):
                return candidate
        raise RuntimeError("unable to sample non-overlapping part spawns")

    def _sample_spawn_candidate(self) -> np.ndarray:
        anchor = self.spawn_anchors[int(self.random_state.integers(len(self.spawn_anchors)))]
        jitter = self.random_state.uniform(-self.spawn_jitter, self.spawn_jitter)
        return anchor + jitter

    def _choose_next_part(self) -> str | None:
        candidates = [
            name
            for name in self.part_order
            if name not in self.completed_parts and name != self.holding_part
        ]
        if not candidates:
            return None
        ee_pos = self.data.site_xpos[self.ee_site_id]
        return min(
            candidates,
            key=lambda name: float(
                np.linalg.norm(self._pick_hover_target(name) - ee_pos)
            ),
        )

    def _pick_target(self, part_name: str) -> np.ndarray:
        return self.data.xpos[self.part_body_ids[part_name]].copy() + self.pick_offset

    def _pick_hover_target(self, part_name: str) -> np.ndarray:
        return (
            self.data.xpos[self.part_body_ids[part_name]].copy()
            + self.pick_hover_offset
        )

    def _bin_hover_target(self, bin_name: str) -> np.ndarray:
        return self.data.site_xpos[self.bin_approach_site_ids[bin_name]].copy()

    def _drop_release_target(self, bin_name: str) -> np.ndarray:
        return self.data.site_xpos[self.bin_approach_site_ids[bin_name]].copy()

    def _read_freejoint_qpos(self, name: str) -> np.ndarray:
        adr = self.part_qpos_adr[name]
        return self.data.qpos[adr : adr + 7]

    def _write_freejoint_qpos(self, name: str, qpos: np.ndarray) -> None:
        adr = self.part_qpos_adr[name]
        self.data.qpos[adr : adr + 7] = qpos
        vel_adr = self.part_qvel_adr[name]
        self.data.qvel[vel_adr : vel_adr + 6] = 0.0

    def _update_task_progress(self) -> None:
        if self.holding_part is None:
            if (
                self.controller_phase == "move_to_pick"
                and self.active_part is not None
                and self.active_part not in self.completed_parts
                and self._ee_close_to_pick(self.active_part)
            ):
                self.holding_part = self.active_part
                self._set_part_collision_enabled(self.holding_part, enabled=False)
                self._update_attachment()
            return

        target_bin = self.part_specs[self.holding_part]["bin"]
        if self.controller_phase == "move_to_drop" and self._ee_ready_to_drop(target_bin):
            self._drop_part_in_bin(self.holding_part, target_bin)

    def _update_attachment(self) -> None:
        if self.holding_part is None:
            return
        ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        target_qpos = np.array(
            [ee_pos[0], ee_pos[1], ee_pos[2] - 0.08, 1.0, 0.0, 0.0, 0.0],
            dtype=float,
        )
        self._write_freejoint_qpos(self.holding_part, target_qpos)
        mujoco.mj_forward(self.model, self.data)

    def _ee_close_to_pick(self, part_name: str, tol: float = 0.035) -> bool:
        target = self._pick_target(part_name)
        return float(np.linalg.norm(self.data.site_xpos[self.ee_site_id] - target)) < tol

    def _ee_ready_to_drop(self, bin_name: str) -> bool:
        ee_pos = self.data.site_xpos[self.ee_site_id]
        target = self._drop_release_target(bin_name)
        xy_error = float(np.linalg.norm(ee_pos[:2] - target[:2]))
        z_error = float(abs(ee_pos[2] - target[2]))
        return xy_error < self.drop_release_xy_tol and z_error < self.drop_release_z_tol

    def _ee_close_to_position(self, target: np.ndarray, tol: float) -> bool:
        return float(np.linalg.norm(self.data.site_xpos[self.ee_site_id] - target)) < tol

    def _drop_part_in_bin(self, part_name: str, bin_name: str) -> None:
        target_pos = self.data.site_xpos[self.bin_site_ids[bin_name]].copy()
        stacked = sum(
            1
            for name in self.completed_parts
            if self.part_specs[name]["bin"] == bin_name
        )
        target_qpos = np.array(
            [
                target_pos[0],
                target_pos[1],
                target_pos[2] - 0.02 + 0.035 * stacked,
                1.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=float,
        )
        self._write_freejoint_qpos(part_name, target_qpos)
        self._set_part_collision_enabled(part_name, enabled=True)
        self.completed_parts.add(part_name)
        self.holding_part = None
        self.active_part = None
        self.last_pick_hover_target = self.home_ee_target.copy()
        mujoco.mj_forward(self.model, self.data)

    def _set_part_collision_enabled(self, part_name: str, enabled: bool) -> None:
        geom_id = self.part_geom_ids[part_name]
        contype, conaffinity = self.part_collision_masks[part_name]
        if enabled:
            self.model.geom_contype[geom_id] = contype
            self.model.geom_conaffinity[geom_id] = conaffinity
            return
        self.model.geom_contype[geom_id] = 0
        self.model.geom_conaffinity[geom_id] = 0

    def _set_robot_configuration(self, q: np.ndarray) -> None:
        self.data.qpos[: self.arm_dofs] = q
        self.data.qvel[: self.arm_dofs] = 0.0
        self.data.ctrl[:] = q
        mujoco.mj_forward(self.model, self.data)

    def _task_reward(self) -> float:
        reward = 0.0
        for name, spec in self.part_specs.items():
            body_pos = self.data.xpos[self.part_body_ids[name]]
            target_pos = self.data.site_xpos[self.bin_site_ids[spec["bin"]]]
            reward -= float(np.linalg.norm(body_pos - target_pos))
        return reward

    def _part_state(self, name: str) -> dict:
        spec = self.part_specs[name]
        spawn_qpos = self.spawn_layout.get(name, self._read_freejoint_qpos(name))
        return {
            "color": spec["color"],
            "target_bin": spec["bin"],
            "spawn_position": spawn_qpos[:3].round(4).tolist(),
            "current_position": self.data.xpos[self.part_body_ids[name]].round(4).tolist(),
            "holding": name == self.holding_part,
            "sorted": name in self.completed_parts,
        }

    def _get_observation(self) -> np.ndarray:
        bin_features = np.concatenate(
            [
                self.data.site_xpos[self.bin_site_ids[name]].copy()
                for name in ("blue", "orange")
            ]
        )
        part_features = []
        for name in self.part_order:
            spec = self.part_specs[name]
            part_features.extend(self.data.xpos[self.part_body_ids[name]].tolist())
            part_features.extend(self.color_encoding[spec["color"]].tolist())
            part_features.append(float(name == self.holding_part))
            part_features.append(float(name in self.completed_parts))
        return np.concatenate(
            [
                self.data.qpos.copy(),
                self.data.qvel.copy(),
                self.data.site_xpos[self.ee_site_id].copy(),
                bin_features,
                np.array(part_features, dtype=float),
            ]
        )

    def _get_info(self) -> dict:
        return {
            "time": float(self.data.time),
            "controller_phase": self.controller_phase,
            "active_part": self.active_part,
            "ee_position": self.data.site_xpos[self.ee_site_id].copy(),
            "holding_part": self.holding_part,
            "sorted_counts": self._sorted_counts(),
            "parts": {
                name: self._part_state(name)
                for name in self.part_order
            },
        }

    def _sorted_counts(self) -> dict[str, int]:
        counts = {"blue": 0, "orange": 0}
        for name in self.completed_parts:
            counts[self.part_specs[name]["bin"]] += 1
        return counts
