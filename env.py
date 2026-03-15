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
        self.ctrl_low = self.model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_high = self.model.actuator_ctrlrange[:, 1].copy()
        self.control_substeps = 10
        self.control_dt = self.model.opt.timestep * self.control_substeps
        self.home_ctrl = np.array([-3.25, -1.72, 1.48, -1.72, -3.05, 0.0], dtype=float)
        self.cycle_duration = 7.0
        self.part_specs = {
            "part_blue_1": {
                "bin": "blue",
                "body": "part_blue_1",
                "joint": "part_blue_1_free",
                "pick_site": "feed_hover_a",
            },
            "part_orange_1": {
                "bin": "orange",
                "body": "part_orange_1",
                "joint": "part_orange_1_free",
                "pick_site": "feed_hover_b",
            },
            "part_blue_2": {
                "bin": "blue",
                "body": "part_blue_2",
                "joint": "part_blue_2_free",
                "pick_site": "feed_hover_c",
            },
        }
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
        self.drop_site_ids = {
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
        self.pick_site_ids = {
            name: mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, spec["pick_site"]
            )
            for name, spec in self.part_specs.items()
        }
        self.part_joint_ids = {
            name: mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, spec["joint"]
            )
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
        mujoco.mj_resetData(self.model, self.data)
        self._set_robot_configuration(self.home_ctrl)
        self.part_home = {
            name: self._read_freejoint_qpos(name).copy() for name in self.part_specs
        }
        self.part_order = list(self.part_specs)
        self.holding_part: str | None = None
        self.completed_parts: set[str] = set()
        self.pick_poses = {
            "part_blue_1": np.array(
                [-3.4393, -2.0198, 1.7075, -1.8917, -1.9208, 0.0], dtype=float
            ),
            "part_orange_1": np.array(
                [-3.2489, -1.8023, 1.8265, -1.4369, -4.6172, 0.0], dtype=float
            ),
            "part_blue_2": np.array(
                [-3.4234, -1.5847, 1.4137, -1.9676, -2.3713, 0.0], dtype=float
            ),
        }
        self.pick_hover_poses = {
            "part_blue_1": np.array(
                [-3.2420, -1.8376, 1.4552, -1.6648, -3.1939, 0.0], dtype=float
            ),
            "part_orange_1": np.array(
                [-3.2557, -1.7294, 1.5261, -1.4520, -4.6452, 0.0], dtype=float
            ),
            "part_blue_2": np.array(
                [-3.4238, -1.5175, 1.1241, -2.0522, -2.3687, 0.0], dtype=float
            ),
        }
        self.drop_targets = {
            name: self.data.site_xpos[site_id].copy()
            for name, site_id in self.drop_site_ids.items()
        }
        self.drop_poses = {
            "blue": np.array(
                [-3.0880, -1.4200, 0.8150, -1.7760, -1.5940, 0.8710], dtype=float
            ),
            "orange": np.array(
                [-3.6700, -1.5130, 1.1450, -2.1370, -1.6740, 0.0], dtype=float
            ),
        }
        self.reset()

    def reset(self) -> tuple[np.ndarray, dict]:
        mujoco.mj_resetData(self.model, self.data)
        self._set_robot_configuration(self.home_ctrl)
        for name, qpos in self.part_home.items():
            self._write_freejoint_qpos(name, qpos)
        mujoco.mj_forward(self.model, self.data)
        self.holding_part = None
        self.completed_parts = set()
        return self._get_observation(), self._get_info()

    def step(self, action: np.ndarray) -> StepResult:
        self.set_arm_configuration(action)
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
            self._update_task_progress()
            if self.holding_part is not None:
                self._update_attachment()

    def scripted_action(self, t: float) -> np.ndarray:
        part_name, target_bin, phase = self._current_cycle_state(t)
        pick = self.pick_poses[part_name]
        pick_hover = self.pick_hover_poses[part_name]
        drop = self.drop_poses[target_bin]

        if phase < 0.24:
            return self._blend(self.home_ctrl, pick_hover, phase / 0.24)
        if phase < 0.34:
            return self._blend(pick_hover, pick, (phase - 0.24) / 0.10)
        if phase < 0.44:
            return pick
        if phase < 0.54:
            return self._blend(pick, pick_hover, (phase - 0.44) / 0.10)
        if phase < 0.82:
            return self._blend(pick_hover, drop, (phase - 0.54) / 0.28)
        if phase < 0.92:
            return drop
        return self._blend(drop, self.home_ctrl, (phase - 0.92) / 0.08)

    def describe_task(self) -> dict:
        return {
            "scene": str(self.xml_path),
            "control_dt": self.control_dt,
            "parts": {
                name: {
                    "target_bin": spec["bin"],
                    "current_position": self.data.xpos[self.part_body_ids[name]]
                    .round(4)
                    .tolist(),
                }
                for name, spec in self.part_specs.items()
            },
            "bins": {
                name: self.data.site_xpos[site_id].round(4).tolist()
                for name, site_id in self.bin_site_ids.items()
            },
        }

    def _current_cycle_state(self, t: float) -> tuple[str, str, float]:
        part_name = self.part_order[
            int(t // self.cycle_duration) % len(self.part_order)
        ]
        return (
            part_name,
            self.part_specs[part_name]["bin"],
            (t % self.cycle_duration) / self.cycle_duration,
        )

    def _blend(self, a: np.ndarray, b: np.ndarray, blend: float) -> np.ndarray:
        blend = float(np.clip(blend, 0.0, 1.0))
        smooth = blend * blend * (3.0 - 2.0 * blend)
        return (1.0 - smooth) * a + smooth * b

    def _read_freejoint_qpos(self, name: str) -> np.ndarray:
        adr = self.part_qpos_adr[name]
        return self.data.qpos[adr : adr + 7]

    def _write_freejoint_qpos(self, name: str, qpos: np.ndarray) -> None:
        adr = self.part_qpos_adr[name]
        self.data.qpos[adr : adr + 7] = qpos
        vel_adr = self.part_qvel_adr[name]
        self.data.qvel[vel_adr : vel_adr + 6] = 0.0

    def _update_task_progress(self) -> None:
        active_part, _, _ = self._current_cycle_state(self.data.time)
        if self.holding_part is None:
            if active_part not in self.completed_parts and self._ee_close_to_pick(
                active_part, tol=0.045
            ):
                self.holding_part = active_part
                self._update_attachment()
            return
        target_bin = self.part_specs[self.holding_part]["bin"]
        if self._ee_close_to_bin(target_bin, tol=0.07):
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

    def _ee_close_to_pick(self, part_name: str, tol: float = 0.03) -> bool:
        target = self.data.site_xpos[self.pick_site_ids[part_name]]
        return (
            float(np.linalg.norm(self.data.site_xpos[self.ee_site_id] - target)) < tol
        )

    def _ee_close_to_bin(self, bin_name: str, tol: float = 0.04) -> bool:
        return (
            float(
                np.linalg.norm(
                    self.data.site_xpos[self.ee_site_id] - self.drop_targets[bin_name]
                )
            )
            < tol
        )

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
        self.completed_parts.add(part_name)
        self.holding_part = None
        mujoco.mj_forward(self.model, self.data)

    def _set_robot_configuration(self, q: np.ndarray) -> None:
        self.data.qpos[:6] = q
        self.data.qvel[:6] = 0.0
        self.data.ctrl[:] = q
        mujoco.mj_forward(self.model, self.data)

    def _task_reward(self) -> float:
        reward = 0.0
        for name, spec in self.part_specs.items():
            body_pos = self.data.xpos[self.part_body_ids[name]]
            target_pos = self.data.site_xpos[self.bin_site_ids[spec["bin"]]]
            reward -= float(np.linalg.norm(body_pos - target_pos))
        return reward

    def _get_observation(self) -> np.ndarray:
        return np.concatenate(
            [
                self.data.qpos.copy(),
                self.data.qvel.copy(),
                self.data.site_xpos[self.ee_site_id].copy(),
            ]
        )

    def _get_info(self) -> dict:
        return {
            "time": float(self.data.time),
            "ee_position": self.data.site_xpos[self.ee_site_id].copy(),
            "holding_part": self.holding_part,
            "sorted_counts": self._sorted_counts(),
        }

    def _sorted_counts(self) -> dict[str, int]:
        counts = {"blue": 0, "orange": 0}
        for name, spec in self.part_specs.items():
            body_pos = self.data.xpos[self.part_body_ids[name]]
            target_pos = self.data.site_xpos[self.bin_site_ids[spec["bin"]]]
            if np.linalg.norm(body_pos - target_pos) < 0.10:
                counts[spec["bin"]] += 1
        return counts
