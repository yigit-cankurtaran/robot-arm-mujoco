from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parent
SCENE_PATH = ROOT / "third_party" / "menagerie" / "universal_robots_ur5e" / "scene.xml"


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
        self.home_ctrl = np.array(
            [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0],
            dtype=float,
        )
        self.ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site"
        )
        self.reset()

    def reset(self) -> tuple[np.ndarray, dict]:
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.data.ctrl[:] = self.home_ctrl
        mujoco.mj_forward(self.model, self.data)
        return self._get_observation(), self._get_info()

    def step(self, action: np.ndarray) -> StepResult:
        self.data.ctrl[:] = np.clip(
            action,
            self.model.actuator_ctrlrange[:, 0],
            self.model.actuator_ctrlrange[:, 1],
        )
        mujoco.mj_step(self.model, self.data)
        return StepResult(
            observation=self._get_observation(),
            reward=0.0,
            terminated=False,
            truncated=False,
            info=self._get_info(),
        )

    def scripted_action(self, t: float) -> np.ndarray:
        phase = (t % 10.0) / 10.0
        action = self.home_ctrl.copy()
        action[0] += 0.55 * np.sin(2.0 * np.pi * phase)
        action[1] += -0.35 * np.sin(2.0 * np.pi * phase)
        action[2] += 0.45 * np.sin(2.0 * np.pi * phase + 0.7)
        action[3] += 0.30 * np.sin(4.0 * np.pi * phase)
        action[4] += 0.25 * np.sin(2.0 * np.pi * phase + 1.4)
        action[5] += 0.40 * np.sin(4.0 * np.pi * phase + 0.3)
        return action

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
        }
