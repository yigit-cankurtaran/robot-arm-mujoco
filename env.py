from __future__ import annotations

import colorsys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parent
SCENE_PATH = (
    ROOT / "third_party" / "menagerie" / "universal_robots_ur5e" / "workcell_scene.xml"
)


def _load_model_with_policy_camera(
    xml_path: Path, camera_name: str
) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(xml_path))
    if camera_name not in {camera.name for camera in spec.cameras}:
        spec.worldbody.add_camera(
            name=camera_name,
            pos=[0.56, 0.0, 1.75],
            quat=[1.0, 0.0, 0.0, 0.0],
            fovy=52.0,
        )
    return spec.compile()


@dataclass
class StepResult:
    observation: dict[str, np.ndarray]
    reward: float
    terminated: bool
    truncated: bool
    info: dict


class FactoryFloorEnv:
    def __init__(
        self,
        xml_path: str | Path = SCENE_PATH,
        *,
        enable_rgb_observation: bool = True,
        camera_name: str = "policy_camera",
        camera_width: int = 240,
        camera_height: int = 240,
        rgb_render_interval: int = 1,
    ):
        self.xml_path = Path(xml_path)
        self.camera_name = camera_name
        self.model = _load_model_with_policy_camera(
            self.xml_path, self.camera_name
        )
        self.data = mujoco.MjData(self.model)
        self.ik_data = mujoco.MjData(self.model)
        self.enable_rgb_observation = enable_rgb_observation
        self.camera_width = camera_width
        self.camera_height = camera_height
        if rgb_render_interval < 1:
            raise ValueError("rgb_render_interval must be at least 1")
        self.rgb_render_interval = rgb_render_interval
        self.rgb_frame_counter = 0
        self.last_rgb: np.ndarray | None = None
        self.policy_scene_option = mujoco.MjvOption()
        mujoco.mjv_defaultOption(self.policy_scene_option)
        self.policy_scene_option.sitegroup[:] = 0
        self.renderer: mujoco.Renderer | None = None
        self.camera_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name
        )
        if self.camera_id < 0:
            raise ValueError(f"camera {self.camera_name!r} does not exist in the model")
        self.base_camera_pos = self.model.cam_pos[self.camera_id].copy()
        self.base_light_pos = self.model.light_pos.copy()
        self.base_light_diffuse = self.model.light_diffuse.copy()

        self.arm_dofs = 6
        self.ctrl_low = self.model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_high = self.model.actuator_ctrlrange[:, 1].copy()
        self.control_substeps = 10
        self.control_dt = self.model.opt.timestep * self.control_substeps
        self.home_ctrl = np.array([-3.25, -1.72, 1.48, -1.72, -3.05, 0.0], dtype=float)

        # Per-joint limits for the commanded minimum-jerk trajectories.  These are
        # deliberately below the UR5e hardware limits while being considerably
        # quicker than the old controller's unconstrained actuator settling.
        self.max_joint_velocity = np.array(
            [1.8, 1.8, 2.2, 2.4, 2.4, 2.4], dtype=float
        )
        self.max_joint_acceleration = np.array(
            [7.0, 7.0, 8.0, 10.0, 10.0, 10.0], dtype=float
        )
        self.max_joint_jerk = np.array(
            [55.0, 55.0, 65.0, 80.0, 80.0, 80.0], dtype=float
        )
        self.min_trajectory_duration = 0.24

        self.ik_iterations = 32
        self.ik_damping = 0.08
        self.ik_step_scale = 0.8
        self.ik_rest_gain = 0.08
        self.ik_tolerance = 0.002
        self.ik_max_update = 0.35

        self.pick_offset = np.array([0.0, 0.0, 0.09], dtype=float)
        self.pick_hover_offset = np.array([0.0, 0.0, 0.15], dtype=float)
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

        # XML names remain implementation details inherited from the Menagerie
        # workcell. Policy-facing identities are deliberately color-neutral.
        self.part_specs = {
            "part_0": {
                "body": "part_blue_1",
                "joint": "part_blue_1_free",
                "geom": "part_blue_1_geom",
                "support_height": 0.025,
            },
            "part_1": {
                "body": "part_orange_1",
                "joint": "part_orange_1_free",
                "geom": "part_orange_1_geom",
                "support_height": 0.03,
            },
            "part_2": {
                "body": "part_blue_2",
                "joint": "part_blue_2_free",
                "geom": "part_blue_2_geom",
                "support_height": 0.03,
            },
        }
        self.part_order = list(self.part_specs)
        self.bin_order = ["bin_0", "bin_1"]

        self.ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site"
        )
        self.bin_site_ids = {
            "bin_0": mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, "bin_blue_target"
            ),
            "bin_1": mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, "bin_orange_target"
            ),
        }
        self.bin_approach_site_ids = {
            "bin_0": mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, "bin_blue_hover"
            ),
            "bin_1": mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, "bin_orange_hover"
            ),
        }
        # The lower orange approach avoids the UR5e's upper-arm/table grazing
        # posture while retaining clearance over the bin walls.
        self.model.site_pos[self.bin_approach_site_ids["bin_1"], 2] = 0.70
        pedestal_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "robot_pedestal"
        )
        # The decorative pedestal encloses the fixed robot base and otherwise
        # creates permanent false contacts with its collision capsules.
        self.model.geom_contype[pedestal_geom_id] = 0
        self.model.geom_conaffinity[pedestal_geom_id] = 0
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
        self.bin_geom_ids = {
            "bin_0": self._body_geom_ids("blue_bin"),
            "bin_1": self._body_geom_ids("orange_bin"),
        }
        self.bin_colors: dict[str, np.ndarray] = {}
        self.part_colors: dict[str, np.ndarray] = {}

        mujoco.mj_resetData(self.model, self.data)
        self._set_robot_configuration(self.home_ctrl)
        self.home_ee_target = self.data.site_xpos[self.ee_site_id].copy()

        self.holding_part: str | None = None
        self.active_part: str | None = None
        self.completed_parts: set[str] = set()
        self.controller_phase = "idle"
        self.last_pick_hover_target = self.home_ee_target.copy()
        self.spawn_layout: dict[str, np.ndarray] = {}

        self.trajectory_phase: str | None = None
        self.trajectory_start = self.home_ctrl.copy()
        self.trajectory_goal = self.home_ctrl.copy()
        self.trajectory_elapsed = 0.0
        self.trajectory_duration = self.min_trajectory_duration
        self.contact_pairs_this_step: set[tuple[int, int]] = set()

        if self.enable_rgb_observation:
            self.renderer = mujoco.Renderer(
                self.model,
                height=self.camera_height,
                width=self.camera_width,
            )

        self.reset()

    def _body_geom_ids(self, body_name: str) -> list[int]:
        body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, body_name
        )
        first_geom = int(self.model.body_geomadr[body_id])
        geom_count = int(self.model.body_geomnum[body_id])
        return list(range(first_geom, first_geom + geom_count))

    def _sample_distinct_colors(self) -> list[np.ndarray]:
        first_hue = float(self.random_state.uniform(0.0, 1.0))
        hue_separation = float(self.random_state.uniform(0.28, 0.50))
        hues = [first_hue, (first_hue + hue_separation) % 1.0]
        colors = []
        for hue in hues:
            saturation = float(self.random_state.uniform(0.72, 0.95))
            value = float(self.random_state.uniform(0.72, 0.95))
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append(np.array([*rgb, 1.0], dtype=float))
        return colors

    def _set_geom_color(self, geom_ids: list[int], color: np.ndarray) -> None:
        for geom_id in geom_ids:
            # Removing the material binding lets each geom receive its own
            # per-episode color instead of sharing the legacy named material.
            self.model.geom_matid[geom_id] = -1
            self.model.geom_rgba[geom_id] = color

    def _randomize_visual_task(self) -> None:
        self.model.cam_pos[self.camera_id] = self.base_camera_pos + self.random_state.uniform(
            [-0.015, -0.015, -0.02], [0.015, 0.015, 0.02]
        )
        if self.model.nlight:
            self.model.light_pos[:] = self.base_light_pos + self.random_state.uniform(
                -0.08, 0.08, self.base_light_pos.shape
            )
            light_scale = self.random_state.uniform(
                0.78, 1.18, (self.model.nlight, 1)
            )
            self.model.light_diffuse[:] = np.clip(
                self.base_light_diffuse * light_scale, 0.0, 1.0
            )

        colors = self._sample_distinct_colors()
        self.bin_colors = {
            bin_name: colors[index].copy()
            for index, bin_name in enumerate(self.bin_order)
        }
        for bin_name in self.bin_order:
            self._set_geom_color(self.bin_geom_ids[bin_name], self.bin_colors[bin_name])

        # Both bins receive at least one part. The remaining part and ordering
        # are sampled, so geometry names never imply a destination.
        assignments = self.bin_order.copy()
        assignments.append(
            self.bin_order[int(self.random_state.integers(len(self.bin_order)))]
        )
        self.random_state.shuffle(assignments)
        self.part_colors = {}
        for part_name, target_bin in zip(self.part_order, assignments, strict=True):
            match_id = self.bin_order.index(target_bin)
            self.part_specs[part_name]["match_id"] = match_id
            self.part_specs[part_name]["target_bin"] = target_bin
            color = self.bin_colors[target_bin].copy()
            self.part_colors[part_name] = color
            self._set_geom_color([self.part_geom_ids[part_name]], color)

    def render_rgb(self) -> np.ndarray:
        if self.renderer is None:
            raise RuntimeError("RGB observations are disabled for this environment")
        self.renderer.update_scene(
            self.data,
            camera=self.camera_name,
            scene_option=self.policy_scene_option,
        )
        return self.renderer.render().copy()

    def render_segmentation(self) -> np.ndarray:
        """Render privileged geom IDs for automatic dataset supervision."""
        if self.renderer is None:
            raise RuntimeError("RGB observations are disabled for this environment")
        self.renderer.enable_segmentation_rendering()
        try:
            self.renderer.update_scene(
                self.data,
                camera=self.camera_name,
                scene_option=self.policy_scene_option,
            )
            return self.renderer.render().copy()
        finally:
            self.renderer.disable_segmentation_rendering()

    def oracle_task_state(self) -> dict:
        """Privileged matching state for dataset labels and evaluation only."""
        return {
            "part_to_bin": {
                name: self.part_specs[name]["target_bin"]
                for name in self.part_order
            },
            "match_ids": {
                name: int(self.part_specs[name]["match_id"])
                for name in self.part_order
            },
            "bin_colors_rgba": {
                name: self.bin_colors[name].copy()
                for name in self.bin_order
            },
            "part_colors_rgba": {
                name: self.part_colors[name].copy()
                for name in self.part_order
            },
        }

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def reset(self) -> tuple[dict[str, np.ndarray], dict]:
        mujoco.mj_resetData(self.model, self.data)
        self._set_robot_configuration(self.home_ctrl)
        for name in self.part_order:
            self._set_part_collision_enabled(name, enabled=True)
        self._randomize_visual_task()
        self._randomize_part_layout()
        mujoco.mj_forward(self.model, self.data)
        self.holding_part = None
        self.active_part = None
        self.completed_parts = set()
        self.controller_phase = "select_part"
        self.last_pick_hover_target = self.home_ee_target.copy()
        self._reset_joint_trajectory()
        self.rgb_frame_counter = 0
        self.last_rgb = None
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
        smooth_target_q = self._trajectory_command(target_q)
        self.set_arm_configuration(smooth_target_q)
        self._advance_controller_state()
        return StepResult(
            observation=self._get_observation(),
            reward=self._task_reward(),
            terminated=False,
            truncated=False,
            info=self._get_info(),
        )

    def _reset_joint_trajectory(self) -> None:
        current = self.data.ctrl.copy()
        self.trajectory_phase = None
        self.trajectory_start = current.copy()
        self.trajectory_goal = current.copy()
        self.trajectory_elapsed = 0.0
        self.trajectory_duration = self.min_trajectory_duration

    def _trajectory_command(self, target_q: np.ndarray) -> np.ndarray:
        """Return a speed-limited C2-continuous joint command for this phase."""
        target_q = np.clip(target_q, self.ctrl_low, self.ctrl_high)
        if self.trajectory_phase != self.controller_phase:
            self.trajectory_phase = self.controller_phase
            self.trajectory_start = self.data.ctrl.copy()
            self.trajectory_goal = target_q.copy()
            self.trajectory_elapsed = 0.0
            delta = np.abs(self.trajectory_goal - self.trajectory_start)

            # Extrema for p(s)=10s^3-15s^4+6s^5 are 1.875 velocity,
            # 5.7735 acceleration, and 60 jerk (with unit duration).
            velocity_time = 1.875 * delta / self.max_joint_velocity
            acceleration_time = np.sqrt(
                5.7735 * delta / self.max_joint_acceleration
            )
            jerk_time = np.cbrt(60.0 * delta / self.max_joint_jerk)
            if float(np.max(delta)) < 1e-6:
                self.trajectory_duration = self.control_dt
            else:
                self.trajectory_duration = max(
                    self.min_trajectory_duration,
                    float(np.max(velocity_time)),
                    float(np.max(acceleration_time)),
                    float(np.max(jerk_time)),
                )

        self.trajectory_elapsed = min(
            self.trajectory_elapsed + self.control_dt, self.trajectory_duration
        )
        s = self.trajectory_elapsed / self.trajectory_duration
        blend = 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5
        return self.trajectory_start + blend * (
            self.trajectory_goal - self.trajectory_start
        )

    def _trajectory_complete(self) -> bool:
        return self.trajectory_elapsed >= self.trajectory_duration

    def set_arm_configuration(self, qpos: np.ndarray) -> None:
        target = np.clip(qpos, self.ctrl_low, self.ctrl_high)
        self.data.ctrl[:] = target
        self.contact_pairs_this_step.clear()
        for _ in range(self.control_substeps):
            if self.holding_part is not None:
                self._update_attachment()
            mujoco.mj_step(self.model, self.data)
            self.contact_pairs_this_step.update(
                (min(int(contact.geom1), int(contact.geom2)),
                 max(int(contact.geom1), int(contact.geom2)))
                for contact in self.data.contact[: self.data.ncon]
            )
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
            if self._trajectory_complete() and self._ee_close_to_position(
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
            if self._trajectory_complete() and self._ee_close_to_position(
                self.last_pick_hover_target, self.phase_position_tol
            ):
                self.controller_phase = "move_to_transfer"
            return

        if self.controller_phase == "move_to_transfer":
            if self.holding_part is None:
                self.controller_phase = "select_part"
                return
            if self._trajectory_complete() and self._ee_close_to_position(
                self.transfer_target, self.phase_position_tol
            ):
                self.controller_phase = "move_to_bin_hover"
            return

        if self.controller_phase == "move_to_bin_hover":
            if self.holding_part is None:
                self.controller_phase = "select_part"
                return
            hover_target = self._bin_hover_target(
                self.part_specs[self.holding_part]["target_bin"]
            )
            if self._trajectory_complete() and self._ee_close_to_position(
                hover_target, self.phase_position_tol
            ):
                self.controller_phase = "move_to_drop"
            return

        if self.controller_phase == "move_to_drop":
            if self.holding_part is None:
                self.active_part = None
                self.controller_phase = "return_home"
            return

        if self.controller_phase == "return_home":
            if self._trajectory_complete() and self._ee_close_to_position(
                self.home_ee_target, self.home_position_tol
            ):
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

        target_bin = self.part_specs[part_name]["target_bin"]
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
                and self._trajectory_complete()
                and self._ee_close_to_pick(self.active_part)
            ):
                self.holding_part = self.active_part
                self._set_part_collision_enabled(self.holding_part, enabled=False)
                self._update_attachment()
            return

        target_bin = self.part_specs[self.holding_part]["target_bin"]
        if (
            self.controller_phase == "move_to_drop"
            and self._trajectory_complete()
            and self._ee_ready_to_drop(target_bin)
        ):
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
            if self.part_specs[name]["target_bin"] == bin_name
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
            target_pos = self.data.site_xpos[
                self.bin_site_ids[spec["target_bin"]]
            ]
            reward -= float(np.linalg.norm(body_pos - target_pos))
        return reward

    def _part_state(self, name: str) -> dict:
        spawn_qpos = self.spawn_layout.get(name, self._read_freejoint_qpos(name))
        return {
            "spawn_position": spawn_qpos[:3].round(4).tolist(),
            "current_position": self.data.xpos[self.part_body_ids[name]].round(4).tolist(),
            "holding": name == self.holding_part,
            "sorted": name in self.completed_parts,
        }

    def _get_observation(self) -> dict[str, np.ndarray]:
        # Part coordinates, match IDs, target bins, and simulator colors are
        # intentionally excluded. A learned policy must infer them from RGB.
        proprioception = np.concatenate(
            [
                self.data.qpos[: self.arm_dofs].copy(),
                self.data.qvel[: self.arm_dofs].copy(),
                self.data.site_xpos[self.ee_site_id].copy(),
            ]
        ).astype(np.float32)
        observation = {"proprioception": proprioception}
        if self.renderer is not None:
            if (
                self.last_rgb is None
                or self.rgb_frame_counter % self.rgb_render_interval == 0
            ):
                self.last_rgb = self.render_rgb()
            observation["rgb"] = self.last_rgb.copy()
            self.rgb_frame_counter += 1
        return observation

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
        counts = {bin_name: 0 for bin_name in self.bin_order}
        for name in self.completed_parts:
            counts[self.part_specs[name]["target_bin"]] += 1
        return counts
