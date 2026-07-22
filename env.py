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
        min_active_parts: int = 1,
        max_active_parts: int = 5,
        randomize_bin_positions: bool = True,
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
        if not 1 <= min_active_parts <= max_active_parts <= 8:
            raise ValueError("active part count must satisfy 1 <= min <= max <= 8")
        self.min_active_parts = min_active_parts
        self.max_active_parts = max_active_parts
        self.randomize_bin_positions = randomize_bin_positions
        self.rgb_frame_counter = 0
        self.last_rgb: np.ndarray | None = None
        self.policy_scene_option = mujoco.MjvOption()
        mujoco.mjv_defaultOption(self.policy_scene_option)
        self.policy_scene_option.sitegroup[:] = 0
        # The physical camera support remains visible in the interactive viewer
        # and remains a collision object, but an underslung inspection camera
        # should not see its own mounting bar across the image.
        camera_pole_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "camera_pole"
        )
        if camera_pole_geom_id >= 0:
            self.model.geom_group[camera_pole_geom_id] = 3
            self.policy_scene_option.geomgroup[3] = 0
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

        # Position-only IK treats wrist orientation as an independent tool
        # servo. The visible fixed-length gripper rotates between audited pick
        # and bin vectors while its base follows the UR attachment site.
        self.gripper_bin_vector = np.array([0.14, 0.15, -0.10], dtype=float)
        self.gripper_length = float(np.linalg.norm(self.gripper_bin_vector))
        raw_pick_vector = np.array([0.05, -0.16, -0.125], dtype=float)
        self.gripper_pick_vector = (
            self.gripper_length
            * raw_pick_vector
            / float(np.linalg.norm(raw_pick_vector))
        )
        self.pick_offset = -self.gripper_pick_vector
        self.pick_hover_offset = self.pick_offset + np.array(
            [0.0, 0.0, 0.08], dtype=float
        )
        self.transfer_target = np.array([0.45, 0.0, 0.90], dtype=float)
        self.home_position_tol = 0.06
        self.phase_position_tol = 0.05
        self.drop_release_xy_tol = 0.05
        self.drop_release_z_tol = 0.05
        self.drop_hover_height = 0.86
        self.drop_release_height = 0.82
        self.drop_settle_duration = 0.55
        self.drop_settle_timeout = 1.50
        self.bin_inner_half_width = 0.085

        self.table_surface_z = 0.49
        self.spawn_surface_z = 0.52
        self.spawn_anchors = [
            np.array([0.28, -0.25], dtype=float),
            np.array([0.38, -0.25], dtype=float),
            np.array([0.48, -0.25], dtype=float),
            np.array([0.28, -0.35], dtype=float),
            np.array([0.38, -0.35], dtype=float),
            np.array([0.48, -0.35], dtype=float),
        ]
        self.spawn_jitter = np.array([0.008, 0.008], dtype=float)
        self.spawn_clearance = 0.075
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
            "part_3": {
                "body": "part_sphere",
                "joint": "part_sphere_free",
                "geom": "part_sphere_geom",
                "support_height": 0.027,
            },
            "part_4": {
                "body": "part_capsule",
                "joint": "part_capsule_free",
                "geom": "part_capsule_geom",
                "support_height": 0.017,
            },
            "part_5": {
                "body": "part_ellipsoid",
                "joint": "part_ellipsoid_free",
                "geom": "part_ellipsoid_geom",
                "support_height": 0.020,
            },
            "part_6": {
                "body": "part_disc",
                "joint": "part_disc_free",
                "geom": "part_disc_geom",
                "support_height": 0.013,
            },
            "part_7": {
                "body": "part_brick",
                "joint": "part_brick_free",
                "geom": "part_brick_geom",
                "support_height": 0.016,
            },
        }
        self.part_order = list(self.part_specs)
        self.active_part_order = self.part_order.copy()
        self.bin_order = ["bin_0", "bin_1"]

        self.ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site"
        )
        gripper_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "simple_gripper_tool"
        )
        self.gripper_mocap_id = int(self.model.body_mocapid[gripper_body_id])
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
        self.bin_body_ids = {
            "bin_0": mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "blue_bin"
            ),
            "bin_1": mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "orange_bin"
            ),
        }
        self.base_bin_body_pos = {
            name: self.model.body_pos[body_id].copy()
            for name, body_id in self.bin_body_ids.items()
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
        self.current_gripper_vector = self.gripper_pick_vector.copy()
        self.released_part: str | None = None
        self.released_target_bin: str | None = None
        self.drop_settle_elapsed = 0.0
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
        self.visual_targets: dict[str, dict[str, np.ndarray | str]] | None = None

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

        # With two or more visible parts, both relational colors are represented.
        # A one-part scene samples either bin and remains a valid matching task.
        if len(self.active_part_order) >= 2:
            assignments = self.bin_order.copy()
            assignments.extend(
                self.bin_order[int(self.random_state.integers(len(self.bin_order)))]
                for _ in range(len(self.active_part_order) - 2)
            )
        else:
            assignments = [
                self.bin_order[int(self.random_state.integers(len(self.bin_order)))]
            ]
        self.random_state.shuffle(assignments)
        self.part_colors = {}
        for part_name, target_bin in zip(
            self.active_part_order, assignments, strict=True
        ):
            match_id = self.bin_order.index(target_bin)
            self.part_specs[part_name]["match_id"] = match_id
            self.part_specs[part_name]["target_bin"] = target_bin
            color = self.bin_colors[target_bin].copy()
            self.part_colors[part_name] = color
            self._set_geom_color([self.part_geom_ids[part_name]], color)

        for part_name in set(self.part_order) - set(self.active_part_order):
            self.part_specs[part_name].pop("match_id", None)
            self.part_specs[part_name].pop("target_bin", None)
            self.model.geom_rgba[self.part_geom_ids[part_name], 3] = 0.0

    def _randomize_task_structure(self) -> None:
        active_count = int(
            self.random_state.integers(
                self.min_active_parts, self.max_active_parts + 1
            )
        )
        chosen = self.random_state.choice(
            self.part_order, size=active_count, replace=False
        )
        chosen_names = {str(name) for name in chosen}
        self.active_part_order = [
            name for name in self.part_order if name in chosen_names
        ]

        for name, body_id in self.bin_body_ids.items():
            self.model.body_pos[body_id] = self.base_bin_body_pos[name]
        if self.randomize_bin_positions:
            # Sample across the far table bands. Bin walls remain inside the
            # tabletop, clear of the feed tray, visible, and inside the UR5e's
            # audited approach corridor.
            positive_center = self.random_state.uniform([0.62, 0.13], [0.64, 0.14])
            negative_center = self.random_state.uniform([0.62, 0.39], [0.64, 0.40])
            self.model.body_pos[self.bin_body_ids["bin_0"], :2] += (
                positive_center - np.array([0.70, 0.26])
            )
            self.model.body_pos[self.bin_body_ids["bin_1"], :2] += (
                negative_center - np.array([0.70, -0.26])
            )

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
                for name in self.active_part_order
            },
            "match_ids": {
                name: int(self.part_specs[name]["match_id"])
                for name in self.active_part_order
            },
            "bin_colors_rgba": {
                name: self.bin_colors[name].copy()
                for name in self.bin_order
            },
            "part_colors_rgba": {
                name: self.part_colors[name].copy()
                for name in self.active_part_order
            },
            "active_parts": self.active_part_order.copy(),
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
            self.model.geom_rgba[self.part_geom_ids[name], 3] = 1.0
        self._randomize_task_structure()
        for name in set(self.part_order) - set(self.active_part_order):
            self._set_part_collision_enabled(name, enabled=False)
        self._randomize_visual_task()
        self._randomize_part_layout()
        mujoco.mj_forward(self.model, self.data)
        self._update_gripper_pose()
        self.holding_part = None
        self.current_gripper_vector = self.gripper_pick_vector.copy()
        self.released_part = None
        self.released_target_bin = None
        self.drop_settle_elapsed = 0.0
        self.active_part = None
        self.completed_parts = set()
        self.visual_targets = None
        self.controller_phase = "select_part"
        self.last_pick_hover_target = self.home_ee_target.copy()
        self._reset_joint_trajectory()
        self.rgb_frame_counter = 0
        self.last_rgb = None
        return self._get_observation(), self._get_info()

    def set_visual_targets(self, commands: list[dict[str, np.ndarray]]) -> dict:
        """Latch RGB-derived task targets for the classical motion controller.

        The nearest-body association below is a simulator-only grasp adapter: the
        current workcell has no physical gripper, so MuJoCo needs an internal body
        name to implement attachment. It does not change a predicted pick point or
        bin match and is never exposed to the visual policy.
        """
        if not commands:
            raise ValueError("visual policy supplied no pick commands")
        unmatched_parts = set(self.active_part_order) - self.completed_parts
        targets: dict[str, dict[str, np.ndarray | str]] = {}
        associations = []
        for command in commands:
            pick = np.asarray(command["pick_position"], dtype=float).copy()
            target = np.asarray(command["bin_position"], dtype=float).copy()
            if pick.shape != (3,) or target.shape != (3,):
                raise ValueError("visual positions must be XYZ vectors")
            if not (0.22 <= pick[0] <= 0.60 and -0.42 <= pick[1] <= 0.15):
                raise ValueError(f"unsafe predicted pick position {pick.tolist()}")
            if not (0.47 <= target[0] <= 0.88 and -0.46 <= target[1] <= 0.46):
                raise ValueError(f"unsafe predicted bin position {target.tolist()}")
            if not unmatched_parts:
                raise ValueError("visual policy predicted too many parts")
            part_name = min(
                unmatched_parts,
                key=lambda name: float(
                    np.linalg.norm(
                        self.data.xpos[self.part_body_ids[name]][:2] - pick[:2]
                    )
                ),
            )
            part_error = float(
                np.linalg.norm(
                    self.data.xpos[self.part_body_ids[part_name]][:2] - pick[:2]
                )
            )
            if part_error > 0.06:
                raise ValueError(
                    f"predicted pick is {part_error:.3f} m from any simulated part"
                )
            bin_name = min(
                self.bin_order,
                key=lambda name: float(
                    np.linalg.norm(
                        self.data.site_xpos[self.bin_site_ids[name]][:2] - target[:2]
                    )
                ),
            )
            bin_error = float(
                np.linalg.norm(
                    self.data.site_xpos[self.bin_site_ids[bin_name]][:2] - target[:2]
                )
            )
            if bin_error > 0.08:
                raise ValueError(
                    f"predicted bin is {bin_error:.3f} m from any simulated bin"
                )
            unmatched_parts.remove(part_name)
            targets[part_name] = {
                "pick_position": pick,
                "bin_position": target,
                "sim_bin": bin_name,
            }
            associations.append(
                {
                    "sim_part": part_name,
                    "sim_bin": bin_name,
                    "pick_error_m": part_error,
                    "bin_error_m": bin_error,
                }
            )
        self.visual_targets = targets
        self.active_part = None
        self.controller_phase = "select_part"
        self._reset_joint_trajectory()
        return {"commands": len(targets), "associations": associations}

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
        if self.controller_phase in {"idle", "return_home", "drop_failed"}:
            # Cartesian position alone has several UR5e solutions. Returning to
            # the known joint posture prevents the next pick from starting on
            # a table-facing IK branch.
            target_q = self.home_ctrl.copy()
        else:
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
            self._update_gripper_pose()
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
            self._update_gripper_pose()
        if self.released_part is not None:
            self.drop_settle_elapsed += self.control_dt
        self._update_task_progress()

    def describe_task(self) -> dict:
        return {
            "scene": str(self.xml_path),
            "control_dt": self.control_dt,
            "controller_phase": self.controller_phase,
            "active_part": self.active_part,
            "parts": {
                name: self._part_state(name)
                for name in self.active_part_order
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
                self.last_pick_hover_target, self._motion_position_tolerance()
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
                self.last_pick_hover_target, self._motion_position_tolerance()
            ):
                self.controller_phase = "move_to_transfer"
            return

        if self.controller_phase == "move_to_transfer":
            if self.holding_part is None:
                self.controller_phase = "select_part"
                return
            if self._trajectory_complete() and self._ee_close_to_position(
                self.transfer_target, self._motion_position_tolerance()
            ):
                self.controller_phase = "move_to_bin_hover"
            return

        if self.controller_phase == "move_to_bin_hover":
            if self.holding_part is None:
                self.controller_phase = "select_part"
                return
            hover_target = self._part_bin_hover_target(self.holding_part)
            if self._trajectory_complete() and self._ee_close_to_position(
                hover_target, self._motion_position_tolerance()
            ):
                self.controller_phase = "move_to_drop"
            return

        if self.controller_phase == "move_to_drop":
            if self.holding_part is None:
                self.controller_phase = "settle_drop"
            return

        if self.controller_phase == "settle_drop":
            if self.released_part is None:
                self.controller_phase = "return_home"
                return
            if self.drop_settle_elapsed >= self.drop_settle_duration:
                if self._released_part_inside_target_bin():
                    self.completed_parts.add(self.released_part)
                    self.released_part = None
                    self.released_target_bin = None
                    self.active_part = None
                    self.last_pick_hover_target = self.home_ee_target.copy()
                    self.controller_phase = "return_home"
                elif self.drop_settle_elapsed >= self.drop_settle_timeout:
                    # Stay visibly failed instead of crediting a missed drop.
                    self.controller_phase = "drop_failed"
            return

        if self.controller_phase == "return_home":
            if self._trajectory_complete() and self._ee_close_to_position(
                self.home_ee_target, self.home_position_tol
            ) and float(
                np.max(np.abs(self.data.qpos[: self.arm_dofs] - self.home_ctrl))
            ) < 0.04:
                self.controller_phase = (
                    "select_part" if self._choose_next_part() is not None else "idle"
                )

    def _controller_target_position(self) -> np.ndarray:
        if self.controller_phase in {"idle", "return_home", "drop_failed"}:
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

        if self.controller_phase == "move_to_bin_hover":
            return self._part_bin_hover_target(part_name)
        if self.controller_phase == "move_to_drop":
            return self._part_drop_release_target(part_name)
        if self.controller_phase == "settle_drop":
            return self._part_bin_hover_target(part_name)
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

        for name in self.active_part_order:
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
                    self.spawn_surface_z + self.part_specs[name]["support_height"],
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
        available_parts = (
            list(self.visual_targets)
            if self.visual_targets is not None
            else self.active_part_order
        )
        candidates = [
            name
            for name in available_parts
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
        if self.visual_targets is not None:
            target = np.asarray(
                self.visual_targets[part_name]["pick_position"], dtype=float
            ) + self.pick_offset
        else:
            target = (
                self.data.xpos[self.part_body_ids[part_name]].copy()
                + self.pick_offset
            )
        target[2] = max(target[2], 0.635)
        return target

    def _pick_hover_target(self, part_name: str) -> np.ndarray:
        if self.visual_targets is not None:
            target = np.asarray(
                self.visual_targets[part_name]["pick_position"], dtype=float
            ) + self.pick_hover_offset
        else:
            target = (
                self.data.xpos[self.part_body_ids[part_name]].copy()
                + self.pick_hover_offset
            )
        target[2] = max(target[2], 0.785)
        return target

    def _bin_hover_target(self, bin_name: str) -> np.ndarray:
        bin_position = self.data.site_xpos[self.bin_site_ids[bin_name]].copy()
        bin_position[:2] -= self.gripper_bin_vector[:2]
        bin_position[2] = self.drop_hover_height
        return bin_position

    def _drop_release_target(self, bin_name: str) -> np.ndarray:
        bin_position = self.data.site_xpos[self.bin_site_ids[bin_name]].copy()
        bin_position[:2] -= self.gripper_bin_vector[:2]
        bin_position[2] = self.drop_release_height
        return bin_position

    def _part_bin_hover_target(self, part_name: str) -> np.ndarray:
        if self.visual_targets is None:
            return self._bin_hover_target(self.part_specs[part_name]["target_bin"])
        bin_position = np.asarray(
            self.visual_targets[part_name]["bin_position"], dtype=float
        ).copy()
        bin_position[:2] -= self.gripper_bin_vector[:2]
        bin_position[2] = self.drop_hover_height
        return bin_position

    def _part_drop_release_target(self, part_name: str) -> np.ndarray:
        if self.visual_targets is None:
            return self._drop_release_target(self.part_specs[part_name]["target_bin"])
        bin_position = np.asarray(
            self.visual_targets[part_name]["bin_position"], dtype=float
        ).copy()
        bin_position[:2] -= self.gripper_bin_vector[:2]
        bin_position[2] = self.drop_release_height
        return bin_position

    def _part_sim_target_bin(self, part_name: str) -> str:
        if self.visual_targets is None:
            return str(self.part_specs[part_name]["target_bin"])
        return str(self.visual_targets[part_name]["sim_bin"])

    def _motion_position_tolerance(self) -> float:
        # Pixel calibration adds roughly 1--2 cm of target uncertainty.
        return 0.075 if self.visual_targets is not None else self.phase_position_tol

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

        target_bin = self._part_sim_target_bin(self.holding_part)
        if (
            self.controller_phase == "move_to_drop"
            and self._trajectory_complete()
            and self._ee_close_to_position(
                self._part_drop_release_target(self.holding_part),
                self._motion_position_tolerance(),
            )
        ):
            self._release_part(self.holding_part, target_bin)

    def _update_attachment(self) -> None:
        if self.holding_part is None:
            return
        ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        tip_pos = ee_pos + self.current_gripper_vector
        target_qpos = np.array(
            [
                tip_pos[0],
                tip_pos[1],
                tip_pos[2],
                1.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=float,
        )
        self._write_freejoint_qpos(self.holding_part, target_qpos)
        mujoco.mj_forward(self.model, self.data)

    def _update_gripper_pose(self) -> None:
        pick_vector = self.gripper_pick_vector
        if self.holding_part is not None and self.controller_phase in {
            "move_to_transfer",
            "move_to_bin_hover",
            "move_to_drop",
        }:
            if self.controller_phase == "move_to_transfer":
                progress = min(
                    self.trajectory_elapsed / max(self.trajectory_duration, 1e-9),
                    1.0,
                )
                blend = 10.0 * progress**3 - 15.0 * progress**4 + 6.0 * progress**5
                direction = (
                    (1.0 - blend) * pick_vector
                    + blend * self.gripper_bin_vector
                )
                self.current_gripper_vector = (
                    self.gripper_length * direction / np.linalg.norm(direction)
                )
            else:
                self.current_gripper_vector = self.gripper_bin_vector.copy()
        elif self.controller_phase == "settle_drop" and self.released_part is not None:
            self.current_gripper_vector = self.gripper_bin_vector.copy()
        else:
            self.current_gripper_vector = pick_vector.copy()

        self.data.mocap_pos[self.gripper_mocap_id] = self.data.site_xpos[
            self.ee_site_id
        ]
        direction = self.current_gripper_vector / np.linalg.norm(
            self.current_gripper_vector
        )
        local_z_world = -direction
        reference = (
            np.array([0.0, 1.0, 0.0], dtype=float)
            if abs(local_z_world[2]) > 0.9
            else np.array([0.0, 0.0, 1.0], dtype=float)
        )
        x_axis = np.cross(reference, local_z_world)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(local_z_world, x_axis)
        rotation = np.column_stack((x_axis, y_axis, local_z_world))
        quaternion = np.empty(4, dtype=float)
        mujoco.mju_mat2Quat(quaternion, rotation.ravel())
        self.data.mocap_quat[self.gripper_mocap_id] = quaternion

    def _ee_close_to_pick(self, part_name: str, tol: float = 0.05) -> bool:
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

    def _release_part(self, part_name: str, bin_name: str) -> None:
        """Open the simplified gripper without changing the part pose."""
        self._set_part_collision_enabled(part_name, enabled=True)
        self.holding_part = None
        self.released_part = part_name
        self.released_target_bin = bin_name
        self.drop_settle_elapsed = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _released_part_inside_target_bin(self) -> bool:
        if self.released_part is None or self.released_target_bin is None:
            return False
        part_pos = self.data.xpos[self.part_body_ids[self.released_part]]
        bin_pos = self.data.site_xpos[self.bin_site_ids[self.released_target_bin]]
        inside_xy = bool(
            np.all(np.abs(part_pos[:2] - bin_pos[:2]) <= self.bin_inner_half_width)
        )
        support_height = float(
            self.part_specs[self.released_part]["support_height"]
        )
        settled_z = (
            self.table_surface_z + support_height
            <= part_pos[2]
            <= 0.68 + support_height
        )
        return inside_xy and settled_z

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
        for name in self.active_part_order:
            spec = self.part_specs[name]
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
            "released_part": self.released_part,
            "sorted_counts": self._sorted_counts(),
            "parts": {
                name: self._part_state(name)
                for name in self.active_part_order
            },
            "active_part_count": len(self.active_part_order),
        }

    def _sorted_counts(self) -> dict[str, int]:
        counts = {bin_name: 0 for bin_name in self.bin_order}
        for name in self.completed_parts:
            counts[self.part_specs[name]["target_bin"]] += 1
        return counts
