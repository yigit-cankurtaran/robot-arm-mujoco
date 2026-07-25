from __future__ import annotations

import copy
import colorsys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parent
SCENE_PATH = (
    ROOT / "third_party" / "menagerie" / "universal_robots_ur5e" / "workcell_scene.xml"
)
GRIPPER_PATH = ROOT / "third_party" / "menagerie" / "robotiq_2f85" / "2f85.xml"


def _load_model_with_policy_camera(
    xml_path: Path, camera_name: str
) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(xml_path))
    gripper_spec = mujoco.MjSpec.from_file(str(GRIPPER_PATH))
    attachment_site = spec.site("attachment_site")
    if attachment_site is None:
        raise ValueError("robot model does not define attachment_site")
    attachment_site.attach_body(
        gripper_spec.body("base_mount"), prefix="gripper/"
    )
    # The upstream 2F-85 model is tuned for elliptic friction cones and a high
    # tangential-to-normal contact impedance ratio.
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10.0
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


@dataclass(frozen=True)
class ContactEvent:
    geom1: int
    geom2: int
    distance: float
    position: tuple[float, float, float]
    simulated_time: float
    controller_phase: str
    expected_part_geom_id: int | None


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
        # Route previews use an independent model so collision masks can omit
        # loose parts without changing the physical simulation.  Copying the
        # compiled model is inexpensive and preserves all numeric IDs.
        self.collision_model = copy.copy(self.model)
        self.collision_data = mujoco.MjData(self.collision_model)
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
        self.ctrl_low = self.model.actuator_ctrlrange[: self.arm_dofs, 0].copy()
        self.ctrl_high = self.model.actuator_ctrlrange[: self.arm_dofs, 1].copy()
        self.control_substeps = 10
        self.control_dt = self.model.opt.timestep * self.control_substeps
        # Downward-facing, table-clear IK branch for the physically attached
        # Robotiq gripper.  Equivalent wrapped joint solutions exist, but this
        # one was selected by collision audit and has good workspace coverage.
        self.home_ctrl = np.array(
            [-3.56075, -1.79627, 1.23290, -1.73263, 4.06687, -2.54526],
            dtype=float,
        )

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
        # Discrete 50 Hz commands and phase boundaries need margin below the
        # continuous minimum-jerk polynomial's analytical peak.
        self.trajectory_jerk_margin = 0.55
        self.min_trajectory_duration = 0.24
        # Loaded motion is intentionally slower than unloaded minimum-jerk
        # motion. These scales are benchmarked independently so speed tuning
        # never has to weaken the underlying velocity/acceleration/jerk limits.
        self.transfer_duration_scale = 2.35
        self.bin_hover_duration_scale = 1.7
        # Flat-sided and upright, axisymmetric parts remain well constrained
        # during the long transfer. Their final bin approach retains more
        # tracking margin because its tested joint path passes closer to the
        # table than the compact rounded/low-profile group.
        self.stable_part_transfer_duration_scale = 2.2
        self.stable_part_bin_hover_duration_scale = 2.4
        # With no neighboring or previously sorted objects, a single-object
        # episode can run much nearer the derivative-limited trajectory.
        self.single_part_transfer_duration_scale = 1.3
        self.single_part_bin_hover_duration_scale = 1.0

        self.ik_iterations = 32
        self.ik_damping = 0.08
        self.ik_step_scale = 0.8
        self.ik_rest_gain = 0.08
        self.ik_tolerance = 0.002
        self.ik_max_update = 0.35

        # The pinch site is position-and-axis controlled. These audited approach
        # directions keep the bulky wrist inboard while the shorter physical
        # fingers reach the tabletop and randomized bins. Rotation about the
        # gripper axis remains free, avoiding an unnecessary IK constraint.
        raw_pick_direction = np.array([0.05, -0.16, -0.125], dtype=float)
        self.gripper_pick_direction = raw_pick_direction / np.linalg.norm(
            raw_pick_direction
        )
        self.pick_direction_candidates = [
            direction / np.linalg.norm(direction)
            for direction in (
                np.array([0.05, -0.16, -0.125], dtype=float),
                np.array([0.05, -0.14, -0.125], dtype=float),
                np.array([0.05, -0.12, -0.125], dtype=float),
                np.array([0.05, -0.10, -0.125], dtype=float),
                np.array([0.02, -0.14, -0.125], dtype=float),
                np.array([0.08, -0.14, -0.125], dtype=float),
            )
        ]
        self.active_pick_direction = self.gripper_pick_direction.copy()
        self.active_pick_direction_index = 0
        raw_bin_direction = np.array([0.628, 0.736, -0.254], dtype=float)
        self.gripper_bin_direction = raw_bin_direction / np.linalg.norm(
            raw_bin_direction
        )
        self.pick_hover_height = 0.16
        self.low_profile_grasp_height = 0.028
        self.grasp_table_clearance = 0.006
        self.max_grasp_center_offset = 0.018
        self.transfer_target = np.array([0.45, 0.0, 0.82], dtype=float)
        self.bin_ik_seed = np.array(
            [-0.0324, -1.3677, -1.0930, -1.1547, 2.1906, -3.4311],
            dtype=float,
        )
        self.bin_approach_ik_seed = np.array(
            [-0.0324, -1.3677, -1.0930, -1.1547, 2.1906, -3.4311],
            dtype=float,
        )
        self.home_position_tol = 0.06
        self.phase_position_tol = 0.025
        self.drop_release_xy_tol = 0.05
        self.drop_release_z_tol = 0.05
        self.drop_hover_height = 0.76
        self.drop_release_height = 0.69
        self.drop_settle_duration = 0.55
        self.drop_settle_timeout = 3.00
        self.grasp_close_duration = 0.70
        self.grasp_settle_duration = 0.20
        self.grasp_loss_timeout = 0.08
        self.gripper_open_ctrl = 0.0
        self.gripper_close_ctrl = 255.0
        self.gripper_release_duration = 0.20
        self.bin_inner_half_width = 0.085

        self.table_surface_z = 0.49
        self.spawn_surface_z = self.table_surface_z
        self.spawn_anchors = [
            np.array([0.25, -0.22], dtype=float),
            np.array([0.385, -0.22], dtype=float),
            np.array([0.52, -0.22], dtype=float),
            np.array([0.25, -0.36], dtype=float),
            np.array([0.385, -0.36], dtype=float),
            np.array([0.52, -0.36], dtype=float),
        ]
        self.spawn_jitter = np.array([0.004, 0.004], dtype=float)
        self.spawn_clearance = 0.115
        self.random_state = np.random.default_rng()

        # XML names remain implementation details inherited from the Menagerie
        # workcell. Policy-facing identities are deliberately color-neutral.
        self.part_specs = {
            "part_0": {
                "body": "part_blue_1",
                "joint": "part_blue_1_free",
                "geom": "part_blue_1_geom",
                "support_height": 0.025,
                "motion_profile": "stable",
            },
            "part_1": {
                "body": "part_orange_1",
                "joint": "part_orange_1_free",
                "geom": "part_orange_1_geom",
                "support_height": 0.03,
                "motion_profile": "stable",
                # This tall cylinder has a narrow pad contact patch. Even
                # without neighboring clutter it needs the cautious transfer
                # scale to avoid slowly walking out of the physical grasp.
                "single_transfer_scale": 2.1,
            },
            "part_2": {
                "body": "part_blue_2",
                "joint": "part_blue_2_free",
                "geom": "part_blue_2_geom",
                "support_height": 0.03,
                "motion_profile": "stable",
            },
            "part_3": {
                "body": "part_sphere",
                "joint": "part_sphere_free",
                "geom": "part_sphere_geom",
                "support_height": 0.027,
                "motion_profile": "cautious",
            },
            "part_4": {
                "body": "part_capsule",
                "joint": "part_capsule_free",
                "geom": "part_capsule_geom",
                "support_height": 0.017,
                "motion_profile": "cautious",
            },
            "part_5": {
                "body": "part_ellipsoid",
                "joint": "part_ellipsoid_free",
                "geom": "part_ellipsoid_geom",
                "support_height": 0.020,
                "motion_profile": "cautious",
            },
            "part_6": {
                "body": "part_disc",
                "joint": "part_disc_free",
                "geom": "part_disc_geom",
                "support_height": 0.013,
                "motion_profile": "cautious",
            },
            "part_7": {
                "body": "part_brick",
                "joint": "part_brick_free",
                "geom": "part_brick_geom",
                "support_height": 0.016,
                "motion_profile": "stable",
            },
        }
        self.part_order = list(self.part_specs)
        self.active_part_order = self.part_order.copy()
        self.bin_order = ["bin_0", "bin_1"]

        self.flange_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site"
        )
        self.ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "gripper/pinch"
        )
        self.gripper_actuator_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper/fingers_actuator"
        )
        self.gripper_adhesion_actuator_ids = [
            mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                f"gripper/{side}_pad_adhesion",
            )
            for side in ("right", "left")
        ]
        self.gripper_pad_geom_ids = {
            "right": {
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM, "gripper/right_pad1"
                ),
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM, "gripper/right_pad2"
                ),
            },
            "left": {
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM, "gripper/left_pad1"
                ),
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM, "gripper/left_pad2"
                ),
            },
        }
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
        self.collision_model.geom_contype[pedestal_geom_id] = 0
        self.collision_model.geom_conaffinity[pedestal_geom_id] = 0
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
        # A pick-route preview answers whether the arm itself clears the
        # workcell. Loose objects are dynamic obstacles whose frozen poses are
        # not meaningful along a hypothetical sweep, and the ellipsoid can
        # make MuJoCo's convex solver spend seconds resolving deep preview-only
        # overlaps. The live simulation still retains every part collision.
        for geom_id in self.part_geom_ids.values():
            self.collision_model.geom_contype[geom_id] = 0
            self.collision_model.geom_conaffinity[geom_id] = 0
        # Contact-free commanded geometry is not enough for a torque-driven
        # arm: tracking lag can cut a corner by a few millimetres. Inflating
        # only the preview model turns that uncertainty into a planning margin
        # without changing any live contact dynamics.
        self.route_collision_margin = 0.008
        self.bin_geom_ids = {
            "bin_0": self._body_geom_ids("blue_bin"),
            "bin_1": self._body_geom_ids("orange_bin"),
        }
        self.robot_root_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "base"
        )
        self.gripper_root_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper/base_mount"
        )
        clearance_geoms = [
            mujoco.mj_name2id(
                self.collision_model,
                mujoco.mjtObj.mjOBJ_GEOM,
                "table_top",
            ),
            *(
                geom_id
                for geom_ids in self.bin_geom_ids.values()
                for geom_id in geom_ids
            ),
        ]
        for geom_id in clearance_geoms:
            self.collision_model.geom_margin[geom_id] = max(
                self.collision_model.geom_margin[geom_id],
                self.route_collision_margin,
            )
        self.bin_colors: dict[str, np.ndarray] = {}
        self.part_colors: dict[str, np.ndarray] = {}

        mujoco.mj_resetData(self.model, self.data)
        self._set_robot_configuration(self.home_ctrl)
        self.home_ee_target = self.data.site_xpos[self.ee_site_id].copy()

        self.holding_part: str | None = None
        self.released_part: str | None = None
        self.released_target_bin: str | None = None
        self.drop_settle_elapsed = 0.0
        self.grasp_elapsed = 0.0
        self.grasp_lost_elapsed = 0.0
        self.gripper_phase_elapsed = 0.0
        self.active_part: str | None = None
        self.completed_parts: set[str] = set()
        self.controller_phase = "idle"
        self.last_pick_hover_target = self.home_ee_target.copy()
        # Returning most of the way along the known-safe home trajectory clears
        # the bin, but the next pick may start before exact home convergence.
        self.inter_pick_home_fraction = 0.8
        self.force_full_home_for_pick = False
        self.route_retry_from_home = False
        self.route_preflight_checks = 0
        self.route_preflight_rejections = 0
        self.spawn_layout: dict[str, np.ndarray] = {}

        self.trajectory_phase: str | None = None
        self.trajectory_start = self.home_ctrl.copy()
        self.trajectory_goal = self.home_ctrl.copy()
        self.trajectory_start_velocity = np.zeros(self.arm_dofs, dtype=float)
        self.trajectory_start_acceleration = np.zeros(
            self.arm_dofs, dtype=float
        )
        self.trajectory_velocity = np.zeros(self.arm_dofs, dtype=float)
        self.trajectory_acceleration = np.zeros(self.arm_dofs, dtype=float)
        self.trajectory_coefficients = np.vstack(
            (
                self.home_ctrl.copy(),
                np.zeros((5, self.arm_dofs), dtype=float),
            )
        )
        self.trajectory_elapsed = 0.0
        self.trajectory_duration = self.min_trajectory_duration
        self.contact_pairs_this_step: set[tuple[int, int]] = set()
        self.contact_events_this_step: list[ContactEvent] = []
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
            # These are world-space centers, not body translations. Both bands
            # are inside the collision-audited UR5e corridor and remain at
            # least one bin width apart. Farther positive-Y placements are
            # position-reachable but drive the upper arm through the tabletop.
            centers = {
                "bin_0": self.random_state.uniform([0.40, 0.28], [0.46, 0.36]),
                "bin_1": self.random_state.uniform([0.60, 0.04], [0.66, 0.10]),
            }
            nominal_centers = {
                "bin_0": np.array([0.70, 0.26]),
                "bin_1": np.array([0.70, -0.26]),
            }
            for bin_name in self.bin_order:
                self.model.body_pos[self.bin_body_ids[bin_name], :2] = (
                    centers[bin_name] - nominal_centers[bin_name]
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
        self.holding_part = None
        self.released_part = None
        self.released_target_bin = None
        self.drop_settle_elapsed = 0.0
        self.grasp_elapsed = 0.0
        self.grasp_lost_elapsed = 0.0
        self.gripper_phase_elapsed = 0.0
        self.active_part = None
        self.completed_parts = set()
        self.visual_targets = None
        self.controller_phase = "select_part"
        self.last_pick_hover_target = self.home_ee_target.copy()
        self.active_pick_direction = self.gripper_pick_direction.copy()
        self.active_pick_direction_index = 0
        self.force_full_home_for_pick = False
        self.route_retry_from_home = False
        self.route_preflight_checks = 0
        self.route_preflight_rejections = 0
        self._reset_joint_trajectory()
        self.rgb_frame_counter = 0
        self.last_rgb = None
        return self._get_observation(), self._get_info()

    def set_visual_targets(self, commands: list[dict[str, np.ndarray]]) -> dict:
        """Latch RGB-derived task targets for the classical motion controller.

        The nearest-body association below maps anonymous RGB detections to
        simulator bodies only so contact can be verified against the intended
        object. It does not alter the predicted pick point or bin match and is
        never exposed to the visual policy.
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
        if self.controller_phase in {
            "idle",
            "return_home",
            "drop_failed",
            "grasp_failed",
            "route_failed",
        }:
            # Cartesian position alone has several UR5e solutions. Returning to
            # the known joint posture prevents the next pick from starting on
            # a table-facing IK branch.
            target_q = self.home_ctrl.copy()
        elif (
            self.trajectory_phase == self.controller_phase
            and not self._trajectory_complete()
        ):
            # The minimum-jerk goal is latched for the duration of a segment.
            # Re-solving identical Cartesian IK on every intermediate tick
            # cannot change the command, so reuse the goal until the segment
            # completes. A moving/settling target is solved again at the
            # endpoint and `_trajectory_command()` starts a smooth correction
            # if it drifted materially.
            target_q = self.trajectory_goal.copy()
        else:
            target_q = (
                self._select_collision_free_pick_target(target_pos)
                if self.controller_phase
                in {"move_to_pick_hover", "move_to_pick"}
                else self._solve_inverse_kinematics(target_pos)
            )
            if target_q is None:
                self._handle_blocked_pick_route()
                target_q = self.home_ctrl.copy()
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
        current = self.data.ctrl[: self.arm_dofs].copy()
        self.trajectory_phase = None
        self.trajectory_start = current.copy()
        self.trajectory_goal = current.copy()
        self.trajectory_start_velocity[:] = 0.0
        self.trajectory_start_acceleration[:] = 0.0
        self.trajectory_velocity[:] = 0.0
        self.trajectory_acceleration[:] = 0.0
        self.trajectory_coefficients = np.vstack(
            (current.copy(), np.zeros((5, self.arm_dofs), dtype=float))
        )
        self.trajectory_elapsed = 0.0
        self.trajectory_duration = self.min_trajectory_duration

    def _trajectory_command(self, target_q: np.ndarray) -> np.ndarray:
        """Return a limit-checked C2-continuous joint command for this phase."""
        target_q = np.clip(target_q, self.ctrl_low, self.ctrl_high)
        target_drifted = bool(
            self._trajectory_complete()
            and np.max(np.abs(target_q - self.trajectory_goal)) > 0.015
        )
        if self.trajectory_phase != self.controller_phase or target_drifted:
            self.trajectory_phase = self.controller_phase
            self.trajectory_start = self.data.ctrl[: self.arm_dofs].copy()
            self.trajectory_goal = target_q.copy()
            self.trajectory_start_velocity = self.trajectory_velocity.copy()
            self.trajectory_start_acceleration = (
                self.trajectory_acceleration.copy()
            )
            self.trajectory_elapsed = 0.0
            delta = np.abs(self.trajectory_goal - self.trajectory_start)

            # These rest-to-rest extrema provide a good initial duration.
            # A general boundary-conditioned quintic is then checked below
            # because a mid-motion replan can start with nonzero velocity and
            # acceleration.
            velocity_time = 1.875 * delta / self.max_joint_velocity
            acceleration_time = np.sqrt(
                5.7735 * delta / self.max_joint_acceleration
            )
            jerk_time = np.cbrt(
                60.0
                * delta
                / (self.trajectory_jerk_margin * self.max_joint_jerk)
            )
            stationary = bool(
                float(np.max(delta)) < 1e-6
                and float(np.max(np.abs(self.trajectory_start_velocity)))
                < 1e-6
                and float(
                    np.max(np.abs(self.trajectory_start_acceleration))
                )
                < 1e-6
            )
            if stationary:
                self.trajectory_duration = self.control_dt
                self.trajectory_coefficients = (
                    self._quintic_coefficients(self.trajectory_duration)
                )
            else:
                initial_duration = max(
                    self.min_trajectory_duration,
                    float(np.max(velocity_time)),
                    float(np.max(acceleration_time)),
                    float(np.max(jerk_time)),
                )
                if self.controller_phase == "move_to_bin_hover":
                    initial_duration *= self._loaded_motion_scale(
                        "bin_hover"
                    )
                elif self.controller_phase == "move_to_transfer":
                    # This phase also rotates from the pick attitude to the bin
                    # attitude. Slow it under load so rounded parts do not gain
                    # enough lateral momentum to slide out of the real pads.
                    initial_duration *= self._loaded_motion_scale(
                        "transfer"
                    )
                (
                    self.trajectory_duration,
                    self.trajectory_coefficients,
                ) = self._limit_checked_trajectory(initial_duration)

        self.trajectory_elapsed = min(
            self.trajectory_elapsed + self.control_dt, self.trajectory_duration
        )
        (
            command,
            self.trajectory_velocity,
            self.trajectory_acceleration,
            _,
        ) = self._quintic_state(
            self.trajectory_coefficients,
            self.trajectory_duration,
            self.trajectory_elapsed,
        )
        return command

    def _quintic_coefficients(self, duration: float) -> np.ndarray:
        """Build normalized-time coefficients for the current boundaries."""
        coefficients = np.zeros((6, self.arm_dofs), dtype=float)
        coefficients[0] = self.trajectory_start
        coefficients[1] = self.trajectory_start_velocity * duration
        coefficients[2] = (
            0.5 * self.trajectory_start_acceleration * duration**2
        )
        remaining_position = self.trajectory_goal - np.sum(
            coefficients[:3], axis=0
        )
        remaining_velocity = -coefficients[1] - 2.0 * coefficients[2]
        remaining_acceleration = -2.0 * coefficients[2]
        boundary_matrix = np.array(
            [
                [1.0, 1.0, 1.0],
                [3.0, 4.0, 5.0],
                [6.0, 12.0, 20.0],
            ],
            dtype=float,
        )
        coefficients[3:] = np.linalg.solve(
            boundary_matrix,
            np.vstack(
                (
                    remaining_position,
                    remaining_velocity,
                    remaining_acceleration,
                )
            ),
        )
        return coefficients

    def _quintic_state(
        self,
        coefficients: np.ndarray,
        duration: float,
        elapsed: float | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate position through jerk for normalized quintic coefficients."""
        times = np.atleast_1d(np.asarray(elapsed, dtype=float))
        s = np.clip(times / duration, 0.0, 1.0)[:, None]
        position = sum(coefficients[degree] * s**degree for degree in range(6))
        velocity = sum(
            degree * coefficients[degree] * s ** (degree - 1)
            for degree in range(1, 6)
        ) / duration
        acceleration = sum(
            degree
            * (degree - 1)
            * coefficients[degree]
            * s ** (degree - 2)
            for degree in range(2, 6)
        ) / duration**2
        jerk = sum(
            degree
            * (degree - 1)
            * (degree - 2)
            * coefficients[degree]
            * s ** (degree - 3)
            for degree in range(3, 6)
        ) / duration**3
        if np.ndim(elapsed) == 0:
            return position[0], velocity[0], acceleration[0], jerk[0]
        return position, velocity, acceleration, jerk

    def _limit_checked_trajectory(
        self, initial_duration: float
    ) -> tuple[float, np.ndarray]:
        """Expand duration until the full polynomial respects every limit."""
        duration = max(initial_duration, self.control_dt)
        for _ in range(40):
            coefficients = self._quintic_coefficients(duration)
            sample_count = max(
                257,
                int(np.ceil(duration / self.control_dt)) * 4 + 1,
            )
            elapsed = np.linspace(0.0, duration, sample_count)
            _, velocity, acceleration, jerk = self._quintic_state(
                coefficients, duration, elapsed
            )
            if (
                np.all(
                    np.max(np.abs(velocity), axis=0)
                    <= self.max_joint_velocity
                )
                and np.all(
                    np.max(np.abs(acceleration), axis=0)
                    <= self.max_joint_acceleration
                )
                and np.all(
                    np.max(np.abs(jerk), axis=0)
                    <= self.trajectory_jerk_margin * self.max_joint_jerk
                )
            ):
                return duration, coefficients
            duration *= 1.12
        raise RuntimeError("unable to construct a limit-compliant trajectory")

    def _reset_active_pick_direction(self) -> None:
        self.active_pick_direction = self.gripper_pick_direction.copy()
        self.active_pick_direction_index = 0
        self.force_full_home_for_pick = False
        self.route_retry_from_home = False

    def _select_collision_free_pick_target(
        self, target_pos: np.ndarray
    ) -> np.ndarray | None:
        """Choose the first pick attitude whose full joint sweep is safe."""
        directions = [self.active_pick_direction, *self.pick_direction_candidates]
        tried: list[np.ndarray] = []
        for direction in directions:
            if any(np.allclose(direction, previous) for previous in tried):
                continue
            tried.append(direction)
            self.active_pick_direction = direction.copy()
            target_q = self._solve_inverse_kinematics(target_pos)
            self.route_preflight_checks += 1
            if (
                self._ik_solution_reaches_target(target_q, target_pos)
                and self._target_trajectory_is_collision_free(target_q)
            ):
                self.active_pick_direction_index = next(
                    (
                        index
                        for index, candidate in enumerate(
                            self.pick_direction_candidates
                        )
                        if np.allclose(candidate, direction)
                    ),
                    -1,
                )
                self.route_retry_from_home = False
                return target_q
            self.route_preflight_rejections += 1
        self.active_pick_direction = self.gripper_pick_direction.copy()
        self.active_pick_direction_index = 0
        return None

    def _ik_solution_reaches_target(
        self,
        target_q: np.ndarray,
        target_pos: np.ndarray,
    ) -> bool:
        """Reject a collision-free IK result that stopped at a joint limit."""
        self.ik_data.qpos[:] = self.data.qpos
        self.ik_data.qvel[:] = 0.0
        self.ik_data.qpos[: self.arm_dofs] = target_q
        mujoco.mj_kinematics(self.model, self.ik_data)
        position_error = float(
            np.linalg.norm(
                self.ik_data.site_xpos[self.ee_site_id] - target_pos
            )
        )
        # Oracle control uses the tighter 2.5 cm motion tolerance. Pixel-based
        # commands retain their calibrated perception allowance.
        allowed_error = max(
            4.0 * self.ik_tolerance,
            0.75 * self._motion_position_tolerance(),
        )
        return position_error <= allowed_error

    def _handle_blocked_pick_route(self) -> None:
        """Fall back to exact home once, then stop if no safe sweep exists."""
        if self.route_retry_from_home:
            self.controller_phase = "route_failed"
            return
        self.force_full_home_for_pick = True
        self.controller_phase = "return_home"

    def _target_trajectory_is_collision_free(
        self, target_q: np.ndarray
    ) -> bool:
        """Preview a proposed segment without changing real simulation state."""
        snapshot = (
            self.trajectory_phase,
            self.trajectory_start.copy(),
            self.trajectory_goal.copy(),
            self.trajectory_start_velocity.copy(),
            self.trajectory_start_acceleration.copy(),
            self.trajectory_velocity.copy(),
            self.trajectory_acceleration.copy(),
            self.trajectory_coefficients.copy(),
            self.trajectory_elapsed,
            self.trajectory_duration,
        )
        try:
            self._trajectory_command(target_q)
            return self._swept_trajectory_is_collision_free(
                self.trajectory_coefficients,
                self.trajectory_duration,
            )
        finally:
            (
                self.trajectory_phase,
                self.trajectory_start,
                self.trajectory_goal,
                self.trajectory_start_velocity,
                self.trajectory_start_acceleration,
                self.trajectory_velocity,
                self.trajectory_acceleration,
                self.trajectory_coefficients,
                self.trajectory_elapsed,
                self.trajectory_duration,
            ) = snapshot

    def _swept_trajectory_is_collision_free(
        self,
        coefficients: np.ndarray,
        duration: float,
    ) -> bool:
        """Kinematically collision-check a trajectory at control-rate samples."""
        sample_count = max(
            33,
            int(np.ceil(duration / self.control_dt)) + 1,
        )
        elapsed = np.linspace(0.0, duration, sample_count)
        positions, _, _, _ = self._quintic_state(
            coefficients, duration, elapsed
        )
        for body_id in self.bin_body_ids.values():
            self.collision_model.body_pos[body_id] = self.model.body_pos[body_id]
        self.collision_data.qpos[:] = self.data.qpos
        self.collision_data.qvel[:] = 0.0
        self.collision_data.ctrl[:] = self.data.ctrl
        for qpos in positions:
            self.collision_data.qpos[: self.arm_dofs] = qpos
            # Only body kinematics and collision detection are needed here.
            # Calling the full position pipeline would also solve tendon
            # constraints for the frozen gripper.
            mujoco.mj_kinematics(
                self.collision_model, self.collision_data
            )
            mujoco.mj_collision(
                self.collision_model, self.collision_data
            )
            if any(
                self._scratch_contact_is_unsafe(
                    int(contact.geom1),
                    int(contact.geom2),
                    None,
                )
                for contact in self.collision_data.contact[
                    : self.collision_data.ncon
                ]
            ):
                return False
        return True

    def _body_descends_from(self, body_id: int, root_id: int) -> bool:
        while body_id > 0:
            if body_id == root_id:
                return True
            body_id = int(self.model.body_parentid[body_id])
        return False

    def _scratch_contact_is_unsafe(
        self,
        geom1: int,
        geom2: int,
        expected_part_geom_id: int | None,
    ) -> bool:
        body1 = int(self.model.geom_bodyid[geom1])
        body2 = int(self.model.geom_bodyid[geom2])
        robot1 = self._body_descends_from(body1, self.robot_root_body_id)
        robot2 = self._body_descends_from(body2, self.robot_root_body_id)
        unsafe = robot1 != robot2

        if robot1 != robot2 and expected_part_geom_id is not None:
            robot_geom = geom1 if robot1 else geom2
            external_geom = geom2 if robot1 else geom1
            robot_body = int(self.model.geom_bodyid[robot_geom])
            if (
                self._body_descends_from(
                    robot_body, self.gripper_root_body_id
                )
                and external_geom == expected_part_geom_id
            ):
                unsafe = False

        if robot1 and robot2 and body1 != body2:
            unsafe = int(self.model.body_parentid[body1]) != body2 and int(
                self.model.body_parentid[body2]
            ) != body1
            if self._body_descends_from(
                body1, self.gripper_root_body_id
            ) and self._body_descends_from(
                body2, self.gripper_root_body_id
            ):
                unsafe = False
        return unsafe

    def _loaded_motion_scale(self, motion: str) -> float:
        """Choose audited loaded timing from crowding and held-part stability."""
        if motion not in {"transfer", "bin_hover"}:
            raise ValueError(f"unknown loaded motion {motion!r}")

        if len(self.active_part_order) == 1:
            if motion == "transfer":
                part_name = self.holding_part or self.active_part
                if part_name is not None:
                    return float(
                        self.part_specs[part_name].get(
                            "single_transfer_scale",
                            self.single_part_transfer_duration_scale,
                        )
                    )
                return self.single_part_transfer_duration_scale
            return self.single_part_bin_hover_duration_scale

        part_name = self.holding_part or self.active_part
        profile = (
            self.part_specs[part_name]["motion_profile"]
            if part_name is not None
            else "cautious"
        )
        if profile == "stable":
            if motion == "transfer":
                return self.stable_part_transfer_duration_scale
            return self.stable_part_bin_hover_duration_scale
        if motion == "transfer":
            return self.transfer_duration_scale
        return self.bin_hover_duration_scale

    def _trajectory_complete(self) -> bool:
        return self.trajectory_elapsed >= self.trajectory_duration

    def set_arm_configuration(self, qpos: np.ndarray) -> None:
        target = np.clip(qpos, self.ctrl_low, self.ctrl_high)
        self.data.ctrl[: self.arm_dofs] = target
        self.data.ctrl[self.gripper_actuator_id] = self._gripper_control_target()
        adhesion_control = float(self._gripper_control_target() > 0.0)
        self.data.ctrl[self.gripper_adhesion_actuator_ids] = adhesion_control
        self.contact_pairs_this_step.clear()
        self.contact_events_this_step.clear()
        for _ in range(self.control_substeps):
            mujoco.mj_step(self.model, self.data)
            expected_part = self.holding_part or (
                self.active_part
                if self.controller_phase in {"move_to_pick", "close_gripper"}
                else self.released_part
                if self.controller_phase == "open_gripper"
                else None
            )
            expected_part_geom_id = (
                self.part_geom_ids[expected_part]
                if expected_part is not None
                else None
            )
            self.contact_pairs_this_step.update(
                (min(int(contact.geom1), int(contact.geom2)),
                 max(int(contact.geom1), int(contact.geom2)))
                for contact in self.data.contact[: self.data.ncon]
            )
            self.contact_events_this_step.extend(
                ContactEvent(
                    geom1=int(contact.geom1),
                    geom2=int(contact.geom2),
                    distance=float(contact.dist),
                    position=tuple(float(value) for value in contact.pos),
                    simulated_time=float(self.data.time),
                    controller_phase=self.controller_phase,
                    expected_part_geom_id=expected_part_geom_id,
                )
                for contact in self.data.contact[: self.data.ncon]
            )
        if self.controller_phase == "close_gripper":
            self.grasp_elapsed += self.control_dt
        else:
            self.grasp_elapsed = 0.0
        if self.controller_phase == "open_gripper":
            self.gripper_phase_elapsed += self.control_dt
        else:
            self.gripper_phase_elapsed = 0.0
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
            self._reset_active_pick_direction()
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
            if self._trajectory_complete() and self._ee_close_to_pick(
                self.active_part, self._motion_position_tolerance()
            ):
                self.grasp_elapsed = 0.0
                self.controller_phase = "close_gripper"
            return

        if self.controller_phase == "close_gripper":
            if (
                self.holding_part == self.active_part
                and self.grasp_elapsed >= self.grasp_settle_duration
                and self._trajectory_complete()
            ):
                self.controller_phase = "lift_with_part"
            elif (
                self.holding_part is None
                and self.grasp_elapsed >= self.grasp_close_duration
            ):
                self.controller_phase = "grasp_failed"
            return

        if self.controller_phase == "lift_with_part":
            if self.holding_part is None:
                self.controller_phase = "grasp_failed"
                return
            if self._trajectory_complete() and self._ee_close_to_position(
                self.last_pick_hover_target, self._motion_position_tolerance()
            ):
                self.controller_phase = "move_to_transfer"
            return

        if self.controller_phase == "move_to_transfer":
            if self.holding_part is None:
                self.controller_phase = "grasp_failed"
                return
            if self._trajectory_complete() and self._ee_close_to_position(
                self.transfer_target, self._motion_position_tolerance()
            ):
                self.controller_phase = "move_to_bin_hover"
            return

        if self.controller_phase == "move_to_bin_hover":
            if self.holding_part is None:
                self.controller_phase = "grasp_failed"
                return
            hover_target = self._part_bin_hover_target(self.holding_part)
            if self._trajectory_complete() and self._ee_close_to_position(
                hover_target, self._motion_position_tolerance()
            ):
                self.controller_phase = "move_to_drop"
            return

        if self.controller_phase == "move_to_drop":
            if self.holding_part is None:
                self.controller_phase = "grasp_failed"
                return
            if self._trajectory_complete() and self._ee_close_to_position(
                self._part_drop_release_target(self.holding_part),
                self._motion_position_tolerance(),
            ):
                target_bin = self._part_sim_target_bin(self.holding_part)
                self._release_part(self.holding_part, target_bin)
                self.gripper_phase_elapsed = 0.0
                self.controller_phase = "open_gripper"
            return

        if self.controller_phase == "open_gripper":
            if (
                self.gripper_phase_elapsed >= self.gripper_release_duration
                and self._trajectory_complete()
            ):
                self.controller_phase = "settle_drop"
            return

        if self.controller_phase == "settle_drop":
            if self.released_part is None:
                self.controller_phase = "return_home"
                return
            if self.drop_settle_elapsed >= self.drop_settle_duration:
                withdrawal_complete = (
                    self._trajectory_complete()
                    and self._ee_close_to_position(
                        self._part_bin_hover_target(self.released_part),
                        self._motion_position_tolerance(),
                    )
                )
                if self._released_part_inside_target_bin() and withdrawal_complete:
                    self.completed_parts.add(self.released_part)
                    self.released_part = None
                    self.released_target_bin = None
                    next_part = self._choose_next_part()
                    if next_part is None:
                        self.active_part = None
                        self.last_pick_hover_target = self.home_ee_target.copy()
                        self.controller_phase = "return_home"
                    else:
                        self.active_part = next_part
                        self._reset_active_pick_direction()
                        self.last_pick_hover_target = self._pick_hover_target(
                            next_part
                        )
                        self.controller_phase = "return_home"
                elif self.drop_settle_elapsed >= self.drop_settle_timeout:
                    # Stay visibly failed instead of crediting a missed drop.
                    self.controller_phase = "drop_failed"
            return

        if self.controller_phase == "return_home":
            if (
                self.active_part is not None
                and not self.force_full_home_for_pick
                and self.trajectory_phase == "return_home"
                and self.trajectory_elapsed
                >= self.inter_pick_home_fraction * self.trajectory_duration
            ):
                self.controller_phase = "move_to_pick_hover"
                return
            if self._trajectory_complete() and self._ee_close_to_position(
                self.home_ee_target, self.home_position_tol
            ) and float(
                np.max(np.abs(self.data.qpos[: self.arm_dofs] - self.home_ctrl))
            ) < 0.04:
                self.controller_phase = (
                    "move_to_pick_hover"
                    if self.active_part is not None
                    else "select_part"
                    if self._choose_next_part() is not None
                    else "idle"
                )
                if self.active_part is not None:
                    self.force_full_home_for_pick = False
                    self.route_retry_from_home = True
            return

        if self.controller_phase == "grasp_failed":
            self.holding_part = None
            self.active_part = None
            self.last_pick_hover_target = self.home_ee_target.copy()
            self.controller_phase = "return_home"

        if self.controller_phase == "route_failed":
            return

    def _controller_target_position(self) -> np.ndarray:
        if self.controller_phase in {
            "idle",
            "return_home",
            "drop_failed",
            "grasp_failed",
            "route_failed",
        }:
            return self.home_ee_target

        if self.controller_phase == "move_to_pick_hover" and self.active_part is not None:
            return self._pick_hover_target(self.active_part)

        if self.controller_phase == "move_to_pick" and self.active_part is not None:
            return self._pick_target(self.active_part)

        if self.controller_phase == "close_gripper" and self.active_part is not None:
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
        if self.controller_phase == "open_gripper" and self.released_part is not None:
            return self._part_drop_release_target(self.released_part)
        if self.controller_phase == "settle_drop":
            return self._part_bin_hover_target(part_name)
        return self.home_ee_target

    def _solve_inverse_kinematics(self, target_pos: np.ndarray) -> np.ndarray:
        q = (
            self.bin_ik_seed.copy()
            if self.controller_phase == "move_to_transfer"
            else self.bin_approach_ik_seed.copy()
            if self.controller_phase in {"move_to_bin_hover", "move_to_drop"}
            else self.data.qpos[: self.arm_dofs].copy()
        )
        jacp = np.zeros((3, self.model.nv), dtype=float)
        jacr = np.zeros((3, self.model.nv), dtype=float)

        for _ in range(self.ik_iterations):
            self.ik_data.qpos[:] = self.data.qpos
            self.ik_data.qvel[:] = 0.0
            self.ik_data.qpos[: self.arm_dofs] = q
            # IK only needs body/site transforms and subtree centers of mass
            # for the geometric Jacobian. Running the complete position
            # pipeline here would also solve contacts for every intermediate
            # iterate, including deliberately bad candidate poses.
            mujoco.mj_kinematics(self.model, self.ik_data)
            mujoco.mj_comPos(self.model, self.ik_data)

            ee_pos = self.ik_data.site_xpos[self.ee_site_id].copy()
            current_quat = np.empty(4, dtype=float)
            mujoco.mju_mat2Quat(
                current_quat, self.ik_data.site_xmat[self.ee_site_id]
            )
            target_quat = np.empty(4, dtype=float)
            mujoco.mju_mat2Quat(
                target_quat, self._gripper_target_rotation().ravel()
            )
            inverse_current = np.empty(4, dtype=float)
            mujoco.mju_negQuat(inverse_current, current_quat)
            delta_quat = np.empty(4, dtype=float)
            mujoco.mju_mulQuat(delta_quat, target_quat, inverse_current)
            rotation_error = np.empty(3, dtype=float)
            mujoco.mju_quat2Vel(rotation_error, delta_quat, 1.0)
            error = np.concatenate((target_pos - ee_pos, rotation_error))
            if float(np.linalg.norm(error)) < self.ik_tolerance:
                break

            mujoco.mj_jacSite(self.model, self.ik_data, jacp, jacr, self.ee_site_id)
            arm_jac = np.vstack((jacp, jacr))[:, : self.arm_dofs]
            regularized = arm_jac @ arm_jac.T + (
                self.ik_damping**2
            ) * np.eye(error.shape[0])
            dq = arm_jac.T @ np.linalg.solve(regularized, error)
            step_norm = float(np.linalg.norm(dq))
            if step_norm > self.ik_max_update:
                dq *= self.ik_max_update / step_norm
            q = np.clip(q + self.ik_step_scale * dq, self.ctrl_low, self.ctrl_high)

        return q

    def _gripper_target_direction(self) -> np.ndarray:
        if self.controller_phase in {
            "move_to_transfer",
            "move_to_bin_hover",
            "move_to_drop",
            "open_gripper",
            "settle_drop",
        }:
            return self.gripper_bin_direction
        return self.active_pick_direction

    def _gripper_target_rotation(self) -> np.ndarray:
        z_axis = self._gripper_target_direction()
        # Keep the jaw-closing axis horizontal so gravity cannot slide thin
        # pieces toward one fingertip during transport.
        y_axis = np.cross(np.array([0.0, 0.0, 1.0]), z_axis)
        y_axis /= np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)
        return np.column_stack((x_axis, y_axis, z_axis))

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
            ).copy()
        else:
            target = self.data.xpos[self.part_body_ids[part_name]].copy()
        # The official pinch site is centered between the pads. Use the live
        # object height after it has settled; a world-Z offset would become a
        # lateral pad error because this collision-safe approach is angled.
        support_height = float(self.part_specs[part_name]["support_height"])
        center_offset = self.grasp_table_clearance + min(
            self.max_grasp_center_offset,
            max(0.0, self.low_profile_grasp_height - support_height),
        )
        target[2] = (
            self.data.xpos[self.part_body_ids[part_name], 2] + center_offset
        )
        return target

    def _pick_hover_target(self, part_name: str) -> np.ndarray:
        if self.visual_targets is not None:
            target = np.asarray(
                self.visual_targets[part_name]["pick_position"], dtype=float
            ).copy()
        else:
            target = self.data.xpos[self.part_body_ids[part_name]].copy()
        target[2] = self.table_surface_z + self.pick_hover_height
        return target

    def _bin_hover_target(self, bin_name: str) -> np.ndarray:
        bin_position = self.data.site_xpos[self.bin_site_ids[bin_name]].copy()
        bin_position[2] = self.drop_hover_height
        return bin_position

    def _drop_release_target(self, bin_name: str) -> np.ndarray:
        bin_position = self.data.site_xpos[self.bin_site_ids[bin_name]].copy()
        bin_position[2] = self.drop_release_height
        return bin_position

    def _part_bin_hover_target(self, part_name: str) -> np.ndarray:
        if self.visual_targets is None:
            return self._bin_hover_target(self.part_specs[part_name]["target_bin"])
        bin_position = np.asarray(
            self.visual_targets[part_name]["bin_position"], dtype=float
        ).copy()
        bin_position[2] = self.drop_hover_height
        return bin_position

    def _part_drop_release_target(self, part_name: str) -> np.ndarray:
        if self.visual_targets is None:
            return self._drop_release_target(self.part_specs[part_name]["target_bin"])
        bin_position = np.asarray(
            self.visual_targets[part_name]["bin_position"], dtype=float
        ).copy()
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
        if self.controller_phase == "close_gripper" and self.active_part is not None:
            if self._has_bilateral_pad_contact(self.active_part):
                self.holding_part = self.active_part
                self.grasp_lost_elapsed = 0.0
            return

        if self.holding_part is None:
            return
        if self.controller_phase in {
            "lift_with_part",
            "move_to_transfer",
            "move_to_bin_hover",
            "move_to_drop",
        }:
            if self._grasp_is_retained(self.holding_part):
                self.grasp_lost_elapsed = 0.0
            else:
                self.grasp_lost_elapsed += self.control_dt
                if self.grasp_lost_elapsed >= self.grasp_loss_timeout:
                    self.holding_part = None

    def _grasp_is_retained(self, part_name: str) -> bool:
        # Bilateral contact is required to declare the initial grasp. During
        # motion MuJoCo may reduce a redundant contact manifold to one pad even
        # while the closed linkage mechanically cages the object. Separation
        # from the pinch frame is the reliable loss criterion; it never applies
        # forces or rewrites the object's state.
        separation = np.linalg.norm(
            self.data.xpos[self.part_body_ids[part_name]]
            - self.data.site_xpos[self.ee_site_id]
        )
        return bool(separation < 0.075)

    def _has_bilateral_pad_contact(self, part_name: str) -> bool:
        part_geom_id = self.part_geom_ids[part_name]
        touching_sides: set[str] = set()
        for geom1, geom2 in self.contact_pairs_this_step:
            if part_geom_id not in {geom1, geom2}:
                continue
            other = geom2 if geom1 == part_geom_id else geom1
            for side, pad_ids in self.gripper_pad_geom_ids.items():
                if other in pad_ids:
                    touching_sides.add(side)
        return touching_sides == set(self.gripper_pad_geom_ids)

    def _gripper_control_target(self) -> float:
        closed_phases = {
            "close_gripper",
            "lift_with_part",
            "move_to_transfer",
            "move_to_bin_hover",
            "move_to_drop",
        }
        return (
            self.gripper_close_ctrl
            if self.controller_phase in closed_phases
            else self.gripper_open_ctrl
        )

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
        """Open the physical fingers without rewriting the part pose."""
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
        self.data.ctrl[: self.arm_dofs] = q
        self.data.ctrl[self.gripper_actuator_id] = self.gripper_open_ctrl
        self.data.ctrl[self.gripper_adhesion_actuator_ids] = 0.0
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
            "pick_direction_index": self.active_pick_direction_index,
            "route_preflight_checks": self.route_preflight_checks,
            "route_preflight_rejections": self.route_preflight_rejections,
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
