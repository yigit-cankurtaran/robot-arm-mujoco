# MuJoCo UR5e Pick-And-Sort Cell

This project uses the official MuJoCo Menagerie `universal_robots_ur5e` model inside a custom visual pick-and-sort workcell. Every episode generates new, unlabeled colors and randomly assigns parts to same-color bins. It includes both an oracle regression mode and a trained RGB-only visual planner feeding the classical motion controller.

Arm model:

- MuJoCo Menagerie UR5e: https://github.com/google-deepmind/mujoco_menagerie/tree/main/universal_robots_ur5e
- MuJoCo Menagerie Robotiq 2F-85: https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotiq_2f85
- Universal Robots UR5e product page: https://www.universal-robots.com/products/ur5-robot/

The imported gripper meshes, model, and BSD-2-Clause license are kept together
under `third_party/menagerie/robotiq_2f85/`.

For a detailed explanation of the entire codebase, start with the
[project handbook](docs/README.md).

## Setup

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

## Run

```bash
.venv/bin/mjpython demo.py
```

The demo opens the normal MuJoCo viewer plus a small panel showing the exact RGB
frame supplied to the future visual policy. The panel runs in a separate process
so its OpenCV/Cocoa event loop cannot conflict with MuJoCo's GLFW window. Press
`q` or Escape in that panel to exit. Use `--no-camera-panel` to hide it.

The regular oracle demo labels that panel `ORACLE OVERLAY` and draws display-only
part/bin contours and matching arrows. `visual_sort_demo.py` labels it
`VISUAL POLICY OVERLAY` and draws the learned detections instead. In both cases
the underlying policy RGB remains unannotated.

After the final object settles in its bin, the arm returns fully home, prints a
success line with object count, simulated time, wall time, and throughput, then
closes both windows and the camera child process automatically.

If an arm or gripper collision geom touches the table, a bin, the camera
support, an unrelated object, or a non-adjacent robot link, the demo stops on
that physics tick instead. It prints the contact pair, phase, simulated time,
and world-space contact point, then writes the complete joint/task snapshot to
`runs/safety_failures/contact_failure_<UTC timestamp>.json` before cleanup.
Normal fingertip contact with the selected object and internal gripper contacts
are intentionally excluded.

To test only the isolated camera process for three seconds:

```bash
.venv/bin/python check_camera_panel.py
```

Use `.venv/bin/python check_env.py` if you make any changes and want to confirm the scene still loads.

Run the deterministic headless motion audit after controller or scene changes:

```bash
.venv/bin/python audit_motion.py --seeds 20
```

It checks full-cycle completion and timing, commanded joint velocity,
acceleration and jerk, tracking error, and robot contacts with the workcell.

Check randomized colors, hidden matching assignments, and rendered observations:

```bash
.venv/bin/python check_visual_observation.py --seeds 10
```

Generate automatically labeled visual-matching data:

```bash
.venv/bin/python generate_visual_dataset.py --samples 1000
```

Samples contain RGB and robot proprioception as policy inputs. MuJoCo instance
masks, poses, and target-bin indices are stored separately as privileged training
labels; they are never included in the policy observation.

## What This Repo Contains

- `third_party/menagerie/universal_robots_ur5e/workcell_scene.xml`
  The custom workcell scene. It includes the official UR5e model, a feed tray,
  two sorting bins, and eight loose bodies spanning boxes, cylinders, a sphere,
  a capsule, an ellipsoid, and a brick.
- `env.py`
  The task environment. It randomizes unlabeled match groups, renders visual observations, and implements the autonomous IK-based sorter.
- `demo.py`
  The interactive viewer entrypoint. It launches MuJoCo and runs the robot through repeated pick-and-sort cycles.
- `check_env.py`
  A quick smoke test that loads the full workcell and prints the initial task description.
- `audit_motion.py`
  A deterministic, headless speed/smoothness/collision regression check.
- `check_visual_observation.py`
  A headless regression check for randomized matching and camera observations.
- `generate_visual_dataset.py`
  Generates RGB matching data with simulator-derived segmentation and pose labels.

## How The Sorting Demo Works

- One to five parts are sampled from eight shapes and spawn at randomized,
  collision-audited reachable positions on the tabletop.
- Two distinct colors are sampled continuously on every reset; there are no fixed color classes or names.
- Each part is randomly assigned to one same-color bin, with both bins guaranteed to receive at least one part.
- The controller uses position-and-orientation inverse kinematics on the Robotiq pinch site to move through pick, close, lift, transfer, open, and place phases instead of replaying fixed joint-space waypoints.
- The policy observation contains a `240x240` RGB image and 15 robot proprioception values. It does not expose object poses, simulator colors, match IDs, or target bins.
- During pickup, the Robotiq 2F-85 closes its four-bar finger linkage. A grasp is accepted only after both silicone pads contact the selected part; MuJoCo contact forces, friction, and gravity determine whether it stays in the gripper.

The original `demo.py` remains an oracle regression mode. The learned RGB mode
is available after training:

```bash
.venv/bin/mjpython visual_sort_demo.py
```

It detects variable-count parts and two bins from RGB, performs relational
same-color matching without named color classes, estimates workspace targets,
and hands those targets to the existing classical controller. It re-observes
and replans when the controller returns to idle, allowing a temporarily
occluded or missed part to be recovered. See the
[learned visual policy guide](docs/learned-visual-policy.md) for reproducible
generation, training, evaluation, results, and limitations.

This gives a presentable robotics baseline now, while leaving extension points for perception, grasp planning, scheduling, or learning later.

## Limitations
- The gripper is physically simulated, but grasp poses are still generated by a
  shape-aware classical controller rather than learned grasp planning. The
  imported Menagerie 2F-85 uses a rigid `0.10 m` wrist spacer, high-friction
  pads, and contact-only adhesion assistance; it never welds or rewrites a held
  object's pose.
- The learned baseline is validated only over the present simulation
  randomization range; real-camera transfer and contact-rich grasping remain
  future work.
