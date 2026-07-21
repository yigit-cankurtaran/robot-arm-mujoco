# MuJoCo UR5e Pick-And-Sort Cell

This project uses the official MuJoCo Menagerie `universal_robots_ur5e` model inside a custom visual pick-and-sort workcell. Every episode generates new, unlabeled colors and randomly assigns parts to same-color bins. The current oracle task planner provides a safe bridge while a learned visual matcher is trained from RGB observations.

Arm model:

- MuJoCo Menagerie UR5e: https://github.com/google-deepmind/mujoco_menagerie/tree/main/universal_robots_ur5e
- Universal Robots UR5e product page: https://www.universal-robots.com/products/ur5-robot/

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
  The custom workcell scene. It includes the official UR5e model, a feed tray, two sorting bins, and three loose parts.
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

- Three parts spawn at randomized reachable positions on the tabletop.
- Two distinct colors are sampled continuously on every reset; there are no fixed color classes or names.
- Each part is randomly assigned to one same-color bin, with both bins guaranteed to receive at least one part.
- The controller uses inverse kinematics on the UR5e attachment site to move through pick, lift, transfer, and place phases instead of replaying fixed joint-space waypoints.
- The policy observation contains a `240x240` RGB image and 15 robot proprioception values. It does not expose object poses, simulator colors, match IDs, or target bins.
- During pickup, the selected part is attached to the UR5e end-effector site and carried to the correct bin.

The scripted controller currently reads the privileged `target_bin` assignment
stored inside the environment. `oracle_task_state()` exposes copies of the same
truth for dataset labels and evaluation. These fields are absent from policy
observations and can be replaced by a learned detector and shared
color-embedding matcher without changing the motion controller.

This gives a presentable robotics baseline now, while leaving extension points for perception, grasp planning, scheduling, or learning later.

## Limitations
- Grasping is still simplified as a scripted attachment to the end-effector site rather than a contact-rich gripper model.
- Visual dataset generation is implemented, but the learned detector/matching network is the next stage; the live sorter still uses the isolated oracle assignment.
