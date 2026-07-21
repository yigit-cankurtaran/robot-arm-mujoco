# MuJoCo UR5e Pick-And-Sort Cell

This project uses the official MuJoCo Menagerie `universal_robots_ur5e` model inside a custom pick-and-sort workcell. The robot sorts blue and orange parts from randomized tabletop spawn locations into matching bins.

Arm model:

- MuJoCo Menagerie UR5e: https://github.com/google-deepmind/mujoco_menagerie/tree/main/universal_robots_ur5e
- Universal Robots UR5e product page: https://www.universal-robots.com/products/ur5-robot/

## Setup

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

## Run

```bash
.venv/bin/mjpython demo.py
```

Use `.venv/bin/mjpython check_env.py` if you make any changes and want to confirm the scene still loads.

Run the deterministic headless motion audit after controller or scene changes:

```bash
.venv/bin/python audit_motion.py --seeds 20
```

It checks full-cycle completion and timing, commanded joint velocity,
acceleration and jerk, tracking error, and robot contacts with the workcell.

## What This Repo Contains

- `third_party/menagerie/universal_robots_ur5e/workcell_scene.xml`
  The custom workcell scene. It includes the official UR5e model, a feed tray, two sorting bins, and three loose parts.
- `env.py`
  The task environment. It loads the scene, keeps track of parts, colors, and target bins, and implements the autonomous IK-based sorter.
- `demo.py`
  The interactive viewer entrypoint. It launches MuJoCo and runs the robot through repeated pick-and-sort cycles.
- `check_env.py`
  A quick smoke test that loads the full workcell and prints the initial task description.
- `audit_motion.py`
  A deterministic, headless speed/smoothness/collision regression check.

## How The Sorting Demo Works

- Three parts spawn at randomized reachable positions on the tabletop.
- Blue parts are assigned to the blue bin and orange parts are assigned to the orange bin.
- The controller uses inverse kinematics on the UR5e attachment site to move through pick, lift, transfer, and place phases instead of replaying fixed joint-space waypoints.
- The environment observation and task description expose each part's color, current position, spawn position, and target bin.
- During pickup, the selected part is attached to the UR5e end-effector site and carried to the correct bin.

This gives a presentable robotics baseline now, while leaving extension points for perception, grasp planning, scheduling, or learning later.

## Limitations
- Grasping is still simplified as a scripted attachment to the end-effector site rather than a contact-rich gripper model.
