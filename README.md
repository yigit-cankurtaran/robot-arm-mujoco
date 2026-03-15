# MuJoCo UR5e Pick-And-Sort Cell

This project uses the official MuJoCo Menagerie `universal_robots_ur5e` model inside a custom pick-and-sort workcell. The robot sorts blue and orange parts from a feed tray into matching bins.

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

## What This Repo Contains

- `third_party/menagerie/universal_robots_ur5e/workcell_scene.xml`
  The custom workcell scene. It includes the official UR5e model, a feed tray, two sorting bins, and three loose parts.
- `env.py`
  The task environment. It loads the scene, keeps track of parts and target bins, and implements the scripted baseline sorter.
- `demo.py`
  The interactive viewer entrypoint. It launches MuJoCo and runs the robot through repeated pick-and-sort cycles.
- `check_env.py`
  A quick smoke test that loads the full workcell and prints the initial task description.

## How The Sorting Demo Works

- Three parts start in the feed tray.
- Blue parts are assigned to the blue bin and orange parts are assigned to the orange bin.
- The baseline controller uses hand-tuned joint-space waypoints for each pickup pose and each drop-off bin.
- During pickup, the selected part is attached to the UR5e end-effector site and carried to the correct bin.

This gives a presentable robotics baseline now, while leaving extension points for perception, grasp planning, scheduling, or learning later.

## Limitations
- Currently a scripted scenario, will add motion planning and such later on.
