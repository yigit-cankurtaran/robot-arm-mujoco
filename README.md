# MuJoCo UR5e Demo

This project uses the official MuJoCo Menagerie `universal_robots_ur5e` model and its stock `scene.xml` so I can inspect the unmodified robot first.

arm model I used:

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
