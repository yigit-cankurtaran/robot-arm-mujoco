from __future__ import annotations

import time

import mujoco
import mujoco.viewer

from env import FactoryFloorEnv


def main() -> None:
    env = FactoryFloorEnv()
    print(env.describe_task())
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.azimuth = 138
        viewer.cam.elevation = -24
        viewer.cam.distance = 2.8
        viewer.cam.lookat[:] = (0.62, 0.0, 0.68)
        while viewer.is_running():
            start = time.perf_counter()
            env.scripted_step()
            viewer.sync()
            remaining = env.control_dt - (time.perf_counter() - start)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
