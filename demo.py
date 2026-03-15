from __future__ import annotations

import time

import mujoco
import mujoco.viewer

from env import FactoryFloorEnv


def main() -> None:
    env = FactoryFloorEnv()
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20
        viewer.cam.distance = 2.4
        viewer.cam.lookat[:] = (0.3, 0.0, 0.4)
        while viewer.is_running():
            start = time.perf_counter()
            env.step(env.scripted_action(env.data.time))
            viewer.sync()
            remaining = env.model.opt.timestep - (time.perf_counter() - start)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
