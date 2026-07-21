from __future__ import annotations

import argparse
import time

from camera_panel import CameraPanelProcess
from env import FactoryFloorEnv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Open the isolated camera panel for a short smoke test."
    )
    parser.add_argument("--seconds", type=float, default=3.0)
    args = parser.parse_args()

    env = FactoryFloorEnv(rgb_render_interval=2)
    panel = CameraPanelProcess()
    try:
        startup_error = panel.start()
        if startup_error is not None:
            raise RuntimeError(startup_error)
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline and not panel.user_requested_close():
            result = env.scripted_step()
            runtime_error = panel.poll_error()
            if runtime_error is not None:
                raise RuntimeError(runtime_error)
            panel.publish(result.observation["rgb"], env.controller_phase)
            time.sleep(env.control_dt)
    finally:
        panel.close()
        env.close()
    print("Camera panel smoke test passed.")


if __name__ == "__main__":
    main()
