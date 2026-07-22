from __future__ import annotations

import argparse
import time

import mujoco.viewer

from camera_panel import CameraPanelProcess
from demo_runtime import sort_is_complete, sort_success_message
from env import FactoryFloorEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive sorter demonstration")
    parser.add_argument(
        "--no-camera-panel",
        action="store_true",
        help="hide the separate window showing the visual policy's RGB input",
    )
    args = parser.parse_args()

    # A 25 Hz camera is ample for the visual matcher and leaves the 50 Hz arm
    # controller enough rendering headroom for a smooth interactive demo.
    env = FactoryFloorEnv(rgb_render_interval=2)
    camera_panel = None if args.no_camera_panel else CameraPanelProcess()
    try:
        if camera_panel is not None:
            startup_error = camera_panel.start()
            if startup_error is not None:
                print(f"Camera panel disabled: {startup_error}")
                camera_panel.close()
                camera_panel = None

        print(env.describe_task())
        run_started = time.perf_counter()
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            viewer.cam.azimuth = 138
            viewer.cam.elevation = -24
            viewer.cam.distance = 2.8
            viewer.cam.lookat[:] = (0.62, 0.0, 0.68)
            while viewer.is_running():
                start = time.perf_counter()
                result = env.scripted_step()
                viewer.sync()
                if camera_panel is not None:
                    if camera_panel.user_requested_close():
                        break
                    runtime_error = camera_panel.poll_error()
                    if runtime_error is not None:
                        print(f"Camera panel disabled: {runtime_error}")
                        camera_panel.close()
                        camera_panel = None
                    else:
                        camera_panel.publish(
                            result.observation["rgb"], env.controller_phase
                        )
                if sort_is_complete(env):
                    print(
                        sort_success_message(
                            env, time.perf_counter() - run_started
                        )
                    )
                    break
                remaining = env.control_dt - (time.perf_counter() - start)
                if remaining > 0:
                    time.sleep(remaining)
    except KeyboardInterrupt:
        print("Stopping demo...")
    finally:
        if camera_panel is not None:
            camera_panel.close()
        env.close()


if __name__ == "__main__":
    main()
