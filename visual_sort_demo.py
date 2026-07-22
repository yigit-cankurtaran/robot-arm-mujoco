from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco.viewer

from camera_panel import CameraPanelProcess
from env import FactoryFloorEnv
from visual_policy import RGBVisualPolicy, draw_task_estimate


def validated_commands(policy: RGBVisualPolicy, rgb):
    estimate = policy.predict(rgb)
    if len(estimate.bins) != 2:
        raise RuntimeError(f"expected two bins, detected {len(estimate.bins)}")
    if not 1 <= len(estimate.parts) <= 3:
        raise RuntimeError(f"expected one-to-three parts, detected {len(estimate.parts)}")
    if len(estimate.picks) != len(estimate.parts):
        raise RuntimeError("not every detected part received a bin match")
    for instance in [*estimate.parts, *estimate.bins]:
        if instance.confidence < 0.70:
            raise RuntimeError(
                f"low-confidence {instance.kind} detection ({instance.confidence:.3f})"
            )
    for pick in estimate.picks:
        if pick.color_margin < 0.010:
            raise RuntimeError(
                f"ambiguous relational color match (margin {pick.color_margin:.3f})"
            )
    commands = [
        {
            "pick_position": pick.part.world_position,
            "bin_position": pick.target_bin.world_position,
        }
        for pick in estimate.picks
    ]
    return estimate, commands


def main() -> None:
    parser = argparse.ArgumentParser(description="RGB-only visual sorting demo")
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("runs/visual_policy/best.pt")
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-camera-panel", action="store_true")
    args = parser.parse_args()

    env = FactoryFloorEnv(rgb_render_interval=2)
    policy = RGBVisualPolicy(args.checkpoint, device=args.device)
    initial_rgb = env.render_rgb()
    estimate, commands = validated_commands(policy, initial_rgb)
    diagnostics = env.set_visual_targets(commands)
    print({"visual_policy": diagnostics})
    overlay = draw_task_estimate(initial_rgb, estimate)

    camera_panel = None if args.no_camera_panel else CameraPanelProcess()
    if camera_panel is not None:
        startup_error = camera_panel.start()
        if startup_error is not None:
            print(f"Camera panel disabled: {startup_error}")
            camera_panel.close()
            camera_panel = None
    try:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            viewer.cam.azimuth = 138
            viewer.cam.elevation = -24
            viewer.cam.distance = 2.8
            viewer.cam.lookat[:] = (0.62, 0.0, 0.68)
            while viewer.is_running():
                started = time.perf_counter()
                result = env.scripted_step()
                rendered_this_tick = (
                    (env.rgb_frame_counter - 1) % env.rgb_render_interval == 0
                )
                if rendered_this_tick:
                    live_estimate = policy.predict(result.observation["rgb"])
                    overlay = draw_task_estimate(
                        result.observation["rgb"], live_estimate
                    )
                viewer.sync()
                if camera_panel is not None:
                    if camera_panel.user_requested_close():
                        break
                    error = camera_panel.poll_error()
                    if error is not None:
                        print(f"Camera panel disabled: {error}")
                        camera_panel.close()
                        camera_panel = None
                    else:
                        camera_panel.publish(overlay, env.controller_phase)
                remaining = env.control_dt - (time.perf_counter() - started)
                if remaining > 0:
                    time.sleep(remaining)
    finally:
        if camera_panel is not None:
            camera_panel.close()
        env.close()


if __name__ == "__main__":
    main()
