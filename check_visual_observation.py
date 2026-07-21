from __future__ import annotations

import argparse
import json

import mujoco
import numpy as np

from env import FactoryFloorEnv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Headless RGB and randomized matching-task smoke test."
    )
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()

    env = FactoryFloorEnv()
    frame_signatures: list[float] = []
    color_signatures: set[tuple[float, ...]] = set()
    assignments: set[tuple[str, ...]] = set()
    failures: list[str] = []
    try:
        for seed in range(args.seeds):
            env.random_state = np.random.default_rng(seed)
            observation, _ = env.reset()
            rgb = observation.get("rgb")
            proprioception = observation.get("proprioception")
            oracle = env.oracle_task_state()

            if rgb is None or rgb.shape != (
                env.camera_height,
                env.camera_width,
                3,
            ):
                failures.append(f"seed {seed}: invalid RGB shape")
                continue
            if rgb.dtype != np.uint8 or float(rgb.std()) < 10.0:
                failures.append(f"seed {seed}: RGB frame is blank or malformed")
            if proprioception is None or proprioception.shape != (15,):
                failures.append(f"seed {seed}: invalid proprioception shape")

            segmentation = env.render_segmentation()
            geom_pixels = segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
            for part_name in env.part_order:
                visible_pixels = int(
                    (
                        (segmentation[..., 0] == env.part_geom_ids[part_name])
                        & geom_pixels
                    ).sum()
                )
                if visible_pixels < 20:
                    failures.append(
                        f"seed {seed}: {part_name} is too occluded ({visible_pixels} pixels)"
                    )

            bin_colors = oracle["bin_colors_rgba"]
            rgb_distance = float(
                np.linalg.norm(bin_colors["bin_0"][:3] - bin_colors["bin_1"][:3])
            )
            if rgb_distance < 0.25:
                failures.append(f"seed {seed}: sampled colors are ambiguous")

            target_bins = oracle["part_to_bin"]
            if set(target_bins.values()) != set(env.bin_order):
                failures.append(f"seed {seed}: a bin has no matching part")
            for part_name, bin_name in target_bins.items():
                if not np.allclose(
                    oracle["part_colors_rgba"][part_name], bin_colors[bin_name]
                ):
                    failures.append(
                        f"seed {seed}: {part_name} does not match {bin_name}"
                    )

            frame_signatures.append(float(rgb.mean()))
            color_signatures.add(
                tuple(np.round(np.concatenate(list(bin_colors.values())), 3))
            )
            assignments.add(tuple(target_bins[name] for name in env.part_order))
    finally:
        env.close()

    if len(color_signatures) < max(2, args.seeds // 2):
        failures.append("colors are not varying sufficiently across resets")
    if len(assignments) < 2 and args.seeds > 1:
        failures.append("part-to-bin assignments are not varying across resets")
    if np.ptp(frame_signatures) < 0.25:
        failures.append("rendered observations are not changing across resets")

    summary = {
        "passed": not failures,
        "seeds": args.seeds,
        "rgb_shape": [env.camera_height, env.camera_width, 3],
        "unique_color_pairs": len(color_signatures),
        "unique_assignments": len(assignments),
        "frame_mean_range": round(float(np.ptp(frame_signatures)), 4),
        "failures": failures,
    }
    print(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
