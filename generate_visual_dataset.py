from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from env import FactoryFloorEnv


def _geom_mask(segmentation: np.ndarray, geom_ids: list[int]) -> np.ndarray:
    object_id = segmentation[..., 0]
    object_type = segmentation[..., 1]
    return (
        np.isin(object_id, np.asarray(geom_ids))
        & (object_type == int(mujoco.mjtObj.mjOBJ_GEOM))
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate RGB visual-matching data with privileged MuJoCo labels."
    )
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", type=Path, default=Path("datasets/visual_matching")
    )
    parser.add_argument(
        "--preview-count",
        type=int,
        default=12,
        help="number of RGB PNG previews to write in addition to compressed samples",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest_path = args.output / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise SystemExit(
            f"{manifest_path} already exists; pass --overwrite to replace samples"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    preview_dir = args.output / "previews"
    if args.preview_count > 0:
        preview_dir.mkdir(parents=True, exist_ok=True)

    env = FactoryFloorEnv()
    # Keep MuJoCo's macOS CGL initialization ahead of OpenCV/Cocoa.
    import cv2

    sample_records = []
    try:
        for sample_index in range(args.samples):
            sample_seed = args.seed + sample_index
            env.random_state = np.random.default_rng(sample_seed)
            observation, _ = env.reset()
            rgb = observation["rgb"]
            segmentation = env.render_segmentation()
            oracle = env.oracle_task_state()

            part_masks = np.stack(
                [
                    (
                        _geom_mask(segmentation, [env.part_geom_ids[name]])
                        if name in env.active_part_order
                        else np.zeros(rgb.shape[:2], dtype=bool)
                    )
                    for name in env.part_order
                ]
            )
            bin_masks = np.stack(
                [
                    _geom_mask(segmentation, env.bin_geom_ids[name])
                    for name in env.bin_order
                ]
            )
            target_bin_indices = np.array(
                [
                    (
                        env.bin_order.index(oracle["part_to_bin"][name])
                        if name in env.active_part_order
                        else -1
                    )
                    for name in env.part_order
                ],
                dtype=np.int64,
            )
            part_present = np.array(
                [name in env.active_part_order for name in env.part_order], dtype=bool
            )
            part_positions = np.full((len(env.part_order), 3), np.nan, dtype=float)
            for part_index, name in enumerate(env.part_order):
                if part_present[part_index]:
                    part_positions[part_index] = env.data.xpos[
                        env.part_body_ids[name]
                    ].copy()
            bin_positions = np.stack(
                [
                    env.data.site_xpos[env.bin_site_ids[name]].copy()
                    for name in env.bin_order
                ]
            )
            bin_colors = np.stack(
                [oracle["bin_colors_rgba"][name][:3] for name in env.bin_order]
            )

            filename = f"sample_{sample_index:06d}.npz"
            np.savez_compressed(
                args.output / filename,
                rgb=rgb,
                proprioception=observation["proprioception"],
                part_masks=part_masks,
                bin_masks=bin_masks,
                part_positions=part_positions,
                part_present=part_present,
                bin_positions=bin_positions,
                target_bin_indices=target_bin_indices,
                bin_colors=bin_colors,
            )
            if sample_index < args.preview_count:
                cv2.imwrite(
                    str(preview_dir / f"sample_{sample_index:06d}.png"),
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                )
            sample_records.append(
                {
                    "file": filename,
                    "seed": sample_seed,
                    "visible_part_pixels": part_masks.sum(axis=(1, 2)).tolist(),
                    "visible_bin_pixels": bin_masks.sum(axis=(1, 2)).tolist(),
                    "active_part_count": int(part_present.sum()),
                }
            )
    finally:
        env.close()

    manifest = {
        "format_version": 2,
        "samples": args.samples,
        "camera": {
            "name": env.camera_name,
            "width": env.camera_width,
            "height": env.camera_height,
        },
        "policy_inputs": ["rgb", "proprioception"],
        "privileged_training_labels": [
            "part_masks",
            "bin_masks",
            "part_positions",
            "part_present",
            "bin_positions",
            "target_bin_indices",
        ],
        "part_order": env.part_order,
        "bin_order": env.bin_order,
        "records": sample_records,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "samples": args.samples,
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
