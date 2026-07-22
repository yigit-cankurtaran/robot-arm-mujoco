from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

from env import FactoryFloorEnv
from visual_policy import RGBVisualPolicy, VisualTaskEstimate


def geom_mask(segmentation: np.ndarray, geom_ids: list[int]) -> np.ndarray:
    return np.isin(segmentation[..., 0], geom_ids) & (
        segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
    )


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = np.logical_and(first, second).sum()
    union = np.logical_or(first, second).sum()
    return float(intersection / max(union, 1))


def greedy_associate(
    predicted: list[np.ndarray], truth: list[np.ndarray]
) -> dict[int, int]:
    candidates = sorted(
        (
            (mask_iou(prediction, target), prediction_index, target_index)
            for prediction_index, prediction in enumerate(predicted)
            for target_index, target in enumerate(truth)
        ),
        reverse=True,
    )
    mapping: dict[int, int] = {}
    used_truth: set[int] = set()
    for score, prediction_index, target_index in candidates:
        if score <= 0 or prediction_index in mapping or target_index in used_truth:
            continue
        mapping[prediction_index] = target_index
        used_truth.add(target_index)
    return mapping


def draw_estimate(rgb: np.ndarray, estimate: VisualTaskEstimate) -> np.ndarray:
    canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for index, instance in enumerate(estimate.bins):
        contours, _ = cv2.findContours(
            instance.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(canvas, contours, -1, (255, 150, 40), 2)
        point = tuple(int(value) for value in instance.centroid_xy)
        cv2.putText(canvas, f"B{index}", point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 150, 40), 2)
    for index, instance in enumerate(estimate.parts):
        contours, _ = cv2.findContours(
            instance.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(canvas, contours, -1, (50, 230, 80), 2)
        point = tuple(int(value) for value in instance.centroid_xy)
        cv2.putText(canvas, f"P{index}", point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 230, 80), 2)
    for pick in estimate.picks:
        start = tuple(int(value) for value in pick.part.centroid_xy)
        end = tuple(int(value) for value in pick.target_bin.centroid_xy)
        cv2.arrowedLine(canvas, start, end, (30, 30, 240), 1, tipLength=0.08)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate held-out RGB visual policy.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=200_000)
    parser.add_argument("--output", type=Path, default=Path("runs/visual_evaluation"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-images", type=int, default=24)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    policy = RGBVisualPolicy(args.checkpoint, device=args.device)
    env = FactoryFloorEnv()
    rows = []
    saved = 0
    try:
        for offset in range(args.seeds):
            seed = args.seed_start + offset
            env.random_state = np.random.default_rng(seed)
            observation, _ = env.reset()
            rgb = observation["rgb"]
            estimate = policy.predict(rgb)
            segmentation = env.render_segmentation()
            oracle = env.oracle_task_state()
            truth_parts = [
                geom_mask(segmentation, [env.part_geom_ids[name]])
                for name in env.active_part_order
            ]
            truth_bins = [
                geom_mask(segmentation, env.bin_geom_ids[name])
                for name in env.bin_order
            ]
            part_map = greedy_associate([item.mask for item in estimate.parts], truth_parts)
            bin_map = greedy_associate([item.mask for item in estimate.bins], truth_bins)
            correct_matches = 0
            evaluated_matches = 0
            position_errors = []
            for predicted_index, truth_index in part_map.items():
                instance = estimate.parts[predicted_index]
                truth_name = env.active_part_order[truth_index]
                truth_position = env.data.xpos[env.part_body_ids[truth_name]]
                position_errors.append(
                    float(np.linalg.norm(instance.world_position[:2] - truth_position[:2]))
                )
                matching_pick = next(
                    (pick for pick in estimate.picks if pick.part is instance), None
                )
                if matching_pick is None:
                    continue
                predicted_bin_index = next(
                    index
                    for index, item in enumerate(estimate.bins)
                    if item is matching_pick.target_bin
                )
                if predicted_bin_index not in bin_map:
                    continue
                evaluated_matches += 1
                truth_bin_name = env.bin_order[bin_map[predicted_bin_index]]
                correct_matches += int(
                    oracle["part_to_bin"][truth_name] == truth_bin_name
                )
            row = {
                "seed": seed,
                "truth_parts": len(truth_parts),
                "predicted_parts": len(estimate.parts),
                "predicted_bins": len(estimate.bins),
                "part_count_correct": len(estimate.parts) == len(truth_parts),
                "bin_count_correct": len(estimate.bins) == 2,
                "associated_parts": len(part_map),
                "evaluated_matches": evaluated_matches,
                "correct_matches": correct_matches,
                "all_matches_correct": (
                    evaluated_matches == len(truth_parts)
                    and correct_matches == len(truth_parts)
                ),
                "mean_part_xy_error_m": (
                    float(np.mean(position_errors)) if position_errors else None
                ),
            }
            rows.append(row)
            should_save = saved < args.save_images and (
                not row["all_matches_correct"] or saved < min(8, args.save_images)
            )
            if should_save:
                cv2.imwrite(
                    str(args.output / f"seed_{seed}_{'ok' if row['all_matches_correct'] else 'fail'}.png"),
                    draw_estimate(rgb, estimate),
                )
                saved += 1
    finally:
        env.close()

    total_truth_parts = sum(row["truth_parts"] for row in rows)
    total_evaluated = sum(row["evaluated_matches"] for row in rows)
    errors = [
        row["mean_part_xy_error_m"]
        for row in rows
        if row["mean_part_xy_error_m"] is not None
    ]
    summary = {
        "checkpoint": str(args.checkpoint),
        "scenes": len(rows),
        "part_count_accuracy": float(np.mean([row["part_count_correct"] for row in rows])),
        "bin_count_accuracy": float(np.mean([row["bin_count_correct"] for row in rows])),
        "part_recall": sum(row["associated_parts"] for row in rows) / max(total_truth_parts, 1),
        "per_part_match_accuracy": sum(row["correct_matches"] for row in rows) / max(total_evaluated, 1),
        "full_scene_match_accuracy": float(np.mean([row["all_matches_correct"] for row in rows])),
        "mean_part_xy_error_m": float(np.mean(errors)) if errors else None,
        "p95_part_xy_error_m": float(np.percentile(errors, 95)) if errors else None,
    }
    (args.output / "records.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
