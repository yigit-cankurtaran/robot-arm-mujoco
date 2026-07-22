from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from env import FactoryFloorEnv
from visual_policy import RGBVisualPolicy
from visual_sort_demo import validated_commands


def main() -> None:
    parser = argparse.ArgumentParser(description="Headless end-to-end visual sort check")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/visual_physical_release_separated_policy/best.pt"),
    )
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=300_000)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    policy = RGBVisualPolicy(args.checkpoint, device=args.device)
    env = FactoryFloorEnv(rgb_render_interval=2)
    records = []
    try:
        for offset in range(args.seeds):
            seed = args.seed_start + offset
            env.random_state = np.random.default_rng(seed)
            observation, _ = env.reset()
            oracle = env.oracle_task_state()
            try:
                estimate, commands = validated_commands(policy, observation["rgb"])
                initial_detected_parts = len(estimate.parts)
                adapter = env.set_visual_targets(commands)
                # The plan is latched for one motion cycle; avoid paying for
                # unused camera frames during this headless dynamics check.
                env.rgb_render_interval = args.max_steps + 1
                env.rgb_frame_counter = 1
            except (RuntimeError, ValueError) as exc:
                records.append({"seed": seed, "accepted": False, "error": str(exc)})
                continue
            steps = 0
            replans = 0
            while steps < args.max_steps:
                env.scripted_step()
                steps += 1
                if (
                    env.controller_phase == "idle"
                    and len(env.completed_parts) == len(env.active_part_order)
                ):
                    break
                if env.controller_phase == "idle" and replans < 3:
                    replan_rgb = env.render_rgb()
                    try:
                        estimate, commands = validated_commands(policy, replan_rgb)
                        env.set_visual_targets(commands)
                        env.rgb_frame_counter = 1
                        replans += 1
                    except (RuntimeError, ValueError):
                        break
            correct_destinations = 0
            for name in env.active_part_order:
                position = env.data.xpos[env.part_body_ids[name]]
                nearest_bin = min(
                    env.bin_order,
                    key=lambda bin_name: float(
                        np.linalg.norm(
                            position[:2]
                            - env.data.site_xpos[env.bin_site_ids[bin_name]][:2]
                        )
                    ),
                )
                correct_destinations += int(
                    nearest_bin == oracle["part_to_bin"][name]
                )
            records.append(
                {
                    "seed": seed,
                    "accepted": True,
                    "truth_parts": len(env.active_part_order),
                    "detected_parts": initial_detected_parts,
                    "steps": steps,
                    "completed": len(env.completed_parts),
                    "replans": replans,
                    "correct_destinations": correct_destinations,
                    "adapter": adapter,
                }
            )
    finally:
        env.close()

    accepted = [record for record in records if record["accepted"]]
    total_parts = sum(record["truth_parts"] for record in accepted)
    summary = {
        "checkpoint": str(args.checkpoint),
        "seeds": args.seeds,
        "plan_acceptance_rate": len(accepted) / max(len(records), 1),
        "completion_rate": float(
            np.mean(
                [
                    record["completed"] == record["truth_parts"]
                    for record in accepted
                ]
            )
        )
        if accepted
        else 0.0,
        "sorted_part_accuracy": sum(
            record["correct_destinations"] for record in accepted
        )
        / max(total_parts, 1),
        "records": records,
    }
    print(json.dumps(summary, indent=2))
    if (
        summary["plan_acceptance_rate"] < 0.9
        or summary["completion_rate"] < 0.9
        or summary["sorted_part_accuracy"] < 0.9
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
