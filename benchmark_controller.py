from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass

import numpy as np

from demo_runtime import sort_has_failed, sort_is_complete
from env import FactoryFloorEnv
from safety_monitor import detect_safety_failure


SINGLE_CASES = (1, 4, 7, 11, 12, 21, 23)
MIXED_CASES = (0, 1, 2, 4)
BASELINE_SINGLE_MEAN_SECONDS = 20.32
BASELINE_MIXED_MEAN_SECONDS = 63.205
MAX_ACCEPTED_SINGLE_MEAN_SECONDS = 13.0
MAX_ACCEPTED_MIXED_MEAN_SECONDS = 52.0


@dataclass
class BenchmarkTrial:
    kind: str
    seed: int
    parts: list[str]
    complete: bool
    safe: bool
    simulated_seconds: float
    wall_seconds: float
    steps: int
    ik_calls: int
    loaded_route_direct_shortcuts: int
    loaded_route_blended_shortcuts: int
    loaded_route_baseline_fallbacks: int
    loaded_route_pass_throughs: int
    loaded_route_predicted_seconds_saved: float
    failure: str | None


def run_trial(kind: str, seed: int, part_count: int, max_steps: int) -> BenchmarkTrial:
    env = FactoryFloorEnv(
        enable_rgb_observation=False,
        min_active_parts=part_count,
        max_active_parts=part_count,
    )
    env.random_state = np.random.default_rng(seed)
    env.reset()
    original_solver = env._solve_inverse_kinematics
    ik_calls = 0

    def counted_solver(
        target: np.ndarray, phase: str | None = None
    ) -> np.ndarray:
        nonlocal ik_calls
        ik_calls += 1
        return original_solver(target, phase)

    env._solve_inverse_kinematics = counted_solver
    started = time.perf_counter()
    failure = None
    try:
        for step in range(max_steps):
            env.scripted_step()
            failure = detect_safety_failure(
                env, time.perf_counter() - started
            )
            if (
                failure is not None
                or sort_is_complete(env)
                or sort_has_failed(env)
            ):
                break
        return BenchmarkTrial(
            kind=kind,
            seed=seed,
            parts=env.active_part_order.copy(),
            complete=sort_is_complete(env),
            safe=failure is None,
            simulated_seconds=round(float(env.data.time), 3),
            wall_seconds=round(time.perf_counter() - started, 3),
            steps=step + 1,
            ik_calls=ik_calls,
            loaded_route_direct_shortcuts=(
                env.loaded_route_direct_shortcuts
            ),
            loaded_route_blended_shortcuts=(
                env.loaded_route_blended_shortcuts
            ),
            loaded_route_baseline_fallbacks=(
                env.loaded_route_baseline_fallbacks
            ),
            loaded_route_pass_throughs=env.loaded_route_pass_throughs,
            loaded_route_predicted_seconds_saved=round(
                env.loaded_route_predicted_seconds_saved, 3
            ),
            failure=(
                failure.contacts[0].pair
                if failure is not None
                else env.controller_phase
                if sort_has_failed(env)
                else None
            ),
        )
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Accuracy-gated physical controller speed benchmark."
    )
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="run one single-object and one mixed regression instead of all 11",
    )
    args = parser.parse_args()
    cases = (
        [("single", 4, 1), ("mixed", 1, 3)]
        if args.quick
        else [
            *(("single", seed, 1) for seed in SINGLE_CASES),
            *(("mixed", seed, 3) for seed in MIXED_CASES),
        ]
    )
    trials = [
        run_trial(kind, seed, part_count, args.max_steps)
        for kind, seed, part_count in cases
    ]
    accepted = [trial for trial in trials if trial.complete and trial.safe]

    def mean_seconds(kind: str) -> float | None:
        values = [
            trial.simulated_seconds
            for trial in accepted
            if trial.kind == kind
        ]
        return round(float(np.mean(values)), 3) if values else None

    single_mean = mean_seconds("single")
    mixed_mean = mean_seconds("mixed")
    speed_gate_passed = bool(
        args.quick
        or (
            single_mean is not None
            and mixed_mean is not None
            and single_mean <= MAX_ACCEPTED_SINGLE_MEAN_SECONDS
            and mixed_mean <= MAX_ACCEPTED_MIXED_MEAN_SECONDS
        )
    )
    summary = {
        "passed": len(accepted) == len(trials) and speed_gate_passed,
        "accepted_trials": len(accepted),
        "trials": len(trials),
        "single_mean_simulated_seconds": single_mean,
        "mixed_mean_simulated_seconds": mixed_mean,
        "speed_gate_passed": speed_gate_passed,
        "maximum_accepted_mean_seconds": {
            "single": MAX_ACCEPTED_SINGLE_MEAN_SECONDS,
            "mixed": MAX_ACCEPTED_MIXED_MEAN_SECONDS,
        },
        "single_speedup_percent": (
            round(
                100.0
                * (BASELINE_SINGLE_MEAN_SECONDS - single_mean)
                / BASELINE_SINGLE_MEAN_SECONDS,
                2,
            )
            if not args.quick and single_mean is not None
            else None
        ),
        "mixed_speedup_percent": (
            round(
                100.0
                * (BASELINE_MIXED_MEAN_SECONDS - mixed_mean)
                / BASELINE_MIXED_MEAN_SECONDS,
                2,
            )
            if not args.quick and mixed_mean is not None
            else None
        ),
        "total_steps": sum(trial.steps for trial in trials),
        "total_ik_calls": sum(trial.ik_calls for trial in trials),
        "loaded_route_selection": {
            "direct_shortcuts": sum(
                trial.loaded_route_direct_shortcuts for trial in trials
            ),
            "blended_shortcuts": sum(
                trial.loaded_route_blended_shortcuts for trial in trials
            ),
            "baseline_fallbacks": sum(
                trial.loaded_route_baseline_fallbacks for trial in trials
            ),
            "pass_throughs": sum(
                trial.loaded_route_pass_throughs for trial in trials
            ),
            "predicted_seconds_saved": round(
                sum(
                    trial.loaded_route_predicted_seconds_saved
                    for trial in trials
                ),
                3,
            ),
        },
        "results": [asdict(trial) for trial in trials],
    }
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
