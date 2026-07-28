from __future__ import annotations

import json
import subprocess
import sys

import numpy as np

from audit_motion import run_trial as run_audit_trial
from benchmark_controller import MIXED_CASES, SINGLE_CASES, run_trial


BASELINE_SINGLE_MEAN_SECONDS = 12.543
BASELINE_MIXED_MEAN_SECONDS = 50.810
TARGET_MIXED_MEAN_SECONDS = 50.50
MIN_RANDOMIZED_COMPLETIONS = 19
MIN_COLLISION_FREE_TRIALS = 18


def main() -> None:
    pytest_result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        check=False,
        capture_output=True,
        text=True,
    )

    benchmark_trials = [
        *(
            run_trial("single", seed, part_count=1, max_steps=5_000)
            for seed in SINGLE_CASES
        ),
        *(
            run_trial("mixed", seed, part_count=3, max_steps=5_000)
            for seed in MIXED_CASES
        ),
    ]
    accepted = [
        trial for trial in benchmark_trials if trial.complete and trial.safe
    ]
    benchmark_pass_throughs = sum(
        trial.loaded_route_pass_throughs for trial in benchmark_trials
    )

    def mean_seconds(kind: str) -> float:
        return float(
            np.mean(
                [
                    trial.simulated_seconds
                    for trial in benchmark_trials
                    if trial.kind == kind
                ]
            )
        )

    single_mean = mean_seconds("single")
    mixed_mean = mean_seconds("mixed")
    audit_trials = [run_audit_trial(seed, 150.0) for seed in range(20)]
    object_drops = [
        trial.seed
        for trial in audit_trials
        if trial.terminal_phase == "object_dropped"
    ]
    completed_trials = sum(trial.completed for trial in audit_trials)
    collision_free_trials = sum(
        trial.unsafe_contacts == 0 for trial in audit_trials
    )
    collision_seeds = [
        trial.seed for trial in audit_trials if trial.unsafe_contacts > 0
    ]

    checks = {
        "tests_passed": pytest_result.returncode == 0,
        "benchmark_accuracy_passed": len(accepted) == len(benchmark_trials),
        "pass_through_exercised": benchmark_pass_throughs > 0,
        "single_speed_preserved": single_mean <= BASELINE_SINGLE_MEAN_SECONDS,
        "mixed_speed_improved": mixed_mean <= TARGET_MIXED_MEAN_SECONDS,
        "no_object_drops": not object_drops,
        "randomized_completions_preserved": (
            completed_trials >= MIN_RANDOMIZED_COMPLETIONS
        ),
        "collision_free_trials_preserved": (
            collision_free_trials >= MIN_COLLISION_FREE_TRIALS
        ),
    }
    summary = {
        "passed": all(checks.values()),
        "checks": checks,
        "tests": pytest_result.stdout.strip().splitlines()[-1:],
        "benchmark": {
            "accepted_trials": len(accepted),
            "trials": len(benchmark_trials),
            "single_mean_simulated_seconds": round(single_mean, 3),
            "mixed_mean_simulated_seconds": round(mixed_mean, 3),
            "loaded_route_pass_throughs": benchmark_pass_throughs,
        },
        "randomized_audit": {
            "completed_trials": completed_trials,
            "collision_free_trials": collision_free_trials,
            "collision_seeds": collision_seeds,
            "object_drop_seeds": object_drops,
            "terminal_failures": {
                str(trial.seed): trial.terminal_phase
                for trial in audit_trials
                if trial.terminal_phase is not None
            },
        },
    }
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
