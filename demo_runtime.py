from __future__ import annotations

from typing import Protocol


class SortEnvironment(Protocol):
    active_part_order: list[str]
    completed_parts: set[str]
    controller_phase: str

    class Data(Protocol):
        time: float

    data: Data


def sort_is_complete(env: SortEnvironment) -> bool:
    """Return true only after every active part is sorted and the arm is home."""
    return (
        bool(env.active_part_order)
        and env.controller_phase == "idle"
        and len(env.completed_parts) == len(env.active_part_order)
    )


def sort_success_message(env: SortEnvironment, wall_seconds: float) -> str:
    count = len(env.completed_parts)
    simulated_seconds = float(env.data.time)
    objects_per_minute = (
        60.0 * count / simulated_seconds if simulated_seconds > 0.0 else 0.0
    )
    noun = "object" if count == 1 else "objects"
    return (
        f"Sort complete: {count} {noun} placed successfully in "
        f"{simulated_seconds:.2f}s simulated time "
        f"({wall_seconds:.2f}s wall time, {objects_per_minute:.2f} objects/min)."
    )
