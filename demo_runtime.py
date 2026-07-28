from __future__ import annotations

from typing import Protocol


TERMINAL_SORT_FAILURE_PHASES = frozenset(
    {"drop_failed", "object_dropped", "route_failed"}
)


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


def sort_has_failed(env: SortEnvironment) -> bool:
    """Return true for stable task failures that require loop shutdown."""
    return env.controller_phase in TERMINAL_SORT_FAILURE_PHASES


def sort_failure_message(env: SortEnvironment, wall_seconds: float) -> str:
    """Create a concise diagnostic for a terminal non-contact task failure."""
    phase = env.controller_phase
    if phase == "object_dropped":
        event = getattr(env, "object_drop_event", None)
        if event is None:
            return f"Sort failed: held object was dropped ({wall_seconds:.2f}s wall)."
        position = ", ".join(f"{value:.3f}" for value in event.part_position)
        return (
            f"Sort failed: {event.part_name} left the gripper during "
            f"{event.controller_phase} at t={event.detected_time:.3f}s simulated "
            f"({wall_seconds:.2f}s wall, separation={event.separation:.3f}m, "
            f"part position=[{position}])."
        )
    if phase == "drop_failed":
        part_name = getattr(env, "released_part", None) or "released object"
        return (
            f"Sort failed: {part_name} did not settle inside its target bin "
            f"by t={float(env.data.time):.3f}s simulated "
            f"({wall_seconds:.2f}s wall)."
        )
    if phase == "route_failed":
        return (
            "Sort failed: no collision-free, reachable pick approach was "
            "found, including from exact home."
        )
    raise ValueError(f"controller phase {phase!r} is not a terminal failure")


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
