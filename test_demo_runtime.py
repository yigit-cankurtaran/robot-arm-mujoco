from __future__ import annotations

from types import SimpleNamespace

from demo_runtime import (
    sort_failure_message,
    sort_has_failed,
    sort_is_complete,
    sort_success_message,
)


def fake_env(*, active: int, completed: int, phase: str, sim_time: float):
    return SimpleNamespace(
        active_part_order=[f"part_{index}" for index in range(active)],
        completed_parts={f"part_{index}" for index in range(completed)},
        controller_phase=phase,
        data=SimpleNamespace(time=sim_time),
    )


def test_sort_completes_only_after_every_part_and_return_home() -> None:
    assert not sort_is_complete(
        fake_env(active=2, completed=2, phase="return_home", sim_time=10.0)
    )
    assert not sort_is_complete(
        fake_env(active=2, completed=1, phase="idle", sim_time=10.0)
    )
    assert sort_is_complete(
        fake_env(active=2, completed=2, phase="idle", sim_time=10.0)
    )


def test_success_message_reports_simulated_and_wall_time() -> None:
    env = fake_env(active=3, completed=3, phase="idle", sim_time=60.0)
    assert sort_success_message(env, 62.5) == (
        "Sort complete: 3 objects placed successfully in 60.00s simulated time "
        "(62.50s wall time, 3.00 objects/min)."
    )


def test_terminal_sort_failure_phases_stop_runtime() -> None:
    for phase in ("drop_failed", "object_dropped", "route_failed"):
        assert sort_has_failed(
            fake_env(active=2, completed=0, phase=phase, sim_time=4.0)
        )
    assert not sort_has_failed(
        fake_env(active=2, completed=0, phase="grasp_failed", sim_time=4.0)
    )


def test_object_drop_message_reports_when_and_where_grasp_was_lost() -> None:
    env = fake_env(
        active=2,
        completed=0,
        phase="object_dropped",
        sim_time=7.0,
    )
    env.object_drop_event = SimpleNamespace(
        part_name="part_3",
        controller_phase="move_to_transfer",
        detected_time=6.75,
        separation=0.081,
        part_position=(0.42, -0.18, 0.61),
    )
    message = sort_failure_message(env, wall_seconds=8.5)
    assert "part_3 left the gripper" in message
    assert "move_to_transfer" in message
    assert "t=6.750s simulated" in message
    assert "separation=0.081m" in message
    assert "part position=[0.420, -0.180, 0.610]" in message
