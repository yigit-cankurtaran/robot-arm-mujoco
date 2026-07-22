from __future__ import annotations

from types import SimpleNamespace

from demo_runtime import sort_is_complete, sort_success_message


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
