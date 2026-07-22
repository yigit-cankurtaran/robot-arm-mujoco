from __future__ import annotations

import queue

from camera_panel import CameraPanelProcess


class FakeEvent:
    def __init__(self) -> None:
        self.was_set = False

    def set(self) -> None:
        self.was_set = True


class FakeQueue:
    def __init__(self, *, full: bool = False) -> None:
        self.full = full
        self.cancelled = False
        self.closed = False

    def put_nowait(self, _item) -> None:
        if self.full:
            raise queue.Full

    def cancel_join_thread(self) -> None:
        self.cancelled = True

    def close(self) -> None:
        self.closed = True


class StubbornProcess:
    def __init__(self) -> None:
        self.alive = True
        self.join_timeouts: list[float] = []
        self.terminated = False
        self.killed = False

    def join(self, timeout: float) -> None:
        self.join_timeouts.append(timeout)

    def is_alive(self) -> bool:
        return self.alive

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True
        self.alive = False


def test_close_escalates_and_releases_queue_threads() -> None:
    panel = CameraPanelProcess.__new__(CameraPanelProcess)
    panel.started = True
    panel.close_event = FakeEvent()
    panel.frame_queue = FakeQueue(full=True)
    panel.error_queue = FakeQueue()
    panel.process = StubbornProcess()

    panel.close()

    assert panel.close_event.was_set
    assert panel.process.terminated
    assert panel.process.killed
    assert panel.process.join_timeouts == [2.0, 1.0, 1.0]
    assert panel.frame_queue.cancelled and panel.frame_queue.closed
    assert panel.error_queue.cancelled and panel.error_queue.closed
    assert not panel.started

    # A second cleanup call must be a harmless no-op.
    panel.close()
    assert panel.process.join_timeouts == [2.0, 1.0, 1.0]
