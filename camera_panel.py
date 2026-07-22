from __future__ import annotations

import multiprocessing as mp
import queue
import signal
from typing import Any

import numpy as np


CAMERA_WINDOW = "Visual policy observation"


def build_camera_panel(
    rgb: np.ndarray, phase: str, scale: int = 2
) -> np.ndarray:
    import cv2

    height, width = rgb.shape[:2]
    resized = cv2.resize(
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        (width * scale, height * scale),
        interpolation=cv2.INTER_NEAREST,
    )
    header = np.full((34, resized.shape[1], 3), 24, dtype=np.uint8)
    cv2.putText(
        header,
        f"POLICY RGB  |  {phase}",
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    return np.vstack([header, resized])


def _camera_worker(
    frame_queue: Any,
    ready_event: Any,
    close_event: Any,
    error_queue: Any,
    scale: int,
) -> None:
    # Ctrl+C belongs to the parent process.  If the child handles it too, it can
    # be interrupted halfway through Queue.get()/Cocoa cleanup while the parent
    # is trying to join it, leaving multiprocessing resources alive.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # This import and all Cocoa calls intentionally live in the child process.
    import cv2

    try:
        cv2.namedWindow(CAMERA_WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow(CAMERA_WINDOW, 20, 20)
        ready_event.set()

        while not close_event.is_set():
            latest = None
            try:
                latest = frame_queue.get(timeout=0.05)
                while True:
                    newer = frame_queue.get_nowait()
                    if newer is None:
                        latest = None
                        close_event.set()
                        break
                    latest = newer
            except queue.Empty:
                pass

            if latest is not None:
                rgb, phase = latest
                cv2.imshow(CAMERA_WINDOW, build_camera_panel(rgb, phase, scale))
            key = cv2.waitKey(1) & 0xFF
            if key in {ord("q"), 27}:
                close_event.set()
    except Exception as exc:
        error_queue.put(f"{type(exc).__name__}: {exc}")
        ready_event.set()
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


class CameraPanelProcess:
    def __init__(self, scale: int = 2):
        context = mp.get_context("spawn")
        self.frame_queue = context.Queue(maxsize=2)
        self.error_queue = context.Queue(maxsize=1)
        self.ready_event = context.Event()
        self.close_event = context.Event()
        self.process = context.Process(
            target=_camera_worker,
            args=(
                self.frame_queue,
                self.ready_event,
                self.close_event,
                self.error_queue,
                scale,
            ),
            name="visual-policy-camera",
            daemon=True,
        )
        self.started = False

    def start(self, timeout: float = 5.0) -> str | None:
        # A spawned interpreter imports the main module before entering
        # _camera_worker().  Inherit SIG_IGN across that bootstrap too, then
        # immediately restore the parent's normal Ctrl+C handling.
        previous_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            self.process.start()
        finally:
            signal.signal(signal.SIGINT, previous_sigint)
        self.started = True
        if not self.ready_event.wait(timeout):
            return "camera process did not initialize within five seconds"
        return self.poll_error()

    def publish(self, rgb: np.ndarray, phase: str) -> None:
        if not self.started or self.close_event.is_set():
            return
        frame = (rgb.copy(), phase)
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                pass

    def poll_error(self) -> str | None:
        try:
            return self.error_queue.get_nowait()
        except queue.Empty:
            return None

    def user_requested_close(self) -> bool:
        return self.started and self.close_event.is_set() and self.poll_error() is None

    def close(self) -> None:
        if not self.started:
            return
        # Make cleanup idempotent before any operation that could itself fail.
        self.started = False
        self.close_event.set()
        try:
            self.frame_queue.put_nowait(None)
        except queue.Full:
            pass

        try:
            self.process.join(timeout=2.0)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=1.0)
            if self.process.is_alive():
                # A Cocoa event loop can occasionally ignore SIGTERM while it
                # is inside native code.  Do not let that child outlive the
                # demo; SIGKILL is the final, bounded fallback.
                self.process.kill()
                self.process.join(timeout=1.0)
        finally:
            # multiprocessing Queue owns a feeder thread.  Waiting for that
            # thread at interpreter shutdown can keep Python alive after both
            # windows have gone away, especially when frames remain buffered.
            for transport_queue in (self.frame_queue, self.error_queue):
                transport_queue.cancel_join_thread()
                transport_queue.close()
