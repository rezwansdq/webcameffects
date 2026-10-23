"""Background worker: camera capture -> tracking -> effects, latest frame kept for the UI."""

import sys
import threading
import time
from dataclasses import dataclass

import cv2

from .effects import GlowRenderer
from .tracking import Tracker

QUALITY_WIDTHS = {"Low": 640, "Balanced": 960, "High": 1280}


@dataclass
class Settings:
    effects_on: bool = True
    aura_hands: bool = True
    aura_arms: bool = True
    aura_head: bool = True
    trails: bool = True
    skeleton: bool = False
    mirror: bool = True
    color_bgr: tuple = (20, 255, 57)  # toxic neon green
    intensity: float = 0.6
    trail_length: float = 0.6
    thickness: float = 6.0
    darkness: float = 0.55
    quality: str = "Balanced"
    camera: int = 0


def open_camera(index):
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF] if sys.platform == "win32" else [cv2.CAP_ANY]
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ok, _ = cap.read()
            if ok:
                return cap
        cap.release()
    return None


class Engine:
    def __init__(self, settings):
        self.settings = settings
        self.lock = threading.Lock()
        self.latest = None  # BGR frame ready for display
        self.status = "Starting camera..."
        self.fps = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)

    def get_frame(self):
        with self.lock:
            frame, self.latest = self.latest, None
        return frame

    def _run(self):
        s = self.settings
        try:
            tracker = Tracker()
        except Exception as e:  # missing models, etc.
            self.status = f"Tracker failed to load: {e}"
            return
        renderer = GlowRenderer()
        cap, cam_index = None, None
        last = time.perf_counter()

        while not self._stop.is_set():
            if cap is None or cam_index != s.camera:
                if cap is not None:
                    cap.release()
                cam_index = s.camera
                self.status = f"Opening camera {cam_index}..."
                cap = open_camera(cam_index)
                renderer.reset()
                if cap is None:
                    self.status = f"Camera {cam_index} not found - pick another in the panel"
                    time.sleep(1.0)
                    continue
                self.status = ""

            ok, frame = cap.read()
            if not ok:
                self.status = "Camera stopped sending frames, reconnecting..."
                cap.release()
                cap = None
                time.sleep(0.5)
                continue

            if s.mirror:
                frame = cv2.flip(frame, 1)
            target_w = QUALITY_WIDTHS.get(s.quality, 960)
            h, w = frame.shape[:2]
            if w != target_w:
                frame = cv2.resize(frame, (target_w, int(h * target_w / w)), interpolation=cv2.INTER_AREA)

            tracks = {"pose": None, "hands": [], "person": None}
            want_aura = s.aura_hands or s.aura_arms or s.aura_head
            if s.effects_on and (want_aura or s.skeleton):
                # Track on a smaller copy; landmarks are scaled back up to frame size.
                track_w = 640
                fh, fw = frame.shape[:2]
                small = frame if fw <= track_w else cv2.resize(frame, (track_w, int(fh * track_w / fw)))
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                tracks = tracker.process(rgb, want_pose=True, want_hands=s.aura_hands or s.skeleton,
                                         want_mask=want_aura)
                k = fw / small.shape[1]
                if k != 1:
                    if tracks["pose"] is not None:
                        tracks["pose"][:, :2] *= k
                    tracks["hands"] = [hnd * k for hnd in tracks["hands"]]

            out = renderer.render(frame, tracks, s, effects_on=s.effects_on)

            now = time.perf_counter()
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(1e-6, now - last))
            last = now
            with self.lock:
                self.latest = out

        if cap is not None:
            cap.release()
        tracker.close()
