"""Load the trackers and render one frame without a camera or window (used by CI)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from webcamfx.effects import GlowRenderer  # noqa: E402
from webcamfx.engine import Settings  # noqa: E402
from webcamfx.tracking import Tracker  # noqa: E402

frame = np.full((360, 640, 3), 40, np.uint8)
tracker = Tracker()
tracks = tracker.process(frame[:, :, ::-1].copy())
out = GlowRenderer().render(frame, tracks, Settings())
assert out.shape == frame.shape and out.dtype == np.uint8
tracker.close()
print("smoke test ok")
