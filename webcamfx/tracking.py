"""Body + hand tracking with MediaPipe Tasks, with light jitter smoothing."""

import os
import sys
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

POSE_MODEL = "pose_landmarker_lite.task"
HAND_MODEL = "hand_landmarker.task"
SEG_MODEL = "selfie_segmenter.tflite"

# Pose landmark indices (33-point BlazePose topology).
NOSE, L_EAR, R_EAR = 0, 7, 8
L_SHOULDER, R_SHOULDER = 11, 12
ARMS = [(11, 13, 15), (12, 14, 16)]  # shoulder, elbow, wrist
POSE_HANDS = [(15, 17, 19, 21), (16, 18, 20, 22)]  # wrist, pinky, index, thumb

POSE_BODY_CONNECTIONS = [
    (11, 12),                      # shoulders
    (11, 13), (13, 15),            # left arm
    (12, 14), (14, 16),            # right arm
    (11, 23), (12, 24), (23, 24),  # torso
    (23, 25), (25, 27),            # left leg
    (24, 26), (26, 28),            # right leg
    (27, 29), (29, 31), (27, 31),  # left foot
    (28, 30), (30, 32), (28, 32),  # right foot
]
# Joints that get a glowing dot.
POSE_BODY_JOINTS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (9, 10), (10, 11), (11, 12),             # middle
    (13, 14), (14, 15), (15, 16),            # ring
    (0, 17), (17, 18), (18, 19), (19, 20),   # pinky
    (5, 9), (9, 13), (13, 17),               # palm
]
HAND_TIPS = [4, 8, 12, 16, 20]


def resource_path(*parts):
    """Locate bundled files both from source and inside a PyInstaller exe."""
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base, *parts)


class Smoother:
    """Exponential smoothing per landmark; resets when tracking is lost."""

    def __init__(self, alpha):
        self.alpha = alpha
        self.state = None

    def __call__(self, pts):
        if pts is None:
            self.state = None
            return None
        if self.state is None or self.state.shape != pts.shape:
            self.state = pts.copy()
        else:
            self.state += self.alpha * (pts - self.state)
        return self.state.copy()


class Tracker:
    def __init__(self):
        base = mp_tasks.BaseOptions
        mode = vision.RunningMode.VIDEO
        self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=base(model_asset_path=resource_path("models", POSE_MODEL)),
            running_mode=mode,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ))
        self.hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=base(model_asset_path=resource_path("models", HAND_MODEL)),
            running_mode=mode,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ))
        self.segmenter = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
            base_options=base(model_asset_path=resource_path("models", SEG_MODEL)),
            running_mode=mode,
            output_confidence_masks=True,
            output_category_mask=False,
        ))
        self.person_mask = None  # temporally smoothed float32 0..1
        self.pose_smooth = Smoother(0.6)
        self.hand_smooth = {"Left": Smoother(0.7), "Right": Smoother(0.7)}
        self.last_ts = 0

    def _timestamp(self):
        # MediaPipe VIDEO mode requires strictly increasing timestamps.
        ts = int(time.monotonic() * 1000)
        if ts <= self.last_ts:
            ts = self.last_ts + 1
        self.last_ts = ts
        return ts

    def process(self, rgb, want_pose=True, want_hands=True, want_mask=True):
        """Return {'pose': (33,3) array of x,y,visibility in pixels or None,
        'hands': list of (21,2) pixel arrays,
        'person': float32 (h,w) 0..1 person mask or None}."""
        h, w = rgb.shape[:2]
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        ts = self._timestamp()
        out = {"pose": None, "hands": [], "person": None}

        if want_mask:
            res = self.segmenter.segment_for_video(image, ts)
            m = res.confidence_masks[0].numpy_view().reshape(h, w).astype(np.float32)
            if self.person_mask is None or self.person_mask.shape != m.shape:
                self.person_mask = m
            else:
                self.person_mask = cv2.addWeighted(self.person_mask, 0.4, m, 0.6, 0)  # reduce flicker
            out["person"] = self.person_mask
        else:
            self.person_mask = None

        pose_pts = None
        if want_pose:
            res = self.pose.detect_for_video(image, ts)
            if res.pose_landmarks:
                lms = res.pose_landmarks[0]
                pose_pts = np.array(
                    [(lm.x * w, lm.y * h, lm.visibility if lm.visibility is not None else 1.0) for lm in lms],
                    dtype=np.float32,
                )
        out["pose"] = self.pose_smooth(pose_pts)

        seen = set()
        if want_hands:
            res = self.hands.detect_for_video(image, ts)
            for lms, handed in zip(res.hand_landmarks, res.handedness):
                label = handed[0].category_name if handed else None
                pts = np.array([(lm.x * w, lm.y * h) for lm in lms], dtype=np.float32)
                smoother = self.hand_smooth.get(label)
                if smoother is None or label in seen:  # unknown or duplicate label: use raw points
                    out["hands"].append(pts)
                    continue
                seen.add(label)
                out["hands"].append(smoother(pts))
        for label, smoother in self.hand_smooth.items():
            if label not in seen:
                smoother(None)
        return out

    def close(self):
        self.pose.close()
        self.hands.close()
        self.segmenter.close()
