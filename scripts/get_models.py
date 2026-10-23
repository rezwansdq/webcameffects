"""Download the MediaPipe model files into ./models (run once before launching or building)."""

import os
import urllib.request

MODELS = {
    "pose_landmarker_lite.task":
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "hand_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "selfie_segmenter.tflite":
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
}

root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
os.makedirs(root, exist_ok=True)
for name, url in MODELS.items():
    path = os.path.join(root, name)
    if os.path.exists(path):
        print(f"have {name}")
        continue
    print(f"downloading {name}...")
    urllib.request.urlretrieve(url, path)
print("models ready in", root)
