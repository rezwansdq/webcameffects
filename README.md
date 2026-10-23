# WebcamFX

Live neon glow effects on your webcam, the Yeat-music-video look. Launch it and it plugs straight into your webcam. It tracks your hands, arms and head and draws a glowing outline **around** them, with motion trails that smear behind you as you move.

## Get the .exe (Windows)

1. Download the latest build: **[WebcamFX.exe](https://github.com/rezwansdq/webcameffects/releases/latest/download/WebcamFX.exe)**
   (or go to the repo's **Releases** page). A new build is published automatically every time `main` changes.
2. Double-click `WebcamFX.exe`.
   - The first launch takes ~10 s while it unpacks.
   - Windows SmartScreen may warn about an unknown app (it isn't code-signed). Click **More info → Run anyway**.
   - If the video stays black, check **Settings → Privacy & security → Camera → Let desktop apps access your camera**, and close any other app that's using the webcam.

## Controls

| Control | What it does |
| --- | --- |
| **EFFECTS: ON/OFF** (or `Space`) | Turns the effects on and off. The camera stays live. |
| Camera | Picks which webcam to use (0 is usually the built-in one). |
| Quality | Low / Balanced / High. Use Low if it feels laggy. |
| Glow around hands / arms / head | Picks which parts get the glowing outline. |
| Motion trails | Afterimages that fade out behind you. |
| Skeleton lines on body | Optional extra: neon lines drawn along your bones and fingers. |
| Mirror view | Selfie-style mirrored image. |
| Color swatches / `+` | Preset colors (toxic green by default) or pick any color. |
| Glow, Trail length, Glow width, Background darkness | Tuning sliders. |
| Fullscreen (or `F11`, `Esc` to exit) | Hides the panel and fills the screen. Good for screen recording. |

To record, use any screen recorder (Xbox Game Bar `Win+Alt+R`, OBS, etc.).

## Run from source

Requires Python 3.12 (64-bit).

```bash
pip install -r requirements.txt
python scripts/get_models.py   # downloads the 3 small MediaPipe models into ./models
python main.py
```

Build the exe yourself on Windows: `pip install pyinstaller` then `python scripts/build_exe.py`. The exe ends up at `dist/WebcamFX.exe`.

## How it works

- **Tracking:** Google MediaPipe models, all running on the CPU: pose (shoulders/elbows/wrists/head), hand landmarks (21 points per hand) and selfie segmentation (the exact outline of your body).
- **Aura:** the landmarks pick out the hands, arms and head from the body outline. A glowing band is drawn just outside their edges. Hands get outlined even when they're in front of your body.
- **Glow:** the outline goes into a fading trail buffer, gets bloomed (blurred at two sizes), colorized, and added on top of a darkened, desaturated camera image.

Code: `webcamfx/tracking.py` (models), `webcamfx/effects.py` (rendering), `webcamfx/engine.py` (camera thread), `webcamfx/app.py` (window + panel).
