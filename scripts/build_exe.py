"""Build a single-file WebcamFX executable with PyInstaller (dist/WebcamFX.exe on Windows)."""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

subprocess.check_call([sys.executable, os.path.join(ROOT, "scripts", "get_models.py")])

import PyInstaller.__main__  # noqa: E402

PyInstaller.__main__.run([
    "--noconfirm",
    "--onefile",
    "--windowed",
    "--name", "WebcamFX",
    "--collect-all", "mediapipe",
    "--add-data", f"{os.path.join(ROOT, 'models')}{os.pathsep}models",
    "--distpath", os.path.join(ROOT, "dist"),
    "--workpath", os.path.join(ROOT, "build"),
    "--specpath", os.path.join(ROOT, "build"),
    os.path.join(ROOT, "main.py"),
])
