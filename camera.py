import sys
import subprocess

import cv2


def _macos_desktop_bounds() -> tuple[int, int] | None:
    try:
        result = subprocess.run(
            [
                "osascript",
                "-e",
                'tell application "Finder" to get bounds of window of desktop',
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=2.0,
        )
    except Exception:
        return None

    try:
        left, top, right, bottom = [
            int(part.strip()) for part in result.stdout.strip().split(",")
        ]
    except Exception:
        return None

    width = max(right - left, 1)
    height = max(bottom - top, 1)
    return width, height


def get_screen_size() -> tuple[int, int]:
    if sys.platform == "darwin":
        screen_size = _macos_desktop_bounds()
        if screen_size is not None:
            return screen_size

    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
        root.destroy()
        return width, height
    except Exception:
        return 1920, 1080


def open_camera() -> cv2.VideoCapture:
    backends = [cv2.CAP_ANY]
    if sys.platform == "darwin":
        backends.append(cv2.CAP_AVFOUNDATION)

    for backend in backends:
        cap = cv2.VideoCapture(0, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            return cap
        cap.release()

    raise RuntimeError("Could not open webcam")
