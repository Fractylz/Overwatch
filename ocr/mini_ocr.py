from functools import lru_cache
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np


@lru_cache(maxsize=1)
def reader():
    import easyocr
    return easyocr.Reader(["en"], gpu=False)


def windows_words(image):
    with TemporaryDirectory() as directory:
        path = Path(directory) / "watch.png"
        photo = image.copy()
        photo.thumbnail((2400, 2400))
        photo.save(path)
        output = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
             Path(__file__).with_name("windows_ocr.ps1").read_text(encoding="utf-8")],
            env={**os.environ, "OVERWATCH_OCR_IMAGE": str(path)},
            capture_output=True, timeout=30, check=True, encoding="utf-8")
    return json.loads(output.stdout)


def read_text(crop, brand=""):
    image = crop.convert("RGB")
    if os.name == "nt":
        try:
            return {"ocr_text": windows_words(image)}
        except (subprocess.SubprocessError, OSError, ValueError):
            pass
    words = reader().readtext(np.array(image), detail=0)
    return {"ocr_text": [s.strip() for s in words if s.strip()]}
