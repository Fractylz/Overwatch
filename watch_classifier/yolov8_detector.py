"""
detector/detect.py - YOLO watch detection, nothing else.

Exposes detect_brand(image_path) -> dict | None
The model loads once when this module is imported.
"""

from pathlib import Path

from PIL import Image
from ultralytics import YOLO

# artifacts/ lives one level up from this detector/ folder
_ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"
_MODEL_PATH = _ARTIFACTS / "watch_detector.pt"
_CONF = 0.25

_model = YOLO(str(_MODEL_PATH))


def detect_brand(image_path, conf=_CONF):
    """Returns {brand, brand_confidence, box, crop} for the top detection, or None."""
    res = _model.predict(str(image_path), conf=conf, verbose=False)[0]
    if len(res.boxes) == 0:
        return None
    i = int(res.boxes.conf.argmax())
    box = [int(v) for v in res.boxes.xyxy[i].tolist()]
    brand = res.names[int(res.boxes.cls[i])]
    score = float(res.boxes.conf[i])
    crop = Image.open(image_path).convert("RGB").crop(tuple(box))
    return {
        "brand": brand,
        "brand_confidence": round(score, 3),
        "box": box,
        "crop": crop,
    }
