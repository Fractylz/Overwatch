from functools import lru_cache
import json
from pathlib import Path
from PIL import Image

MODEL = Path(__file__).resolve().parents[1] / "artifacts/watch_detector.pt"


@lru_cache(maxsize=1)
def detector(path, modified):
    from ultralytics import YOLO
    return YOLO(str(path))


def detect_brand(image_path, conf=None):
    if conf is None:
        settings = MODEL.with_name("detector_settings.json")
        conf = json.loads(settings.read_text())["confidence"] if settings.exists() else .25
    image = (image_path if isinstance(image_path, Image.Image) else Image.open(image_path)).convert("RGB")
    model = detector(MODEL, MODEL.stat().st_mtime_ns)
    result = model.predict(image, conf=conf, imgsz=model.overrides.get("imgsz", 224), verbose=False)[0]
    if not len(result.boxes):
        return None
    i = int(result.boxes.conf.argmax())
    box = tuple(round(v) for v in result.boxes.xyxy[i].tolist())
    return dict(brand=result.names[int(result.boxes.cls[i])].replace("_", " "),
                brand_confidence=round(float(result.boxes.conf[i]), 3), box=box, crop=image.crop(box))
