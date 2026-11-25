"""
e2e_prediction.py - thin orchestrator: detector + ocr + price model.

    image -> detect_brand() -> read_text() -> features -> price -> JSON

Run from the PROJECT ROOT so the detector/ and ocr/ packages import:
    python e2e_prediction.py                 # random image from data/samples/
    python e2e_prediction.py path\\to\\pic.jpg
"""

import sys
import json
import random
from pathlib import Path

from watch_classifier.yolov8_detector import detect_brand
from ocr.mini_ocr import read_text

PROJECT_DIR = Path(__file__).resolve().parent
SAMPLES_DIR = PROJECT_DIR / "data" / "images"
PRICE_MODEL_PATH = PROJECT_DIR / "artifacts" / "price_model.joblib"


def pick_random_image():
    imgs = [
        p
        for p in SAMPLES_DIR.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ]
    if not imgs:
        raise SystemExit(
            f"No images in {SAMPLES_DIR} - drop some watch photos there first."
        )
    return random.choice(imgs)


def build_features(detection, ocr):
    """Brand = YOLO, model = OCR guess; the rest can't be read from a photo, so
    they're placeholders until a real app gets them from the listing / user."""
    return {
        "brand": detection["brand"],
        "model_family": ocr["candidate_model"] or "Unknown",
        "year": 2020,  # TODO: from listing / user
        "condition": "Good",  # TODO
        "material": "Steel",  # TODO
        "size_mm": 40,  # TODO
        "box_papers": "Unknown",  # TODO
    }


def predict_price(features):
    if not PRICE_MODEL_PATH.exists():
        return None
    import joblib
    import pandas as pd

    model = joblib.load(PRICE_MODEL_PATH)
    return float(model.predict(pd.DataFrame([features]))[0])


def run(image_path):
    detection = detect_brand(image_path)
    if detection is None:
        return {"image": str(image_path), "error": "no watch detected"}
    ocr = read_text(detection["crop"], detection["brand"])
    features = build_features(detection, ocr)
    price = predict_price(features)
    return {
        "image": str(image_path),
        "brand": detection["brand"],
        "brand_confidence": detection["brand_confidence"],
        "box": detection["box"],
        "ocr_text": ocr["ocr_text"],
        "candidate_model": ocr["candidate_model"],
        "price_model_input": features,
        "estimated_price": (
            round(price, 2) if price is not None else "price model not trained yet"
        ),
    }


if __name__ == "__main__":
    image = sys.argv[1] if len(sys.argv) > 1 else pick_random_image()
    print(json.dumps(run(image), indent=2))
