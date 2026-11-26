"""Inspect a photo, match dial/listing text, and estimate using available metadata."""
import argparse
import json
from PIL import Image, ImageOps
from listing import catalogue, identify, estimate
from watch_classifier.yolov8_detector import MODEL, detect_brand
from ocr.mini_ocr import read_text


def inspect_photo(image):
    image = ImageOps.exif_transpose(image).convert("RGB")
    out = dict(brand="", box=None, brand_confidence=None, ocr_text=[], messages=[])
    crop = image
    if MODEL.exists():
        detection = detect_brand(image)
        if detection:
            crop = detection.pop("crop")
            out.update(detection)
        else:
            out["messages"].append("No detection. OCR used the full image.")
    else:
        out["messages"].append("Detector weights unavailable. OCR used the full image.")
    try:
        out.update(read_text(crop))
    except (ImportError, RuntimeError, OSError) as exc:
        out["messages"].append(f"OCR unavailable ({type(exc).__name__}). Enter the visible text below.")
    return out


def run(image_path, listing_text="", **attributes):
    with Image.open(image_path) as image:
        detection = inspect_photo(image)
    combined = " ".join([*detection["ocr_text"], listing_text])
    candidate = identify(combined, catalogue(), detection["brand"])
    return dict(image=str(image_path), **detection, candidate=candidate,
                estimated_price=estimate({**attributes, **candidate}) if candidate else None)


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("image")
    cli.add_argument("--text", default="")
    cli.add_argument("--year", type=int)
    cli.add_argument("--size-mm", type=float)
    cli.add_argument("--condition")
    cli.add_argument("--material")
    cli.add_argument("--box-papers")
    args = vars(cli.parse_args())
    print(json.dumps(run(args.pop("image"), args.pop("text"), **args), indent=2))
