from pathlib import Path
import json

from flask import Flask, abort, jsonify, render_template, request, send_from_directory
from PIL import Image, ImageOps

from listing import ROOT, bundle, catalogue, estimate, history, identify, save_listing
from e2e_prediction import inspect_photo
from family_search import suggest

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024


@app.get("/")
def studio():
    return render_template("studio.html", materials=sorted({
        row["material"] for row in bundle()["catalogue"] if isinstance(row["material"], str)
    }))


@app.post("/api/scan")
def scan():
    photo = request.files.get("photo")
    if not photo:
        return jsonify(error="Choose a watch photo first."), 400
    try:
        with Image.open(photo.stream) as original:
            if original.width * original.height > 25_000_000:
                raise ValueError
            image = ImageOps.exif_transpose(original).convert("RGB")
    except (OSError, ValueError, Image.DecompressionBombError):
        return jsonify(error="Choose a readable photo under 25 megapixels."), 400
    result = inspect_photo(image)
    dial = " ".join(result["ocr_text"])
    return jsonify(**result, suggestions=suggest(image, dial),
                   match=identify(dial, catalogue(), result["brand"]),
                   image_width=image.width, image_height=image.height)


@app.get("/gallery/<name>")
def gallery(name):
    return send_from_directory(ROOT / "data/family_photos/gallery", name)


@app.get("/api/catalogue")
def selectors():
    rows = catalogue()
    available = {}
    for row in rows:
        available.setdefault(row["brand"], set()).add(row["model_family"])
    return jsonify({brand: sorted(families) for brand, families in available.items()})


@app.post("/api/match")
def match():
    payload = request.get_json()
    return jsonify(identify(payload.get("text", ""), catalogue(), payload.get("brand", "")))


def listing_payload():
    raw = request.get_json()
    info = {key: str(raw.get(key) or "").strip() for key in (
        "brand", "model_family", "reference", "condition", "material", "box_papers"
    )}
    if not info["brand"] or not info["model_family"]:
        abort(400)
    for name, lower, upper, convert in (("year", 1750, 2100, int), ("size_mm", 5, 100, float)):
        try:
            value = convert(raw[name]) if raw.get(name) else None
            if value is not None and not lower <= value <= upper:
                raise ValueError
        except (ValueError, TypeError):
            abort(400)
        info[name] = value
    return info


@app.post("/api/price")
def pricing():
    info = listing_payload()
    family = info["model_family"]
    rows = [r for r in bundle()["catalogue"] if r["brand"] == info["brand"]]
    exact = [r for r in rows if r["model_family"] == family]
    related = [r for r in rows if r["model_family"].casefold().startswith(family.casefold() + " ")]
    comparables = [
        {"family": r["model_family"], "reference": r["reference"], "price": r["price"],
         "currency": r["currency"], "year": int(r["year"]) if r["year"] == r["year"] else None}
        for r in sorted(exact or related, key=lambda row: row["price"])
    ]
    return jsonify(estimate=estimate(info), comparables=comparables,
                   related=sorted({r["model_family"] for r in related}))


@app.route("/api/drafts", methods=["GET", "POST"])
def drafts():
    if request.method == "GET":
        return jsonify(history())
    info = listing_payload()
    photo_name = str(request.get_json().get("photo_name") or "")[:255]
    return jsonify(id=save_listing(dict(**info, estimated_price=estimate(info),
                                      photo_name=photo_name or None)))


@app.get("/api/evaluation")
def evaluation():
    files = {"detector": "test_metrics.json", "clusters": "visual_groups.json"}
    reports = {}
    for key, name in files.items():
        path = ROOT / "artifacts" / name
        reports[key] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    import pandas as pd
    photo_test = ROOT / "artifacts/family_test_results.csv"
    if photo_test.exists():
        frame = pd.read_csv(photo_test)
        reports["family"] = {"photos": len(frame), "top1": int(frame.top1.sum()), "top3": int(frame.top3.sum())}
    reports["pricing"] = bundle()["report"]
    return jsonify(reports)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8502)
