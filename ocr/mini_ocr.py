"""
ocr/read.py - dial-text OCR, nothing else.

Exposes read_text(crop, brand) -> {ocr_text, candidate_model}
The EasyOCR reader loads once when this module is imported.
"""

import numpy as np
import easyocr

# dial boilerplate to ignore when guessing a model name
_STOPWORDS = {
    "swiss",
    "made",
    "automatic",
    "chronometer",
    "chronograph",
    "watch",
    "quartz",
    "superlative",
    "officially",
    "certified",
    "cosc",
    "water",
    "resistant",
    "geneve",
    "paris",
    "date",
}

_reader = easyocr.Reader(["en"], gpu=False)  # CPU-only


def read_text(crop, brand):
    """OCR the crop; return raw tokens + a best-effort model-name guess."""
    tokens = [
        t.strip() for t in _reader.readtext(np.array(crop), detail=0) if t.strip()
    ]
    brand_words = set(brand.lower().replace("-", "_").split("_"))
    candidates = [
        t
        for t in tokens
        if t.isalpha()
        and len(t) > 3
        and t.lower() not in _STOPWORDS
        and t.lower() not in brand_words
    ]
    candidate_model = max(candidates, key=len).title() if candidates else None
    return {"ocr_text": tokens, "candidate_model": candidate_model}
