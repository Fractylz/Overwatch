from functools import lru_cache
from pathlib import Path
import re
import json
import sqlite3
import joblib
import numpy as np
import pandas as pd
from price_classifier.train_price_model import CAT, NUM

ROOT = Path(__file__).resolve().parent
PRICE = ROOT / "artifacts/price_model.joblib"


def normalized(text):
    return " ".join(re.findall(r"[a-z0-9]+", str(text).casefold()))


@lru_cache(maxsize=2)
def _load(path, modified):
    return joblib.load(path)


def bundle():
    return _load(PRICE, PRICE.stat().st_mtime_ns)


def catalogue():
    families = pd.read_csv(ROOT / "data/model_families.csv").assign(reference="")
    return [*bundle()["catalogue"], *families.to_dict("records")]


def visible_brands(text, rows):
    words = " " + normalized(text) + " "
    return {r["brand"] for r in rows if " " + normalized(r["brand"]) + " " in words}


def identify(text, rows, brand=""):
    """Prioritize exact reference tokens, then longest literal model name; reject ties."""
    words = " " + normalized(text) + " "
    mentioned = visible_brands(text, rows)
    if len(mentioned) > 1:
        return None
    explicit_brand = next(iter(mentioned), "")
    family_brand = explicit_brand or brand
    compact_tokens = {re.sub(r"[^a-z0-9]", "", t.casefold()) for t in re.findall(r"[\w./-]+", text)}
    matches = []
    for row in rows:
        if explicit_brand and normalized(row["brand"]) != normalized(explicit_brand):
            continue
        ref = str(row["reference"])
        family = normalized(row["model_family"])
        ref_key = re.sub("[^a-z0-9]", "", ref.casefold())
        if ref_key and ref_key in compact_tokens:
            matches.append(((2, len(ref_key)), row, "reference", ref))
        elif (not family_brand or normalized(row["brand"]) == normalized(family_brand)) and family and " " + family + " " in words:
            matches.append(((1, len(family)), row, "model name", ""))
    if not matches:
        return None
    score = max(m[0] for m in matches)
    winners = {(r["brand"], r["model_family"], ref, via) for s,r,via,ref in matches if s == score}
    if len(winners) != 1:
        return None
    b, m, ref, via = winners.pop()
    return dict(brand=b, model_family=m, reference=ref, evidence=via)


def estimate(info):
    data = bundle()
    rows = [r for r in data["catalogue"] if r["brand"] == info["brand"] and r["model_family"] == info["model_family"]]
    if not rows:
        return None
    features = {key: info.get(key) or "Unknown" for key in CAT}
    features.update({key: float(info[key]) if info.get(key) else np.nan for key in NUM})
    amount = float(np.expm1(data["model"].predict(pd.DataFrame([features]))[0]))
    return dict(amount=round(amount, 2), currency=data["report"]["currency"], examples=len(rows),
                observed_min=min(r["price"] for r in rows), observed_max=max(r["price"] for r in rows))


def save_listing(record, db=None):
    path = db or ROOT / "data/listings.sqlite"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS listings(id INTEGER PRIMARY KEY, created TEXT DEFAULT CURRENT_TIMESTAMP, payload TEXT NOT NULL)")
        return conn.execute("INSERT INTO listings(payload) VALUES (?)", (json.dumps(record),)).lastrowid


def history():
    path = ROOT / "data/listings.sqlite"
    if not path.exists():
        return []
    with sqlite3.connect(path) as conn:
        return [dict(id=i, saved=t, **json.loads(p)) for i,t,p in conn.execute("SELECT id,created,payload FROM listings ORDER BY id DESC")]
