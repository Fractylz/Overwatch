"""
img_downloader.py (hardened)

Reads the per-brand CSVs produced by scraper.py (data/scraped/<brand>.csv, where
column 0 = image URL) and saves the images into data/images/<brand>/ — the
ImageFolder layout the classifier expects.

Improvements over the original:
  - skips files already downloaded (safe to re-run / resume after a crash)
  - per-row try/except so one bad URL doesn't kill the whole run
  - Windows-safe filenames
  - polite delay + request timeout
  - utf-8 CSV reading (handles accented names / currency symbols)
  - per-brand saved / skipped / failed counts

Standard library only — nothing extra to install.
"""

import os
import csv
import re
import time
import socket
import urllib.request

# ---------------- config ----------------
BRANDS = [  # v0 subset — keep these slugs IDENTICAL to scraper.py's
    "rolex",
    "omega",
    "cartier",
    "audemarspiguet",
    "seiko",
    # add the rest of the repo's list back when you scale past the prototype
]
SCRAPED_DIR = "data/scraped"
IMAGES_DIR = "data/images"
DELAY = 0.3  # seconds between downloads — be polite to the host
TIMEOUT = 15  # seconds per request
HAS_HEADER = False  # set True if your CSVs have a header row

socket.setdefaulttimeout(TIMEOUT)

opener = urllib.request.build_opener()
opener.addheaders = [
    (
        "User-Agent",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    )
]
urllib.request.install_opener(opener)


def safe(text):
    """Make a string safe inside a filename on any OS."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(text))[:40]


for brand in BRANDS:
    csv_path = os.path.join(SCRAPED_DIR, f"{brand}.csv")
    out_dir = os.path.join(IMAGES_DIR, brand)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        print(f"[skip] {csv_path} not found — run scraper.py for '{brand}' first")
        continue

    saved = skipped = failed = 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        if HAS_HEADER:
            next(reader, None)
        for index, row in enumerate(reader):
            if not row:  # blank line
                continue
            url = row[0].strip()
            tag = safe(row[1]) if len(row) > 1 else ""
            fname = f"{brand}-{index}-{tag}.jpg" if tag else f"{brand}-{index}.jpg"
            dest = os.path.join(out_dir, fname)

            if os.path.exists(dest):  # resume: don't re-download
                skipped += 1
                continue
            try:
                urllib.request.urlretrieve(url, dest)
                saved += 1
                time.sleep(DELAY)
            except Exception as e:
                failed += 1
                print(f"  [fail] {brand} row {index}: {e}")

    print(f"{brand}: {saved} saved, {skipped} already there, {failed} failed")

print("\nDone.")
