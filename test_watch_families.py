"""Evaluate photo-only family ranking on the held-out manifest split."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from PIL import Image

from family_search import ROOT, suggest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/family_test_results.csv")
    args = parser.parse_args()

    with (ROOT / "data/family_manifest.csv").open(newline="", encoding="utf-8-sig") as stream:
        test_photos = [row for row in csv.DictReader(stream) if row["split"] == "test"]
    if not test_photos:
        parser.error("The family manifest contains no test photos.")

    results = []
    by_family = defaultdict(list)
    for row in test_photos:
        with Image.open(ROOT / row["photo"]) as image:
            labels = [hit["family"] for hit in suggest(image)][:3]
        first, second, third = (labels + [""] * 3)[:3]
        family = row["family"]
        result = dict(expected=family, photo=row["photo"], source=row["source"],
                      first=first, second=second, third=third,
                      top1=first == family, top3=family in labels)
        results.append(result)
        by_family[family].append(result)

    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    for name, rows in [("Overall", results), *sorted(by_family.items())]:
        total = len(rows)
        top1 = sum(row["top1"] for row in rows)
        top3 = sum(row["top3"] for row in rows)
        print(f"{name}: Top-1 {top1}/{total}, Top-3 {top3}/{total}")


if __name__ == "__main__":
    main()
