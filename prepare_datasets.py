"""Prepare both Roboflow exports for detection and Kaggle images for clustering."""
import argparse
import csv
import hashlib
import os
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory
import yaml

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
PROJECTS = [("special-problem-ehhbd", "classify_watch", 4), ("project-jisou", "watch-nefxk", 1)]
BRANDS = ["Audemars Piguet", "Breguet", "Breitling", "Maurice Lacroix", "Omega", "Patek Philippe",
          "Rado", "Rolex", "Tissot", "Vacheron Constantin"]
ALIASES = {re.sub("[^a-z]", "", b.lower()): i for i,b in enumerate(BRANDS)}
ALIASES["vacheron"] = BRANDS.index("Vacheron Constantin")
EXT = {".jpg",".jpeg",".png",".webp"}


def download():
    import kagglehub
    from roboflow import Roboflow
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        raise ValueError("Set ROBOFLOW_API_KEY or pass two extracted exports to merge.")
    client = Roboflow(api_key=key)
    roots = []
    for workspace, project, version in PROJECTS:
        dataset = client.workspace(workspace).project(project).version(version).download(
            "yolov8", location=str(DATA / "exports" / project))
        roots.append(Path(dataset.location))
    merge(roots)
    images = kagglehub.dataset_download("mathewkouch/a-dataset-of-watches")
    print("Kaggle image folder:", images)
    print(f'python cluster_images.py "{images}"')


def merge(roots):
    DATA.mkdir(exist_ok=True)
    dest = DATA / "detection"
    if dest.exists():
        raise ValueError(f"{dest} already exists. Move it before rebuilding.")
    seen_hash, seen_name, audit = set(), set(), []
    sources = []
    for root in roots:
        cfg = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))
        names = cfg["names"]
        mapping = {int(k): ALIASES.get(re.sub("[^a-z]", "", v.lower())) for k,v in
                   (enumerate(names) if isinstance(names, list) else names.items())}
        sources.append((root, mapping))
    with TemporaryDirectory(dir=DATA) as temp:
        out = Path(temp) / "detection"
        # Reserve held-out originals before processing augmented training copies.
        for split, folder in [("test","test"), ("val","valid"), ("train","train")]:
            for source, mapping in sources:
                sub = source / folder
                if split == "val" and not sub.exists():
                    sub = source / "val"
                for pic in sorted((sub / "images").glob("*")):
                    if pic.suffix.lower() not in EXT:
                        continue
                    label = sub / "labels" / (pic.stem+".txt")
                    if not label.exists():
                        continue
                    digest = hashlib.sha256(pic.read_bytes()).hexdigest()
                    origin = pic.stem.lower().split(".rf.")[0]
                    if digest in seen_hash or origin in seen_name:
                        continue
                    converted = []
                    for line in label.read_text().splitlines():
                        fields = line.split()
                        if len(fields) >= 7 and len(fields) % 2 == 1:
                            coords = [min(1., max(0., float(v))) for v in fields[1:]]
                            xs, ys = coords[::2], coords[1::2]
                            xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
                            fields = [fields[0], str((xmin+xmax)/2), str((ymin+ymax)/2),
                                      str(xmax-xmin), str(ymax-ymin)]
                        if len(fields) != 5:
                            raise ValueError(f"Expected YOLO bounding boxes in {label}")
                        cls = mapping.get(int(fields[0]))
                        if cls is None:
                            continue
                        x,y,w,h = map(float, fields[1:])
                        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                            raise ValueError(f"Invalid box in {label}")
                        converted.append(" ".join([str(cls), *fields[1:]]))
                    if not converted:
                        continue
                    seen_hash.add(digest)
                    seen_name.add(origin)
                    for name in ["images","labels"]:
                        (out/split/name).mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(pic, out/split/"images"/(digest+pic.suffix.lower()))
                    (out/split/"labels"/(digest+".txt")).write_text("\n".join(converted)+"\n")
                    audit.append((split, source.name, pic.name, digest))
        if not all((out/s/"images").exists() for s in ["train","val"]):
            raise ValueError("Merged exports need labelled training and validation images.")
        cfg = {"path":str(dest.resolve()), "train":"train/images", "val":"val/images", "names":BRANDS}
        if (out/"test/images").exists():
            cfg["test"] = "test/images"
        (out/"data.yaml").write_text(yaml.safe_dump(cfg))
        with (out/"manifest.csv").open("w",newline="",encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["split","source","filename","sha256"])
            writer.writerows(audit)
        out.rename(dest)
    print(f"Prepared {len(audit)} annotated images in {dest}")


def train(args):
    from ultralytics import YOLO
    model = YOLO(args.initial)
    model.train(data=str(DATA/"detection/data.yaml"), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, workers=0, device=args.device, optimizer="AdamW", lr0=.002,
                warmup_epochs=1, close_mosaic=3, amp=False, plots=False,
                project=str(ROOT/"runs"), name="brands", seed=42)
    weights = Path(model.trainer.best)
    out = ROOT / "artifacts/watch_detector.pt"
    out.parent.mkdir(exist_ok=True)
    shutil.copyfile(weights, out.with_suffix(".tmp"))
    os.replace(out.with_suffix(".tmp"), out)
    if (DATA/"detection/test/images").exists():
        YOLO(str(out)).val(data=str(DATA/"detection/data.yaml"), split="test", workers=0)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download")
    sub.add_parser("merge").add_argument("exports", type=Path, nargs=2)
    training = sub.add_parser("train")
    training.add_argument("--epochs", type=int, default=16)
    training.add_argument("--imgsz", type=int, default=224)
    training.add_argument("--batch", type=int, default=16)
    training.add_argument("--device", default="cpu")
    training.add_argument("--initial", default="yolov8n.yaml",
                          help="YOLOv8n architecture YAML or a pretrained checkpoint")
    a = p.parse_args()
    if a.cmd == "download":
        download()
    elif a.cmd == "merge":
        merge(a.exports)
    else:
        train(a)
