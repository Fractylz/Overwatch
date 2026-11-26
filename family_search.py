from pathlib import Path
from functools import lru_cache
import re
import csv
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
INDEX = ROOT / "artifacts/family_gallery.npz"

@lru_cache(maxsize=1)
def encoder():
    import torch
    from torchvision.models import resnet18, ResNet18_Weights
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    weights = ResNet18_Weights.DEFAULT
    network = resnet18(weights=weights).eval()
    network.fc = torch.nn.Identity()
    return network, weights.transforms()

def embed(images):
    import torch
    network, transform = encoder()
    with torch.inference_mode():
        batch = torch.stack([transform(ImageOps.exif_transpose(i).convert("RGB")) for i in images])
        return torch.nn.functional.normalize(network(batch)).numpy()

@lru_cache(maxsize=1)
def index():
    with np.load(INDEX) as archive:
        return archive["vectors"], archive["labels"], archive["photos"]

def suggest(photo, dial_text=""):
    vectors, labels, paths = index()
    similarity = vectors @ embed([photo])[0]
    words = set(re.findall(r"[a-z0-9]+", dial_text.lower()))
    families = []
    for family in sorted(set(labels)):
        indices = np.flatnonzero(labels == family)
        closest = indices[np.argsort(-similarity[indices])[:3]]
        score = float(np.mean(similarity[closest]))
        # Require a distinctive word taken from the model-family name.
        distinctive = {"submariner","daytona","reverso","primero"}
        hits = words & distinctive & set(family.lower().split())
        score += .16 * bool(hits)
        families.append(dict(family=str(family), photo=str(paths[closest[0]]), score=score,
                             evidence="Dial text and photo" if hits else "Photo similarity"))
    return sorted(families, key=lambda row: -row["score"])[:3]

def build():
    with (ROOT / "data/family_manifest.csv").open(newline="", encoding="utf8") as stream:
        entries = [r for r in csv.DictReader(stream) if r["split"]=="gallery"]
    blocks = []
    for i in range(0,len(entries),12):
        pictures = []
        for row in entries[i:i+12]:
            with Image.open(ROOT/row["photo"]) as im:
                pictures.append(im.convert("RGB"))
        blocks.append(embed(pictures))
    np.savez_compressed(INDEX, vectors=np.vstack(blocks), labels=[r["family"] for r in entries],
                        photos=[r["photo"] for r in entries])

if __name__ == "__main__":
    build()
