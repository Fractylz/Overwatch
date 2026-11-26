"""CLI version of the upstream CLIP and K-means notebook."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize


def group(embeddings, max_k=15):
    n = len(embeddings)
    if n < 3:
        raise ValueError("Clustering needs at least three images.")
    features = PCA(n_components=min(50,n-1,embeddings.shape[1]), random_state=42).fit_transform(normalize(embeddings))
    best = None
    for k in range(2, min(max_k, n-1)+1):
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(features)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(features, labels, sample_size=min(n, 3000), random_state=42)
        if best is None or score > best[0]:
            best = (score, labels)
    if best is None:
        raise ValueError("Images have indistinguishable embeddings.")
    return best


def main():
    import torch
    import open_clip
    from PIL import Image
    p = argparse.ArgumentParser()
    p.add_argument("images", type=Path)
    p.add_argument("--out", type=Path, default=Path("artifacts/clusters.csv"))
    a = p.parse_args()
    paths = sorted(f for f in a.images.rglob("*") if f.suffix.lower() in {".png",".jpg",".jpeg",".webp"})
    if len(paths) < 3:
        p.error("Supply a directory with at least three watch photographs.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, transform = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    model = model.to(device).eval()
    vectors = []
    with torch.inference_mode():
        for i in range(0,len(paths),16):
            batch = []
            for path in paths[i:i+16]:
                with Image.open(path) as img:
                    batch.append(transform(img.convert("RGB")))
            vectors.extend(model.encode_image(torch.stack(batch).to(device)).float().cpu().numpy())
    score, labels = group(np.asarray(vectors))
    a.out.parent.mkdir(exist_ok=True, parents=True)
    pd.DataFrame({"image":[str(f.resolve()) for f in paths],"cluster":labels}).to_csv(a.out,index=False)
    print(f"{len(paths)} images; {len(set(labels))} clusters; silhouette {score:.3f}; {a.out}")


if __name__ == "__main__":
    main()
