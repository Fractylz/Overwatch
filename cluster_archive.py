from pathlib import Path
import zipfile, shutil, json
from PIL import Image
import cv2, numpy as np, pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from threadpoolctl import threadpool_limits

root=Path(__file__).resolve().parents[1]
def descriptor(img):
    img=cv2.resize(img,(96,96),interpolation=cv2.INTER_AREA)
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    blocks=[]
    for y in (0, 48):
        for x in (0,48):
            region=hsv[y:y+48,x:x+48]
            hist=cv2.calcHist([region],[0,1],None,[12,4],[0,180,0,256]).ravel()
            blocks.extend(hist / max(1,hist.sum()))
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blocks.extend(cv2.resize(gray,(12,12),interpolation=cv2.INTER_AREA).ravel()/255)
    return blocks

def image_embeddings(archive, names):
    """Extract normalized ImageNet ResNet-18 features from product photos."""
    import torch
    from torchvision.models import resnet18, ResNet18_Weights
    from io import BytesIO
    weights = ResNet18_Weights.DEFAULT
    torch.set_num_threads(2)
    encoder = resnet18(weights=weights).eval()
    encoder.fc = torch.nn.Identity()
    transform = weights.transforms()
    embeddings = []
    with torch.inference_mode():
        for offset in range(0, len(names), 16):
            batch=[]
            for name in names[offset:offset+16]:
                with Image.open(BytesIO(archive.read(name))) as image:
                    batch.append(transform(image.convert("RGB")))
            vectors = encoder(torch.stack(batch))
            embeddings.extend(torch.nn.functional.normalize(vectors).numpy())
    return embeddings


def run(archive, out, encoder):
    out.mkdir(parents=True,exist_ok=True)
    names,feat=[],[]
    with zipfile.ZipFile(archive) as z:
        for n in sorted(z.namelist()):
            if not n.lower().endswith(('.jpeg','.jpg','.png','.webp')):
                continue
            img=cv2.imdecode(np.frombuffer(z.read(n),dtype=np.uint8),cv2.IMREAD_COLOR)
            if img is not None:
                names.append(n)
                if encoder == "colour":
                    feat.append(descriptor(img))
        if encoder == "resnet18":
            feat = image_embeddings(z, names)
        with threadpool_limits(2):
            x=np.asarray(feat,dtype=np.float32)
            if encoder == "colour":
                x=StandardScaler().fit_transform(x)
            x=PCA(n_components=min(32,len(x)-1,x.shape[1]),random_state=42).fit_transform(x)
            scores=[]
            for k in range(2,min(len(x),13)):
                model=MiniBatchKMeans(n_clusters=k,n_init=5,batch_size=256,random_state=42).fit(x)
                if len(set(model.labels_))>1:
                    score=silhouette_score(x,model.labels_,sample_size=min(1500,len(x)),random_state=42)
                    scores.append((float(score), model))
            score, best=max(scores,key=lambda s:s[0])
        df=pd.DataFrame({'image':names,'cluster':best.labels_})
        df.to_csv(out/'visual_groups.csv',index=False)
        tiles=[]
        for k in range(best.n_clusters):
            idx=np.flatnonzero(best.labels_==k)
            top=idx[np.argsort(np.linalg.norm(x[idx]-best.cluster_centers_[k],axis=1))[:4]]
            imgs=[]
            for i in top:
                img=cv2.imdecode(np.frombuffer(z.read(names[i]),np.uint8),cv2.IMREAD_COLOR)
                imgs.append(cv2.resize(img,(100,100)))
            while len(imgs)<4:
                imgs.append(np.full((100,100,3),255,np.uint8))
            row=cv2.copyMakeBorder(np.hstack(imgs),30,0,0,0,cv2.BORDER_CONSTANT,value=(245,245,245))
            cv2.putText(row,f'Group {k} - {len(idx)} photos',(8,21),cv2.FONT_HERSHEY_SIMPLEX,.54,(35,35,35),1,cv2.LINE_AA)
            tiles.append(row)
        cv2.imwrite(str(out/'visual_groups.jpg'),np.vstack(tiles))
        (out/'visual_groups.json').write_text(json.dumps({'images':len(x),'groups':best.n_clusters,'silhouette':round(score,3),
            'method':('ResNet-18 image embeddings' if encoder == 'resnet18' else 'HSV histograms and grayscale layout') + ', PCA and mini-batch K-means' },indent=2))

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('archive',type=Path)
    p.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'artifacts')
    p.add_argument('--encoder',choices=['resnet18','colour'],default='resnet18')
    a=p.parse_args()
    run(a.archive,a.out,a.encoder)
