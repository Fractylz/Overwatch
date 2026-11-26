import json
from pathlib import Path
import shutil
import os
os.environ["YOLO_OFFLINE"]="true"

from ultralytics import YOLO
import pandas as pd
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent
ART=ROOT/"artifacts"

def predict_folder(model, pics, imgsz, threshold):
    for start in range(0,len(pics),8):
        yield from model.predict([str(p) for p in pics[start:start+8]],batch=8,device="cpu",
                                 imgsz=imgsz,conf=threshold,verbose=False)


def top_box_predictions(model, split, imgsz, threshold):
    testdir=ROOT/"data/detection"/split
    pics=sorted((testdir/"images").glob("*"))
    decisions=[]
    for result in predict_folder(model,pics,imgsz,threshold):
        path=Path(result.path)
        labels=np.loadtxt(testdir/"labels"/(path.stem+".txt"),ndmin=2)
        found=False
        predicted=""
        overlap=0.
        confidence=0.
        if len(result.boxes):
            i=int(result.boxes.conf.argmax())
            px=result.boxes.xyxyn[i].cpu().numpy()
            cls=int(result.boxes.cls[i])
            predicted=model.names[cls]
            confidence=float(result.boxes.conf[i])
            for cat,x,y,w,h in labels:
                if int(cat)!=cls:
                    continue
                gt=[x-w/2,y-h/2,x+w/2,y+h/2]
                inter=max(0,min(gt[2],px[2])-max(gt[0],px[0])) * max(0,min(gt[3],px[3])-max(gt[1],px[1]))
                union=w*h+(px[2]-px[0])*(px[3]-px[1])-inter
                overlap=max(overlap,float(inter/union) if union>0 else 0.)
            found=overlap>=.5
        decisions.append(dict(image=path.name,expected=" / ".join(sorted({model.names[int(v)] for v in labels[:,0]})),
                              prediction=predicted,score=round(confidence,4),same_class_iou=round(overlap,4),correct=found))
    return decisions

if __name__=="__main__":
    import sys
    sys.path.insert(0,str(ROOT))
    from e2e_prediction import run
    from watch_classifier.yolov8_detector import detect_brand
    from listing import save_listing
    import argparse
    parser=argparse.ArgumentParser(description="Evaluate and install a trained checkpoint for the app.")
    parser.add_argument("run", type=Path, help="YOLO run folder containing results.csv and weights/best.pt")
    checkpoint_run=parser.parse_args().run.resolve()
    model_path=checkpoint_run/"weights/best.pt"
    shutil.copy2(model_path,ART/"watch_detector.pt")
    model=YOLO(str(ART/"watch_detector.pt"))
    imgsz=int(model.overrides.get("imgsz",224))
    results=model.val(data=str(ROOT/"data/detection/data.yaml"),split="test",imgsz=imgsz,device="cpu",
                      workers=0,batch=16,plots=False,project=str(ROOT/"runs"),name="test_evaluation")
    sources=pd.read_csv(ROOT/"data/detection/manifest.csv")
    log=pd.read_csv(checkpoint_run/"results.csv")
    log.to_csv(ART/"detector_training.csv",index=False)
    def metrics(p,r,m50,m):
        return {k:round(float(v),5) for k,v in zip(("precision","recall","mAP50","mAP50-95"),(p,r,m50,m))}
    classes=[]
    for j,idx in enumerate(results.box.ap_class_index):
        classes.append({"brand":model.names[int(idx)],"instances":int(results.nt_per_class[int(idx)]),
                        **metrics(*results.box.class_result(j))})
    report={"train_images":int((sources.split=="train").sum()),"validation_images":int((sources.split=="val").sum()),
            "test_images":int((sources.split=="test").sum()),"classes":len(model.names),"input_pixels":imgsz,
            "epochs":len(log),"overall":metrics(*results.box.mean_results()),"per_brand":classes}
    # Verify the application's top-box rule at the actual decision threshold.
    validation=top_box_predictions(model,"val",imgsz,.05)
    options=[]
    for threshold in [.05,.075,.1,.125,.15,.175,.2,.225,.25,.3,.35,.4,.5]:
        detected=[d for d in validation if d["prediction"] and d["score"]>=threshold]
        correct=sum(bool(d["correct"]) for d in detected)
        f1=2*correct/(len(validation)+len(detected))
        options.append((f1,threshold))
    val_f1,threshold=max(options)
    settings={"confidence":threshold,"imgsz":imgsz,"validation_top_box_f1":round(val_f1,5)}
    (ART/"detector_settings.json").write_text(json.dumps(settings,indent=2))
    decisions=top_box_predictions(model,"test",imgsz,threshold)
    pd.DataFrame(decisions).to_csv(ART/"test_predictions.csv",index=False)
    report["app_top_box"]={"confidence_threshold":threshold,"required_iou":.5,"correct":sum(bool(d["correct"]) for d in decisions),
                           "no_detection":sum(not d["prediction"] for d in decisions),"total":len(decisions)}
    (ART/"test_metrics.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    held=sources[sources.split=="test"].sort_values("filename").iloc[0]
    path=next((ROOT/"data/detection/test/images").glob(held.sha256+".*"))
    (ROOT/"examples").mkdir(exist_ok=True)
    sample=ROOT/"examples"/"heldout_watch.jpg"
    shutil.copyfile(path,sample)
    expected=(ROOT/"data/detection/test/labels"/(held.sha256+".txt")).read_text().splitlines()
    out=run(sample)
    out["image"]="examples/heldout_watch.jpg"
    out["expected_brands"]=sorted({model.names[int(line.split()[0])] for line in expected})
    out["source_filename"]=held.filename
    (ROOT/"examples/photo_result.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
    assert len(model.names)==10 and len(classes)>0
    print(json.dumps({"metrics":report["overall"],"example":out},ensure_ascii=True))
