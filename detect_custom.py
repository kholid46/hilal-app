import torch
import pandas as pd
from pathlib import Path
import os
from PIL import Image

def run_detection(source_path):
    weights = "best.pt"
    model = torch.hub.load("ultralytics/yolov5", "custom", path=weights, force_reload=True)
    model.conf = 0.25
    results = model(source_path)
    results.save()

    out_path = list(Path("runs/detect").glob("exp*/"))[-1]
    result_img = list(out_path.glob("*.jpg"))[0] if list(out_path.glob("*.jpg")) else None

    data = []
    for i, pred in enumerate(results.pred):
        if pred is not None and len(pred):
            for *box, conf, cls in pred.tolist():
                data.append({
                    "label": model.names[int(cls)],
                    "confidence": round(conf, 3),
                    "xmin": round(box[0]),
                    "ymin": round(box[1]),
                    "xmax": round(box[2]),
                    "ymax": round(box[3]),
                })

    df = pd.DataFrame(data)
    csv_path = out_path / "hasil_deteksi.csv"
    df.to_csv(csv_path, index=False)

    return str(result_img), str(csv_path), df
