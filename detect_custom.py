# detect_custom.py
import sys
from pathlib import Path

# Pastikan path YOLOv5 ditambahkan
FILE = Path(yolov5).resolve()
ROOT = FILE.parent / "yolov5"
sys.path.append(str(ROOT))

from yolov5.detect import run as detect_run

def run_detection(weights, source, conf=0.25):
    detect_run(weights=weights, source=source, conf_thres=conf, save_txt=True, save_conf=True)
