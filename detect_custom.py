import sys
from pathlib import Path

# Misal YOLOv5 ada di subfolder 'yolov5'
yolov5_path = Path(__file__).parent / "yolov5"
sys.path.append(str(yolov5_path.resolve()))

from yolov5.detect import run as detect_run

def run_detection(weights, source, conf=0.25):
    detect_run(weights=weights, source=source, conf_thres=conf, save_txt=True, save_conf=True)
