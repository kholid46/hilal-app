import sys
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device
import torch

def run_detection(source_path):
    weights = "best.pt"
    device = select_device("")
    model = DetectMultiBackend(weights, device=device)
    model.eval()
    
    dataset = LoadImages(source_path, img_size=640, stride=32, auto=True)
    names = model.names

    data_list = []
    output_image_path = "output.jpg"

    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(device).float() / 255.0
        if im.ndimension() == 3:
            im = im.unsqueeze(0)
        
        pred = model(im, augment=False, visualize=False)
        pred = non_max_suppression(pred, 0.25, 0.45, agnostic=False)

        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0s.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    data_list.append({
                        "filename": path,
                        "label": names[int(cls)],
                        "confidence": round(float(conf), 3),
                        "xmin": int(xyxy[0]),
                        "ymin": int(xyxy[1]),
                        "xmax": int(xyxy[2]),
                        "ymax": int(xyxy[3]),
                    })

        cv2.imwrite(output_image_path, im0s)

    df = pd.DataFrame(data_list)
    df.to_csv("hasil_deteksi.csv", index=False)
    return output_image_path, "hasil_deteksi.csv", df
