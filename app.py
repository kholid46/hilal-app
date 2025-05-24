import streamlit as st
import os
from pathlib import Path
import uuid
from PIL import Image
import torch
import sys

# Menambahkan path YOLOv5 ke sys.path
sys.path.append(str(Path(__file__).resolve().parent / 'yolov5'))

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device
from utils.datasets import letterbox

# Konfigurasi direktori
INPUT_DIR = Path("input")
OUTPUT_DIR = Path("runs/detect")
LABELS_DIR = OUTPUT_DIR / "labels"

# Buat direktori jika belum ada
for dir in [INPUT_DIR, OUTPUT_DIR, LABELS_DIR]:
    dir.mkdir(parents=True, exist_ok=True)

# Judul aplikasi
st.set_page_config(page_title="Deteksi Hilal", layout="centered")
st.title("🌙 Deteksi Hilal Otomatis dengan YOLOv5")
st.write("Unggah gambar hilal, dan model akan mendeteksi keberadaannya.")

# Upload file
uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_ext = uploaded_file.name.split('.')[-1]
    unique_name = f"{uuid.uuid4()}.{file_ext}"
    input_path = INPUT_DIR / unique_name

    # Simpan file yang diunggah
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load model
    device = select_device('')
    model = attempt_load('best.pt', map_location=device)
    model.eval()

    # Baca gambar
    img = Image.open(input_path)
    img = letterbox(img, 640, stride=32, auto=True)[0]  # Resize
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0  # Normalize
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inferensi
    pred = model(img)[0]
    pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

    # Proses hasil
    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img.size).round()
            for *xyxy, conf, cls in reversed(det):
                label = f'{model.names[int(cls)]} {conf:.2f}'
                st.write(f"Deteksi: {label}")
        else:
            st.warning("⚠️ Hilal tidak terdeteksi dalam gambar ini.")

    # Tampilkan gambar asli
    st.image(input_path, caption="Gambar yang diunggah", use_column_width=True)

    # Bersihkan file yang diunggah
    input_path.unlink()
