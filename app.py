import streamlit as st
import subprocess
import os
from pathlib import Path
from PIL import Image

st.title("🌓 Deteksi Hilal dengan YOLOv5")

# Folder input/output
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("runs/detect")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Upload file
uploaded_file = st.file_uploader("Unggah Gambar (.jpg, .png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_path = UPLOAD_DIR / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.image(file_path, caption="Gambar Diupload", use_column_width=True)

    # Jalankan deteksi YOLOv5 (subprocess ke detect.py)
    st.info("🔍 Menjalankan deteksi hilal...")
    result = subprocess.run([
        "python", "yolov5/detect.py",
        "--weights", "best.pt",
        "--source", str(file_path),
        "--save-txt",
        "--save-conf"
    ], capture_output=True, text=True)

    # Cek hasil
    latest_exp = sorted(OUTPUT_DIR.glob("exp*"), key=os.path.getmtime)[-1]
    result_image_path = latest_exp / uploaded_file.name

    if result_image_path.exists():
        st.success("✅ Deteksi selesai!")
        st.image(str(result_image_path), caption="Hasil Deteksi", use_column_width=True)

        with open(result_image_path, "rb") as f:
            st.download_button("📥 Unduh Gambar Hasil", f, file_name=uploaded_file.name)

        labels_path = latest_exp / "labels" / uploaded_file.name.replace(".jpg", ".txt").replace(".png", ".txt")
        if not labels_path.exists():
            st.warning("⚠️ Hilal tidak terdeteksi.")
    else:
        st.error("❌ Gagal mendeteksi hilal.")
        st.code(result.stderr)
