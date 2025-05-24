import streamlit as st
import os
from pathlib import Path
import shutil
import uuid
from PIL import Image
from yolov5 import detect

# Konfigurasi direktori
INPUT_DIR = Path("input")
OUTPUT_DIR = Path("runs/detect")
LABELS_DIR = OUTPUT_DIR / "labels"

# Buat direktori jika belum ada
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LABELS_DIR.mkdir(parents=True, exist_ok=True)

# Judul aplikasi
st.set_page_config(page_title="Deteksi Hilal", layout="centered")
st.title("🌙 Deteksi Hilal Otomatis dengan YOLOv5")
st.write("Unggah gambar atau video hilal, dan model akan mendeteksi keberadaannya.")

# Upload file
uploaded_file = st.file_uploader("Unggah Gambar atau Video", type=["jpg", "jpeg", "png", "mp4", "mov", "avi"])

if uploaded_file is not None:
    file_ext = uploaded_file.name.split('.')[-1]
    unique_name = f"{uuid.uuid4()}.{file_ext}"
    input_path = INPUT_DIR / unique_name

    # Simpan file yang diunggah
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Jalankan deteksi YOLOv5
    st.info("🔍 Mendeteksi hilal...")
    detect.run(
        weights="best.pt",
        source=str(input_path),
        conf_thres=0.25,
        save_txt=True,
        save_conf=True
    )

    # Cari direktori hasil terbaru
    latest_exp_dir = max(OUTPUT_DIR.glob("exp*"), key=os.path.getmtime)
    result_file = latest_exp_dir / uploaded_file.name

    # Tampilkan hasil gambar/video
    if result_file.exists():
        if result_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            st.image(str(result_file), caption="Hasil Deteksi", use_column_width=True)
        elif result_file.suffix.lower() in [".mp4", ".mov", ".avi"]:
            st.video(str(result_file))

        # Unduh hasil
        with open(result_file, "rb") as f:
            st.download_button(
                label="📥 Unduh Hasil Deteksi",
                data=f,
                file_name=uploaded_file.name,
                mime="application/octet-stream"
            )

        # Cek apakah ada label hilal (output YOLO)
        label_file = latest_exp_dir / "labels" / uploaded_file.name.replace(f".{file_ext}", ".txt")
        if not label_file.exists():
            st.warning("⚠️ Hilal tidak terdeteksi dalam file ini.")
        else:
            st.success("✅ Hilal berhasil terdeteksi!")

    else:
        st.error("❌ Gagal menampilkan hasil deteksi.")

    # Bersihkan input
    input_path.unlink(missing_ok=True)

