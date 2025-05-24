import streamlit as st
from yolov5 import detect
import os
from pathlib import Path

# Judul aplikasi
st.title("Deteksi Hilal Menggunakan YOLOv5")

# Upload file gambar atau video
uploaded_file = st.file_uploader("Unggah Gambar atau Video", type=["jpg", "jpeg", "png", "mp4", "mov"])

# Jalankan deteksi jika ada file
if uploaded_file is not None:
    input_path = Path("input") / uploaded_file.name
    output_path = Path("runs/detect/exp")

    # Simpan file yang diunggah
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Jalankan deteksi YOLOv5
    detect.run(weights="best.pt", source=str(input_path), save_txt=True, save_conf=True)

    # Tampilkan hasil
    result_image = output_path / uploaded_file.name
    if result_image.exists():
        st.success("Deteksi selesai. Berikut hasilnya:")
        st.image(str(result_image))
        with open(result_image, "rb") as img_file:
            st.download_button("📥 Unduh Hasil Deteksi", img_file, file_name=uploaded_file.name)

        # Cek apakah ada hilal terdeteksi
        label_file = output_path / "labels" / uploaded_file.name.replace(".jpg", ".txt")
        if not label_file.exists():
            st.warning("⚠️ Hilal tidak terdeteksi.")
    else:
        st.error("Gagal mendeteksi hilal.")

