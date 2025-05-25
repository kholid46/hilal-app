if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        input_path = temp_file.name

    st.info("🚀 Proses deteksi berjalan...")
    output_image_path, csv_path, hasil_df = run_detection(input_path)

    if hasil_df is not None and not hasil_df.empty:
        st.image(output_image_path, caption="Hasil Deteksi", use_column_width=True)
        st.success("✅ Hilal terdeteksi.")
        st.dataframe(hasil_df)

        st.download_button(
            "📥 Unduh CSV Deteksi",
            data=hasil_df.to_csv(index=False),
            file_name="hasil_deteksi.csv",
            mime="text/csv"
        )

        with open(output_image_path, "rb") as img_file:
            st.download_button(
                label="📥 Unduh Gambar Hasil Deteksi",
                data=img_file,
                file_name="hasil_deteksi.jpg",
                mime="image/jpeg"
            )
    else:
        st.warning("⚠️ Tidak ada hilal terdeteksi.")
