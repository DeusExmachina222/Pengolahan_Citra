import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def main():
    st.set_page_config(layout="wide")
    st.title("Tugas: Thresholding dan Equalization Citra 📸")
    
    # 1. Input satu Citra
    uploaded_file = st.file_uploader("Upload satu Citra (JPG, PNG, BMP)", type=["jpg", "jpeg", "png", "bmp"])
    
    if uploaded_file is not None:
        # Baca file gambar
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Konversi ke RGB (dari BGR default OpenCV) dan Grayscale
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        st.header("Citra Asli")
        col_ori1, col_ori2 = st.columns(2)
        with col_ori1:
            st.image(img_rgb, caption="Citra RGB (Asli)", use_column_width=True)
        with col_ori2:
            st.image(img_gray, caption="Citra Grayscale", use_column_width=True)
            
        st.divider()
        
        # ---
        # 1. Tampilkan Histogram RGB dan Grayscale (Grafik dan Angka)
        # ---
        st.header("1. Analisis Histogram")
        st.subheader("Grafik Histogram")

        col_hist1, col_hist2 = st.columns(2)
        
        # Kalkulasi dan Plotting Histogram RGB
        with col_hist1:
            fig_rgb, ax_rgb = plt.subplots()
            colors = ('r', 'g', 'b')
            hist_data_rgb = {}
            
            for i, col in enumerate(colors):
                hist = cv2.calcHist([img_rgb], [i], None, [256], [0, 256])
                ax_rgb.plot(hist, color=col)
                hist_data_rgb[col.upper()] = hist.ravel() 
            
            ax_rgb.set_title("Histogram RGB")
            ax_rgb.set_xlabel("Intensitas Piksel")
            ax_rgb.set_ylabel("Jumlah Piksel")
            ax_rgb.legend(['Red', 'Green', 'Blue'])
            ax_rgb.set_xlim([0, 256])
            st.pyplot(fig_rgb)

        # Kalkulasi dan Plotting Histogram Grayscale
        with col_hist2:
            fig_gray, ax_gray = plt.subplots()
            hist_gray = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
            ax_gray.plot(hist_gray, color='gray')
            
            ax_gray.set_title("Histogram Grayscale (Asli)")
            ax_gray.set_xlabel("Intensitas Piksel")
            ax_gray.set_ylabel("Jumlah Piksel")
            ax_gray.set_xlim([0, 256])
            st.pyplot(fig_gray)
            
        st.subheader("Data Angka Histogram")
        col_data1, col_data2 = st.columns(2)
        with col_data1:
            st.write("Data Histogram RGB (256x3):")
            df_rgb = pd.DataFrame(hist_data_rgb)
            st.dataframe(df_rgb, height=300)
            
        with col_data2:
            st.write("Data Histogram Grayscale (256x1):")
            df_gray = pd.DataFrame(hist_gray, columns=["Jumlah Piksel"])
            st.dataframe(df_gray, height=300)
            
        st.divider()

        # ---
        # 2. Tampilkan Nilai Threshold dan Citra Biner (UPDATED)
        # ---
        st.header("2. Thresholding Biner (Otsu)")
        
        # Gunakan Otsu's Binarization untuk menemukan threshold
        nilai_threshold, img_biner = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        st.write(f"**Nilai Threshold (Otsu) yang ditemukan:** `{int(nilai_threshold)}`")
        st.write("Metode Otsu secara otomatis menentukan nilai threshold optimal yang memisahkan dua puncak dalam histogram (biasanya foreground dan background).")
        
        # Buat layout kolom untuk perbandingan
        col_thresh1, col_thresh2 = st.columns(2)
        
        with col_thresh1:
            st.subheader("Histogram Asli vs. Garis Threshold")
            
            # Hitung histogram grayscale (sama seperti di atas, untuk plot di sini)
            hist_gray_for_thresh = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
            
            fig_thresh, ax_thresh = plt.subplots()
            
            # 1. Plot histogramnya
            ax_thresh.plot(hist_gray_for_thresh, color='gray')
            ax_thresh.set_title("Histogram Grayscale dengan Garis Threshold")
            ax_thresh.set_xlabel("Intensitas Piksel")
            ax_thresh.set_ylabel("Jumlah Piksel")
            ax_thresh.set_xlim([0, 256])
            
            # 2. Tentukan nilai thresholdnya (INI YANG PENTING)
            # Gambar garis vertikal merah di posisi nilai_threshold
            ax_thresh.axvline(x=nilai_threshold, color='r', linestyle='dashed', linewidth=2)
            
            # Tambahkan legenda
            ax_thresh.legend([f'Threshold di {int(nilai_threshold)}', 'Histogram'], loc="upper right")
            
            st.pyplot(fig_thresh)

        with col_thresh2:
            st.subheader("Citra Biner Hasil Threshold")
            st.image(img_biner, caption=f"Citra Biner (Threshold = {int(nilai_threshold)})", use_column_width=True)

        st.divider()

        # ---
        # 3. Lakukan Histogram Equalization (Uniform)
        # ---
        st.header("3. Histogram Equalization (Uniform)")
        
        # Lakukan equalization pada citra grayscale
        img_equalized = cv2.equalizeHist(img_gray)
        
        st.write("Histogram Equalization (HE) mendistribusikan ulang intensitas piksel sehingga histogramnya menjadi lebih 'seragam' atau 'rata'. Ini biasanya meningkatkan kontras pada gambar.")
        
        col_eq1, col_eq2 = st.columns(2)
        
        with col_eq1:
            st.image(img_equalized, caption="Citra Hasil Equalization (Grayscale)", use_column_width=True)
            
        with col_eq2:
            # Tampilkan histogram dari citra yang sudah di-equalize untuk membuktikan
            fig_eq, ax_eq = plt.subplots()
            hist_eq = cv2.calcHist([img_equalized], [0], None, [256], [0, 256])
            ax_eq.plot(hist_eq, color='gray')
            ax_eq.set_title("Histogram Setelah Equalization")
            ax_eq.set_xlabel("Intensitas Piksel")
            ax_eq.set_ylabel("Jumlah Piksel")
            ax_eq.set_xlim([0, 256])
            st.pyplot(fig_eq)

if __name__ == "__main__":
    main()
