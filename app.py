import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier

def extract_avg_rgb(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    avg_color = img.mean(axis=0).mean(axis=0)
    return avg_color

# Load dataset citra
X = []
y = []
for fname in os.listdir("skin_samples"):
    path = os.path.join("skin_samples", fname)
    label = fname.split("_")[0]
    img = cv2.imread(path)
    feat = extract_avg_rgb(img)
    X.append(feat)
    y.append(label)

# Train model
model = KNeighborsClassifier(n_neighbors=1)
model.fit(X, y)

st.title("Rekomendasi Warna Kerudung Berbasis Citra Warna Kulit")

uploaded_file = st.file_uploader("Upload foto warna kulit (misal: foto tangan)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Citra yang diupload", width=200)
    img_array = np.array(img)
    if len(img_array.shape) == 3:
        img_array = cv2.resize(img_array, (100, 100))
        avg_rgb = extract_avg_rgb(img_array)
        label = model.predict([avg_rgb])[0]

        st.write(f"Warna kulit terdeteksi: **{label}**")

        # Rekomendasi berdasarkan warna kulit
        if label == "putih":
            rekomendasi = ["maroon.jpg", "emerald.jpg", "hitam.jpg", "merah_muda.jpg", "lavender.jpg", "peach.jpg"]
        elif label == "sawo":
            rekomendasi = ["lavender.jpg", "burgundy.jpg", "hitam.jpg", "mustard.jpg", "olive.jpg"]
        elif label == "kuning":
            rekomendasi = ["coral.jpg", "hijau_mint.jpg", "hitam.jpg", "biru_muda.jpg"]
        elif label == "gelap":
            rekomendasi = ["ungu_tua.jpg", "maroon.jpg", "emerald.jpg"]
        else:
            rekomendasi = ["hitam.jpg"]  # Default

        st.subheader("Rekomendasi Warna Kerudung:")
        for r in rekomendasi:
            image_path = os.path.join("kerudung_images", r)
            if os.path.exists(image_path):
                st.image(image_path, caption=r.split(".")[0], width=150)
            else:
                st.write(f"Gambar '{r}' tidak ditemukan.")
