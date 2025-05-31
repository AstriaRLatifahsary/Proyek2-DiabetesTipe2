# Proyek 2 - Diabetes Tipe 2

Aplikasi web Flask untuk klasifikasi risiko Diabetes Tipe 2 berbasis machine learning (Random Forest atau KNN). Pengguna mengisi data kesehatan dan gaya hidup, lalu aplikasi mengklasifikasikan risiko diabetes serta menampilkan faktor utama dan rekomendasi spesifik.

## Fitur

- Interface web yang responsif dan mudah digunakan
- Formulir input data kesehatan dan gaya hidup dengan penjelasan
- Klasifikasi risiko diabetes tipe 2 berdasarkan input
- Analisis faktor utama risiko (max 3 faktor utama ditampilkan)
- Rekomendasi spesifik sesuai faktor risiko

## Struktur Proyek

```text
app.py                  # Aplikasi Flask utama
models/                 # Model machine learning & scaler
    random_forest_model.pkl
    knn_model.pkl
    scaler.pkl
    feature_names.pkl
static/                 # File statis (CSS, JS, gambar)
    css/style.css
    images/bg.jpg, appvengers_logo.png, ...
templates/              # Template HTML (index, form, result)
data/                   # Dataset pelatihan
notebooks/              # Notebook EDA & modeling
requirements.txt        # Dependensi Python
README.md               # Dokumentasi proyek
```

## Cara Menjalankan

1. **Clone repository**
2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Pastikan file model ada di folder `models/`**
4. **Jalankan aplikasi**

   ```bash
   python app.py
   ```

5. **Buka browser** ke `http://127.0.0.1:5000/`

## Tim Pengembang

Kelompok B1 - Appvengers:

- Alya Naila Putri Ashadilla (231524036)
- Astria Rizka Latifahsary (231524037)
- Devi Febrianti (231524039)
- Muhammad Gianluigi Julian (231524054)
- Muhammad Hasbi Asshidiqi (231524055)

## Lisensi

MIT License
