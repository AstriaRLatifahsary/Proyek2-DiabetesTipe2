from flask import Flask, request, render_template, url_for
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# ====== Konfigurasi Path Aset ======
MODEL_PATH = 'models/random_forest_model.pkl' # Ganti jika menggunakan random_forest_model.pkl
SCALER_PATH = 'models/scaler.pkl'
FEATURE_NAMES_PATH = 'models/feature_names.pkl'
NUMERIC_COLS_TO_SCALE = ['BMI', 'MentHlth', 'PhysHlth']

# ====== Muat Model dan Aset ======
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    print("Model, scaler, dan nama fitur berhasil dimuat.")
    print(f"Feature Names loaded: {feature_names}") # Debug: Print loaded feature names
except FileNotFoundError as e:
    print(f"Error: File aset tidak ditemukan. Pastikan ada di folder 'models/'. {e}")
    model = None
    scaler = None
    feature_names = None

# ====== Rute Aplikasi ======
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/form')
def form():
    if model is None or scaler is None or feature_names is None:
         return "Server Error: Aset model tidak dapat dimuat. Periksa log.", 500
    return render_template('form.html')

@app.route('/classify', methods=['POST'])
def classify():
    # Pastikan aset dimuat sebelum memproses klasifikasi
    if model is None or scaler is None or feature_names is None:
         return "Server Error: Aset model tidak dapat dimuat. Periksa log.", 500

    try:
        data = request.form.to_dict()
        print(f"Raw form data: {data}") # Debug: Print raw data from form

        input_data_dict = {col: None for col in feature_names}

        for col in feature_names:
            if col in data:
                try:
                    if col in NUMERIC_COLS_TO_SCALE:
                        input_data_dict[col] = float(data[col])
                    else:
                        input_data_dict[col] = int(data[col])
                except ValueError:
                    print(f"ValueError: Could not convert input for column {col}: {data[col]}")
                    return render_template('result.html', classification_text=f'Input tidak valid untuk kolom: {col}')
            else:
                print(f"Error: Missing column in form data: {col}")
                return render_template('result.html', classification_text=f'Input untuk kolom {col} tidak ditemukan.')

        print(f"Input data dict before DataFrame: {input_data_dict}")
        input_df = pd.DataFrame([input_data_dict], index=[0])
        print(f"DataFrame after creation (before scaling):")
        print(input_df)
        print(input_df.info())
        print(f"Columns before sorting: {input_df.columns.tolist()}")

        cols_to_scale_present = [col for col in NUMERIC_COLS_TO_SCALE if col in input_df.columns]
        if cols_to_scale_present:
            try:
                 input_df[cols_to_scale_present] = scaler.transform(input_df[cols_to_scale_present])
                 print(f"DataFrame after scaling:")
                 print(input_df)
            except Exception as e:
                 print(f"Error during scaling: {e}")
                 return render_template('result.html', classification_text=f'Error saat penskalaan data: {e}')
        else:
            print("Peringatan: Kolom numerik untuk scaling tidak ditemukan di input data.")

        input_df = input_df.reindex(columns=feature_names, fill_value=0)
        print(f"DataFrame just before classification (with correct order):")
        print(input_df)
        print(f"Columns just before classification: {input_df.columns.tolist()}")

        prediction = model.predict(input_df)
        print(f"Raw classification result: {prediction}")

        risk = 'high' if prediction[0] == 1 else 'low'
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(input_df)
            print(f"Classification probabilities [0=Non-Diabetes, 1=Diabetes]: {proba}")
            confidence = proba[0][prediction[0]]
        else:
            confidence = None

        # --- Faktor utama dan rekomendasi spesifik ---
        factors = []
        recs = []
        # Ambil threshold sederhana, bisa disesuaikan dengan feature_names dataset Anda
        if prediction[0] == 1:
            # BMI tinggi
            if 'BMI' in input_data_dict and input_data_dict['BMI'] is not None and input_data_dict['BMI'] > 27:
                factors.append('Indeks Massa Tubuh (BMI) tinggi')
                recs.append('Menurunkan berat badan dan menjaga pola makan sehat')
            # Tidak aktif
            if 'PhysActivity' in input_data_dict and input_data_dict['PhysActivity'] == 0:
                factors.append('Kurang aktivitas fisik')
                recs.append('Tingkatkan aktivitas fisik, misal rutin berjalan kaki')
            # Konsumsi buah kurang
            if 'Fruits' in input_data_dict and input_data_dict['Fruits'] == 0:
                factors.append('Kurang konsumsi buah')
                recs.append('Perbanyak konsumsi buah setiap hari')
            # Konsumsi sayur kurang
            if 'Veggies' in input_data_dict and input_data_dict['Veggies'] == 0:
                factors.append('Kurang konsumsi sayur')
                recs.append('Perbanyak konsumsi sayur setiap hari')
            # Merokok
            if 'Smoker' in input_data_dict and input_data_dict['Smoker'] == 1:
                factors.append('Kebiasaan merokok')
                recs.append('Berhenti merokok untuk menurunkan risiko')
            # Tekanan darah tinggi
            if 'HighBP' in input_data_dict and input_data_dict['HighBP'] == 1:
                factors.append('Tekanan darah tinggi')
                recs.append('Kontrol tekanan darah secara rutin')
            # Kolesterol tinggi
            if 'HighChol' in input_data_dict and input_data_dict['HighChol'] == 1:
                factors.append('Kolesterol tinggi')
                recs.append('Perbaiki pola makan dan cek kolesterol secara rutin')
            # Alkohol berat
            if 'HvyAlcoholConsump' in input_data_dict and input_data_dict['HvyAlcoholConsump'] == 1:
                factors.append('Konsumsi alkohol berat')
                recs.append('Kurangi atau hindari konsumsi alkohol')
            # Hari kesehatan fisik buruk
            if 'PhysHlth' in input_data_dict and input_data_dict['PhysHlth'] is not None and input_data_dict['PhysHlth'] > 10:
                factors.append('Sering mengalami hari kesehatan fisik buruk')
                recs.append('Perbaiki pola hidup dan konsultasi ke dokter')
            # Hari kesehatan mental buruk
            if 'MentHlth' in input_data_dict and input_data_dict['MentHlth'] is not None and input_data_dict['MentHlth'] > 10:
                factors.append('Sering mengalami hari kesehatan mental buruk')
                recs.append('Kelola stres dan konsultasi ke profesional')
            # Riwayat penyakit jantung/serangan jantung
            if 'HeartDiseaseorAttack' in input_data_dict and input_data_dict['HeartDiseaseorAttack'] == 1:
                factors.append('Riwayat penyakit jantung/serangan jantung')
                recs.append('Konsultasi ke dokter untuk penanganan lebih lanjut')
            # Riwayat stroke
            if 'Stroke' in input_data_dict and input_data_dict['Stroke'] == 1:
                factors.append('Riwayat stroke')
                recs.append('Konsultasi ke dokter untuk penanganan lebih lanjut')
            # Pilih maksimal 3 faktor utama
            if len(factors) > 3:
                factors = factors[:3]
            # Gabungkan rekomendasi unik
            if recs:
                recommendation = '; '.join(dict.fromkeys(recs))
            else:
                recommendation = 'Segera konsultasikan ke dokter dan perbaiki pola hidup.'
            result = 'Mengidap diabetes'
            explanation = 'Data Anda menunjukkan risiko tinggi diabetes tipe 2.'
        else:
            result = 'Tidak mengidap diabetes'
            explanation = 'Data Anda menunjukkan risiko rendah diabetes tipe 2.'
            factors = []
            recommendation = 'Tetap jaga pola hidup sehat dan lakukan pemeriksaan rutin.'

        print(f"Final classification string: {result}")

        return render_template('result.html', classification_text=result, risk=risk, explanation=explanation, recommendation=recommendation, factors=factors)

    except Exception as e:
        print(f"Terjadi error tak terduga saat klasifikasi: {e}")
        return render_template('result.html', classification_text=f'Terjadi Error: {e}')

# ====== Jalankan Aplikasi ======
if __name__ == '__main__':
    app.run(debug=True)
    # Jika berjalan di Colab dan ingin diakses publik, Anda perlu ngrok:
    # from flask_ngrok import run_with_ngrok
    # run_with_ngrok(app)
    # app.run()