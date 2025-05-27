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
    if model is None or scaler is None or feature_names is None:
         return "Server Error: Aset model tidak dapat dimuat. Periksa log.", 500
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Pastikan aset dimuat sebelum memproses prediksi
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
                        # Convert to float for numeric columns that will be scaled
                        input_data_dict[col] = float(data[col])
                    else:
                        # Convert to int for other (mostly categorical/discrete) columns
                        input_data_dict[col] = int(data[col])
                except ValueError:
                    # Tangani non-numeric input jika diharapkan numerik
                    print(f"ValueError: Could not convert input for column {col}: {data[col]}") # Debug: Log conversion error
                    return render_template('result.html', prediction_text=f'Input tidak valid untuk kolom: {col}')
            else:
                # Tangani jika ada kolom yang tidak ada di form data
                print(f"Error: Missing column in form data: {col}") # Debug: Log missing column error
                return render_template('result.html', prediction_text=f'Input untuk kolom {col} tidak ditemukan.')

        print(f"Input data dict before DataFrame: {input_data_dict}") # Debug: Print dict before DataFrame creation

        # Create DataFrame from dictionary
        # Use index=[0] to ensure it's a single row DataFrame
        input_df = pd.DataFrame([input_data_dict], index=[0])
        print(f"DataFrame after creation (before scaling):") # Debug
        print(input_df)
        print(input_df.info()) # Debug: Check data types and non-nulls
        print(f"Columns before sorting: {input_df.columns.tolist()}") # Debug

        # Skalakan kolom numerik
        # Pastikan nama kolom numerik untuk scaling benar dan ada di input_df
        cols_to_scale_present = [col for col in NUMERIC_COLS_TO_SCALE if col in input_df.columns]
        if cols_to_scale_present:
            try:
                 # Apply scaling
                 input_df[cols_to_scale_present] = scaler.transform(input_df[cols_to_scale_present])
                 print(f"DataFrame after scaling:") # Debug
                 print(input_df)
            except Exception as e:
                 print(f"Error during scaling: {e}") # Debug: Log scaling error
                 return render_template('result.html', prediction_text=f'Error saat penskalaan data: {e}')
        else:
            print("Peringatan: Kolom numerik untuk scaling tidak ditemukan di input data.")

        # PASTIKAN URUTAN KOLOM SAMA DENGAN TRAINING
        # Ini adalah langkah krusial. Memilih ulang kolom sesuai urutan feature_names
        # Menggunakan reindex akan memastikan urutan yang benar dan menangani jika ada kolom ekstra
        input_df = input_df.reindex(columns=feature_names, fill_value=0) # Use reindex for safety
        print(f"DataFrame just before prediction (with correct order):") # Debug
        print(input_df)
        print(f"Columns just before prediction: {input_df.columns.tolist()}") # Debug: Final check on column order

        # Lakukan prediksi menggunakan DataFrame
        prediction = model.predict(input_df)
        print(f"Raw prediction result: {prediction}") # Debug: Print the raw prediction array

        # If using RandomForest, check probabilities
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(input_df)
            print(f"Prediction probabilities [0=Non-Diabetes, 1=Diabetes]: {proba}") # Debug: Print probabilities


        # Berikan hasil prediksi
        if prediction[0] == 0:
            result = 'Tidak mengidap diabetes'
        else:
            result = 'Mengidap diabetes atau pra-diabetes'

        print(f"Final prediction string: {result}") # Debug: Print the final result string

        return render_template('result.html', prediction_text=result)

    except Exception as e:
        # Tangani error tak terduga lainnya
        print(f"Terjadi error tak terduga saat prediksi: {e}")
        return render_template('result.html', prediction_text=f'Terjadi Error: {e}')

# ====== Jalankan Aplikasi ======
if __name__ == '__main__':
    app.run(debug=True)
    # Jika berjalan di Colab dan ingin diakses publik, Anda perlu ngrok:
    # from flask_ngrok import run_with_ngrok
    # run_with_ngrok(app)
    # app.run()