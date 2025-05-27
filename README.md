# DiabetesTipe2Web

A Flask web application for predicting Type 2 Diabetes risk using a trained machine learning model (Random Forest or KNN). Users input health indicators, and the app predicts diabetes risk based on the provided data.

## Features

- User-friendly web interface for inputting health data
- Predicts diabetes or pre-diabetes risk
- Uses pre-trained machine learning models (Random Forest, KNN)
- Scales numeric features for accurate predictions
- Displays prediction results clearly

## Project Structure

```text
app.py                  # Main Flask application
models/                 # Contains trained models and scaler
    random_forest_model.pkl
    knn_model.pkl
    scaler.pkl
    feature_names.pkl
static/                 # Static files (CSS, JS, images)
templates/              # HTML templates (index, input, result)
data/                   # Dataset(s) used for training
notebooks/              # Jupyter notebooks for EDA/modeling
requirements.txt        # Python dependencies
README.md               # Project documentation
```

## Setup & Usage

1. **Clone the repository**

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure model files are present in `models/`**

4. **Run the app**

   ```bash
   python app.py
   ```

5. **Open your browser** and go to `http://127.0.0.1:5000/`

## Notes

- If you see a server error, ensure all model/scaler files exist in the `models/` folder.
- You can retrain models and update the `.pkl` files as needed.

## Authors

- Appvengers Team

## License

MIT License
