# Crop-Yield-Prediction
Predict agricultural crop yields using machine learning models trained on historical data, weather patterns, and soil conditions across Indian states.

# 🌾 CropYield AI — Smart Crop Prediction System

A production-ready, end-to-end Machine Learning pipeline and Web Application designed to accurately forecast agricultural crop yields across Indian states. 

By fusing massive historical agricultural datasets containing over 19,000 records with historical state-level **weather patterns** (temperature, rainfall, humidity) and **soil compositions** (Nitrogen, Phosphorous, Potassium, pH), this system implements an optimized ensemble architecture to achieve extremely high predictive accuracy.

## 🚀 Key Features

* **Advanced Data Fusion**: Dynamically merges isolated datasets—historical crop production, local climatic metrics (1997–2020), and soil N-P-K constraints—into a rich, multi-dimensional feature space.
* **Ensemble ML Architecture**: Trains and evaluates multiple state-of-the-art algorithms simultaneously:
  * **Random Forest Regressor** (Chosen best-performing model: **94.7%+ R² Score**)
  * Histogram-based Gradient Boosting
  * Gradient Boosting Regressor
* **Automated Data Cleaning**: Features robust ETL preprocessing directly in the pipeline, stripping out dimensional anomalies (like nut-based vs. tonne-based yields) to collapse Mean Absolute Error (MAE).
* **Premium Glassmorphism UI**: A fully responsive, dark-themed, interactive frontend built without any heavy CSS frameworks, communicating securely via a localized REST API wrapper.
* **Real-time Live Inference**: Instantly cross-references the user's input against the stored datasets to assemble the required feature-vector and outputs the predicted yield alongside critical environmental metrics.

## 💻 Tech Stack

* **Machine Learning**: `scikit-learn`, `pandas`, `numpy`, `joblib`
* **Backend Framework**: `Flask`, `Flask-CORS`
* **Frontend**: Pure HTML5, Vanilla JavaScript, CSS3 (Modern Glassmorphism)

---

## 🛠️ Installation & Setup

Because this project uses a pre-trained serialized model (`model.pkl`), anyone pulling the repository can boot up the application in seconds without needing to retrain the pipeline!

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/cropyield-ai.git
cd cropyield-ai
```

### 2. Create & Activate a Virtual Environment
```bash
# For macOS / Linux:
python3 -m venv venv
source venv/bin/activate

# For Windows:
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Web Server
Launch the Flask backend wrapper:
```bash
python app.py
```
Open your browser and navigate to: 👉 `http://localhost:5000`

---

## 🔬 Model Retraining (Optional)

If you modify the core dataset `crop_yield.csv` or wish to tweak hyper-parameters, you can fire the entire data-preprocessing and model training pipeline from scratch:

```bash
python src/main.py
```
This script will:
1. Reload, merge, and clean the 4 distinct CSV files.
2. Label-encode categorical strings.
3. Automatically evaluate Random Forest vs. Gradient Boosted trees.
4. Export the single highest-accuracy model directly to `model.pkl` and `validators.pkl`.
5. Dump the live metrics to `metrics.pkl` so the Web UI updates its dashboard instantly.

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
