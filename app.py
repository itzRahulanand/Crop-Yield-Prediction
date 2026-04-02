"""
Crop Yield Prediction — Flask Web Application
Serves a premium UI and prediction API.
"""

import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

# ── Paths ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))
CORS(app)

# ── Load saved artifacts ─────────────────────────────────────────────────
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
encoders = joblib.load(os.path.join(BASE_DIR, "encoders.pkl"))
feature_cols = joblib.load(os.path.join(BASE_DIR, "feature_cols.pkl"))
form_options = joblib.load(os.path.join(BASE_DIR, "form_options.pkl"))

# Weather + soil lookup tables
weather_df = pd.read_csv(os.path.join(BASE_DIR, "state_weather_data_1997_2020.csv"))
weather_df.columns = weather_df.columns.str.strip().str.lower()
weather_df["state"] = weather_df["state"].str.strip()

soil_df = pd.read_csv(os.path.join(BASE_DIR, "state_soil_data.csv"))
soil_df.columns = soil_df.columns.str.strip().str.lower()
soil_df["state"] = soil_df["state"].str.strip()
soil_df.rename(columns={"n": "soil_n", "p": "soil_p", "k": "soil_k", "ph": "soil_ph"}, inplace=True)


# ── Routes ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/options")
def api_options():
    """Return dropdown options for the frontend form."""
    return jsonify(form_options)


@app.route("/api/metrics")
def api_metrics():
    """Return the saved performance metrics from the trained model."""
    try:
        metrics = joblib.load(os.path.join(BASE_DIR, "metrics.pkl"))
        return jsonify(metrics)
    except FileNotFoundError:
        return jsonify({"error": "Metrics object missing"}), 404


@app.route("/predict", methods=["POST"])
def predict():
    """Accept form data, build feature vector, return prediction."""
    try:
        data = request.json

        crop = data["crop"].strip()
        state = data["state"].strip()
        season = data["season"].strip()
        year = int(data["year"])
        area = float(data["area"])
        fertilizer = float(data["fertilizer"])
        pesticide = float(data["pesticide"])

        # Encode categoricals
        crop_enc = encoders["crop"].transform([crop])[0]
        state_enc = encoders["state"].transform([state])[0]
        season_enc = encoders["season"].transform([season])[0]

        # Lookup weather (use closest available year if exact not found)
        w = weather_df[weather_df["state"] == state]
        w_year = w[w["year"] == year]
        if w_year.empty:
            w_year = w[w["year"] == w["year"].max()]
        if w_year.empty:
            return jsonify({"error": f"No weather data for state: {state}"}), 400

        avg_temp = w_year.iloc[0]["avg_temp_c"]
        rainfall = w_year.iloc[0]["total_rainfall_mm"]
        humidity = w_year.iloc[0]["avg_humidity_percent"]

        # Lookup soil
        s = soil_df[soil_df["state"] == state]
        if s.empty:
            return jsonify({"error": f"No soil data for state: {state}"}), 400

        soil_n = s.iloc[0]["soil_n"]
        soil_p = s.iloc[0]["soil_p"]
        soil_k = s.iloc[0]["soil_k"]
        soil_ph = s.iloc[0]["soil_ph"]

        # Build feature vector in exact column order
        features = {
            "year": year,
            "area": area,
            "fertilizer": fertilizer,
            "pesticide": pesticide,
            "avg_temp_c": avg_temp,
            "total_rainfall_mm": rainfall,
            "avg_humidity_percent": humidity,
            "soil_n": soil_n,
            "soil_p": soil_p,
            "soil_k": soil_k,
            "soil_ph": soil_ph,
            "crop_enc": crop_enc,
            "season_enc": season_enc,
            "state_enc": state_enc,
        }

        X = np.array([[features[col] for col in feature_cols]])
        prediction = model.predict(X)[0]

        return jsonify({
            "predicted_yield": round(float(prediction), 4),
            "unit": "tons per hectare",
            "model_features": {
                "temperature": f"{avg_temp}°C",
                "rainfall": f"{rainfall} mm",
                "humidity": f"{humidity}%",
                "soil_npk": f"N={soil_n}, P={soil_p}, K={soil_k}",
                "soil_ph": soil_ph,
            },
        })

    except KeyError as e:
        return jsonify({"error": f"Missing field: {e}"}), 400
    except ValueError as e:
        return jsonify({"error": f"Invalid value: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)