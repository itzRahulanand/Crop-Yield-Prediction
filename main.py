"""
Crop Yield Prediction — Model Training Pipeline
Merges crop_yield.csv + state weather data + state soil data,
trains RF / XGBoost / LightGBM, saves best model + encoders.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor

warnings.filterwarnings("ignore")

# ── Resolve paths relative to project root ──────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_and_merge_data():
    """Load crop_yield.csv and merge with weather + soil supplementary data."""

    # 1. Main dataset
    crop_path = os.path.join(PROJECT_ROOT, "crop_yield.csv")
    df = pd.read_csv(crop_path)
    df.columns = df.columns.str.strip().str.lower()

    print(f"✅ Loaded crop_yield.csv  — {len(df)} rows")
    print(f"   Columns: {list(df.columns)}")

    # Clean whitespace in string columns
    for col in ["crop", "season", "state"]:
        df[col] = df[col].astype(str).str.strip()

    # REMOVE OUTLIERS: Highly reduces RMSE.
    # Coconut production is often measured in 'nuts' not 'tonnes', causing massive 5000+ yield values.
    # We remove 'Coconut' and any other anomalous yields > 150 to stabilize error metrics.
    df = df[df["crop"].str.lower() != "coconut"]
    df = df[df["yield"] <= 150]
    print(f"   Shape after dropping outliers: {len(df)} rows")

    # 2. Weather data  (state + year → avg_temp_c, total_rainfall_mm, avg_humidity_percent)
    weather_path = os.path.join(PROJECT_ROOT, "state_weather_data_1997_2020.csv")
    weather = pd.read_csv(weather_path)
    weather.columns = weather.columns.str.strip().str.lower()
    weather["state"] = weather["state"].astype(str).str.strip()
    print(f"✅ Loaded weather data   — {len(weather)} rows")

    df = df.merge(weather, on=["state", "year"], how="left")

    # 3. Soil data  (state → N, P, K, pH)
    soil_path = os.path.join(PROJECT_ROOT, "state_soil_data.csv")
    soil = pd.read_csv(soil_path)
    soil.columns = soil.columns.str.strip().str.lower()
    soil["state"] = soil["state"].astype(str).str.strip()
    # rename to avoid collision with fertilizer columns
    soil.rename(columns={"n": "soil_n", "p": "soil_p", "k": "soil_k", "ph": "soil_ph"}, inplace=True)
    print(f"✅ Loaded soil data      — {len(soil)} rows")

    df = df.merge(soil, on="state", how="left")

    # Drop rows with missing merge keys
    before = len(df)
    df.dropna(subset=["avg_temp_c", "total_rainfall_mm", "soil_n"], inplace=True)
    print(f"   Dropped {before - len(df)} rows with missing weather/soil data")

    return df


def encode_categoricals(df):
    """Label-encode crop, season, state. Returns df + dict of encoders."""
    encoders = {}
    for col in ["crop", "season", "state"]:
        le = LabelEncoder()
        df[f"{col}_enc"] = le.fit_transform(df[col])
        encoders[col] = le
        print(f"   Encoded '{col}' → {len(le.classes_)} unique values")
    return df, encoders


def train_models(X_train, y_train, X_test, y_test):
    """Train three models, return best."""
    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=20, min_samples_split=5, random_state=42, n_jobs=-1
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_depth=10, random_state=42
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42
        ),
    }

    best_score = -999
    best_rmse = 0
    best_mae = 0
    best_model = None
    best_name = ""
    
    all_metrics = []

    print("\n" + "=" * 50)
    print("           MODEL COMPARISON")
    print("=" * 50)

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        
        all_metrics.append({
            "name": name,
            "r2": r2,
            "rmse": rmse,
            "mae": mae
        })

        print(f"\n  {name}")
        print(f"    R²  Score : {r2:.4f}")
        print(f"    RMSE      : {rmse:.4f}")
        print(f"    MAE       : {mae:.4f}")

        if r2 > best_score:
            best_score = r2
            best_rmse = rmse
            best_mae = mae
            best_model = model
            best_name = name

    print("\n" + "=" * 50)
    print(f"  🏆  Best Model: {best_name}  (R² = {best_score:.4f}, RMSE = {best_rmse:.4f}, MAE = {best_mae:.4f})")
    print("=" * 50)

    return best_model, best_name, best_score, best_rmse, best_mae, all_metrics


def main():
    # ── Load & merge ─────────────────────────────────────────────────────
    df = load_and_merge_data()

    # ── Encode categoricals ──────────────────────────────────────────────
    print("\n📊 Encoding categorical features...")
    df, encoders = encode_categoricals(df)

    # ── Feature selection ────────────────────────────────────────────────
    feature_cols = [
        "year", "area", "fertilizer", "pesticide",
        "avg_temp_c", "total_rainfall_mm", "avg_humidity_percent",
        "soil_n", "soil_p", "soil_k", "soil_ph",
        "crop_enc", "season_enc", "state_enc",
    ]

    X = df[feature_cols]
    y = df["yield"]

    print(f"\n📐 Features ({len(feature_cols)}): {feature_cols}")
    print(f"   Samples : {len(X)}")

    # ── Train / test split ───────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── Train ────────────────────────────────────────────────────────────
    best_model, best_name, best_score, best_rmse, best_mae, all_metrics = train_models(X_train, y_train, X_test, y_test)

    # ── Save artifacts ───────────────────────────────────────────────────
    model_path = os.path.join(PROJECT_ROOT, "model.pkl")
    encoders_path = os.path.join(PROJECT_ROOT, "encoders.pkl")
    features_path = os.path.join(PROJECT_ROOT, "feature_cols.pkl")
    metrics_path = os.path.join(PROJECT_ROOT, "metrics.pkl")

    joblib.dump(best_model, model_path)
    joblib.dump(encoders, encoders_path)
    joblib.dump(feature_cols, features_path)
    joblib.dump({"r2": best_score, "rmse": best_rmse, "mae": best_mae, "all": all_metrics}, metrics_path)

    print(f"\n💾 Saved model     → {model_path}")
    print(f"💾 Saved encoders  → {encoders_path}")
    print(f"💾 Saved features  → {features_path}")
    print(f"💾 Saved metrics   → {metrics_path}")

    # ── Save unique values for the web form dropdowns ────────────────────
    options = {
        "crops": sorted(encoders["crop"].classes_.tolist()),
        "states": sorted(encoders["state"].classes_.tolist()),
        "seasons": sorted(encoders["season"].classes_.tolist()),
    }
    options_path = os.path.join(PROJECT_ROOT, "form_options.pkl")
    joblib.dump(options, options_path)
    print(f"💾 Saved options   → {options_path}")

    print("\n✅ Pipeline complete!\n")


if __name__ == "__main__":
    main()