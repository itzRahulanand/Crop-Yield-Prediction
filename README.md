# Crop Yield Prediction

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web%20App-000000?style=flat-square&logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![R2](https://img.shields.io/badge/R%C2%B2-0.947-2d6a4f?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)

A machine learning web application that predicts crop yield (tons per hectare) for Indian states from crop, season, input-usage, weather and soil data.

## Highlights

- **Explains 94.7% of yield variance (R² = 0.947, MAE = 1.03 t/ha)** on a held-out 20% test set by selecting the best of 3 ensemble regressors.
- **Enriched 19,000+ production records (1997–2020)** by merging them with state-level weather and soil datasets, giving the model 14 input features.
- **Improved data quality** by filtering out records in incompatible units (such as coconut counts) and anomalous yields above 150 t/ha before training.
- **Serves predictions through a Flask web app**, with JSON endpoints for form options, model metrics and predictions.

## Model comparison

All models were trained on the same 80:20 train-test split (`random_state=42`). The model with the highest R² is saved automatically.

| Model | R² | RMSE (t/ha) | MAE (t/ha) |
|---|---|---|---|
| **Gradient Boosting** (selected) | **0.947** | **2.73** | 1.03 |
| HistGradientBoosting | 0.944 | 2.82 | 1.10 |
| Random Forest | 0.942 | 2.86 | 0.81 |

## Features used

| Group | Features |
|---|---|
| Categorical | Crop, Season, State |
| Time and land | Year, Area (ha) |
| Inputs | Fertilizer (kg), Pesticide (kg) |
| Climate | Average temperature (°C), Annual rainfall (mm), Humidity (%) |
| Soil | Nitrogen (N), Phosphorus (P), Potassium (K), pH |

## Project structure

```text
Crop-Yield-Prediction/
├── app.py                           # Flask server: UI + prediction API
├── predict.py                       # Command-line prediction helper
├── src/
│   └── main.py                      # Training pipeline: merge data, train 3 models, save the best
├── templates/
│   └── index.html                   # Web interface
├── crop_yield.csv                   # Production data (19,000+ rows)
├── state_weather_data_1997_2020.csv # State-level weather data
├── state_soil_data.csv              # State-level soil data
├── model.pkl                        # Trained model
├── encoders.pkl                     # Label encoders for categorical features
├── feature_cols.pkl                 # Feature order used by the model
├── form_options.pkl                 # Dropdown values for the UI
├── metrics.pkl                      # Saved evaluation metrics
└── requirements.txt
```

## Getting started

```bash
git clone https://github.com/itzRahulanand/Crop-Yield-Prediction.git
cd Crop-Yield-Prediction
pip install -r requirements.txt

# Optional: retrain the model
python src/main.py

# Start the web app
python app.py
```

Then open http://127.0.0.1:5000 in your browser.

## API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web interface |
| `GET` | `/api/options` | Valid crops, seasons and states for the form |
| `GET` | `/api/metrics` | R², RMSE and MAE of the deployed model |
| `POST` | `/predict` | Returns the predicted yield for the submitted inputs |

## Tech stack

Python · pandas · NumPy · scikit-learn · Flask · HTML/CSS/JavaScript

## License

MIT. See [LICENSE](LICENSE).
