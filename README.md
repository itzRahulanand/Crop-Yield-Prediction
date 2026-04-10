# Crop Yield Prediction System

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=flat-square&logo=flask&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![R2 Score](https://img.shields.io/badge/R2%20Score-94.72%25-2d6a4f?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)

An intelligent machine learning system designed to predict agricultural crop yields (tons per hectare) based on a comprehensive set of variables including location, crop variety, seasonal weather patterns, soil composition, and chemical inputs (fertilizers and pesticides).

---

## Technical Overview

The system uses a Gradient Boosting regression engine trained on over 19,000 historical records from across India. It provides high-accuracy estimates by correlating multi-dimensional data points that traditionally impact agricultural production.

### Core Prediction Parameters

The model evaluates 14 critical features for every prediction:

1.  **Temporal Data**: Year (captures long-term climate and technology trends).
2.  **Land Metrics**: Area under cultivation (hectares).
3.  **Chemical Inputs**: Fertilizer and Pesticide usage (kg).
4.  **Climate Metrics**: Average Temperature (C), Total Rainfall (mm), and Humidity (%).
5.  **Soil Composition**: Nitrogen (N), Phosphorus (P), Potassium (K), and Soil pH.
6.  **Categorical Identifiers**: Crop type, Season, and State.

---

## Model Performance

The system compares multiple ensemble models (Random Forest, HistGradientBoosting, and GradientBoosting) and automatically deploys the one with the highest R-squared value on the validation set.

### Current Benchmarks

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **R-squared (R2)** | 94.72% | Model explains 94.7% of yield variance |
| **Mean Absolute Error (MAE)** | 1.03 | Average prediction error is 1.03 tons/ha |
| **Root Mean Squared Error (RMSE)** | 2.73 | Standard deviation of prediction residuals |

*Note: Performance was optimized by implementing outlier detection to remove incompatible units (e.g. coconut nut counts) and anomalous yield values (>150).*

---

## Project Structure

```text
crop yield prediction/
|-- app.py                      # Flask backend serving predictions
|-- src/
|   |-- main.py                 # Training pipeline (ETL + Model Selection)
|-- crop_yield.csv              # Main production dataset (19,000+ records)
|-- state_weather_data.csv      # Supplementary historical weather patterns
|-- state_soil_data.csv         # Supplementary regional soil benchmarks
|-- model.pkl                   # Serialized champion regressor
|-- encoders.pkl                # Categorical label encoders
|-- form_options.pkl            # Pre-computed dropdown options for UI
|-- metrics.pkl                 # Cached performance metrics
|-- templates/
|   |-- index.html              # Responsive web interface
|-- requirements.txt             # Project dependencies
```

---

## Installation and Usage

### Prerequisites

- Python 3.9 or higher
- Pip package manager

### Setup

1.  Standardize your environment by installing required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2.  (Optional) Run the training pipeline to regenerate the model using the latest data:
    ```bash
    python src/main.py
    ```

### Running the Web Application

To launch the predictive interface:

```bash
python app.py
```

The application will be accessible at `http://127.0.0.1:5000`.

---

## Machine Learning Workflow

1.  **Data Ingestion**: Merging production data with high-resolution weather and soil datasets via state-level keys.
2.  **Preprocessing**: Handling missing values through targeted deletion and normalizing disparate units.
3.  **Feature Engineering**: Label encoding categorical strings (Crop, Season, State) for numeric compatibility.
4.  **Model Selection**: Training competing architectures and selecting the "Champion" based on minimizing MAE and maximizing R-squared scores.
5.  **Artifact Generation**: Saving the trained model, feature scalers, and categorical mappings for sub-second inference in production.
