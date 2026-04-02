import joblib
import numpy as np

model = joblib.load("model.pkl")

def predict(input_data):
    prediction = model.predict([input_data])
    return prediction[0]

if __name__ == "__main__":
    sample = [30, 60, 2, 0.5, 100, 40, 80, 6.5, 25, 1, 0.02]
    print("Predicted Yield:", predict(sample))