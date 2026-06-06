import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.config import BEST_MODEL_PATH
from data_preprocessing import engineer_features
from model_training import load_saved_model


def make_prediction(area, bedrooms, bathrooms, age, location, property_type):
    model, preprocessor = load_saved_model(BEST_MODEL_PATH)
    input_df = pd.DataFrame([{
        "Area": area,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Age": age,
        "Location": location,
        "Property_Type": property_type,
    }])
    input_df = engineer_features(input_df)
    input_processed = preprocessor.transform(input_df)
    prediction = model.predict(input_processed)[0]
    return round(float(prediction), 2)


if __name__ == "__main__":
    price = make_prediction(2000, 3, 2, 10, "Urban", "House")
    print(f"Predicted Price: {price}")
