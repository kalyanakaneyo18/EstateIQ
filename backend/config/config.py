import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_RAW_PATH = BASE_DIR / "data" / "raw" / "house_prices.csv"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
CLEANED_DATA_PATH = DATA_PROCESSED_DIR / "cleaned_data.csv"
MODELS_DIR = BASE_DIR / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"

RANDOM_STATE = 18
TEST_SIZE = 0.2
