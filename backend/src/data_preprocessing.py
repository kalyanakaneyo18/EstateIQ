import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.config import DATA_RAW_PATH, CLEANED_DATA_PATH, RANDOM_STATE


def load_data(path=None):
    path = path or DATA_RAW_PATH
    df = pd.read_csv(path)
    return df


def clean_data(df):
    df = df.copy()
    if "Property_ID" in df.columns:
        df = df.drop(columns=["Property_ID"])
    df = df.drop_duplicates()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower, upper)
    return df


def engineer_features(df):
    df = df.copy()
    df["Area_per_Bedroom"] = df["Area"] / (df["Bedrooms"] + 1)
    df["Area_per_Bathroom"] = df["Area"] / (df["Bathrooms"] + 1)
    return df


def get_preprocessor():
    numeric_features = ["Area", "Bedrooms", "Bathrooms", "Age", "Area_per_Bedroom", "Area_per_Bathroom"]
    categorical_features = ["Location", "Property_Type"]
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop="first", sparse_output=False)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor, numeric_features, categorical_features


def prepare_data(df):
    df = clean_data(df)
    df = engineer_features(df)
    return df


if __name__ == "__main__":
    df = load_data()
    print(f"Loaded data shape: {df.shape}")
    df_clean = prepare_data(df)
    print(f"Cleaned data shape: {df_clean.shape}")
    CLEANED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(CLEANED_DATA_PATH, index=False)
    print(f"Cleaned data saved to {CLEANED_DATA_PATH}")
