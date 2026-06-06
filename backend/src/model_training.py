import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.config import CLEANED_DATA_PATH, BEST_MODEL_PATH, RANDOM_STATE, TEST_SIZE
from data_preprocessing import load_data, prepare_data, get_preprocessor


def train_models(X_train, y_train):
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
    return trained


def evaluate_models(trained_models, X_test, y_test):
    results = []
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        results.append({
            "Model": name,
            "MAE": round(mae, 2),
            "MSE": round(mse, 2),
            "RMSE": round(rmse, 2),
            "R2": round(r2, 4),
        })
    return pd.DataFrame(results)


def select_best_model(results_df, trained_models):
    best_row = results_df.loc[results_df["R2"].idxmax()]
    best_name = best_row["Model"]
    print(f"Best model: {best_name}")
    print(f"R² Score: {best_row['R2']}")
    print(f"RMSE: {best_row['RMSE']}")
    return trained_models[best_name], best_name


def save_model(model, preprocessor, path=None):
    path = path or BEST_MODEL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "preprocessor": preprocessor}, path)
    print(f"Model saved to {path}")


def load_saved_model(path=None):
    path = path or BEST_MODEL_PATH
    artifact = joblib.load(path)
    return artifact["model"], artifact["preprocessor"]


if __name__ == "__main__":
    df = load_data()
    df = prepare_data(df)
    preprocessor, numeric_features, categorical_features = get_preprocessor()
    X = df.drop(columns=["Price"])
    y = df["Price"]
    X_processed = preprocessor.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    models = train_models(X_train, y_train)
    results = evaluate_models(models, X_test, y_test)
    print("\nModel Comparison:")
    print(results.to_string(index=False))
    best_model, best_name = select_best_model(results, models)
    save_model(best_model, preprocessor)
    print(f"\nBest model ({best_name}) saved successfully!")
