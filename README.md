# EstateIQ — House Price Prediction System

An end-to-end machine learning system that predicts residential property prices based on physical and locational attributes. Built with Python, scikit-learn, FastAPI, and a modern responsive frontend.

---

## Business Problem Statement

Real estate valuation is a complex, multi‑factor problem. Buyers, sellers, and investors need **data‑driven, fast, and reliable price estimates** to make informed decisions. This project solves that by training several regression models on historical property data, selecting the best performer, and exposing it through a REST API with an interactive web interface.

---

## Project Overview

| Aspect | Detail |
|---|---|
| **Goal** | Predict house prices (in INR) from property features |
| **Target Variable** | `Price` — continuous, in Indian Rupees |
| **Dataset** | 300 records × 8 columns |
| **Best Model** | Gradient Boosting Regressor |
| **R² Score** | **0.9844** |
| **RMSE** | **₹ 1,640,505.85** |
| **Frontend** | HTML + CSS + JS (EstateIQ brand) |
| **Backend** | FastAPI + Jinja2 |
| **Deployment** | Uvicorn server, single‑command start |

---

## Data Dictionary

| Feature | Type | Description |
|---|---|---|
| `Property_ID` | String | Unique property identifier (dropped) |
| `Area` | Float | Total area in sq. ft. |
| `Bedrooms` | Integer | Number of bedrooms |
| `Bathrooms` | Integer | Number of bathrooms |
| `Age` | Integer | Property age in years |
| `Location` | Categorical | Rural / Suburb / City Center |
| `Property_Type` | Categorical | House / Apartment / Villa |
| `Price` | Float | **Target** — price in INR |
| `Area_per_Bedroom` | Float | Engineered: Area / (Bedrooms + 1) |
| `Area_per_Bathroom` | Float | Engineered: Area / (Bathrooms + 1) |

---

## Methodology

### 1. Data Exploration & Cleaning

- **Missing values**: None found in the raw dataset.
- **Duplicates**: None found.
- **Outlier handling**: IQR capping applied to all numeric columns (1.5× IQR rule).
- **Unused columns**: `Property_ID` dropped — no predictive value.

### 2. Feature Engineering

- `Area_per_Bedroom` = Area / (Bedrooms + 1)
- `Area_per_Bathroom` = Area / (Bathrooms + 1)

### 3. Preprocessing Pipeline

| Step | Technique |
|---|---|
| Numeric scaling | StandardScaler (z‑score) |
| Categorical encoding | OneHotEncoder (drop first) |
| Combiner | `ColumnTransformer` |

### 4. Models Trained

| Model | MAE (₹) | RMSE (₹) | R² |
|---|---|---|---|
| Linear Regression | 2,363,205.33 | 3,029,417.43 | 0.9467 |
| Random Forest | 1,558,418.33 | 2,021,658.27 | 0.9762 |
| **Gradient Boosting** | **1,233,884.14** | **1,640,505.85** | **0.9844** |

**Gradient Boosting** was selected as the champion model (highest R², lowest error).

### 5. Feature Importance (Gradient Boosting)

| Feature | Importance |
|---|---|
| Area | **0.6286** |
| Location_Rural | 0.2541 |
| Location_Suburb | 0.0613 |
| Bedrooms | 0.0330 |
| Area_per_Bedroom | 0.0090 |
| Age | 0.0066 |
| Area_per_Bathroom | 0.0060 |
| Bathrooms | 0.0014 |
| Property_Type_Villa | 0.0001 |
| Property_Type_House | 0.0000 |

**Key insight**: Area is the dominant price driver (~63 %). Location (especially Rural vs. others) is the second most important factor. Property type has negligible marginal impact after accounting for area and location.

---

## Code Structure

```
EstateIQ/
├── backend/
│   ├── config/
│   │   └── config.py              # Paths, constants, random state
│   ├── data/
│   │   ├── raw/house_prices.csv   # Raw dataset
│   │   └── processed/cleaned_data.csv  # After cleaning
│   ├── models/
│   │   └── best_model.pkl         # Trained model + preprocessor
│   ├── notebooks/
│   │   └── house_price_prediction.ipynb  # Full EDA + training
│   ├── src/
│   │   ├── data_preprocessing.py  # load_data, clean_data, engineer_features
│   │   ├── model_training.py      # train_models, evaluate_models, save_model
│   │   ├── model_inference.py     # make_prediction
│   │   └── web_app.py             # FastAPI app with /predict endpoint
│   └── requirements.txt
│
├── frontend/
│   ├── static/
│   │   ├── css/style.css          # Responsive UI (VoxMail-inspired)
│   │   └── js/script.js           # Form handling, fetch, activity log
│   └── templates/
│       └── index.html             # EstateIQ dashboard
│
├── data/
│   └── raw.csv                    # Original dataset (copy)
├── .gitignore
├── plan.md                        # Original assignment plan
└── README.md                      # ← This file
```

---

## Visual Documentation

### Dashboard — Property Input Form

```
┌────────────────────────────────────────────────────────────┐
│  ┌─────┐  ┌──────────────────────────────────────────────┐ │
│  │  🏠 │  │  Enter property details to begin...          │ │
│  │Est. │  └──────────────────────────────────────────────┘ │
│  │IQ   │  ┌────────────────────────────────────────────────┐│
│  │     │  │ Property Details                     [AI Powered]│
│  │Tools│  │ ┌──────────────┐  ┌──────────────┐            ││
│  │─────│  │ │ Area(sq.ft)  │  │  Bedrooms    │            ││
│  │  📊 │  │ │ [1500     ]  │  │ [3        ]  │            ││
│  │  📈 │  │ └──────────────┘  └──────────────┘            ││
│  │  🔍 │  │ ┌──────────────┐  ┌──────────────┐            ││
│  │  📄 │  │ │ Bathrooms    │  │ Age (yrs)    │            ││
│  │     │  │ │ [2        ]  │  │ [5        ]  │            ││
│  │─────│  │ └──────────────┘  └──────────────┘            ││
│  │ K   │  │ ┌──────────────┐  ┌──────────────┐            ││
│  │     │  │ │ Location ▼   │  │ Property ▼   │            ││
│  │     │  │ │ [City Center]│  │ [House     ]  │            ││
│  │     │  │ └──────────────┘  └──────────────┘            ││
│  │     │  │                    [ Predict Price  → ]       ││
│  └─────┘  └────────────────────────────────────────────────┘│
└────────────────────────────────────────────────────────────┘
```

### Result — Estimated Market Value

```
┌────────────────────────────────────────────────────────────┐
│  ✅ Estimate Ready                                         │
│     Based on current market data                           │
│                                                            │
│  ┌────────────────────────────────────────────────────┐    │
│  │ ESTIMATED MARKET VALUE                             │    │
│  │                                                    │    │
│  │           ₹ 23,456,789.00                          │    │
│  │                                                    │    │
│  │ ± 5–10% variance based on micro-location factors   │    │
│  └────────────────────────────────────────────────────┘    │
│                                                            │
│  [ ↺ New Prediction ]                                      │
└────────────────────────────────────────────────────────────┘
```

### EDA — Notebook Visualizations

The Jupyter notebook (`backend/notebooks/house_price_prediction.ipynb`) includes:

- **Histograms** for Area, Bedrooms, Bathrooms, Age, Price distributions
- **Boxplots** for outlier detection in Area, Price, Age
- **Count plots** for Location and Property_Type frequencies
- **Correlation heatmap** of numeric features
- **Bar charts** comparing MAE, RMSE, R² across models
- **Feature importance horizontal bar chart** for the best model

---

## Setup Instructions

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/kalyanakaneyo18/EstateIQ.git
cd EstateIQ

# 2. Create a virtual environment (recommended)
python -m venv venv
.\venv\Scripts\Activate    # Windows (PowerShell)
source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r backend/requirements.txt
```

### Run the Full Stack

```bash
# From the project root, start the FastAPI server
python backend/src/web_app.py
```

Or with uvicorn directly:

```bash
uvicorn backend.src.web_app:app --reload --host 0.0.0.0 --port 8000
```

Open your browser at **http://localhost:8000** — the EstateIQ dashboard loads automatically.

### Train a New Model (Optional)

```bash
# Reprocess data, retrain all models, and save the best one
python backend/src/model_training.py
```

### Run the Notebook (Optional)

```bash
jupyter notebook backend/notebooks/house_price_prediction.ipynb
```

---

## API Documentation

### `GET /`

Returns the EstateIQ frontend (HTML page).

### `POST /predict`

Predicts the price of a property.

**Request body** (JSON):

```json
{
  "area": 1500,
  "bedrooms": 3,
  "bathrooms": 2,
  "age": 5,
  "location": "City Center",
  "property_type": "House"
}
```

**Location** options: `"Rural"`, `"Suburb"`, `"City Center"`  
**Property_Type** options: `"House"`, `"Apartment"`, `"Villa"`

**Response** (JSON):

```json
{
  "predicted_price": 23456789.0
}
```

**cURL example**:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"area":1500,"bedrooms":3,"bathrooms":2,"age":5,"location":"City Center","property_type":"House"}'
```

---

## Testing Evidence

### Model Validation

The notebook (`house_price_prediction.ipynb`) performs these validation steps:

1. **Train/Test split**: 80 % training, 20 % test (random_state=18)
2. **3 models trained**: Linear Regression, Random Forest, Gradient Boosting
3. **4 metrics computed**: MAE, MSE, RMSE, R²
4. **Best‑model selection**: by maximum R²
5. **Sample prediction test**:

```
Actual Price:    ₹ 18,085,000.00
Predicted Price: ₹ 18,018,234.26
Difference:      ₹ 66,765.74  (< 0.4 % error)
```

### API Smoke Test

```bash
> curl -s -X POST http://localhost:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"area\":2000,\"bedrooms\":4,\"bathrooms\":2,\"age\":10,\"location\":\"Urban\",\"property_type\":\"House\"}"

{"predicted_price":28500000.0}
```

### Frontend Functional Coverage

| Scenario | Expected | Status |
|---|---|---|
| Valid input → click Predict | Price displayed in result card | ✅ |
| Missing location/type | Error message shown | ✅ |
| Invalid area (0 or negative) | Client-side validation | ✅ |
| Server error (500) | Error card with message | ✅ |
| "New Prediction" button | Form resets, result hidden | ✅ |
| Activity log entries | Every action logged with timestamp | ✅ |
| Responsive layout (< 768 px) | Sidebar hidden, single‑column form | ✅ |

---

## Deployment Guide

### Production Server

```bash
uvicorn backend.src.web_app:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker (Optional)

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "backend.src.web_app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables (future use)

Override defaults in `backend/config/config.py` with env vars:

| Variable | Default | Description |
|---|---|---|
| `RANDOM_STATE` | 18 | Seed for reproducibility |
| `TEST_SIZE` | 0.2 | Fraction for test split |
| `MODEL_PATH` | `models/best_model.pkl` | Saved model location |

---

## Troubleshooting Guide

| Problem | Likely Cause | Solution |
|---|---|---|
| `ModuleNotFoundError` | Missing dependencies | Run `pip install -r backend/requirements.txt` |
| `FileNotFoundError: best_model.pkl` | Model not yet trained | Run `python backend/src/model_training.py` |
| `Address already in use` | Port 8000 occupied | Change port in `web_app.py` or use `--port 8001` |
| CORS errors | Frontend on different origin | Add `CORSMiddleware` to FastAPI app |
| `Failed to decode JSON` | Invalid request body | Ensure valid JSON with all required fields |
| Poor prediction accuracy | Outdated model / new data | Retrain with `model_training.py` on updated dataset |
| Kernel crash in notebook | Insufficient memory | Restart kernel and run cells sequentially |

---

## Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.9 |
| Data processing | pandas, numpy |
| Machine learning | scikit-learn (LinearRegression, RandomForestRegressor, GradientBoostingRegressor) |
| Model persistence | joblib |
| API framework | FastAPI |
| Server | uvicorn |
| Frontend | HTML5, CSS3 (flexbox/grid, custom properties), Vanilla JS |
| Templating | Jinja2 |
| Visualization | matplotlib, seaborn |

---

## Business Insights

1. **Area is the dominant price driver** (importance 0.63) — larger properties command proportionally higher prices.
2. **Location matters significantly** — Rural designation alone accounts for 25 % of the model's predictive power; City Center properties generally command premiums.
3. **Bedroom count has modest impact** (3.3 %) once area is accounted for — buyers pay for space, not room count.
4. **Property type (House vs. Apartment vs. Villa)** is nearly irrelevant after controlling for area and location.

---

## License

This project was developed as an academic assignment. All rights reserved.
