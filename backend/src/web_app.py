from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.model_inference import make_prediction

app = FastAPI(title="House Price Prediction API")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "frontend" / "templates"))
static_dir = BASE_DIR / "frontend" / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


class PredictionInput(BaseModel):
    area: float
    bedrooms: int
    bathrooms: int
    age: int
    location: str
    property_type: str


class PredictionOutput(BaseModel):
    predicted_price: float


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    price = make_prediction(
        area=input_data.area,
        bedrooms=input_data.bedrooms,
        bathrooms=input_data.bathrooms,
        age=input_data.age,
        location=input_data.location,
        property_type=input_data.property_type,
    )
    return PredictionOutput(predicted_price=price)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
