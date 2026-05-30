import contextlib
import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.data_utils import load_trained_model, prediction_label, prepare_image_bytes


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "final_best_model.keras"

app = FastAPI(title="EchoMind", version="1.0.0")
app.mount("/static", StaticFiles(directory=BASE_DIR / "app" / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "app" / "templates")

model = None


@app.on_event("startup")
def load_model_on_startup():
    global model
    with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
        model = load_trained_model(MODEL_PATH)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        image = prepare_image_bytes(image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Could not read this image.") from exc

    with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
        score = float(model.predict(image, verbose=0)[0][0])

    return {
        "prediction": prediction_label(score),
        "sick_probability": round(score, 6),
        "normal_probability": round(1.0 - score, 6),
        "threshold": 0.5,
    }
