from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from contextlib import asynccontextmanager

class TextRequest(BaseModel):
    text: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, vectorizer
    try:
        model = joblib.load("classifier.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
    except Exception as e:
        raise RuntimeError(f"Model yüklenemedi: {e}")
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/predict")
def predict(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Boş metin gönderilemez.")
    features = vectorizer.transform([request.text])
    pred_class = model.predict(features)[0]
    proba = np.max(model.predict_proba(features))
    return {
        "class": pred_class,
        "probability": round(float(proba), 4)
    }
