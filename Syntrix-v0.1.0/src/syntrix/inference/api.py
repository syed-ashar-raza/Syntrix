from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, make_asgi_app

from syntrix.models import build_model
from syntrix.utils.device import get_device


device = get_device()
model = None

REQUESTS = Counter(
    "syntrix_inference_requests_total",
    "Inference requests",
)

LATENCY = Histogram(
    "syntrix_inference_latency_seconds",
    "Inference latency",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    artifact = Path("artifacts/model.pt")

    if artifact.exists():
        checkpoint = torch.load(
            artifact,
            map_location=device,
            weights_only=True,
        )

        model = build_model(checkpoint["model_name"]).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()

    yield


app = FastAPI(
    title="Syntrix Inference API",
    version="0.1.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    values: list[float]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "device": str(device),
    }


@app.post("/v1/predict")
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifact not found. Train the model first.",
        )

    if len(req.values) != 784:
        raise HTTPException(
            status_code=400,
            detail="values must contain exactly 784 grayscale pixels.",
        )

    REQUESTS.inc()

    with LATENCY.time():
        x = torch.tensor(
            req.values,
            dtype=torch.float32,
            device=device,
        ).reshape(1, 1, 28, 28)

        x = (x - 0.2860) / 0.3530

        with torch.inference_mode():
            probabilities = model(x).softmax(-1)[0]
            predicted_class = int(probabilities.argmax())

    return {
        "class_id": predicted_class,
        "confidence": float(probabilities[predicted_class]),
    }


app.mount("/metrics", make_asgi_app())
