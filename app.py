"""API d'inférence Fashion-MNIST. Lancer : python -m uvicorn app:app."""
from pathlib import Path
import time

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

app = FastAPI(
    title="Fashion-MNIST API",
    description="API de classification des vêtements Fashion-MNIST",
    version="1.0",
)
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "fashion_mnist.keras"
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

# Un seul chargement par processus serveur, sans restaurer l'optimiseur.
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
if model.input_shape != (None, 28, 28) or model.output_shape != (None, 10):
    raise RuntimeError("Le modèle ne respecte pas le contrat Fashion-MNIST.")


class ImageInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pixels: list[float] = Field(
        description="784 nombres finis, en ordre ligne par ligne, dans [0, 255]. "
        "Si le maximum dépasse 1, l'image est divisée par 255."
    )


class PredictionOutput(BaseModel):
    class_id: int = Field(ge=0, le=9)
    class_name: str
    confidence: float = Field(ge=0, le=1)
    latency_ms: float = Field(ge=0)


@app.get("/")
def home():
    return {"message": "Bienvenue dans Fashion-MNIST API"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_name": MODEL_PATH.name,
        "input_shape": str(model.input_shape),
        "output_shape": str(model.output_shape),
    }


@app.post("/predict", response_model=PredictionOutput,
          responses={400: {"description": "Nombre de pixels ou valeurs invalides"}})
def predict(data: ImageInput):
    if len(data.pixels) != 784:
        raise HTTPException(400, "L'entrée doit contenir exactement 784 pixels.")
    # Vérification avant float32 pour éviter un débordement à la conversion.
    if any(not np.isfinite(value) or value < 0 or value > 255
           for value in data.pixels):
        raise HTTPException(400, "Les pixels doivent être des nombres finis entre 0 et 255.")
    image = np.asarray(data.pixels, dtype=np.float32).reshape(1, 28, 28)
    if image.max() > 1:
        image = image / np.float32(255.0)

    # Inclut le modèle, softmax et la matérialisation du résultat NumPy.
    # Exclut validation, prétraitement, sérialisation JSON et transport HTTP.
    start = time.perf_counter()
    logits = model(image, training=False)
    probabilities = tf.nn.softmax(logits, axis=-1).numpy()[0]
    latency_ms = (time.perf_counter() - start) * 1000
    class_id = int(np.argmax(probabilities))
    return {
        "class_id": class_id,
        "class_name": CLASS_NAMES[class_id],
        "confidence": float(probabilities[class_id]),
        "latency_ms": round(latency_ms, 4),
    }
