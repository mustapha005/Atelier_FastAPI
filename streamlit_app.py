"""Interface française : python -m streamlit run streamlit_app.py."""
import json
import os
from pathlib import Path

import numpy as np
import requests
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Fashion Lab", page_icon="👕", layout="wide")
st.title("Fashion Lab")
st.caption("Classification de vêtements · Mustapha Aarab et Aya Agrigah")
st.write("Choisissez un exemple ou importez une image pour reconnaître son type de vêtement.")
API_URL = os.environ.get("FASHION_API_URL", "").rstrip("/")


@st.cache_resource
def local_backend():
    # Réutilise exactement le contrat et le modèle de l'API, sans serveur HTTP.
    from app import ImageInput, predict
    return ImageInput, predict

with st.sidebar:
    st.header("Le service")
    try:
        if API_URL:
            response = requests.get(API_URL + "/health", timeout=5)
            response.raise_for_status()
            health = response.json()
            if not health.get("model_loaded"):
                raise ValueError("Modèle indisponible")
            st.success("API connectée")
            st.caption(health.get("model_name", "Fashion-MNIST"))
        else:
            st.info("Mode autonome · modèle embarqué")
    except (requests.RequestException, ValueError):
        st.warning("API indisponible")
        st.code("python -m uvicorn app:app --port 8000")
    st.divider()
    st.write("10 classes · Images de 28 × 28 pixels")
    st.caption("Le modèle est entraîné sur des vêtements isolés, en niveaux de gris, sur fond noir.")

predict_tab, results_tab, about_tab = st.tabs(["Prédire", "Résultats du TP", "À propos"])
with predict_tab:
    source = st.radio("Image à analyser", ["Exemple Fashion-MNIST", "Importer une image"], horizontal=True)
    pixels = None
    true_class = None
    if source == "Exemple Fashion-MNIST":
        samples = np.load(ROOT / "examples.npz", allow_pickle=False)
        index = st.selectbox("Exemple du jeu de test", range(20), format_func=lambda i: f"Image {i}")
        pixels = samples["images"][index]
        names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
        true_class = names[int(samples["labels"][index])]
    else:
        uploaded = st.file_uploader("Image PNG ou JPEG", type=["png", "jpg", "jpeg"])
        invert = st.checkbox("Inverser les couleurs pour obtenir un fond noir")
        st.caption("L’image sera convertie en niveaux de gris puis redimensionnée à 28 × 28. Les photos ordinaires peuvent donner des résultats peu fiables.")
        if uploaded is not None:
            try:
                if uploaded.size > 5 * 1024 * 1024:
                    raise ValueError("La taille maximale est de 5 Mo.")
                with Image.open(uploaded) as image:
                    image = ImageOps.exif_transpose(image).convert("L")
                    image = image.resize((28, 28), Image.Resampling.LANCZOS)
                    if invert:
                        image = ImageOps.invert(image)
                    pixels = np.asarray(image)
            except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
                st.error(f"Impossible de lire cette image : {exc}")
    left, right = st.columns([1, 2], gap="large")
    with left:
        if pixels is not None:
            st.image(Image.fromarray(pixels).resize((280, 280), Image.Resampling.NEAREST), caption="Image envoyée au modèle", width=280)
            if true_class:
                st.caption(f"Classe réelle : {true_class}")
    with right:
        st.subheader("Résultat de la classification")
        if st.button("Analyser l’image", type="primary", disabled=pixels is None):
            try:
                with st.spinner("Analyse en cours…"):
                    payload = {"pixels": pixels.reshape(-1).tolist()}
                    if API_URL:
                        response = requests.post(API_URL + "/predict", json=payload, timeout=60)
                        response.raise_for_status()
                        result = response.json()
                    else:
                        ImageInput, predict = local_backend()
                        result = predict(ImageInput(**payload))
                st.success(result["class_name"])
                confidence, latency = st.columns(2)
                confidence.metric("Confiance", f"{result['confidence']:.2%}")
                latency.metric("Latence d’inférence", f"{result['latency_ms']:.2f} ms")
                st.progress(float(result["confidence"]))
                if true_class:
                    if result["class_name"] == true_class:
                        st.success("La prédiction correspond à la classe réelle.")
                    else:
                        st.warning(f"Prédiction incorrecte. Classe réelle : {true_class}.")
                st.caption("La confiance est un score du modèle, pas une garantie de justesse.")
                with st.expander("Réponse JSON"):
                    st.json(result)
                st.download_button("Télécharger le résultat", json.dumps(result, indent=2), "prediction.json", "application/json")
            except (requests.RequestException, ValueError, KeyError, ImportError, OSError) as exc:
                st.error(f"La prédiction a échoué. Vérifiez le modèle et les dépendances, ou le serveur en mode API. {exc}")

with results_tab:
    data = json.loads((ROOT / "resultats/resultats.json").read_text(encoding="utf-8"))
    st.subheader("Campagne de référence du 19 novembre 2026")
    st.caption("Résultats enregistrés sur les 20 premières images, après échauffement. Ces mesures ne sont pas recalculées à l’ouverture de la page.")
    a, b, c = st.columns(3)
    a.metric("Prédictions correctes", f"{data['correct']} / {data['count']}")
    b.metric("Taux de réussite", f"{data['accuracy']:.0%}")
    c.metric("Latence moyenne", f"{data['mean_latency_ms']:.2f} ms")
    st.dataframe([{"Image": r["image"], "Classe réelle": r["true_class"], "Prédiction": r["class_name"], "Confiance": round(r["confidence"],4), "Latence ms": r["latency_ms"], "Correct": r["correct"]} for r in data["predictions"]], hide_index=True)
    st.image(str(ROOT / "captures/predictions_20.png"))
    st.download_button("Télécharger le tableau CSV", (ROOT / "resultats/predictions.csv").read_bytes(), "predictions.csv", "text/csv")

with about_tab:
    st.subheader("Du modèle à l’API")
    st.write("L’interface utilise le modèle embarqué en mode autonome. Si FASHION_API_URL est configurée, elle envoie 784 pixels à FastAPI par HTTP. Les deux modes utilisent le même prétraitement et la même fonction de prédiction.")
    st.write("Architecture : Flatten → Dense 128 avec ReLU → Dense 10. Le modèle contient 101 770 paramètres.")
    report = ROOT / "rapport/Rapport_FastAPI_Aarab_Agrigah.docx"
    st.download_button("Télécharger le rapport Word", report.read_bytes(), report.name, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
