"""Tests HTTP réels et figures : python test_api.py --count 20.

Démarrer auparavant : python -m uvicorn app:app --host 127.0.0.1 --port 8000
Les résultats sont écrits dans resultats/ et les figures dans captures/.
"""
import argparse
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import platform
import statistics
import time

ROOT = Path(__file__).resolve().parent
# Cache local au projet : aucun changement du cache global de l'utilisateur.
os.environ.setdefault("KERAS_HOME", str(ROOT / ".keras"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
import tensorflow as tf

CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def check_prediction(result):
    assert set(result) == {"class_id", "class_name", "confidence", "latency_ms"}, result
    assert type(result["class_id"]) is int and 0 <= result["class_id"] <= 9
    assert result["class_name"] == CLASS_NAMES[result["class_id"]]
    assert math.isfinite(result["confidence"]) and 0 <= result["confidence"] <= 1
    assert math.isfinite(result["latency_ms"]) and result["latency_ms"] >= 0


def run(base_url, count, show=False):
    result_dir = ROOT / "resultats"
    capture_dir = ROOT / "captures"
    result_dir.mkdir(exist_ok=True)
    capture_dir.mkdir(exist_ok=True)
    base_url = base_url.rstrip("/")
    session = requests.Session()
    health_response = session.get(base_url + "/health", timeout=30)
    health_response.raise_for_status()
    health = health_response.json()
    assert health["status"] == "healthy" and health["model_loaded"] is True
    assert health["input_shape"] == "(None, 28, 28)"
    assert health["output_shape"] == "(None, 10)"
    home = session.get(base_url + "/", timeout=30)
    home.raise_for_status()
    assert "message" in home.json()
    schema = session.get(base_url + "/openapi.json", timeout=30)
    schema.raise_for_status()
    assert "post" in schema.json()["paths"]["/predict"]

    (_, _), (images, labels) = tf.keras.datasets.fashion_mnist.load_data()
    if not 20 <= count <= len(images):
        raise ValueError("Choisir un nombre d'images entre 20 et 10000.")

    # Un échauffement explicite est exclu du tableau et des statistiques.
    warmup = session.post(base_url + "/predict",
                          json={"pixels": images[0].reshape(-1).tolist()}, timeout=60)
    warmup.raise_for_status()
    check_prediction(warmup.json())
    rows = []
    print("Image | Classe réelle | Classe prédite | Confiance | Latence | Résultat")
    for index in range(count):
        payload = {"pixels": images[index].reshape(-1).tolist()}
        start = time.perf_counter()
        response = session.post(base_url + "/predict", json=payload, timeout=30)
        http_ms = (time.perf_counter() - start) * 1000
        response.raise_for_status()
        result = response.json()
        check_prediction(result)
        true_id = int(labels[index])
        row = {"image": index, "true_id": true_id, "true_class": CLASS_NAMES[true_id],
               **result, "http_ms": http_ms, "correct": result["class_id"] == true_id}
        rows.append(row)
        print(f"{index:2} | {row['true_class']:11} | {row['class_name']:11} | "
              f"{row['confidence']:.4f} | {row['latency_ms']:.4f} ms | "
              f"{'Correct' if row['correct'] else 'Incorrect'}")

    invalid = session.post(base_url + "/predict", json={"pixels": [0, 128, 255]}, timeout=30)
    assert invalid.status_code == 400, invalid.text
    assert invalid.json() == {"detail": "L'entrée doit contenir exactement 784 pixels."}
    checks = [{"name": "3 pixels", "status": invalid.status_code, "passed": True}]
    for name, payload, expected in [
        ("785 pixels", {"pixels": [0] * 785}, 400),
        ("valeur négative", {"pixels": [-1] + [0] * 783}, 400),
        ("valeur supérieure à 255", {"pixels": [256] + [0] * 783}, 400),
        ("champ absent", {}, 422),
        ("valeur non numérique", {"pixels": ["abc"] + [0] * 783}, 422),
    ]:
        response = session.post(base_url + "/predict", json=payload, timeout=30)
        assert response.status_code == expected, (name, response.text)
        checks.append({"name": name, "status": response.status_code, "passed": True})

    normalized = session.post(base_url + "/predict", json={
        "pixels": (images[0].astype(np.float32) / 255).reshape(-1).tolist()}, timeout=30)
    normalized.raise_for_status()
    check_prediction(normalized.json())
    assert normalized.json()["class_id"] == rows[0]["class_id"]
    assert abs(normalized.json()["confidence"] - rows[0]["confidence"]) < 1e-5
    checks.append({"name": "équivalence pixels bruts et normalisés", "status": 200, "passed": True})
    summary = {
        "executed_at": datetime.now().astimezone().isoformat(),
        "base_url": base_url, "python": platform.python_version(),
        "tensorflow": tf.__version__, "platform": platform.platform(),
        "count": count, "correct": sum(row["correct"] for row in rows),
        "accuracy": sum(row["correct"] for row in rows) / count,
        "mean_latency_ms": statistics.mean(row["latency_ms"] for row in rows),
        "min_latency_ms": min(row["latency_ms"] for row in rows),
        "max_latency_ms": max(row["latency_ms"] for row in rows),
        "mean_http_ms": statistics.mean(row["http_ms"] for row in rows),
        "warmup_latency_ms": warmup.json()["latency_ms"],
        "warmup_excluded": True, "checks": checks,
        "invalid_response": {"status": invalid.status_code, "body": invalid.json()},
        "health": health, "predictions": rows,
    }
    (result_dir / "resultats.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (result_dir / "image_0_payload.json").write_text(json.dumps({"pixels": images[0].reshape(-1).tolist()}), encoding="utf-8")
    with (result_dir / "predictions.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Les figures utilisent exclusivement les réponses reçues par HTTP.
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(images[0], cmap="gray", vmin=0, vmax=255)
    ax.set_axis_off()
    ax.set_title(f"Image 0 · Réel : {rows[0]['true_class']}\n"
                 f"Prédit : {rows[0]['class_name']} · Confiance : {rows[0]['confidence']:.2%}\n"
                 f"Latence : {rows[0]['latency_ms']:.4f} ms", fontsize=12)
    fig.tight_layout()
    fig.savefig(capture_dir / "image_testee.png", dpi=180)
    plt.close(fig)
    columns = 5
    nrows = math.ceil(count / columns)
    fig, axes = plt.subplots(nrows, columns, figsize=(13, nrows * 2.8), squeeze=False)
    for index, ax in enumerate(axes.flat):
        ax.set_axis_off()
        if index >= count:
            continue
        row = rows[index]
        ax.imshow(images[index], cmap="gray", vmin=0, vmax=255)
        color = "#1E8449" if row["correct"] else "#C0392B"
        ax.set_title(f"#{index} Réel : {row['true_class']}\nPrédit : {row['class_name']}\n"
                     f"{row['confidence']:.1%} | {row['latency_ms']:.2f} ms", fontsize=9, color=color)
    fig.tight_layout()
    fig.savefig(capture_dir / "predictions_20.png", dpi=160)
    plt.close(fig)
    print(f"\nRéussite : {summary['correct']}/{count} ({summary['accuracy']:.1%})")
    print(f"Latence moyenne : {summary['mean_latency_ms']:.4f} ms (hors échauffement)")
    print(f"Entrée incorrecte : HTTP {invalid.status_code}, {invalid.json()}")
    print(f"Contrôles supplémentaires : {len(checks)}/{len(checks)} réussis")
    if show:
        from PIL import Image
        Image.open(capture_dir / "image_testee.png").show()
        Image.open(capture_dir / "predictions_20.png").show()
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--show", action="store_true", help="Ouvrir les figures après les tests")
    args = parser.parse_args()
    try:
        run(args.base_url, args.count, args.show)
    except requests.RequestException as exc:
        raise SystemExit(f"Échec HTTP. Vérifier que le serveur est lancé : {exc}") from exc
