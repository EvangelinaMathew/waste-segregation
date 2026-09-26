"""
app.py — Flask REST API
========================
POST /classify   — accepts a multipart image upload, preprocesses it,
                   runs it through the saved MobileNetV2 model, and
                   returns JSON {"category": "...", "confidence": 0.XX}

GET  /health     — simple liveness check

Run:
    python backend/app.py
"""

import os
import io
from pathlib import Path

import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import tensorflow as tf

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).resolve().parent / "model" / "waste_classifier.h5"
IMG_SIZE   = (224, 224)
CLASSES    = ["Biodegradable", "Hazardous", "Non-Recyclable", "Recyclable"]

# Bin colour / disposal hint returned alongside the category
BIN_INFO = {
    "Biodegradable": {
        "bin_colour": "Green",
        "bin_label": "Compost / Organic Bin",
        "tip": "Compostable — place in the green organic bin.",
    },
    "Recyclable": {
        "bin_colour": "Blue",
        "bin_label": "Recycling Bin",
        "tip": "Recyclable — rinse and place in the blue recycling bin.",
    },
    "Non-Recyclable": {
        "bin_colour": "Black",
        "bin_label": "General Waste Bin",
        "tip": "Non-recyclable — place in the black general waste bin.",
    },
    "Hazardous": {
        "bin_colour": "Red",
        "bin_label": "Hazardous Waste Bin",
        "tip": "Hazardous — take to a designated hazardous waste facility.",
    },
}

# ── App & model ───────────────────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")

app = Flask(__name__)
CORS(app)   # allow requests from file:// (origin "null") and any other origin

# Load model once at startup
_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Run backend/train_model.py first."
            )
        print(f"Loading model from {MODEL_PATH} …", flush=True)
        _model = tf.keras.models.load_model(str(MODEL_PATH))
        print("Model loaded.", flush=True)
    return _model


# ── Preprocessing helper ──────────────────────────────────────────────────────
def preprocess_image(file_bytes: bytes) -> np.ndarray:
    """
    Mirror the same pipeline used in preprocess.py + train_model.py:
      1. Open & convert to RGB
      2. Resize to 224×224
      3. Apply MobileNetV2 preprocess_input (scales [0,255] → [-1,1])
    Returns a (1, 224, 224, 3) float32 array ready for model.predict().
    """
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)   # add batch dim


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": _model is not None})


@app.route("/classify", methods=["POST"])
def classify():
    # ── Validate upload ───────────────────────────────────────────────────────
    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Use field name 'image'."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    allowed = {"jpg", "jpeg", "png", "webp", "bmp", "gif"}
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in allowed:
        return jsonify({"error": f"Unsupported file type: .{ext}"}), 415

    # ── Read & preprocess ─────────────────────────────────────────────────────
    try:
        file_bytes = file.read()
        img_tensor = preprocess_image(file_bytes)
    except Exception as exc:
        return jsonify({"error": f"Could not decode image: {exc}"}), 422

    # ── Inference ─────────────────────────────────────────────────────────────
    try:
        model  = get_model()
        preds  = model.predict(img_tensor, verbose=0)[0]   # shape: (4,)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        return jsonify({"error": f"Inference error: {exc}"}), 500

    # ── Build response ────────────────────────────────────────────────────────
    top_idx    = int(np.argmax(preds))
    category   = CLASSES[top_idx]
    confidence = float(preds[top_idx])

    # All class probabilities (for a nice debug view in the frontend)
    all_scores = {cls: round(float(preds[i]), 4) for i, cls in enumerate(CLASSES)}

    response = {
        "category":   category,
        "confidence": round(confidence, 4),
        **BIN_INFO[category],
        "all_scores": all_scores,
    }
    return jsonify(response)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Pre-load the model so the first request isn't slow
    try:
        get_model()
    except FileNotFoundError as e:
        print(f"\n⚠️  WARNING: {e}")
        print("   The server will start, but /classify will return 503 until the model exists.\n")

    app.run(host="0.0.0.0", port=5050, debug=False)
