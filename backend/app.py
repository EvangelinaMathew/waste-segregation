import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import io, json
from pathlib import Path
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import tensorflow as tf

MODEL_DIR = Path(__file__).resolve().parent / "model"
IMG_SIZE = (224, 224)
THRESHOLD = 0.6           # custom model must be at least this sure
GENERAL_THRESHOLD = 0.20  # fallback model (1000 classes spreads confidence thinner)

GREEN = ("Green", "Wet / Biodegradable", "Compostable. Put it in the green bin.")
BLUE  = ("Blue", "Dry / Recyclable", "Recyclable. Keep it clean and dry, blue bin.")
RED   = ("Red", "Hazardous", "Hazardous. Never mix with normal waste; hand over at a collection point.")
REWST = ("Red", "Hazardous / E-waste", "E-waste. Give it to an e-waste drive or recycler.")
BLACK = ("Black", "Non-recyclable", "Non-recyclable. Put it in the black general waste bin.")

# Layer 1: classes of YOUR trained model (names = folder names in data/processed)
BIN = {
    "plastic_bottle": BLUE, "paper": BLUE, "glass": BLUE, "metal_can": BLUE,
    "plastic_cup": BLUE,
    "food_waste": GREEN, "egg_shells_tea": GREEN, "yard_waste": GREEN,
    "battery": RED, "paint_chemical": RED, "ewaste": REWST,
    "plastic_bag": BLACK, "thermocol": BLACK, "diaper_sanitary": BLACK,
    "ceramic": BLACK, "pen": BLACK, "paper_cup": BLACK,
}

# Layer 2: ImageNet names (general fallback). Unused names are harmless.
IMAGENET_BIN = {
    "ballpoint": BLACK, "fountain_pen": BLACK, "rubber_eraser": BLACK,
    "water_bottle": BLUE, "pop_bottle": BLUE, "beer_bottle": BLUE,
    "wine_bottle": BLUE, "water_jug": BLUE, "pill_bottle": BLUE,
    "carton": BLUE, "envelope": BLUE,
    "plastic_bag": BLACK, "coffee_mug": BLACK, "toilet_tissue": BLACK, "diaper": BLACK,
    "cellular_telephone": REWST, "laptop": REWST, "notebook": REWST,
    "mouse": REWST, "remote_control": REWST, "ipod": REWST,
    "hand-held_computer": REWST, "computer_keyboard": REWST, "joystick": REWST,
    "digital_watch": REWST,
    "banana": GREEN, "orange": GREEN, "lemon": GREEN, "Granny_Smith": GREEN,
    "strawberry": GREEN, "pineapple": GREEN, "broccoli": GREEN,
    "cucumber": GREEN, "head_cabbage": GREEN, "bell_pepper": GREEN,
    "corn": GREEN, "mushroom": GREEN,
}

app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model(MODEL_DIR / "waste_classifier.keras")
labels = json.load(open(MODEL_DIR / "labels.json"))
general = tf.keras.applications.MobileNetV2(weights="imagenet")   # downloads ~14 MB first time
model.predict(np.zeros((1, 224, 224, 3), dtype="float32"), verbose=0)   # warm-up
general.predict(np.zeros((1, 224, 224, 3), dtype="float32"), verbose=0)
print("Models ready. Classes:", labels, flush=True)


def preprocess_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB").resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)   # same as training
    return np.expand_dims(arr, 0)


def none_result(conf):
    return jsonify({"item": "none", "bin": None, "color": None,
                    "tip": None, "confidence": round(conf, 4)})


@app.route("/health")
def health():
    return jsonify({"status": "ok", "classes": labels})


@app.route("/classify", methods=["POST"])
def classify():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Use field name 'image'."}), 400
    try:
        x = preprocess_image(request.files["image"].read())
    except Exception as exc:
        return jsonify({"error": f"Could not decode image: {exc}"}), 422

    # Layer 1: your trained model
    probs = model.predict(x, verbose=0)[0]
    i = int(np.argmax(probs))
    item, conf = labels[i], float(probs[i])
    if item != "background" and conf >= THRESHOLD and item in BIN:
        color, kind, tip = BIN[item]
        return jsonify({"item": item, "bin": kind, "color": color, "tip": tip,
                        "confidence": round(conf, 4), "source": "custom"})

    # Layer 2: general ImageNet model
    gp = general.predict(x, verbose=0)
    top = tf.keras.applications.mobilenet_v2.decode_predictions(gp, top=3)[0]
    print("general top3:", [(n, round(float(p), 2)) for _, n, p in top], flush=True)
    for _, name, p in top:
        if name in IMAGENET_BIN and float(p) >= GENERAL_THRESHOLD:
            color, kind, tip = IMAGENET_BIN[name]
            return jsonify({"item": name.replace("_", " "), "bin": kind, "color": color,
                            "tip": tip, "confidence": round(float(p), 4),
                            "source": "general"})
    return none_result(conf)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False)