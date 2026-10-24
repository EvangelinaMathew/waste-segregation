import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import io, json
from pathlib import Path
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageOps
from ultralytics import YOLO
import tensorflow as tf

MODEL_DIR = Path(__file__).resolve().parent / "model"
IMG_SIZE = (224, 224)
THRESHOLD = 0.7           # custom model must be at least this sure
GENERAL_THRESHOLD = 0.20  # fallback for unknown items
OVERRIDE_THRESHOLD = 0.25 # pens and electronics: general model gets priority

GREEN = ("Green", "Wet / Biodegradable")
BLUE  = ("Blue", "Dry / Recyclable")
RED   = ("Red", "Hazardous")
REWST = ("Red", "E-waste (Hazardous)")
BLACK = ("Black", "Non-recyclable")

# Guidelines: by bin colour, with item-specific ones where it matters
GUIDE = {
    "Green": ["Remove any plastic or packaging.",
              "Put it in the green bin, or compost it at home.",
              "Keep it away from dry waste."],
    "Blue":  ["Empty it and give it a quick rinse.",
              "Let it dry and flatten it if possible.",
              "Put it in the blue bin."],
    "Red":   ["Never put it in a normal bin.",
              "Keep it away from water and heat.",
              "Hand it over at a hazardous waste collection point."],
    "Black": ["It cannot be recycled.",
              "Wrap it if it is sharp or dirty.",
              "Put it in the black general waste bin."],
}
EWASTE_GUIDE = ["Never put it in a normal bin.",
                "Take out the batteries if you can.",
                "Give it to an e-waste drive or an authorised recycler."]
ITEM_GUIDE = {
    "battery": ["Never throw batteries in normal bins; they can leak or catch fire.",
                "Tape the ends of used batteries.",
                "Drop them at a battery or e-waste collection point."],
    "paint_chemical": ["Keep the lid tightly closed.",
                       "Never pour it down a drain.",
                       "Hand it over at a hazardous waste collection point."],
    "glass": ["Rinse the glass.",
              "Wrap broken glass in paper before binning.",
              "Put it in the blue bin."],
    "plastic_bag": ["Reuse bags where possible.",
                    "Dirty or foil-lined wrappers cannot be recycled.",
                    "Put it in the black bin."],
}
NICE = {"egg_shells_tea": "Egg shells / tea bags", "paint_chemical": "Paint / chemicals",
        "ewaste": "E-waste", "diaper_sanitary": "Diaper / sanitary",
        "metal_can": "Metal can", "yard_waste": "Yard waste", 
        "plastic_container": "Plastic container", "plastic_cutlery": "Plastic cutlery",}

# Layer 1: classes of YOUR trained model
BIN = {
    "plastic_bottle": BLUE, "paper": BLUE, "glass": BLUE, "metal_can": BLUE,
    "plastic_cup": BLUE,
    "food_waste": GREEN, "egg_shells_tea": GREEN, "yard_waste": GREEN,
    "battery": RED, "paint_chemical": RED, "ewaste": REWST,
    "plastic_bag": BLACK, "thermocol": BLACK, "diaper_sanitary": BLACK,
    "ceramic": BLACK, "pen": BLACK, "paper_cup": BLACK,
    "pencil": BLACK, "carton": BLUE, "plastic_container": BLUE, "plastic_cutlery": BLACK,
}

# Objects the general ImageNet model recognises better than our thin classes.
# Names that don't exist in ImageNet are simply never matched, so extras are harmless.
PENS = ["ballpoint", "fountain_pen", "rubber_eraser"]
ELECTRONICS = ["cellular_telephone", "iPod", "hand-held_computer", "laptop", "notebook",
               "mouse", "remote_control", "computer_keyboard", "joystick",
               "digital_watch", "digital_clock", "monitor", "desktop_computer",
               "screen", "television", "modem", "hard_disc", "projector", "printer",
               "loudspeaker", "microphone", "radio", "tape_player", "cassette_player",
               "CD_player", "power_drill", "electric_fan", "hair_dryer", "iron",
               "toaster", "microwave", "electric_guitar", "camera", "reflex_camera",
               "Polaroid_camera", "binoculars", "switch", "plug", "wall_clock",
               "stopwatch", "analog_clock"]
OVERRIDE = {n: BLACK for n in PENS}
OVERRIDE.update({n: REWST for n in ELECTRONICS})

IMAGENET_BIN = {
    "water_bottle": BLUE, "pop_bottle": BLUE, "beer_bottle": BLUE,
    "wine_bottle": BLUE, "water_jug": BLUE, "pill_bottle": BLUE,
    "carton": BLUE, "envelope": BLUE,
    "plastic_bag": BLACK, "coffee_mug": BLACK, "toilet_tissue": BLACK, "diaper": BLACK,
    "banana": GREEN, "orange": GREEN, "lemon": GREEN, "Granny_Smith": GREEN,
    "strawberry": GREEN, "pineapple": GREEN, "broccoli": GREEN,
    "cucumber": GREEN, "head_cabbage": GREEN, "bell_pepper": GREEN,
    "corn": GREEN, "mushroom": GREEN,
}
IMAGENET_BIN.update(OVERRIDE)

app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model(MODEL_DIR / "waste_classifier.keras")
labels = json.load(open(MODEL_DIR / "labels.json"))
general = tf.keras.applications.MobileNetV2(weights="imagenet")
model.predict(np.zeros((1, 224, 224, 3), dtype="float32"), verbose=0)
general.predict(np.zeros((1, 224, 224, 3), dtype="float32"), verbose=0)
print("Models ready. Classes:", labels, flush=True)


# ---- YOLO detector (block 3A) ----
YOLO_CONF = 0.35   # lower = more boxes, higher = fewer wrong ones
yolo = YOLO(str(MODEL_DIR / "yolo_waste.pt"))
yolo.predict(np.zeros((640, 640, 3), dtype="uint8"), verbose=False)

YOLO_BIN = {
    "cardboard": BLUE, "paper": BLUE, "glass": BLUE, "metal": BLUE,
    "plastic": BLUE, "plastic_container": BLUE,
    "plastic_bag": BLACK, "paper_cup": BLACK, "stationery": BLACK,
    "thermocol": BLACK, "trash": BLACK, "fabric": BLACK,
}


def prep(img):
    """PIL RGB image -> (1,224,224,3) model input, same as training."""
    arr = np.array(img.resize(IMG_SIZE, Image.LANCZOS), dtype=np.float32)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return np.expand_dims(arr, 0)


def build(key, name, row, conf, source, box=None):
    color, kind = row
    if kind.startswith("E-waste"):
        guide = EWASTE_GUIDE
    else:
        guide = ITEM_GUIDE.get(key) or GUIDE[color]
    pretty = NICE.get(key) or name.replace("_", " ").capitalize()
    return {"item": key, "name": pretty, "bin": kind, "color": color,
            "guideline": guide, "tip": guide[-1],
            "confidence": round(float(conf), 4), "source": source, "box": box}


def reply(dets, conf=0.0):
    if not dets:
        return jsonify({"item": "none", "name": None, "bin": None, "color": None,
                        "guideline": [], "tip": None, "confidence": round(conf, 4),
                        "detections": []})
    out = dict(max(dets, key=lambda d: d["confidence"]))   # best one = main card
    out["detections"] = dets                               # all boxes for the overlay
    return jsonify(out)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "classes": labels})


@app.route("/classify", methods=["POST"])
def classify():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Use field name 'image'."}), 400
    try:
        raw = request.files["image"].read()
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
        x = prep(img)
    except Exception as exc:
        return jsonify({"error": f"Could not decode image: {exc}"}), 422
    W, H = img.size

    # 1) General model: pens and electronics get priority (unchanged)
    gp = general.predict(x, verbose=0)
    top = tf.keras.applications.mobilenet_v2.decode_predictions(gp, top=5)[0]
    print("general top5:", [(n, round(float(p), 2)) for _, n, p in top], flush=True)
    for _, n, p in top:
        if n in OVERRIDE and float(p) >= OVERRIDE_THRESHOLD:
            return reply([build(n, n, OVERRIDE[n], p, "general")])

    # 2) YOLO finds the objects, custom classifier refines each crop
    res = yolo.predict(img, conf=YOLO_CONF, imgsz=640, verbose=False)[0]
    boxes = sorted(res.boxes, key=lambda b: -float(b.conf))[:3]
    dets = []
    for b in boxes:
        ycls, yconf = res.names[int(b.cls)], float(b.conf)
        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
        box = [x1 / W, y1 / H, x2 / W, y2 / H]
        pad = 0.1 * max(x2 - x1, y2 - y1)
        crop = img.crop((max(0, x1 - pad), max(0, y1 - pad),
                         min(W, x2 + pad), min(H, y2 + pad)))
        probs = model.predict(prep(crop), verbose=0)[0]
        i = int(np.argmax(probs))
        item, conf = labels[i], float(probs[i])
        print("yolo:", ycls, round(yconf, 2), "| custom:", item, round(conf, 2), flush=True)
        if item != "background" and conf >= THRESHOLD and item in BIN:
            dets.append(build(item, item, BIN[item], conf, "yolo+custom", box))
        elif ycls in YOLO_BIN:
            dets.append(build(ycls, ycls, YOLO_BIN[ycls], yconf, "yolo", box))
    if dets:
        return reply(dets)

    # 3) No boxes: whole-frame fallback (your old logic)
    probs = model.predict(x, verbose=0)[0]
    i = int(np.argmax(probs))
    item, conf = labels[i], float(probs[i])
    print("custom:", item, round(conf, 2), flush=True)
    if item != "background" and conf >= THRESHOLD and item in BIN:
        return reply([build(item, item, BIN[item], conf, "custom")])
    for _, n, p in top[:3]:
        if n in IMAGENET_BIN and float(p) >= GENERAL_THRESHOLD:
            return reply([build(n, n, IMAGENET_BIN[n], p, "general")])
    return reply([], conf)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False)