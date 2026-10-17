import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import json
from pathlib import Path
from collections import Counter
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

ROOT = Path(__file__).resolve().parent.parent
VAL = ROOT / "data" / "processed" / "val"
MD = Path(__file__).resolve().parent / "model"

labels = json.load(open(MD / "labels.json"))
model = tf.keras.models.load_model(MD / "waste_classifier.keras")

BINCOL = {
    "plastic_bottle": "Blue", "paper": "Blue", "glass": "Blue", "metal_can": "Blue",
    "carton": "Blue", "plastic_container": "Blue",
    "food_waste": "Green", "egg_shells_tea": "Green", "yard_waste": "Green",
    "battery": "Red", "paint_chemical": "Red", "ewaste": "Red",
    "plastic_bag": "Black", "thermocol": "Black", "diaper_sanitary": "Black",
    "ceramic": "Black", "pen": "Black", "pencil": "Black", "paper_cup": "Black",
    "plastic_cutlery": "Black", "background": "None",
}

gen = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input)
g = gen.flow_from_directory(str(VAL), target_size=(224, 224), batch_size=32,
                            class_mode="categorical", classes=labels, shuffle=False)
probs = model.predict(g, verbose=0)
pred = probs.argmax(1)
true = g.classes

print("\nItem-level accuracy:", round(float((pred == true).mean()), 3))
bin_ok = np.mean([BINCOL.get(labels[p]) == BINCOL.get(labels[t]) for p, t in zip(pred, true)])
print("BIN-level accuracy: ", round(float(bin_ok), 3), "  <- the number that matters\n")

print("Per class (item accuracy, then bin accuracy):")
for i, name in enumerate(labels):
    idx = np.where(true == i)[0]
    if len(idx) == 0:
        continue
    a = np.mean(pred[idx] == i)
    b = np.mean([BINCOL.get(labels[p]) == BINCOL.get(name) for p in pred[idx]])
    print(f"  {name:<18} item {a:.2f}   bin {b:.2f}   (n={len(idx)})")

print("\nTop confusions that put the item in the WRONG bin:")
c = Counter()
for p, t in zip(pred, true):
    if BINCOL.get(labels[p]) != BINCOL.get(labels[t]):
        c[(labels[t], labels[p])] += 1
for (t, p), n in c.most_common(10):
    print(f"  {t}  ->  {p}   x{n}")