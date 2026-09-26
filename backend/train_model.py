"""
train_model.py — Stage 3: Detection & Classification Module
===========================================================
Fine-tunes MobileNetV2 (ImageNet weights) on the 4 waste categories
produced by preprocess.py.  Two-phase training:
  Phase 1 — train only the new classification head (frozen base)
  Phase 2 — unfreeze the top N layers and fine-tune end-to-end

Saves the final model to backend/model/waste_classifier.h5

Run:
    python backend/train_model.py
"""

import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── Config ────────────────────────────────────────────────────────────────────
REPO_ROOT      = Path(__file__).resolve().parent.parent
PROCESSED_DIR  = REPO_ROOT / "data" / "processed"
MODEL_DIR      = Path(__file__).resolve().parent / "model"
MODEL_PATH     = MODEL_DIR / "waste_classifier.h5"

IMG_SIZE       = (224, 224)
BATCH_SIZE     = 16          # small batch → fits comfortably on laptop GPU/CPU
EPOCHS_PHASE1  = 5           # head-only warm-up
EPOCHS_PHASE2  = 10          # fine-tune top layers
UNFREEZE_LAST  = 30          # how many MobileNetV2 layers to unfreeze in phase 2
CLASSES        = ["Biodegradable", "Hazardous", "Non-Recyclable", "Recyclable"]
NUM_CLASSES    = len(CLASSES)
SEED           = 42


# ── Data generators ───────────────────────────────────────────────────────────
def make_generators():
    # preprocess_input expects raw uint8 [0, 255] and maps to [-1, 1].
    # Do NOT also set rescale — that would pre-shrink pixels to [0, 1] and
    # cause preprocess_input to map everything to [-1, ~-0.99], destroying signal.
    train_aug = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.15,
        shear_range=0.1,
        fill_mode="nearest",
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    )
    val_aug = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
    )

    train_gen = train_aug.flow_from_directory(
        str(PROCESSED_DIR / "train"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASSES,
        seed=SEED,
        shuffle=True,
    )
    val_gen = val_aug.flow_from_directory(
        str(PROCESSED_DIR / "val"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASSES,
        seed=SEED,
        shuffle=False,
    )
    return train_gen, val_gen


# ── Model construction ────────────────────────────────────────────────────────
def build_model() -> tf.keras.Model:
    base = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False   # freeze entire base for phase 1

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model, base


# ── Callbacks ─────────────────────────────────────────────────────────────────
def make_callbacks(name_suffix: str):
    ckpt_path = str(MODEL_DIR / f"best_{name_suffix}.keras")
    return [
        callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


# ── Training ──────────────────────────────────────────────────────────────────
def train():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Building data generators…")
    train_gen, val_gen = make_generators()
    print(f"  Classes: {train_gen.class_indices}")
    print(f"  Train samples: {train_gen.samples}  |  Val samples: {val_gen.samples}")

    model, base = build_model()
    model.summary(line_length=80)

    # ── Phase 1: train head only ──────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("PHASE 1 — Training classification head (base frozen)")
    print("═" * 60)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_PHASE1,
        callbacks=make_callbacks("phase1"),
        verbose=1,
    )

    # ── Phase 2: fine-tune top layers ─────────────────────────────────────────
    print("\n" + "═" * 60)
    print(f"PHASE 2 — Fine-tuning top {UNFREEZE_LAST} layers of MobileNetV2")
    print("═" * 60)
    base.trainable = True
    # Freeze all layers EXCEPT the last UNFREEZE_LAST
    for layer in base.layers[:-UNFREEZE_LAST]:
        layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),   # lower LR for fine-tune
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_PHASE2,
        callbacks=make_callbacks("phase2"),
        verbose=1,
    )

    # ── Save final model ──────────────────────────────────────────────────────
    model.save(str(MODEL_PATH))
    print(f"\n✅  Model saved to: {MODEL_PATH}")

    # Quick eval
    print("\nFinal evaluation on validation set:")
    loss, acc = model.evaluate(val_gen, verbose=0)
    print(f"  Val loss: {loss:.4f}  |  Val accuracy: {acc:.4f}")


if __name__ == "__main__":
    # Suppress TF info/warning noise in demo
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    tf.get_logger().setLevel("ERROR")
    train()
