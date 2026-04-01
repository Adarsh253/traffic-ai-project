"""
MobileNetV2 Transfer Learning — Traffic Sign Classification
────────────────────────────────────────────────────────────
Rewrite of the original simple CNN script using transfer learning.

Changes from original:
  - IMG_SIZE      : 32×32  →  96×96  (MobileNetV2 minimum is 32, sweet spot ~96)
  - Backbone      : scratch CNN  →  MobileNetV2 (ImageNet, frozen base)
  - Head          : Flatten+Dense  →  GlobalAvgPool+BN+Dropout+Dense
  - Classes       : hard-coded 5  →  auto-detected (supports 40+)
  - Training      : 1 phase  →  Phase 1 (head only) + Phase 2 (fine-tune)
  - Preprocessing : rescale /255  →  MobileNetV2 preprocess_input (maps to [-1,1])
  - Augmentation  : horizontal_flip=True  →  False (signs are NOT symmetric)
  - Metrics       : accuracy only  →  accuracy + top-3 accuracy
  - Artefacts     : model.h5 only  →  model + class_indices.json + training curves PNG

Expected folder layout (unchanged, ImageDataGenerator-compatible):
  data/
    train/<class_name>/*.jpg
    test/<class_name>/*.jpg
  model/
    mobilenetv2_traffic.h5
    class_indices.json
    training_curves.png
"""

import os
import json
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib
matplotlib.use("Agg")           # headless — safe for servers without a display
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
IMG_SIZE         = (96, 96)         # 96×96 — good balance for MobileNetV2
IMG_SHAPE        = (96, 96, 3)
BATCH_SIZE       = 32
EPOCHS_HEAD      = 15               # Phase 1: classification head only
EPOCHS_FINETUNE  = 10               # Phase 2: unfreeze top base layers
UNFREEZE_LAYERS  = 30               # how many top MobileNetV2 layers to unfreeze
LR_HEAD          = 1e-3             # higher LR safe when base is frozen
LR_FINETUNE      = 1e-5             # very low LR to avoid catastrophic forgetting
DROPOUT_RATE     = 0.4
DENSE_UNITS      = 256

TRAIN_DIR        = "data/train"
TEST_DIR         = "data/test"
MODEL_DIR        = "model"
MODEL_PATH       = os.path.join(MODEL_DIR, "mobilenetv2_traffic.h5")
CLASS_IDX_PATH   = os.path.join(MODEL_DIR, "class_indices.json")
PLOT_PATH        = os.path.join(MODEL_DIR, "training_curves.png")

os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# Auto-detect number of classes
# ─────────────────────────────────────────────
class_names = sorted([
    d for d in os.listdir(TRAIN_DIR)
    if os.path.isdir(os.path.join(TRAIN_DIR, d)) and not d.startswith(".")
])
NUM_CLASSES = len(class_names)

print("\n" + "=" * 60)
print("  MobileNetV2 Transfer Learning — Traffic Signs")
print("=" * 60)
print(f"  Classes detected  : {NUM_CLASSES}")
print(f"  Input size        : {IMG_SIZE}")
print(f"  Batch size        : {BATCH_SIZE}")
print(f"  Phase-1 epochs    : {EPOCHS_HEAD}  (frozen base)")
print(f"  Phase-2 epochs    : {EPOCHS_FINETUNE}  (fine-tune top {UNFREEZE_LAYERS} layers)")
print("=" * 60 + "\n")
for i, name in enumerate(class_names):
    print(f"  [{i:>3}] {name}")
print()

# ─────────────────────────────────────────────
# Data Generators
# NOTE: preprocess_input replaces rescale=1/255 — it maps pixels to [-1, 1]
#       which is what MobileNetV2's ImageNet weights expect.
# NOTE: horizontal_flip is False — traffic signs are NOT horizontally symmetric
#       (a mirrored stop sign is not a stop sign).
# ─────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15,
    width_shift_range=0.12,
    height_shift_range=0.12,
    shear_range=0.08,
    zoom_range=0.15,
    brightness_range=[0.75, 1.25],
    horizontal_flip=False,
    fill_mode="nearest",
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
    seed=SEED,
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
    seed=SEED,
)

assert train_generator.num_classes == NUM_CLASSES, (
    f"Generator found {train_generator.num_classes} classes but "
    f"expected {NUM_CLASSES}. Check {TRAIN_DIR}."
)

print(f"  Training samples  : {train_generator.samples}")
print(f"  Test samples      : {test_generator.samples}\n")

# ─────────────────────────────────────────────
# Model — MobileNetV2 backbone + custom head
# ─────────────────────────────────────────────
def build_model(num_classes: int):
    """
    Functional API build so base_model is accessible separately
    for the fine-tuning phase.
    """
    # ── Frozen backbone ──────────────────────
    base = MobileNetV2(
        input_shape=IMG_SHAPE,
        include_top=False,          # strip ImageNet classifier
        weights="imagenet",
    )
    base.trainable = False          # freeze all 154 layers for Phase 1

    # ── Custom classification head ───────────
    inputs  = keras.Input(shape=IMG_SHAPE, name="image_input")
    x       = base(inputs, training=False)   # training=False keeps BN in inference mode
    x       = layers.GlobalAveragePooling2D(name="gap")(x)
    x       = layers.Dense(DENSE_UNITS, activation="relu", name="dense_1")(x)
    x       = layers.BatchNormalization(name="bn_head")(x)
    x       = layers.Dropout(DROPOUT_RATE, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name="mobilenetv2_traffic")
    return model, base


model, base_model = build_model(NUM_CLASSES)
model.summary(line_length=80)
print(f"\n  Base model total layers : {len(base_model.layers)}")

# ─────────────────────────────────────────────
# Callbacks (reused across both phases)
# ─────────────────────────────────────────────
def make_callbacks(phase_tag: str) -> list:
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-8,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(
            os.path.join(MODEL_DIR, f"log_phase{phase_tag}.csv"),
            append=False,
        ),
    ]

# ─────────────────────────────────────────────
# Phase 1 — Head only (base frozen)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  PHASE 1 — Training classification head  (base frozen)")
print("=" * 60 + "\n")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LR_HEAD),
    loss="categorical_crossentropy",
    metrics=[
        "accuracy",
        keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc"),
    ],
)

history1 = model.fit(
    train_generator,
    epochs=EPOCHS_HEAD,
    validation_data=test_generator,
    callbacks=make_callbacks("1"),
    verbose=1,
)

# ─────────────────────────────────────────────
# Phase 2 — Fine-tune top N base layers
# ─────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  PHASE 2 — Fine-tuning top {UNFREEZE_LAYERS} base layers")
print(f"{'='*60}\n")

base_model.trainable = True
for layer in base_model.layers[:-UNFREEZE_LAYERS]:
    layer.trainable = False

n_trainable = sum(1 for l in base_model.layers if l.trainable)
print(f"  Unfrozen base layers : {n_trainable} / {len(base_model.layers)}\n")

# Must re-compile after changing trainability
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LR_FINETUNE),
    loss="categorical_crossentropy",
    metrics=[
        "accuracy",
        keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc"),
    ],
)

history2 = model.fit(
    train_generator,
    epochs=EPOCHS_FINETUNE,
    validation_data=test_generator,
    callbacks=make_callbacks("2"),
    verbose=1,
)

# ─────────────────────────────────────────────
# Final evaluation
# ─────────────────────────────────────────────
print(f"\n{'='*60}")
print("  FINAL EVALUATION ON TEST SET")
print(f"{'='*60}")

results = model.evaluate(test_generator, verbose=1)
for name, val in zip(model.metrics_names, results):
    print(f"  {name:<20} : {val:.4f}")

# ─────────────────────────────────────────────
# Save class index mapping  (used by Streamlit app)
# ─────────────────────────────────────────────
class_indices = {str(v): k for k, v in train_generator.class_indices.items()}
with open(CLASS_IDX_PATH, "w") as f:
    json.dump(class_indices, f, indent=2)
print(f"\n  Class indices saved : {CLASS_IDX_PATH}")

# ─────────────────────────────────────────────
# Plot training curves (both phases stitched)
# ─────────────────────────────────────────────
def stitch(h1, h2, key):
    """Concatenate phase-1 and phase-2 values for a given metric key."""
    return h1.history.get(key, []) + h2.history.get(key, [])

phase1_len   = len(history1.history["accuracy"])
total_epochs = phase1_len + len(history2.history["accuracy"])
xs           = list(range(1, total_epochs + 1))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor("#0d0d0d")

for ax in axes:
    ax.set_facecolor("#161616")
    ax.tick_params(colors="#aaa")
    ax.spines[:].set_color("#333")
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color("#aaa")

# ── Accuracy panel ──
axes[0].plot(xs, stitch(history1, history2, "accuracy"),     color="#00e5ff", lw=2, label="Train")
axes[0].plot(xs, stitch(history1, history2, "val_accuracy"), color="#ff5252", lw=2, label="Val")
axes[0].axvline(phase1_len + 0.5, color="#ffd740", lw=1.2, ls="--", label="Fine-tune start")
axes[0].set_title("Accuracy", color="#fff", pad=10)
axes[0].set_xlabel("Epoch", color="#aaa")
axes[0].legend(facecolor="#1e1e1e", labelcolor="#ccc", edgecolor="#333")

# ── Loss panel ──
axes[1].plot(xs, stitch(history1, history2, "loss"),     color="#00e5ff", lw=2, label="Train")
axes[1].plot(xs, stitch(history1, history2, "val_loss"), color="#ff5252", lw=2, label="Val")
axes[1].axvline(phase1_len + 0.5, color="#ffd740", lw=1.2, ls="--", label="Fine-tune start")
axes[1].set_title("Loss", color="#fff", pad=10)
axes[1].set_xlabel("Epoch", color="#aaa")
axes[1].legend(facecolor="#1e1e1e", labelcolor="#ccc", edgecolor="#333")

fig.suptitle("MobileNetV2 Transfer Learning — Training Curves", color="#fff", fontsize=13)
plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"  Training curves saved : {PLOT_PATH}")

# ─────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────
best_p1 = max(history1.history["val_accuracy"])
best_p2 = max(history2.history["val_accuracy"])

print("\n" + "=" * 60)
print("  TRAINING SUMMARY")
print("=" * 60)
print(f"  Phase 1 best val accuracy  : {best_p1 * 100:.2f}%  ({phase1_len} epochs)")
print(f"  Phase 2 best val accuracy  : {best_p2 * 100:.2f}%  ({total_epochs - phase1_len} epochs)")
print(f"  Model saved to             : {MODEL_PATH}")
print(f"  Class map saved to         : {CLASS_IDX_PATH}")
print(f"  Curves saved to            : {PLOT_PATH}")
print("=" * 60 + "\n")