# ============================================================================
# AGRISHIELD – FULL END-TO-END AI PIPELINE (SINGLE FILE)
# Dataset → Training → Prediction → UI
# ============================================================================

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd
import gradio as gr
import plotly.graph_objects as go
import datetime
import os

# ============================================================================
# GLOBAL CONFIG
# ============================================================================
MODEL_PATH = "agrishield_model.h5"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5   # Increase to 20+ for real training

# ============================================================================
# DATASET LOADING (ONLINE)
# ============================================================================
def load_dataset():
    print("📥 Loading PlantVillage dataset...")
    (train_ds, val_ds), ds_info = tfds.load(
        "plant_village",
        split=["train[:80%]", "train[80%:]"],
        as_supervised=True,
        with_info=True
    )

    class_names = ds_info.features["label"].names

    def preprocess(img, label):
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        return img, label

    train_ds = train_ds.map(preprocess).batch(BATCH_SIZE).prefetch(2)
    val_ds = val_ds.map(preprocess).batch(BATCH_SIZE).prefetch(2)

    return train_ds, val_ds, class_names


# ============================================================================
# MODEL BUILDING
# ============================================================================
def build_model(num_classes):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    base.trainable = False

    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(base.input, outputs)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ============================================================================
# TRAIN OR LOAD MODEL
# ============================================================================
if not os.path.exists(MODEL_PATH):
    train_ds, val_ds, CLASS_NAMES = load_dataset()
    model = build_model(len(CLASS_NAMES))
    print("🚀 Training model...")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
    model.save(MODEL_PATH)
else:
    print("✅ Loading saved model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    _, _, CLASS_NAMES = load_dataset()


# ============================================================================
# ENVIRONMENTAL ANALYTICS
# ============================================================================
def fungal_risk(ph, moisture, temp, rainfall):
    risk = 20
    if 20 <= temp <= 30: risk += 25
    if moisture > 70: risk += 30
    if rainfall > 100: risk += 15
    if ph < 5.5 or ph > 7.5: risk += 10
    return min(risk, 100)


def confidence_gauge(conf):
    return go.Figure(go.Indicator(
        mode="gauge+number",
        value=conf,
        title={"text": "Model Confidence"},
        gauge={"axis": {"range": [0, 100]},
               "bar": {"color": "#22c55e"}}
    ))


# ============================================================================
# PREDICTION PIPELINE
# ============================================================================
def predict(image, ph, moisture, temp, rainfall):
    if image is None:
        return "No Image", "-", "-", None

    img = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    img = np.expand_dims(img, 0)

    preds = model.predict(img)[0]
    idx = np.argmax(preds)

    disease = CLASS_NAMES[idx]
    confidence = round(float(preds[idx]) * 100, 2)
    risk = fungal_risk(ph, moisture, temp, rainfall)

    return disease, f"{confidence}%", f"{risk}%", confidence_gauge(confidence)


# ============================================================================
# UI (GRADIO DASHBOARD)
# ============================================================================
with gr.Blocks(title="AgriShield – AI Crop Disease Detection") as app:
    gr.Markdown("## 🌿 AgriShield AI – Real Leaf Disease Detection")

    with gr.Row():
        with gr.Column(scale=1):
            img = gr.Image(label="Upload Leaf Image", type="numpy")
            ph = gr.Slider(0, 14, 6.5, label="Soil pH")
            moisture = gr.Slider(0, 100, 60, label="Soil Moisture (%)")
            temp = gr.Slider(-10, 50, 25, label="Temperature (°C)")
            rainfall = gr.Slider(0, 300, 100, label="Rainfall (mm)")
            btn = gr.Button("Analyze Crop")

        with gr.Column(scale=2):
            disease = gr.Textbox(label="Predicted Disease")
            confidence = gr.Textbox(label="AI Confidence")
            risk = gr.Textbox(label="Environmental Risk")
            gauge = gr.Plot()

    btn.click(
        predict,
        inputs=[img, ph, moisture, temp, rainfall],
        outputs=[disease, confidence, risk, gauge]
    )

app.launch()
