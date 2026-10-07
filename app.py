"""
Rice Disease Detection - Flask web app
Run:  python app.py   ->   open http://127.0.0.1:5000
Needs: models/best_model.keras and models/class_names.json (created by train_models.ipynb)
"""
import io
import json
import os

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.keras")
CLASS_PATH = os.path.join(BASE_DIR, "models", "class_names.json")
ALLOWED = {"jpg", "jpeg", "png", "bmp", "webp"}
LOW_CONFIDENCE = 60.0  # below this we warn the user

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB upload limit

# safe_mode=False is needed because the model contains a preprocessing Lambda layer
# (it is OUR own model, so this is safe).
model = tf.keras.models.load_model(MODEL_PATH, safe_mode=False)
with open(CLASS_PATH) as f:
    CLASS_NAMES = json.load(f)
IMG_SIZE = tuple(model.input_shape[1:3])

# Short general info for the result page. Matching is done on keywords in the class name.
# This is general guidance only - farmers should confirm with a local agriculture expert.
DISEASE_INFO = {
    "blast": "Fungal disease causing diamond-shaped spots with grey centres. Use resistant varieties, avoid excess nitrogen and ask an expert about a suitable fungicide.",
    "brown": "Fungal disease (brown spot) linked to poor soil nutrition and water stress. Improve soil fertility, use clean seed and balanced fertiliser.",
    "bacterial": "Bacterial blight causes yellow-to-white streaks along the leaf edges. Use resistant varieties, avoid excess nitrogen and keep the field drained.",
    "blight": "Blight causes drying and streaking of leaves. Remove infected plants, avoid excess nitrogen and manage field water.",
    "tungro": "Viral disease spread by green leafhoppers; leaves turn yellow-orange and plants stay stunted. Control the insect vector and use resistant varieties.",
    "smut": "Leaf smut shows small black raised spots on leaves. Use clean seed, avoid excess nitrogen and keep the field weed-free.",
    "sheath": "Sheath blight shows oval greenish-grey lesions near the water line. Reduce plant density and avoid excess nitrogen.",
    "healthy": "The leaf looks healthy. Keep monitoring the crop regularly.",
}


def get_info(label):
    low = label.lower()
    for key, text in DISEASE_INFO.items():
        if key in low:
            return text
    return "No stored information for this class. Please consult a local agriculture expert."


def allowed_file(name):
    return "." in name and name.rsplit(".", 1)[1].lower() in ALLOWED


@app.route("/")
def index():
    return render_template("index.html", classes=CLASS_NAMES)


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files or request.files["image"].filename == "":
        return jsonify(error="Please choose an image first."), 400
    file = request.files["image"]
    if not allowed_file(file.filename):
        return jsonify(error="Only JPG, PNG, BMP or WEBP images are allowed."), 400
    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB").resize(IMG_SIZE)
    except Exception:
        return jsonify(error="Could not read this image file."), 400

    x = np.expand_dims(np.asarray(img, dtype=np.float32), axis=0)  # raw 0-255, model preprocesses itself
    probs = model.predict(x, verbose=0)[0]

    top = np.argsort(probs)[::-1][:3]
    top3 = [{"label": CLASS_NAMES[i], "confidence": round(float(probs[i]) * 100, 2)} for i in top]
    best = top3[0]
    return jsonify(
        disease=best["label"],
        confidence=best["confidence"],
        low_confidence=best["confidence"] < LOW_CONFIDENCE,
        info=get_info(best["label"]),
        top3=top3,
    )


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="Image is too large (max 8 MB)."), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
