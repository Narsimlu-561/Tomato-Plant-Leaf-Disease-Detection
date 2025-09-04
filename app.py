from flask import Flask, render_template, request, jsonify
#

from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import io
from PIL import Image

# Import DualECALayer from eca.py
from eca import DualECALayer 
app = Flask(__name__)
CORS(app)  # allow frontend to connect

MODEL_PATH = "./model/model.keras"


#
@app.route("/")
def home():
    return render_template("index.html")
#


# Pass custom_objects to load_model

try:
    model = load_model(MODEL_PATH, custom_objects={"DualECALayer": DualECALayer}, compile=False)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


CLASS_NAMES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

# Hardcoded descriptions and recommendations
DISEASE_INFO = {
    "Tomato___Bacterial_spot": {
        "cause": "Caused by bacteria Xanthomonas campestris pv. vesicatoria. It leads to small, water-soaked spots on leaves and fruit.",
        "recommendations": [
            "Use copper-based fungicides.",
            "Remove and destroy infected plant parts.",
            "Avoid overhead watering to reduce spread."
        ]
    },
    "Tomato___Early_blight": {
        "cause": "Caused by the fungus Alternaria solani. Characterized by target-like spots on older leaves.",
        "recommendations": [
            "Apply fungicides containing mancozeb or chlorothalonil.",
            "Ensure good air circulation.",
            "Prune lower leaves to prevent soil splash."
        ]
    },
    "Tomato___Late_blight": {
        "cause": "A destructive disease caused by the oomycete Phytophthora infestans. It thrives in cool, wet conditions.",
        "recommendations": [
            "Apply fungicides preventively.",
            "Remove and destroy all infected plant material.",
            "Space plants properly for air circulation."
        ]
    },
    "Tomato___Leaf_Mold": {
        "cause": "Caused by the fungus Fulvia fulva. It primarily affects leaves, causing olive-green velvety patches on the underside.",
        "recommendations": [
            "Improve air circulation and reduce humidity.",
            "Use fungicides with azoxystrobin or pyraclostrobin.",
            "Plant resistant tomato varieties."
        ]
    },
    "Tomato___Septoria_leaf_spot": {
        "cause": "A fungal disease caused by Septoria lycopersici. It forms small, circular spots with dark borders and gray centers.",
        "recommendations": [
            "Apply fungicides containing chlorothalonil or mancozeb.",
            "Practice crop rotation.",
            "Keep plants off the ground and use mulch."
        ]
    },
    "Tomato___Spider_mites": {
        "cause": "Caused by tiny arachnids, usually Tetranychus urticae. They suck sap, leading to stippling and webbing on leaves.",
        "recommendations": [
            "Use insecticidal soaps or horticultural oils.",
            "Release beneficial predatory mites.",
            "Hose down plants with a strong stream of water."
        ]
    },
    "Tomato___Target_Spot": {
        "cause": "Caused by the fungus Corynespora cassiicola. It creates small, dark, circular spots on leaves that resemble targets.",
        "recommendations": [
            "Apply fungicides.",
            "Remove crop residue.",
            "Avoid splashing water on foliage."
        ]
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "cause": "A viral disease transmitted by whiteflies (Bemisia tabaci). It causes upward curling and yellowing of leaves.",
        "recommendations": [
            "Control whitefly populations with insecticides.",
            "Use reflective mulches.",
            "Plant resistant varieties."
        ]
    },
    "Tomato___Tomato_mosaic_virus": {
        "cause": "A viral disease that causes mottled, light, and dark green patterns on leaves, often leading to stunted growth.",
        "recommendations": [
            "Remove and destroy infected plants.",
            "Disinfect tools and hands after handling infected plants.",
            "Use virus-free seeds and transplants."
        ]
    },
    "Tomato___healthy": {
        "cause": "The plant shows no signs of disease and appears to be in good health.",
        "recommendations": [
            "Continue with proper plant care.",
            "Ensure adequate watering and sunlight.",
            "Monitor the plant regularly for any signs of stress or disease."
        ]
    }
}

def preprocess_image(img, target_size=(128, 128)):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    try:
        img = Image.open(io.BytesIO(file.read()))
        processed_img = preprocess_image(img)
        preds = model.predict(processed_img)[0]
        confidence = float(np.max(preds))
        predicted_class = CLASS_NAMES[np.argmax(preds)]
        
        # Get description and recommendations based on the predicted class
        info = DISEASE_INFO.get(predicted_class, {
            "cause": "No information available.",
            "recommendations": ["No recommendations available."]
        })

        return jsonify({
            "disease": predicted_class.replace("Tomato___", "").replace("_", " "),
            "confidence": round(confidence * 100, 2),  # percentage
            "cause": info["cause"],
            "recommendations": info["recommendations"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
