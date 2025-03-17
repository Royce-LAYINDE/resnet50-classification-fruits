from flask import Flask, request, jsonify
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import torch
import json


# Chargement du modèle et de l'image processor 
chemin = "mymodel"  

model = AutoModelForImageClassification.from_pretrained(chemin)
image_processor = AutoImageProcessor.from_pretrained(chemin)

# Chargement du fichier JSON
id_to_label_path = chemin +"/id_to_label.json"

with open(id_to_label_path, "r") as f:
    id_to_label = json.load(f)


app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "Aucune image envoyée"}), 400

    fichier = request.files['image']
    image = Image.open(fichier).convert("RGB") 

    inputs = image_processor(image, return_tensors="pt")

    # prédiction
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        class_pred = outputs.logits.argmax(-1).item()

    
    label_pred = id_to_label[str(class_pred)]  

    return jsonify({"prediction": label_pred})
