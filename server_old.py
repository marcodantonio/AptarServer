from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from dotenv import load_dotenv
import supervision as sv

load_dotenv()

app = Flask(__name__)
CORS(app)

# Inizializza il modello YOLOv8
model_path = os.path.join(os.getenv("DIRECTORY_YOLO_MODEL"), os.getenv("YOLO_MODEL"))
yolo_model = YOLO(model_path).cuda() if torch.cuda.is_available() else YOLO(model_path)

class_name_dict = yolo_model.names


@app.route('/detect_objects', methods=['POST'])
def detect_objects():
    print("Richiesta ricevuta da:", request.remote_addr)

    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image = request.files['image'].read()  # Leggi l'immagine come bytes

    # Effettua il rilevamento degli oggetti
    with torch.no_grad():
        results = yolo_model(image)

    # Estrai le informazioni rilevanti
    labels = []
    for result in results.pred[0]:
        label = class_name_dict[result[5].item()]
        confidence = result[4].item()
        labels.append({'label': label, 'confidence': confidence})

    # Chiamata a process_image per salvare l'immagine con i bounding box
    output_image = process_image(image, results, labels)

    # Restituisci la risposta JSON contenente le etichette e le confidenze degli oggetti rilevati
    return jsonify({'objects': labels, 'image_path': output_image})


def process_image(image_bytes, results, labels):
    # ... Effettua le operazioni necessarie per aggiungere i bounding box all'immagine ...

    # Salva l'immagine con i bounding box nella cartella di destinazione (se necessario)
    output_directory = os.path.join(os.getenv("DIRECTORY_OUTPUT"), "images_with_boxes")
    os.makedirs(output_directory, exist_ok=True)

    output_image_path = os.path.join(output_directory, 'output_image.jpg')
    cv2.imwrite(output_image_path, image_with_bboxes)

    return output_image_path


@app.route('/get_image/<path:image_path>', methods=['GET'])
def get_image(image_path):
    # Questa route consente al client di scaricare l'immagine con i bounding box
    return send_file(image_path)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
