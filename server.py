# SERVER - AGGIUNGERE DESCRIZIONE


##################################################################################################################################
##                                                                                                                              ##
##                                                  IMPORT + CONFIG ENV E CORS                                                  ##
##                                                                                                                              ##
##################################################################################################################################

import io
import os
import torch
import supervision as sv
from PIL.Image import Image
from flask_cors import CORS
from ultralytics import YOLO
from dotenv import load_dotenv
from flask import Flask, request, jsonify

# Carica le variabili di configurazione dal file .env
load_dotenv()

app = Flask(__name__)
CORS(app)


##################################################################################################################################
##                                                                                                                              ##
##                                                 ESTRAZIONE RISULTATI                                                         ##
##                                                                                                                              ##
##################################################################################################################################

def plot_bboxes(results):
    xyxys, confidences, class_ids = [], [], []

    for result in results:
        boxes = result.boxes.cpu().numpy()
        xyxys.append(boxes.xyxy)
        confidences.append(boxes.conf)
        class_ids.append(boxes.cls)

    return xyxys, confidences, class_ids


class ObjectDetection:

    ##############################################################################################################################
    ##                                                                                                                          ##
    ##                                                      COSTRUTTORE                                                         ##
    ##                                                                                                                          ##
    ##############################################################################################################################

    def __init__(self, source, output_dir):
        self.cap = None
        self.source = source
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

        model_path = os.path.join(os.getenv("DIRECTORY_YOLO_MODEL"), os.getenv("YOLO_MODEL"))
        self.model = YOLO(model_path).to(self.device)
        self.class_name_dict = self.model.model.names
        self.box_annotator = sv.BoxAnnotator(sv.ColorPalette.default(), thickness=3, text_thickness=3, text_scale=1.5)

    ##############################################################################################################################
    ##                                                                                                                          ##
    ##                                                  FUNZIONI INTRINSECHE                                                    ##
    ##                                                                                                                          ##
    ##############################################################################################################################

    def predict(self, frame):
        results = self.model(frame)
        return results

    def process_image(self, frame):
        # Questa funzione ora restituirà le classi e le confidenze invece di salvare l'immagine
        if frame is not None:
            results = self.predict(frame)
            _, confidences, class_ids = plot_bboxes(results)
            classes = [self.class_name_dict[int(cid)] for cid in class_ids]
            return classes, confidences
        else:
            print(f"Impossibile leggere l'immagine")
            return [], []


@app.route('/detect_objects', methods=['POST'])
def detect_objects():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    # Se necessario, puoi salvare l'immagine o processarla qui.
    image = request.files['image'].read()
    # Assumo che l'immagine sia nel formato PIL. Se hai un formato diverso, convertilo di conseguenza
    frame = Image.open(io.BytesIO(image))

    detector = ObjectDetection(None, None)  # Come non hai specificato una sorgente o una directory, ho impostato entrambi a None
    classes, confidences = detector.process_image(frame)

    # Costruire la risposta con le classi e le confidenze
    response_data = [{"class": cls, "confidence": conf} for cls, conf in zip(classes, confidences)]

    return jsonify(response_data)


if __name__ == '__main__':
    ssl_cert = os.environ.get('SSL_CERT')
    ssl_key = os.environ.get('SSL_KEY')

    if ssl_cert and ssl_key:
        ssl_context = (ssl_cert, ssl_key)
    else:
        ssl_context = None

    app.run(debug=os.environ.get('FLASK_DEBUG', 'False'), host=os.environ.get('FLASK_HOST', '0.0.0.0'),
            port=int(os.environ.get('FLASK_PORT', 5000)), ssl_context=ssl_context)
