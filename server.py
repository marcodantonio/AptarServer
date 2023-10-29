# SERVER - AGGIUNGERE DESCRIZIONE


##################################################################################################################################
##                                                                                                                              ##
##                                                  IMPORT + CONFIG ENV E CORS                                                  ##
##                                                                                                                              ##
##################################################################################################################################

# Import
import os
import cv2
import json
import uuid
import torch
import logging
import numpy as np
import supervision as sv
from flask_cors import CORS
from ultralytics import YOLO
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, request, jsonify

# Verifica se l'applicazione sta girando all'interno di un container Docker
def running_in_docker():
    # Verifica l'esistenza del file .dockerenv
    if os.path.exists('/.dockerenv'):
        return True

    # Controlla il file /proc/1/cgroup
    try:
        with open('/proc/1/cgroup', 'rt') as ifh:
            return 'docker' in ifh.read()
    except FileNotFoundError:
        pass

    return False

if not running_in_docker():
    from dotenv import load_dotenv
    load_dotenv()

app = Flask(__name__)
CORS(app)

app.logger.setLevel(logging.INFO)


##################################################################################################################################
##                                                                                                                              ##
##                                                 ESTRAZIONE RISULTATI                                                         ##
##                                                                                                                              ##
##################################################################################################################################

def plot_bboxes(results):
    xyxys, confidences, class_ids = [], [], []

    # Estrae i risultati
    for result in results:
        boxes = result.boxes.cpu().numpy()
        xyxys.append(boxes.xyxy)
        confidences.append(boxes.conf)
        class_ids.append(boxes.cls)

    return results[0].plot(), xyxys, confidences, class_ids


##################################################################################################################################
##                                                                                                                              ##
##                                               NORMALIZZAZIONE COORDINATE                                                     ##
##                                                                                                                              ##
##################################################################################################################################

def normalize_and_draw_bboxes(frame, xyxys, confidences, class_ids, class_name_dict):
    labels = []  # Lista per le coordinate normalizzate dei bounding box

    for image_idx in range(len(xyxys)):  # Attraversa tutte le immagini
        image_bounding_boxes = xyxys[image_idx]
        image_class_ids = class_ids[image_idx]
        image_confidences = confidences[image_idx]

        for bounding_box_idx in range(len(image_bounding_boxes)):  # Attraversa tutti i bounding box per un'immagine
            xyxy = image_bounding_boxes[bounding_box_idx]
            if len(xyxy) != 4:
                print(f"Coordinate xyxy non valide: {xyxy}")
                continue

            x1, y1, x2, y2 = map(int, xyxy)
            class_id = image_class_ids[bounding_box_idx]
            label = class_name_dict[class_id.item()]  # Usiamo item() per ottenere il valore scalare
            confidence = image_confidences[bounding_box_idx]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness=3)
            cv2.putText(frame, f"{label}: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            # Calcola coordinate normalizzate
            image_height, image_width, _ = frame.shape
            x1_norm = x1 / image_width
            y1_norm = y1 / image_height
            x2_norm = x2 / image_width
            y2_norm = y2 / image_height

            labels.append(f"{label} {x1_norm} {y1_norm} {x2_norm} {y2_norm}\n")

    return labels


##################################################################################################################################
##                                                                                                                              ##
##                                                         EXTRA                                                                ##
##                                                                                                                              ##
##################################################################################################################################

def get_scalar_value(item):
    if isinstance(item, np.ndarray):
        return item.ravel()[0]
    return item


class ObjectDetection:

    ##############################################################################################################################
    ##                                                                                                                          ##
    ##                                                      COSTRUTTORE                                                         ##
    ##                                                                                                                          ##
    ##############################################################################################################################

    def __init__(self, source, output_dir):
        self.cap = None

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

        # Ottieni il percorso del modello da una variabile di ambiente
        model_path = os.getenv("YOLO_MODEL")

        if not model_path:
            raise ValueError("YOLO_MODEL environment variable is not set")

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

    def process_image(self, image_path, frame):
        if frame is not None:
            results = self.predict(frame)
            plot_results, xyxys, confidences, class_ids = plot_bboxes(results)
            if os.getenv("DIRECTORY_OUTPUT"):
                output_file = os.path.join(os.getenv("DIRECTORY_OUTPUT"), f"{os.path.basename(image_path)}")
                self.save_results(frame, xyxys, confidences, class_ids, output_file, ".jpg")
            else:
                print("La variabile d'ambiente DIRECTORY_OUTPUT non è configurata.")
        else:
            print(f"Impossibile leggere l'immagine da {image_path}")

    ##############################################################################################################################
    ##                                                                                                                          ##
    ##                                                  SALVA RISULTATO                                                         ##
    ##                                                                                                                          ##
    ##############################################################################################################################

    def save_results(self, frame, xyxys, confidences, class_ids, output_file, output_extension):
        # Verifica se ci sono rilevamenti (match) in uno qualsiasi degli elenchi xyxys
        app.logger.info(f"xyxys contiene: {xyxys}")

        if any(len(x) > 0 for x in xyxys):
            # Ci sono rilevamenti (match)
            output_directory = os.path.join(os.getenv("DIRECTORY_OUTPUT"), "matched images")
            boxes_output_directory = os.path.join(os.getenv("DIRECTORY_OUTPUT"), "matched images with boxes")
            label_directory = os.path.join(os.getenv("DIRECTORY_OUTPUT"), "matched labels")
            labelstudio_directory = os.path.join(os.getenv("DIRECTORY_OUTPUT"), "matched labelstudio")

            os.makedirs(output_directory, exist_ok=True)
            os.makedirs(boxes_output_directory, exist_ok=True)
            os.makedirs(label_directory, exist_ok=True)
            os.makedirs(labelstudio_directory, exist_ok=True)

            # Genera un nome di file univoco utilizzando il nome del file originale (senza estensione)
            base_filename = os.path.splitext(os.path.basename(output_file))[0]
            counter = 1

            while True:
                output_filename = f"{base_filename}_{counter}{output_extension}"
                label_filename = f"{base_filename}_{counter}.txt"
                labelstudio_filename = f"{base_filename}_{counter}.json"

                if not os.path.exists(os.path.join(output_directory, output_filename)) and not os.path.exists(
                        os.path.join(boxes_output_directory, output_filename)) and not os.path.exists(
                    os.path.join(label_directory, label_filename)) and not os.path.exists(
                    os.path.join(labelstudio_directory, labelstudio_filename)):
                    break

                counter += 1

            output_file = os.path.join(output_directory, output_filename)
            cv2.imwrite(output_file, frame)

            # Crea una copia dell'immagine originale per disegnare i bounding box
            image_with_boxes = frame.copy()

            # Disegna i bounding box sull'immagine copia
            for xyxy, class_id, confidence in zip(xyxys[0], class_ids[0], confidences[0]):
                x1, y1, x2, y2 = map(int, xyxy)
                label = self.class_name_dict[class_id.item()]  # Usiamo item() per ottenere il valore scalare
                confidence = confidence.item()

                cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), thickness=3)
                cv2.putText(image_with_boxes, f"{label}: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                            (0, 255, 0), 2)

            # Salva l'immagine con i bounding box nella cartella "matched images with boxes"
            boxes_output_file = os.path.join(boxes_output_directory, output_filename)
            cv2.imwrite(boxes_output_file, image_with_boxes)

            # Salva le coordinate normalizzate in un file label.txt solo se ci sono rilevamenti e non sono vuote
            labels = []
            for xyxy, class_id, confidence in zip(xyxys[0], class_ids[0], confidences[0]):
                x1, y1, x2, y2 = map(int, xyxy)
                label = self.class_name_dict[class_id.item()]  # Usiamo item() per ottenere il valore scalare
                confidence = confidence.item()
                labels.append(f"{label} {confidence:.2f} {x1} {y1} {x2} {y2}\n")

            label_file = os.path.join(label_directory, label_filename)
            with open(label_file, "w") as f:
                f.writelines(labels)

            # Genera il file labelstudio
            labelstudio_data = []
            for xyxy, class_id, confidence in zip(xyxys[0], class_ids[0], confidences[0]):
                label_data = self.generate_labelstudio_json(frame_path=os.path.basename(output_file), xyxy=xyxy,
                                                            class_id=class_id, confidence=confidence)
                labelstudio_data.append(label_data)

            labelstudio_output_filename = os.path.splitext(os.path.basename(output_file))[0] + ".json"
            labelstudio_output_file = os.path.join(labelstudio_directory, labelstudio_output_filename)

            self.save_labelstudio_json(labelstudio_data, labelstudio_output_file)

        else:
            # Nessun rilevamento (nessun match)
            output_directory = os.path.join(os.getenv("DIRECTORY_OUTPUT"), "unmatched images")
            os.makedirs(output_directory, exist_ok=True)
            output_file = os.path.join(output_directory, os.path.basename(output_file))
            cv2.imwrite(output_file, frame)

    ##############################################################################################################################
    ##                                                                                                                          ##
    ##                                                  LABELSTUDIO JSON                                                        ##
    ##                                                                                                                          ##
    ##############################################################################################################################

    def generate_labelstudio_json(self, frame_path, xyxy, class_id, confidence):
        x1, y1, x2, y2 = map(int, xyxy)
        label = self.class_name_dict[class_id.item()]  # Usiamo item() per ottenere il valore scalare
        confidence = confidence.item()

        label_data = {"frame_path": frame_path, "from_name": "label", "to_name": "image", "type": "rectanglelabels",
                      "value": {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1, "rectanglelabels": [label]}}

        return label_data

    def save_labelstudio_json(self, labelstudio_data, labelstudio_output_file):
        os.makedirs(os.path.dirname(labelstudio_output_file), exist_ok=True)

        with open(labelstudio_output_file, "w") as json_file:
            json.dump(labelstudio_data, json_file, indent=4)

    def process_image_from_request(self, image_bytes):
        # Converti l'immagine da bytes a una matrice OpenCV
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return {"error": "Non è stata fornita un'immagine valida."}, 400

        # Processa l'immagine come facevi nel tuo script originale
        results, _, _, _ = plot_bboxes(self.predict(frame))

        app.logger.info(results)

        # Qui, puoi convertire i risultati in un formato che desideri restituire come risposta
        # Per ora, restituirò solo una risposta di esempio.
        response_data = {"detections": "Esempio di rilevamenti"  # Modifica questa parte per adattarla alle tue esigenze
        }

        return response_data

##################################################################################################################################
##                                                                                                                              ##
##                                                          POST                                                                ##
##                                                                                                                              ##
##################################################################################################################################




@app.route('/detect_objects', methods=['POST'])
def detect_objects():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image = request.files['image'].read()

    # Genera un nome file univoco basato su timestamp e UUID
    current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_filename = f"image_{current_timestamp}_{uuid.uuid4().hex}.jpg"

    detector = ObjectDetection(source=unique_filename, output_dir=os.getenv("DIRECTORY_OUTPUT"))

    response_data = detector.process_image_from_request(image)

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
