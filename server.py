"""
Descrizione del Server:

Questo server Flask si occupa di elaborare le immagini inviate tramite una chiamata POST, utilizzando un modello YOLO
per il riconoscimento degli oggetti. Le immagini processate e i risultati del riconoscimento (coordinate dei bounding
boxes, classi degli oggetti, e confidenze) vengono salvati in diverse cartelle di output. Il server supporta anche
l'esportazione dei dati per LabelStudio in formato JSON.

Funzionalità principali:
- Ricezione ed elaborazione di immagini tramite richieste POST.
- Salvataggio delle immagini originali, immagini con bounding boxes, etichette normalizzate, e dati per LabelStudio.
- Gestione delle variabili di configurazione tramite file .env.
- Logging delle attività del server.
- Supporto per la configurazione SSL.

Utilizzo:
Il server può essere avviato eseguendo il file Python e accetta richieste POST all'endpoint '/detect_objects'.
"""

##################################################################################################################################
##                                                                                                                              ##
##                                                      IMPORT + CONFIG ENV E CORS                                              ##
##                                                                                                                              ##
##################################################################################################################################

# Import
import os
import cv2
import json
import torch
import shutil
import logging
import numpy as np
from flask_cors import CORS
from ultralytics import YOLO
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_file

# Carica le variabili di configurazione dal file .env
load_dotenv()

# Flask e CORS
app = Flask(__name__)
CORS(app)
app.logger.setLevel(logging.INFO)


##################################################################################################################################
##                                                                                                                              ##
##                                                      ESTRAZIONE RISULTATI                                                    ##
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
##                                                      ESTRAZIONE RISULTATI                                                    ##
##                                                                                                                              ##
##################################################################################################################################

def save_label(xyxys, class_ids, confidences, frame, label_file_path):
    labels = []
    image_height, image_width = frame.shape[:2]  # Ottiene le dimensioni dell'immagine

    for xyxy, class_id, confidence in zip(xyxys[0], class_ids[0], confidences[0]):
        x1, y1, x2, y2 = map(int, xyxy)

        # Chiama la funzione di normalizzazione
        norm_center_x, norm_center_y, norm_bbox_width, norm_bbox_height = normalize_label(x1, y1, x2, y2, image_width,
                                                                                          image_height)

        # Aggiungi l'etichetta normalizzata alla lista, includendo la confidence
        label_str = (f"{int(class_id)} {norm_center_x:.6f} {norm_center_y:.6f} "
                     f"{norm_bbox_width:.6f} {norm_bbox_height:.6f}\n")
        labels.append(label_str)

    with open(label_file_path, "w") as f:
        f.writelines(labels)


def move_files(unique_filename, source, dest):
    # Estrai il nome del file senza estensione
    file_name_without_extension, _ = os.path.splitext(unique_filename)

    # Liste delle sottocartelle e delle loro corrispondenti estensioni
    subfolders_extensions = {'matched_images': '.jpg', 'matched_images_with_boxes': '.jpg', 'matched_labels': '.txt',
                             'matched_labelstudio': '.json'}

    for subfolder, extension in subfolders_extensions.items():
        # Percorso completo sorgente e destinazione
        source_file = file_name_without_extension + extension
        dest_file = source_file

        source_path = os.path.join(source, subfolder, source_file)
        dest_path = os.path.join(dest, subfolder, dest_file)

        # Verifica se il file esiste e, in caso affermativo, spostalo
        if os.path.exists(source_path):
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)  # Crea la cartella di destinazione se non esiste
            shutil.move(source_path, dest_path)
        else:
            app.logger.error(f"File non trovato: {source_path}")


def update_label(frame, detect_response, unique_filename, class_id_dict):
    image_height, image_width, _ = frame.shape
    base_path = 'output/matched_labels'
    file_name, _ = os.path.splitext(unique_filename)
    txt_file_path = os.path.join(base_path, f"{file_name}.txt")

    try:
        with open(txt_file_path, 'r') as file:
            lines = file.readlines()

        if not lines:
            raise ValueError("Il file è vuoto o non contiene dati validi.")

        file_bboxes = {}
        for i, line in enumerate(lines):
            parts = line.split()
            if len(parts) >= 5:
                # Arrotonda le coordinate già normalizzate alla seconda cifra decimale
                bbox_key = tuple(round(float(x), 2) for x in parts[1:5])
                file_bboxes[bbox_key] = i

        for detection in detect_response:
            class_name = detection['class']
            if class_name in class_id_dict:
                # Normalizza e arrotonda i bounding box da detect_response
                x1, y1, x2, y2 = detection['bbox']
                norm_bbox = normalize_label(x1, y1, x2, y2, image_width, image_height)
                norm_bbox_rounded = tuple(round(b, 2) for b in norm_bbox)

                if norm_bbox_rounded in file_bboxes:
                    line_index = file_bboxes[norm_bbox_rounded]
                    parts = lines[line_index].split()
                    parts[0] = str(class_id_dict[class_name])  # Aggiorna l'ID di classe
                    lines[line_index] = " ".join(parts) + "\n"

        with open(txt_file_path, 'w') as file:
            file.writelines(lines)

    except IOError as e:
        app.logger.error(f"Errore di I/O: {e}")
    except ValueError as e:
        app.logger.error(f"Errore di valore: {e}")


def normalize_label(x1, y1, x2, y2, image_width, image_height):
    # Formatta i valori con 6 cifre decimali
    norm_center_x = "{:.6f}".format(((x1 + x2) / 2) / image_width)
    norm_center_y = "{:.6f}".format(((y1 + y2) / 2) / image_height)
    norm_bbox_width = "{:.6f}".format((x2 - x1) / image_width)
    norm_bbox_height = "{:.6f}".format((y2 - y1) / image_height)

    return float(norm_center_x), float(norm_center_y), float(norm_bbox_width), float(norm_bbox_height)


class ObjectDetection:

    ##############################################################################################################################
    ##                                                                                                                          ##
    ##                                                      COSTRUTTORE                                                         ##
    ##                                                                                                                          ##
    ##############################################################################################################################

    def __init__(self):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

        self.model = YOLO(os.path.join("models", os.environ.get('YOLO_MODEL'))).to(self.device)
        self.class_name_dict = self.model.names
        self.class_id_dict = {v: k for k, v in self.class_name_dict.items()}

    ##############################################################################################################################
    ##                                                                                                                          ##
    ##                                                      PROCESS IMAGE                                                       ##
    ##                                                                                                                          ##
    ##############################################################################################################################

    def process_image(self, frame, unique_filename):

        # Processa l'immagine
        plot_results, xyxys, confidences, class_ids = plot_bboxes(self.model(frame))

        # Salva il risultato in cartelle
        self.save_results(frame, xyxys, confidences, class_ids, unique_filename)

        # Converti i risultati in un formato da restituire come risposta
        detections = [{"class": self.class_name_dict[int(cls)], "confidence": float(conf), "bbox": [float(x) for x in box]} for
                      (box, conf, cls) in zip(xyxys[0], confidences[0], class_ids[0])]

        response_data = {"detections": detections}

        # stampa dei risultati di prova
        app.logger.info(response_data)

        return response_data

    ##############################################################################################################################
    ##                                                                                                                          ##
    ##                                                      SALVA RISULTATO                                                     ##
    ##                                                                                                                          ##
    ##############################################################################################################################

    def save_results(self, frame, xyxys, confidences, class_ids, unique_filename):

        # Percorsi cartelle di output
        matched_img_dir = "output/matched_images"
        matched_img_with_boxes_dir = "output/matched_images_with_boxes"
        matched_labels_dir = "output/matched_labels"
        matched_labelstudio_dir = "output/matched_labelstudio"
        unmatched_img_dir = "output/unmatched_images"

        # Verifica se ci sono rilevamenti (match) in uno qualsiasi degli elenchi xyxys
        if any(len(x) > 0 for x in xyxys):
            os.makedirs(matched_img_dir, exist_ok=True)
            os.makedirs(matched_img_with_boxes_dir, exist_ok=True)
            os.makedirs(matched_labels_dir, exist_ok=True)
            os.makedirs(matched_labelstudio_dir, exist_ok=True)

            # Salva l'immagine originale
            cv2.imwrite(os.path.join(matched_img_dir, unique_filename), frame)

            # Salva l'immagine con i bounding box
            cv2.imwrite(os.path.join(matched_img_with_boxes_dir, unique_filename),
                        self.save_image_with_boxes(frame, xyxys, class_ids, confidences))
            # Salva le coordinate normalizzate
            save_label(xyxys, class_ids, confidences, frame,
                       os.path.splitext(os.path.join(matched_labels_dir, unique_filename))[0] + ".txt")

            # Salva il file labelstudio
            self.save_labelstudio(xyxys, class_ids, confidences, unique_filename,
                                  os.path.splitext(os.path.join(matched_labelstudio_dir, unique_filename))[0] + ".json")

        else:
            # Nessun rilevamento (nessun match)
            os.makedirs(unmatched_img_dir, exist_ok=True)
            cv2.imwrite(os.path.join(unmatched_img_dir, unique_filename), frame)

    def save_image_with_boxes(self, frame, xyxys, class_ids, confidences):
        for xyxy, class_id, confidence in zip(xyxys[0], class_ids[0], confidences[0]):
            x1, y1, x2, y2 = map(int, xyxy)
            label = self.class_name_dict[class_id.item()]  # item() per ottenere il valore scalare
            confidence = confidence.item()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness=3)
            cv2.putText(frame, f"{label}: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        return frame

    ##############################################################################################################################
    ##                                                                                                                          ##
    ##                                                      LABELSTUDIO JSON                                                    ##
    ##                                                                                                                          ##
    ##############################################################################################################################

    def generate_labelstudio_json(self, xyxy, class_id, frame_path):
        x1, y1, x2, y2 = map(int, xyxy)
        label = self.class_name_dict[class_id.item()]  # Usiamo item() per ottenere il valore scalare

        label_data = {"frame_path": frame_path, "from_name": "label", "to_name": "image", "type": "rectanglelabels",
                      "value": {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1, "rectanglelabels": [label]}}

        return label_data

    def save_labelstudio(self, xyxys, class_ids, confidences, frame_path, labelstudio_output_file):
        labelstudio_data = []

        for xyxy, class_id, confidence in zip(xyxys[0], class_ids[0], confidences[0]):
            label_data = self.generate_labelstudio_json(xyxy, class_id, frame_path)
            labelstudio_data.append(label_data)

        os.makedirs(os.path.dirname(labelstudio_output_file), exist_ok=True)

        with open(labelstudio_output_file, "w") as json_file:
            json.dump(labelstudio_data, json_file, indent=4)


##################################################################################################################################
##                                                                                                                              ##
##                                                          CHIAMATE POST                                                       ##
##                                                                                                                              ##
##################################################################################################################################

app.config['ALLOWED_CLASSES'] = list(ObjectDetection().class_name_dict.values())


@app.route('/detect_objects', methods=['POST'])
def detect_objects():
    try:
        if 'image' not in request.files:
            raise ValueError('No image provided')
        if 'unique_filename' not in request.form:
            raise ValueError('No unique_filename provided')

        frame = cv2.imdecode(np.frombuffer(request.files['image'].read(), np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Non è stata fornita un'immagine valida.")

        response_data = ObjectDetection().process_image(frame, request.form['unique_filename'])
        return jsonify(response_data)

    except ValueError as e:
        app.logger.warning(f"Errore di valore: {e}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        app.logger.error(f"Errore generico: {e}")
        return jsonify({'error': 'Si è verificato un errore durante l\'elaborazione dell\'immagine'}), 500


@app.route('/user_response', methods=['POST'])
def user_response():
    try:
        data = request.get_json()
        if not data:
            raise ValueError('Nessun dato ricevuto')

        unique_filename_with_ext = data.get('unique_filename')
        detect_response = data.get('detectResponse')
        user_response = data.get('userResponse')
        is_modified = data.get('isModified', False)  # Default a False se non fornito

        unique_filename, _ = os.path.splitext(unique_filename_with_ext)
        frame = cv2.imread(os.path.join('output/matched_images/', unique_filename_with_ext))
        if frame is None:
            raise ValueError("Immagine non trovata o non leggibile")

        # Gestisci la logica in base a se le rilevazioni sono state modificate
        if user_response == 'yes':
            if is_modified:
                update_label(frame, detect_response, unique_filename, ObjectDetection().class_id_dict)

            # Poi esegui il move dei file
            move_files(unique_filename, "output", "need_validation")
        else:
            move_files(unique_filename, "output", "wrong_detections")

        return jsonify({'message': 'Risposta ricevuta con successo'}), 200

    except ValueError as e:
        app.logger.error(f"Errore di valore: {e}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        app.logger.error(f"Errore generico: {e}")
        return jsonify({'error': 'Si è verificato un errore durante la ricezione della risposta utente'}), 500


@app.route('/get_image', methods=['POST'])
def get_image():
    # Ottieni il filename dal form data
    unique_filename = request.form.get('unique-filename')

    if not unique_filename:
        return "Unique filename is missing", 400

    # Percorso dell'immagine: cambia questo con il percorso effettivo dove le immagini sono salvate
    image_path = os.path.join('output/matched_images_with_boxes', unique_filename)

    if not os.path.exists(image_path):
        return "File not found.", 404

    # Invia il file come risposta
    return send_file(image_path, mimetype='image/jpeg')


##################################################################################################################################
##                                                                                                                              ##
##                                                          CHIAMATA MAIN                                                       ##
##                                                                                                                              ##
##################################################################################################################################

if __name__ == '__main__':

    # Verifica che la variabile d'ambiente YOLO_MODEL sia impostata
    if not os.path.join("models", os.environ.get('YOLO_MODEL')):
        raise ValueError("YOLO_MODEL environment variable is not set")

    ssl_cert = os.path.join("ssl", os.environ.get('SSL_CERT'))
    ssl_key = os.path.join("ssl", os.environ.get('SSL_KEY'))

    if ssl_cert and ssl_key:
        ssl_context = (ssl_cert, ssl_key)
    else:
        ssl_context = None

    app.run(debug=os.environ.get('FLASK_DEBUG', 'False'), host=os.environ.get('FLASK_HOST', '0.0.0.0'),
            port=int(os.environ.get('FLASK_PORT', 5000)), ssl_context=ssl_context)
