# Usa l'immagine base di Python 3.11 specificando la piattaforma
FROM python:3.11-slim-buster

# Imposta la directory di lavoro
WORKDIR /home/server

# Crea tutte le directory e sottocartelle necessarie
RUN mkdir -p /output/{matched_images,matched_images_with_boxes,matched_labels,matched_labelstudio,unmatched_images} \
    /need_validation/{matched_images,matched_images_with_boxes,matched_labels,matched_labelstudio,unmatched_images} \
    /validated/{matched_images,matched_images_with_boxes,matched_labels,matched_labelstudio,unmatched_images} \
    /wrong_detections/{matched_images,matched_images_with_boxes,matched_labels,matched_labelstudio,unmatched_images} \
    /ssl /models

# Copia i file necessari nell'immagine
COPY requirements.txt server.py ./
COPY models/YoloV8m.pt /models/

# Installa le dipendenze necessarie e pulisci dopo l'uso per ridurre la dimensione dell'immagine
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev libgl1-mesa-glx libglib2.0-0 && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --upgrade pip && \
    apt-get purge -y --auto-remove gcc python3-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Esegui il server Flask
CMD ["python", "server.py"]