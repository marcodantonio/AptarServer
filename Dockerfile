# Usa l'immagine base di Python 3.12 specificando la piattaforma
FROM python:3.11-slim-buster

# Imposta la directory di lavoro
WORKDIR /home/server

# Crea la directory "output" nella directory radice del container
RUN mkdir /output

# Crea altre sottocartelle all'interno della directory "output"
RUN mkdir /output/matched_images
RUN mkdir /output/matched_images_with_boxes
RUN mkdir /output/matched_labels
RUN mkdir /output/matched_labelstudio
RUN mkdir /output/unmatched_images

# Crea le altre cartelle
RUN mkdir /ssl
RUN mkdir /models

# Copia i file necessari nell'immagine
COPY requirements.txt .
COPY server.py .
COPY models/YoloV8m.pt models

# Installa le dipendenze necessarie e rimuovi il pacchetto dopo l'uso per ridurre la dimensione dell'immagine
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev libgl1-mesa-glx libglib2.0-0 && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --upgrade pip && \
    apt-get purge -y gcc python3-dev && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Esegui il server Flask
CMD ["python", "server.py"]
