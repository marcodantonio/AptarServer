FROM python:3.11-slim-buster

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev libgl1-mesa-glx libglib2.0-0

# Imposta la directory di lavoro
WORKDIR /home/server

# Copia il file requirements.txt e installa i requisiti come utente root
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --upgrade pip

# Pulizia per ridurre la dimensione dell'immagine
RUN apt-get purge -y --auto-remove gcc python3-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Definisci le variabili d'ambiente per l'utente e il gruppo (imposta valori predefiniti)
ENV PUID=1001
ENV PGID=1001

# Crea un utente personalizzato con UID e GID specifici
RUN groupadd -g $PGID server && \
    useradd -u $PUID -g $PGID -m server

# Cambia la proprietà della directory di lavoro all'utente server
RUN chown -R server:server /home/server

# Copia i file necessari nell'immagine
COPY --chown=server:server server.py ./

# Esegui come utente 'server'
USER server

# Crea tutte le directory e sottocartelle necessarie (come utente 'server')
RUN mkdir -p ./output/{matched_images,matched_images_with_boxes,matched_labels,matched_labelstudio,unmatched_images} \
    ./need_validation/{matched_images,matched_images_with_boxes,matched_labels,matched_labelstudio,unmatched_images} \
    ./validated/{matched_images,matched_images_with_boxes,matched_labels,matched_labelstudio,unmatched_images} \
    ./wrong_detections/{matched_images,matched_images_with_boxes,matched_labels,matched_labelstudio,unmatched_images} \
    ./ssl ./models

# Copia il modello YoloV8m (come utente 'server')
COPY --chown=server:server models/YoloV8m.pt ./models/

# Copia l'entrypoint script nel container e rendilo eseguibile
COPY entrypoint.sh /entrypoint.sh
RUN chown server:server /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "server.py"]
