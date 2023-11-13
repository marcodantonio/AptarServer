# Usa l'immagine base di Python 3.11 specificando la piattaforma
FROM python:3.11-slim-buster

# Installa le dipendenze necessarie e pulisci dopo l'uso per ridurre la dimensione dell'immagine
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev libgl1-mesa-glx libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Imposta la directory di lavoro (questa sarà la home di 'server')
WORKDIR /home/server

# Definisci le variabili d'ambiente per l'utente e il gruppo (imposta valori predefiniti)
ENV PUID=1001
ENV PGID=1001

# Crea un utente personalizzato con UID e GID specifici
RUN groupadd -g $PGID server && \
    useradd -u $PUID -g $PGID -m server

# Cambia la proprietà della directory di lavoro all'utente server
RUN chown -R server:server /home/server

# Copia i file necessari nell'immagine
COPY --chown=server:server requirements.txt server.py ./

# Esegui come utente 'server'
USER server

# Installa le dipendenze Python (come utente 'server')
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --upgrade pip

# Crea tutte le directory e sottocartelle necessarie all'interno di /home/server (come utente 'server')
RUN mkdir -p ./output/{matched_images,matched_images_with_boxes,matched_labels,matched_labelstudio,unmatched_images} \
    ./need_validation/{matched_images,matched_images_with_boxes,matched_labels,matched_labelstudio,unmatched_images} \
    ./validated/{matched_images,matched_images_with_boxes,matched_labels,matched_labelstudio,unmatched_images} \
    ./wrong_detections/{matched_images,matched_images_with_boxes,matched_labels,matched_labelstudio,unmatched_images} \
    ./ssl ./models

# Copia il modello YoloV8m (come utente 'server')
COPY --chown=server:server models/YoloV8m.pt ./models/

# Esegui il server Flask
CMD ["python", "server.py"]
