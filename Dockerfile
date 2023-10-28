# Usa l'immagine base di Python 3.12 specificando la piattaforma
FROM python:3.11-slim-buster

# Imposta la directory di lavoro
WORKDIR /home/AptarServer

# Copia i file necessari nell'immagine
COPY requirements.txt .

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
