# Usa l'immagine base di Python 3.11 specificando la piattaforma
FROM python:3.11-slim-buster

# Imposta la directory di lavoro
WORKDIR /app

# Copia i file necessari nell'immagine
COPY requirements.txt .
COPY server.py .
COPY cert.pem .
COPY key.pem .

# Installa le dipendenze necessarie e rimuovi il pacchetto dopo l'uso per ridurre la dimensione dell'immagine
RUN apt-get update && \
    apt-get install -y gcc python3-dev libgl1-mesa-glx && \
    docker exec -it nome_container /bin/bash
    pip install --no-cache-dir -r requirements.txt && \
    apt-get purge -y gcc python3-dev && \
    apt-get autoremove -y && \
    apt-get clean

# Esponi la porta su cui il tuo server Flask ascolterà
EXPOSE 5000

# Esegui il server Flask
CMD ["python", "server.py"]
