# Usa l'immagine base di Python 3.11
FROM python:3.11-slim

# Imposta la directory di lavoro
WORKDIR /app

# Copia i file necessari nell'immagine
COPY requirements.txt .
COPY server.py .
COPY cert.pem .
COPY key.pem .
COPY .env .

# Installa le dipendenze
RUN pip install --no-cache-dir -r requirements.txt

# Esponi la porta su cui il tuo server Flask ascolterà
EXPOSE 5000

# Esegui il server Flask
CMD ["python", "server.py"]
