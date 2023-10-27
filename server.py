import os
from flask_cors import CORS
from dotenv import load_dotenv
from flask import Flask, request, jsonify

# Carica le variabili di configurazione dal file .env
load_dotenv()

app = Flask(__name__)
CORS(app)

@app.route('/detect_objects', methods=['POST'])
def detect_objects():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    # Se necessario, puoi salvare l'immagine o processarla qui.
    # image = request.files['image']
    print("Immagine ricevuta!")  # Stampa un messaggio quando l'immagine viene ricevuta
    return jsonify({'message': 'Image received successfully!'})

if __name__ == '__main__':
    ssl_cert = os.environ.get('SSL_CERT')
    ssl_key = os.environ.get('SSL_KEY')

    if ssl_cert and ssl_key:
        ssl_context = (ssl_cert, ssl_key)
    else:
        ssl_context = None

    app.run(debug=os.environ.get('FLASK_DEBUG', 'False'),
            host=os.environ.get('FLASK_HOST', '0.0.0.0'),
            port=int(os.environ.get('FLASK_PORT', 5000)),
            ssl_context=ssl_context)

