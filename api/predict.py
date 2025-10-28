from http.server import BaseHTTPRequestHandler
import json
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import base64

MODEL_PATH = 'model/model.keras'
model = tf.keras.models.load_model(MODEL_PATH)

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data)

        if 'image' not in data:
            self.send_response(400)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': 'No image data found'}).encode('utf-8'))
            return

        try:
            img_data = base64.b64decode(data['image'].split(',')[1])
            img = Image.open(io.BytesIO(img_data)).convert('RGB')

            # --- PREPROCESSING GAMBAR (SUDAH DIPERBAIKI) ---
            # Menyamakan ukuran dengan saat training
            img = img.resize((150, 150)) # <--- PERUBAHAN KRUSIAL DI SINI
            
            img_array = np.array(img)
            img_array = img_array / 255.0  # Normalisasi (ini sudah benar)
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)
            score = float(prediction[0][0])

            label = "Pneumonia" if score > 0.5 else "Normal"
            confidence = score if score > 0.5 else 1 - score

            result = {
                'prediction': label,
                'confidence': f"{confidence * 100:.2f}%"
            }

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode('utf-8'))

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            # Mengirim pesan error yang lebih detail ke frontend untuk debugging
            self.wfile.write(json.dumps({'error': f"Server error: {str(e)}"}).encode('utf-8'))