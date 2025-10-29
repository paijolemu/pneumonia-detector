import os
import numpy as np
import cv2
from PIL import Image
from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
from ctransformers import AutoModelForCausalLM

app = Flask(__name__)

# --- KONFIGURASI ---
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- MUAT MODEL DETEKSI PNEUMONIA (HANYA INI YANG DIMUAT DI AWAL) ---
try:
    model_pneumonia = tf.keras.models.load_model('model.keras')
    print("Model Deteksi Pneumonia berhasil dimuat.")
except Exception as e:
    print(f"Error memuat model pneumonia: {e}")
    model_pneumonia = None

# --- [PERUBAHAN BESAR] KITA TIDAK MEMUAT LLM DI SINI ---
# Variabel global untuk menyimpan model LLM setelah dimuat
model_llm = None 
print("Model LLM akan dimuat saat pertama kali dibutuhkan (lazy loading).")


# --- KONFIGURASI GRAD-CAM ---
IMAGE_SIZE = (150, 150) 
CLASS_NAMES = ['Normal', 'Pneumonia'] 
last_conv_layer_name = "expanded_conv_project_BN" # Pastikan nama ini sudah benar

# --- FUNGSI-FUNGSI BANTUAN (TIDAK BERUBAH) ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image_for_model(image_path, target_size):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[0]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.6):
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + img
    cv2.imwrite(cam_path, superimposed_img)

# --- [PERUBAHAN BESAR] FUNGSI PENJELASAN DENGAN LAZY LOADING ---
def generate_llm_explanation(prediction, confidence):
    global model_llm # Gunakan variabel global

    # Jika model LLM belum dimuat, muat sekarang
    if model_llm is None:
        try:
            print("Model LLM belum dimuat. Memuat sekarang (ini hanya terjadi sekali)...")
            model_path = os.path.join('model_llm', 'tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf')
            model_llm = AutoModelForCausalLM.from_pretrained(
                model_path,
                model_type='llama',
                max_new_tokens=512,
                temperature=0.7
            )
            print("Model LLM Lokal berhasil dimuat.")
        except Exception as e:
            print(f"Error memuat model LLM Lokal: {e}")
            return "Gagal memuat Model AI Penjelas."

    # Template prompt
    # --- TEMPLATE BARU YANG LEBIH TEGAS ---
    # --- TEMPLATE BARU YANG JAUH LEBIH EFEKTIF ---
    template = f"""<|system|>
        Anda adalah seorang asisten medis AI yang simpatik. Tugas Anda adalah menjelaskan hasil analisis kepada pasien dalam Bahasa Indonesia yang mudah dimengerti.</s>
        <|user|>
        Hasil analisis saya adalah 'Normal' dengan keyakinan 95%. Apa artinya?</s>
        <|assistant|>
        Tentu. Hasil analisis menunjukkan bahwa, berdasarkan gambar yang Anda berikan, model AI kami dengan keyakinan sebesar 95% tidak menemukan tanda-tanda pneumonia. Ini adalah indikasi awal yang baik. Namun, 'tingkat keyakinan' ini bukanlah diagnosis pasti. Untuk kepastian, sangat penting untuk tetap berkonsultasi dengan dokter.</s>
        <|user|>
        Hasil analisis saya adalah '{prediction}' dengan keyakinan {confidence}%. Apa artinya?</s>
        <|assistant|>
        """
    try:
        response = model_llm(template)
        return response
    except Exception as e:
        print(f"Error saat memanggil LLM Lokal: {e}")
        return "Terjadi kesalahan saat mencoba menghasilkan penjelasan dari AI."

# --- ROUTE FLASK (TIDAK BERUBAH) ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # ... (Semua kode di dalam fungsi predict ini tidak berubah sama sekali) ...
    if model_pneumonia is None:
        return render_template('index.html', error="Model AI Deteksi tidak berhasil dimuat.")

    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            img_array = preprocess_image_for_model(filepath, IMAGE_SIZE)
            prediction_score = model_pneumonia.predict(img_array)[0][0]
            
            heatmap_filename = None
            if prediction_score > 0.5:
                class_name = CLASS_NAMES[1]
                confidence_val = prediction_score * 100
                heatmap = make_gradcam_heatmap(img_array, model_pneumonia, last_conv_layer_name)
                heatmap_filename = 'heatmap_' + filename
                heatmap_path = os.path.join(app.config['UPLOAD_FOLDER'], heatmap_filename)
                save_and_display_gradcam(filepath, heatmap, heatmap_path)
            else:
                class_name = CLASS_NAMES[0]
                confidence_val = (1 - prediction_score) * 100

            confidence_str = f"{confidence_val:.2f}"
            llm_explanation = generate_llm_explanation(class_name, confidence_str)

            return render_template('index.html', 
                                   filename=filename, 
                                   prediction=class_name,
                                   confidence=confidence_str,
                                   heatmap_filename=heatmap_filename,
                                   llm_explanation=llm_explanation)

        except Exception as e:
            return render_template('index.html', error=f"Terjadi kesalahan saat prediksi: {e}")
    else:
        return render_template('index.html', error="Format file tidak valid. Unggah file .png, .jpg, atau .jpeg")


if __name__ == '__main__':
    app.run(debug=True)