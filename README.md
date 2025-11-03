🧠 Pneumonia Detection using Deep Learning & LLM Integration

Proyek ini saya buat untuk mengembangkan sistem deteksi pneumonia berbasis citra X-ray menggunakan deep learning yang sederhana namun efektif. Tujuannya adalah menciptakan pipeline end-to-end yang mampu memproses gambar X-ray paru-paru, melakukan klasifikasi otomatis (normal atau pneumonia), serta memberikan penjelasan berbasis bahasa alami melalui integrasi dengan LLM (TinyLlama).

Selama pengembangan, saya membangun sistem ini dari nol dalam waktu 5 hari — dimulai dari pemilihan dataset, pelatihan model, evaluasi performa, hingga pembuatan API dan container Docker agar mudah dijalankan di mana saja.

Langkah-langkah utama yang saya lakukan dalam proyek ini adalah:

Eksplorasi Dataset – menggunakan dataset publik chest X-ray, memisahkan data train, validation, dan test secara stratified.

Preprocessing Data – normalisasi piksel, augmentasi (rotasi, shift, flip) untuk mencegah overfitting.

Training Model – menerapkan arsitektur MobileNetV2 (transfer learning) menggunakan Keras dengan optimasi Adam dan learning rate scheduler.

Evaluasi Performa – menghitung metrik utama seperti akurasi, sensitivitas, spesifisitas, F1-score, serta AUC untuk menilai performa model secara menyeluruh.

Integrasi LLM (TinyLlama) – menambahkan modul kecil berbasis LLM untuk menjelaskan hasil prediksi dengan bahasa manusia, membantu interpretasi model.

Deployment – membuat API sederhana menggunakan Flask (app.py), menulis Dockerfile untuk containerisasi, serta mendesain antarmuka web minimalis di folder templates/.

Hasil akhir menunjukkan bahwa model ini mampu mendeteksi pneumonia dengan akurasi sekitar 89%, sensitivitas 90%, dan AUC 0.95. Sistem ini kemudian dikemas secara lengkap sehingga pengguna cukup mengunggah gambar X-ray dan langsung mendapatkan hasil prediksi beserta penjelasan singkat.

Proyek ini menunjukkan bagaimana kecerdasan buatan (AI) dapat membantu proses analisis radiologi secara cepat dan transparan, sekaligus menjadi contoh penerapan AI + LLM dalam bidang medis dengan sumber daya terbatas.

⚠️ Catatan: Proyek ini bersifat penelitian/eksperimen dan tidak dimaksudkan untuk penggunaan klinis langsung tanpa validasi medis yang menyeluruh.
========================================================== english ==================================================================
🧠 Pneumonia Detection using Deep Learning & LLM Integration

This project was developed to build an AI-based system for detecting pneumonia from chest X-ray images using deep learning. The main goal is to design an end-to-end pipeline capable of processing X-ray images, performing automated classification (normal or pneumonia), and generating natural language explanations through integration with a lightweight LLM (TinyLlama).

I built this project from scratch in just 5 days, covering every stage — from dataset selection, model training, and performance evaluation, to API development and Docker containerization for easy deployment.

🔧 Development Steps

Dataset Exploration – used a public chest X-ray dataset, split into train/validation/test sets with stratification.

Data Preprocessing – performed pixel normalization and data augmentation (rotation, shifting, flipping) to improve model generalization.

Model Training – implemented a MobileNetV2 architecture with transfer learning using Keras, optimized with the Adam optimizer and learning rate scheduling.

Model Evaluation – calculated accuracy, sensitivity, specificity, F1-score, and AUC to assess model performance comprehensively.

LLM Integration (TinyLlama) – added a lightweight language model to automatically generate human-readable explanations for the model’s predictions.

Deployment – created a Flask-based API (app.py), wrote a Dockerfile for containerization, and designed a simple web interface under the templates/ directory.

The final model achieved an accuracy of ~89%, sensitivity of 90%, and AUC of 0.95, demonstrating reliable performance in distinguishing pneumonia cases from normal chest X-rays. The system is fully packaged so that users can simply upload an image and receive both the prediction and an explanatory summary instantly.

This project showcases how Artificial Intelligence (AI) can assist radiology analysis efficiently and transparently, serving as an example of AI + LLM synergy applied in medical imaging within limited computing resources.

⚠️ Disclaimer: This project is intended for research and educational purposes only. It should not be used for clinical decision-making without proper medical validation.
