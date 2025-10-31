
# %%
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
import os
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc


base_dir = '/content/drive/MyDrive/Colab Notebooks/chest_xray' # load
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')


IMG_WIDTH, IMG_HEIGHT = 150, 150
BATCH_SIZE = 32

# Data Augmentation dan Preprocessing

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # <--- INI KUNCINYA: 20% dari data training akan jadi data validasi
)


test_datagen = ImageDataGenerator(rescale=1./255)

# Data Generator
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True,
    subset='training' # set training
)

# Generator untuk data validasi (20% dari data training)
validation_generator = train_datagen.flow_from_directory(
    train_dir, # <--- Sumbernya sama dengan train_generator
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False, # ga perlu di-shuffle untuk validasi
    subset='validation' # set validasi
)

# Generator data testing
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=1,
    class_mode='binary',
    shuffle=False
)

print("\nData generator siap digunakan.")
print("Label kelas:", train_generator.class_indices)


# %%

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
base_model.trainable = False # Awalnya ga kepake /frezze

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Compile pertama kali
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

print("--- Arsitektur Awal (Hanya Head yang Dilatih) ---")
model.summary()

# --- SIAPKAN UNTUK FINE-TUNING ---
# unfreeze lapisan-lapisan setelah blok ke-140
base_model.trainable = True
fine_tune_at = 140 # angka 140 adalah yg biasa di pakai

# Freeze semua lapisan sebelum `fine_tune_at`
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Re-compile model dengan learning rate KECIL untuk fine-tuning
model.compile(optimizer=Adam(learning_rate=0.00001), # Learning rate 10x lebih kecil, biasanya 0.0001
              loss='binary_crossentropy',
              metrics=['accuracy']) # itung pakai akurasi 

print("\n=== Arsitektur Setelah Siap Fine-Tuning ===")
model.summary()

# %%
# setelah beberapa training hasilnya masih jelek untuk spesivitas, jadi kita akan mengatasinya menggunakan class weight / bobot agar metrics spesifisitas naik
from sklearn.utils import class_weight

# Hitung class weights
# Ini akan memberikan bobot lebih tinggi pada kelas 'NORMAL' yang jumlahnya lebih sedikit
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)

EPOCHS = 10

history = model.fit( # ini training data
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    class_weight=class_weights 
)
# %%
# evaluate model
# Visualisasi Grafik Training (Akurasi & Loss)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(EPOCHS)

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Prediksi pada Test Set
print("\n--- Mengevaluasi Model pada Test Set ---")
y_pred_proba = model.predict(test_generator, steps=test_generator.samples)
y_pred = (y_pred_proba > 0.5).astype("int32").reshape(-1)
y_true = test_generator.classes

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=train_generator.class_indices.keys(),
            yticklabels=train_generator.class_indices.keys())
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Sensitivitas, Spesifisitas, dan Laporan Klasifikasi --> ini rumuasnya yee
TN, FP, FN, TP = cm.ravel()
sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
print(f"Sensitivity (Recall): {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=train_generator.class_indices.keys()))

# Kurva ROC-AUC --> berbentuk matrics 
fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
# %%
# =================================================================
# (VERSI VERIFIKASI)
# =================================================================
# Tujuan: Verifikasi prediksi dengan ground truth & simpan model.

# catatan penting!! ini pakai grad-cam / heatcam gunannya adalah untuk mengetahui bagaimana komputer mengambil keputusan

# --- Fungsi Bantuan untuk Grad-CAM (tidak ada perubahan) ---
def get_img_array(img_path, size):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    array = tf.keras.preprocessing.image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    # ... (kode fungsi sama seperti sebelumnya) ...
    grad_model = Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[0]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam(img_path, heatmap, alpha=0.6):
    # ... (kode fungsi sama seperti sebelumnya) ...
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    return superimposed_img

# --- Proses Grad-CAM dengan Verifikasi ---
last_conv_layer_name = "out_relu"

try:
    # Ambil path gambar
    pneumonia_dir = os.path.join(test_dir, 'PNEUMONIA')
    normal_dir = os.path.join(test_dir, 'NORMAL')
    pneumonia_img_path = os.path.join(pneumonia_dir, os.listdir(pneumonia_dir)[5]) # Ambil gambar ke-5 untuk variasi
    normal_img_path = os.path.join(normal_dir, os.listdir(normal_dir)[5])

    print(f"Menganalisis gambar PNEUMONIA: {pneumonia_img_path.split('/')[-1]}")
    print(f"Menganalisis gambar NORMAL: {normal_img_path.split('/')[-1]}")

    # Proses gambar PNEUMONIA
    img_array_p = get_img_array(pneumonia_img_path, size=(IMG_WIDTH, IMG_HEIGHT))
    preds_p = model.predict(img_array_p / 255.0)
    pred_label_p = "PNEUMONIA" if preds_p[0][0] > 0.5 else "NORMAL"
    true_label_p = "PNEUMONIA" # Kita tahu karena mengambil dari folder PNEUMONIA
    heatmap_p = make_gradcam_heatmap(img_array_p / 255.0, model, last_conv_layer_name)
    superimposed_img_p = display_gradcam(pneumonia_img_path, heatmap_p)

    # Proses gambar NORMAL
    img_array_n = get_img_array(normal_img_path, size=(IMG_WIDTH, IMG_HEIGHT))
    preds_n = model.predict(img_array_n / 255.0)
    pred_label_n = "PNEUMONIA" if preds_n[0][0] > 0.5 else "NORMAL"
    true_label_n = "NORMAL" # Kita tahu karena mengambil dari folder NORMAL
    heatmap_n = make_gradcam_heatmap(img_array_n / 255.0, model, last_conv_layer_name)
    superimposed_img_n = display_gradcam(normal_img_path, heatmap_n)

    # Tampilkan hasil dengan label yang jelas
    plt.figure(figsize=(14, 7))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(superimposed_img_p, cv2.COLOR_BGR2RGB))
    plt.title(f"Label Asli: {true_label_p}\nPrediksi Model: {pred_label_p}",
              color=('green' if true_label_p == pred_label_p else 'red')) # Warna judul hijau jika benar, merah jika salah
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(superimposed_img_n, cv2.COLOR_BGR2RGB))
    plt.title(f"Label Asli: {true_label_n}\nPrediksi Model: {pred_label_n}",
              color=('green' if true_label_n == pred_label_n else 'red'))
    plt.axis('off')
    plt.show()

except Exception as e:
    print(f"Error saat membuat Grad-CAM: {e}")

# Simpan Model Final
model.save('pneumonia_detection.keras') # ini nama modelnya, ada di file explorer
print("\nModel berhasil disimpan sebagai 'pneumonia_detection.keras'")
# %%
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# ========== PARAMETER (sesuaikan dengan training) ==========
IMG_WIDTH, IMG_HEIGHT = 150, 150
MODEL_PATH = 'pneumonia_detection_model.keras'
# Ganti dengan path gambar yang ingin Anda uji
IMAGE_TO_TEST = '/content/drive/MyDrive/Colab Notebooks/chest_xray/test/PNEUMONIA/person101_bacteria_484.jpeg' # ini ganti dengan path yg mau di uji
# IMAGE_TO_TEST = '/content/drive/MyDrive/Colab Notebooks/chest_xray/test/NORMAL/IM-0019-0001.jpeg'

# ========== FUNGSI BANTUAN 
def get_img_array(img_path, size):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    array = tf.keras.preprocessing.image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[0]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam(img_path, heatmap, alpha=0.6):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    return superimposed_img

# ========== PROSES UTAMA ==========

# 1. Muat model yang sudah dilatih
print("Memuat model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model berhasil dimuat.")

# 2. Muat dan preprocess gambar
img_array = get_img_array(IMAGE_TO_TEST, size=(IMG_WIDTH, IMG_HEIGHT))
# Normalisasi gambar persis seperti saat training
img_array_preprocessed = img_array / 255.0

# 3. Lakukan prediksi
prediction = model.predict(img_array_preprocessed)[0][0]
confidence = prediction if prediction > 0.5 else 1 - prediction
label = "PNEUMONIA" if prediction > 0.5 else "NORMAL"

print(f"\nPrediksi: {label}")
print(f"Confidence: {confidence * 100:.2f}%")

# 4. Buat dan tampilkan Grad-CAM
last_conv_layer_name = "out_relu" # Pastikan nama layer ini sama
heatmap = make_gradcam_heatmap(img_array_preprocessed, model, last_conv_layer_name)
superimposed_img = display_gradcam(IMAGE_TO_TEST, heatmap)

# Tampilkan hasil
plt.figure(figsize=(8, 8))
plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
plt.title(f"Prediksi: {label} ({confidence * 100:.2f}%)")
plt.axis('off')
plt.show()
# %%
# prediksi juga
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os


MODEL_PATH = '/content/model_prediksi.keras' # Ganti dengan nama model Anda jika berbeda
IMG_WIDTH, IMG_HEIGHT = 150, 150
class_labels = {0: 'NORMAL', 1: 'PNEUMONIA'} # Sesuai dengan output generator

# Contoh path untuk gambar PNEUMONIA
image_path_pneumonia = '/content/drive/MyDrive/Colab Notebooks/chest_xray/test/PNEUMONIA/person101_bacteria_486.jpeg'

# Contoh path untuk gambar NORMAL
image_path_normal = '/content/drive/MyDrive/Colab Notebooks/chest_xray/test/NORMAL/IM-0011-0001-0001.jpeg'

# --- GANTI BARIS INI UNTUK MEMILIH GAMBAR ANDA ---
IMAGE_TO_PREDICT = image_path_pneumonia
# ----------------------------------------------------


# cek
try:
    print(f"Memuat model dari: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model berhasil dimuat.")
except Exception as e:
    print(f"Error! Gagal memuat model: {e}")
    print("Pastikan nama file model sudah benar dan berada di direktori yang sama.")



print(f"Memproses gambar: {IMAGE_TO_PREDICT.split('/')[-1]}")

# a. Muat gambar dan ubah ukurannya
img = tf.keras.preprocessing.image.load_img(
    IMAGE_TO_PREDICT, target_size=(IMG_WIDTH, IMG_HEIGHT)
)

# b. Ubah gambar menjadi array numpy
img_array = tf.keras.preprocessing.image.img_to_array(img)

img_array = np.expand_dims(img_array, axis=0)

# d. Normalisasi nilai piksel (rescale)
preprocessed_image = img_array / 255.0

prediction_proba = model.predict(preprocessed_image)[0][0]


threshold = 0.5 # ini penting banget, tapi tetep di 0.5 aja agar seimbang

if prediction_proba > threshold:
    predicted_class_index = 1
    confidence = prediction_proba * 100
else:
    predicted_class_index = 0
    confidence = (1 - prediction_proba) * 100

predicted_label = class_labels[predicted_class_index]

# visual hasil
print(f"\n--- HASIL PREDIKSI ---")
print(f"Probabilitas Mentah dari Model: {prediction_proba:.4f}")
print(f"Prediksi: Gambar ini adalah **{predicted_label}**")
print(f"Tingkat Keyakinan (Confidence): {confidence:.2f}%")

# Tampilkan gambar asli dengan judul prediksinya
img_display = plt.imread(IMAGE_TO_PREDICT)
plt.imshow(img_display, cmap='gray')
plt.title(f"Prediksi: {predicted_label} ({confidence:.2f}%)")
plt.axis('off') # Sembunyikan sumbu x dan y
plt.show()