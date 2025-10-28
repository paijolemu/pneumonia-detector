'use client'; // Ini wajib ada agar bisa menggunakan state dan interaksi pengguna

import { useState } from 'react';
import styles from './page.module.css'; // Kita akan buat file styling ini

export default function Home() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setResult(null); // Reset hasil saat gambar baru dipilih
      setError(null);
      
      // Buat preview gambar untuk ditampilkan di halaman
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(selectedFile);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError("Silakan pilih file gambar terlebih dahulu.");
      return;
    }

    setLoading(true);
    setResult(null);
    setError(null);

    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = async () => {
      const base64Image = reader.result;

      try {
        // Kirim gambar ke backend Python kita di /api/predict
        const response = await fetch('/api/predict', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ image: base64Image }),
        });

        if (!response.ok) {
          const errData = await response.json();
          throw new Error(errData.error || `Error: Terjadi kesalahan di server`);
        }

        const data = await response.json();
        setResult(data); // Simpan hasil prediksi
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false); // Selesai loading
      }
    };
  };

  return (
    <main className={styles.main}>
      <div className={styles.container}>
        <h1 className={styles.title}>Deteksi Pneumonia</h1>
        <p className={styles.description}>
          Unggah gambar X-Ray paru-paru untuk mendeteksi pneumonia.
        </p>

        <form onSubmit={handleSubmit} className={styles.form}>
          <input 
            type="file" 
            accept="image/jpeg, image/png"
            onChange={handleFileChange} 
            className={styles.input}
          />
          <button type="submit" disabled={!file || loading} className={styles.button}>
            {loading ? 'Menganalisis...' : 'Deteksi Sekarang'}
          </button>
        </form>

        {error && <p className={styles.error}>Error: {error}</p>}

        <div className={styles.resultsContainer}>
          {preview && (
            <div className={styles.imagePreview}>
              <h3>Gambar Anda:</h3>
              <img src={preview} alt="Preview" />
            </div>
          )}

          {result && (
            <div className={styles.result}>
              <h3>Hasil Prediksi:</h3>
              <p className={styles.prediction}>
                {result.prediction}
              </p>
              <p className={styles.confidence}>
                Tingkat Keyakinan: {result.confidence}
              </p>
            </div>
          )}
        </div>
      </div>
    </main>
  );
}