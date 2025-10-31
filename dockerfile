# 1. Mulai dari sebuah Image dasar resmi yang sudah ada Python 3.9
FROM python:3.9-slim

# 2. Tentukan folder kerja di dalam container
WORKDIR /app

# 3. Salin file requirements.txt terlebih dahulu (untuk efisiensi cache)
COPY requirements.txt .

# 4. Install semua library Python yang dibutuhkan
RUN pip install --no-cache-dir -r requirements.txt

# 5. Salin semua file proyek lainnya ke dalam container
COPY . .

# 6. Perintahkan Docker untuk menjalankan aplikasi Flask saat container dimulai
# Gunakan "0.0.0.0" agar bisa diakses dari luar container
CMD ["flask", "run", "--host=0.0.0.0"]