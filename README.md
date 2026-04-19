# 🧠 Nusantara Phone Mapper

Mapping Fonetik Antar Bahasa Indonesia dan Sunda

---

## 📌 Deskripsi Project

Project ini bertujuan untuk melakukan **mapping kata antar bahasa (Indonesia ↔ Sunda)** berdasarkan **kemiripan fonetik** menggunakan pendekatan sederhana yang terinspirasi dari paper *Cross-Language Phoneme Mapping (CrossPhon)*.

Alih-alih menggunakan model deep learning kompleks, project ini menggunakan:

* Representasi fonetik sederhana (huruf → phoneme)
* Vector fonetik
* Perhitungan jarak (distance)

---

## 🚀 Fitur Utama

* 🔤 Konversi teks ke representasi fonetik sederhana
* 📊 Mapping kata berdasarkan kemiripan bunyi
* 🌏 Cross-language mapping (Indonesia → Sunda)
* 📁 Support dataset real (Mozilla Common Voice)

---

## 🧱 Struktur Project

```
nusantara-phone-mapper/
│
├── app.py
├── requirements.txt
├── test_dataset.py
│
├── services/
│   ├── ipa.py
│   ├── phoneme_mapper.py
│   ├── dataset_loader.py
│   ├── mapping_engine.py
│
├── dataset/
│   ├── indonesia/
│   └── sunda/
│
└── venv/
```

---

## ⚙️ Instalasi

### 1. Clone / buka project

```bash
cd nusantara-phone-mapper
```

### 2. Buat virtual environment

```bash
python -m venv venv
```

### 3. Aktifkan venv

#### Windows (PowerShell):

```bash
venv\Scripts\activate
```

Jika error:

```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

### 4. Install dependency

```bash
pip install flask flask-cors
```

---

## ▶️ Menjalankan Project

### Jalankan backend Flask:

```bash
python app.py
```

Buka di browser:

```
http://127.0.0.1:5000
```

---

## 🧪 Testing API

### Endpoint:

```
POST /api/map
```

### Contoh request:

```json
{
  "source": "makan",
  "targets": ["dahar", "minum", "tidur"]
}
```

### Contoh response:

```json
{
  "source": "makan",
  "targets": ["dahar", "minum", "tidur"],
  "result": ["dahar", 22]
}
```

---

## 📊 Menggunakan Dataset

### Dataset yang digunakan:

* 🇮🇩 Common Voice Indonesian
* 🇸🇺 Common Voice Sundanese

---

### Jalankan pipeline dataset:

```bash
python test_dataset.py
```

---

### Output:

```
INDO: [...]
SUNDA: [...]

=== MAPPING ===
makan -> ('dahar', 22)
minum -> ('inum', 5)
```

---

## 🔄 Pipeline Sistem

```
Dataset (TSV)
      ↓
Load sentences
      ↓
Extract words
      ↓
Text → Phoneme (ipa.py)
      ↓
Vector mapping
      ↓
Distance calculation
      ↓
Best match
```

---

## 🧠 Konsep yang Digunakan

* Phonetic representation (simplified IPA)
* Feature vector (manner, place, voicing)
* Distance calculation:

  * Manhattan-like
  * Custom phonetic distance
* Cross-language similarity

---

## ⚠️ Limitasi

* Tidak menggunakan IPA asli (disederhanakan)
* Belum support alignment panjang kata berbeda
* Belum menggunakan model deep learning
* Mapping masih berbasis rule

---

## 🔥 Pengembangan Selanjutnya

* Integrasi IPA asli (Epitran / G2P)
* Alignment antar kata (edit distance / DTW)
* Evaluasi akurasi
* Visualisasi hasil mapping
* Integrasi frontend (Vue.js)

---

## 👨‍💻 Author

Project Deep Learning
Semester 6

---

## 💬 Catatan

Project ini dibuat sebagai implementasi sederhana dari konsep phonetic mapping lintas bahasa dengan pendekatan yang ringan dan praktis.
