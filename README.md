# Nusantara Phone Mapper

Dokumen ini berisi:
1. Penjelasan fungsi setiap file utama di project.
2. Kamus istilah teknis yang sering muncul di kode.

## Ringkasan Tujuan Project

Project ini membuat pemetaan kata Indonesia ke kata Sunda berdasarkan kemiripan bunyi (fonetik), lalu menghasilkan data latih untuk model baseline deep learning.

Alur besar:

1. Baca dataset TSV Indonesia dan Sunda.
2. Bersihkan kata dan buat daftar kata unik.
3. Cari pasangan kata paling mirip secara fonetik.
4. Simpan hasil mapping dan training data.
5. Latih model baseline MLP.

## Penjelasan Tiap File

### app.py

Fungsi file:

1. Menyediakan API backend sederhana menggunakan Flask.
2. Menyediakan endpoint untuk cek server aktif.
3. Menyediakan endpoint untuk melakukan mapping kata.

Fungsi penting:

1. home
      Arti: health check endpoint.
      Tujuan: memastikan backend aktif.

2. map_api
      Arti: endpoint mapping.
      Tujuan: menerima source dan targets dari request, lalu mengembalikan hasil terbaik dari map_word.

### services/dataset_loader.py

Fungsi file:

1. Menangani pembacaan file TSV lintas format kolom.
2. Menangani pembersihan token kata untuk proses mapping/training.

Fungsi penting:

1. load_tsv_sentences
      Arti: loader data kalimat dari TSV.
      Tujuan: membaca kolom teks yang tersedia (sentence/transcription/prompt) sampai batas limit.

2. extract_words
      Arti: ekstraksi kata.
      Tujuan: mengubah kalimat menjadi kata unik yang sudah dibersihkan (lowercase, regex, minimal panjang > 2).

### services/encoder.py

Fungsi file:

1. Mengubah kata menjadi vektor angka fixed-length untuk model.

Fungsi penting:

1. encode_word
      Arti: character encoder sederhana.
      Tujuan: memetakan karakter ke indeks dan melakukan padding sampai panjang tetap.

### services/ipa.py

Fungsi file:

1. Konversi kata menjadi representasi fonetik sederhana berbasis karakter.

Fungsi penting:

1. text_to_ipa
      Arti: text ke phoneme list.
      Tujuan: memetakan huruf tertentu ke simbol fonetik sederhana untuk scoring bunyi.

### services/phoneme_mapper.py

Fungsi file:

1. Menyediakan representasi fitur fonetik.
2. Menghitung jarak bunyi antar fonem.
3. Mencari kandidat kata target terbaik untuk satu kata sumber.

Komponen penting:

1. ipa_to_vector
      Arti: fonem ke fitur numerik.
      Tujuan: merepresentasikan fonem dengan fitur artikulasi.

2. phonetic_distance
      Arti: fungsi jarak fonetik.
      Tujuan: menghitung perbedaan dua fonem berdasarkan fitur.

3. LENGTH_PENALTY
      Arti: penalti beda panjang fonem.
      Tujuan: menghindari kata beda panjang terlihat terlalu mirip.

4. map_word
      Arti: pemilih best-match kata.
      Tujuan: membandingkan source ke semua target, hitung total score + normalized score, lalu ambil skor terbaik.

### services/mapping_engine.py

Fungsi file:

1. Menjalankan mapping untuk banyak kata sekaligus.

Fungsi penting:

1. map_all
      Arti: batch mapping function.
      Tujuan: memanggil map_word untuk setiap source word sampai limit.

### test_dataset.py

Fungsi file:

1. Menjalankan pipeline data end-to-end.
2. Menghasilkan output mapping dan training data.

Komponen penting:

1. DATA_LIMIT
      Arti: jumlah maksimum kalimat yang dibaca per dataset.

2. MAPPING_LIMIT
      Arti: jumlah maksimum source words yang diproses.

3. NORMALIZED_SCORE_THRESHOLD
      Arti: batas kualitas hasil mapping.

Output file:

1. results.json
      Berisi hasil mapping terfilter.
2. training_data.json
      Berisi pasangan input-target untuk training model.

### train_model.py

Fungsi file:

1. Menjalankan training baseline model.
2. Menyimpan checkpoint model dan loss history.

Fungsi penting:

1. parse_args
      Arti: pembaca konfigurasi CLI.
      Tujuan: menerima hyperparameter seperti epochs, batch-size, learning rate.

2. load_training_data
      Arti: loader training set.
      Tujuan: membaca training_data.json, encode input-target ke tensor.

3. main
      Arti: eksekusi training loop.
      Tujuan: setup DataLoader, model, optimizer, jalankan epoch loop, lalu simpan model dan loss history.

Output training:

1. model_baseline.pt
      Checkpoint model (binary PyTorch).
2. loss_history.json
      Riwayat loss per epoch.

### results.json

Fungsi file:

1. Menyimpan hasil mapping terpilih yang lolos threshold.
2. Menjadi bahan analisis kualitas mapping.

### training_data.json

Fungsi file:

1. Menyimpan data latih hasil pipeline.
2. Menjadi input utama train_model.py.

### loss_history.json

Fungsi file:

1. Menyimpan kurva loss training.
2. Dipakai untuk evaluasi apakah training membaik.

## Kamus Istilah Teknis

1. Pipeline
      Alur berurutan dari data mentah sampai output akhir.

2. Dataset
      Kumpulan data mentah yang dipakai sistem.

3. Preprocessing
      Tahap pembersihan dan normalisasi data sebelum dipakai model.

4. Token
      Unit teks kecil, biasanya kata.

5. Mapping
      Proses mencocokkan item sumber ke item target terbaik.

6. Phoneme / Fonem
      Unit bunyi dasar dalam bahasa.

7. Feature
      Ciri numerik yang dipakai untuk membedakan data.

8. Vector
      Representasi data dalam bentuk angka.

9. Distance
      Nilai perbedaan dua data, biasanya makin kecil makin mirip.

10. Normalized score
       Skor yang disesuaikan agar adil antar panjang kata.

11. Threshold
       Batas nilai untuk memutuskan hasil diterima atau tidak.

12. Baseline model
       Model awal sederhana sebagai acuan eksperimen.

13. MLP
       Jaringan saraf feed-forward sederhana (Multi-Layer Perceptron).

14. Training
       Proses model belajar dari data.

15. Epoch
       Satu putaran penuh model melihat seluruh data training.

16. Batch
       Sebagian data yang diproses dalam satu langkah update.

17. Mini-batch
       Strategi training menggunakan batch kecil.

18. DataLoader
       Komponen PyTorch untuk batching dan shuffle data.

19. Shuffle
       Mengacak urutan data agar training lebih stabil.

20. Loss
       Nilai error model; target training adalah menurunkan loss.

21. MSE Loss
       Mean Squared Error, rata-rata kuadrat selisih prediksi-target.

22. Optimizer
       Algoritma update bobot model (contoh: Adam).

23. Learning Rate
       Besar langkah update bobot tiap iterasi.

24. Backpropagation
       Proses menghitung gradien error untuk update model.

25. Checkpoint
       File simpan state model setelah training.

26. Inference
       Proses prediksi memakai model yang sudah dilatih.

27. Hyperparameter
       Parameter yang ditentukan sebelum training (epochs, batch-size, lr).

28. CLI Argument
       Parameter yang dikirim lewat command line.

29. Reproducibility
       Kemampuan mengulang eksperimen dengan hasil konsisten.

30. End-to-end
       Sistem yang berjalan lengkap dari awal hingga akhir tanpa putus.

## Perintah Penting

1. Jalankan pipeline data:

```bash
python test_dataset.py
```

2. Jalankan training baseline:

```bash
python train_model.py --epochs 60 --batch-size 16 --log-every 20
```

3. Jalankan API backend:

```bash
python app.py
```

## Catatan

1. model_baseline.pt adalah file binary, jadi tidak bisa dibuka sebagai teks biasa.
2. Untuk membaca info model, gunakan torch.load dari Python.
