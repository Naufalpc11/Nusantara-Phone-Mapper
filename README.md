# Nusantara Phone Mapper

Repo ini berisi pipeline sederhana untuk:
1) mengambil kata dari dataset TSV,
2) melakukan *phonetic mapping* (rule-based) dari kata bahasa Indonesia → kandidat kata bahasa Sunda,
3) (opsional) melatih model baseline dari pasangan hasil mapping.

## Struktur Singkat

- `test_dataset.py` — entrypoint untuk membuat hasil mapping + dataset training (`results.json`, `training_data.json`).
- `train_model.py` — entrypoint untuk training model baseline dari `training_data.json` dan menyimpan checkpoint.
- `app.py` — backend API (Flask) untuk melakukan mapping satu kata terhadap daftar target (rule-based).
- `services/`
  - `dataset_loader.py` — baca TSV dan ekstraksi kata.
   - `pre_processing.py` — normalisasi teks, tokenisasi, dan deduplikasi kata.
  - `mapping_engine.py` — mapping banyak kata (loop).
  - `phoneme_mapper.py` — scoring fonetik untuk memilih target terbaik.
  - `ipa.py` — konversi teks → “IPA” sederhana (berbasis karakter).
  - `encoder.py` — encoding kata → vektor indeks karakter (dipakai saat training).

## Alur Program

### A) Saat Menjalankan `test_dataset.py`

Tujuan: membuat pasangan mapping (Indo → Sunda) lalu menyimpan `training_data.json`.

1. **Load kalimat dari TSV**
   - `test_dataset.py` memanggil `load_tsv_sentences(path, limit)` dari `services/dataset_loader.py`.
   - Fungsi ini membaca TSV dengan dukungan dua format: TSV ber-header yang punya kolom seperti `sentence` / `transcription` / `prompt`, dan TSV tanpa header seperti `utt_spk_text.tsv` di mana teks diambil dari kolom ketiga.

2. **Ekstraksi kata**
    - Kalimat diubah menjadi list kata unik dengan `extract_words(sentences)` dari `services/pre_processing.py`.
    - Proses pembersihan:
       - normalisasi Unicode ringan,
       - lowercase,
       - hapus tanda baca dan simbol,
       - tokenisasi kata,
       - buang kata dengan panjang ≤ 2,
       - deduplikasi dengan urutan yang stabil.

3. **Mapping banyak kata**
   - `test_dataset.py` memanggil `map_all(indo_words, sunda_words, limit)` di `services/mapping_engine.py`.
   - `map_all()` melakukan loop kata sumber (Indo) dan untuk tiap kata memanggil `map_word(source_word, target_words)`.

4. **Scoring & pilih kandidat terbaik (rule-based)**
   - `map_word()` berada di `services/phoneme_mapper.py`.
   - Untuk scoring fonetik:
     1) `text_to_ipa(word)` dari `services/ipa.py` mengubah tiap karakter tertentu menjadi simbol fonem sederhana.
     2) Tiap fonem diubah ke fitur numerik dengan `ipa_to_vector()`.
     3) Jarak dua fonem dihitung dengan `phonetic_distance()`.
     4) Ditambah penalti beda panjang (`LENGTH_PENALTY`) dan dinormalisasi:
        - `normalized_score = total_score / aligned_len`
     5) Kandidat target terbaik dipilih dengan normalized score terkecil.

5. **Filter hasil & simpan output JSON**
   - Hasil mapping di-filter oleh `NORMALIZED_SCORE_THRESHOLD`.
   - Output:
     - `results.json` — dict `{source_word: {target, score, normalized_score}}`
     - `training_data.json` — list of pairs `[{input, target, normalized_score}, ...]`

Ringkasnya:
`dataset_loader.py` → `mapping_engine.py` → `phoneme_mapper.py` → `ipa.py` → simpan `results.json` + `training_data.json`.

### B) Saat Menjalankan `train_model.py`

Tujuan: melatih model baseline dari pasangan `training_data.json`.

1. **Load `training_data.json`**
   - `train_model.py` membaca JSON via `load_training_data()`.

2. **Encode kata menjadi vektor angka (di sinilah `encoder.py` dipakai)**
   - `train_model.py` mengimpor `encode_word()` dari `services/encoder.py`.
   - `encode_word(word, max_len=10)` mengubah huruf `a-z` menjadi indeks (a=1..z=26), dipotong/padding jadi panjang tetap 10.
   - Hasil encoding dipakai untuk membuat tensor:
     - `x_tensor` dari `input`
     - `y_tensor` dari `target`

3. **Training model**
   - Model baseline berupa MLP kecil:
     - `Linear(10 → 32) → ReLU → Linear(32 → 10)`
   - Objective: memprediksi vektor target dari vektor input.
   - Loss: `MSELoss`, optimizer: `Adam`.

4. **Simpan artefak**
   - Checkpoint: `model_baseline.pt`
   - History loss: `loss_history.json`

Ringkasnya:
`training_data.json` → `encoder.py` → training di `train_model.py` → simpan `model_baseline.pt`.

### C) Saat Menjalankan `app.py`

Tujuan: menyediakan endpoint API untuk mapping (rule-based).

- `app.py` membuat endpoint `POST /api/map`.
- Endpoint memanggil `map_word(source, targets)` dari `services/phoneme_mapper.py`.
- Catatan: API ini **tidak** memakai `encoder.py` maupun `model_baseline.pt`.

## Cara Menjalankan (Windows / PowerShell)

1. Install dependency:

```powershell
pip install -r requirements.txt
```

Kalau kamu pakai virtual env di repo ini (`.venv/`), lebih aman jalankan script pakai interpreter itu:

```powershell
& ".\\.venv\\Scripts\\python.exe" .\\test_dataset.py
& ".\\.venv\\Scripts\\python.exe" .\\train_model.py
& ".\\.venv\\Scripts\\python.exe" .\\app.py
```

2. Generate mapping + training data:

```powershell
python .\test_dataset.py
```

3. Train baseline model:

```powershell
python .\train_model.py
```

4. Jalankan API:

```powershell
python .\app.py
```

## Output File

- `results.json` — hasil mapping terfilter (dari `test_dataset.py`)
- `training_data.json` — pasangan training (dari `test_dataset.py`)
- `model_baseline.pt` — checkpoint model (dari `train_model.py`)
- `loss_history.json` — loss per-epoch (dari `train_model.py`)

## Catatan Penting

- Mapping di `test_dataset.py` dan `app.py` adalah **rule-based** (fonetik sederhana), bukan hasil model.
- Model baseline di `train_model.py` belajar dari pasangan hasil rule-based, dengan representasi input/target yang sangat sederhana (indeks karakter). Ini baseline untuk eksperimen, bukan model fonetik yang kuat.
