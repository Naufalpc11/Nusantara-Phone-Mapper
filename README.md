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
  - `audio/` — dataset, fitur log-Mel, model CTC, token fonem, dan metrik audio.
  - `data/` — pembacaan TSV, preprocessing teks, dan pembuatan training data.
  - `evaluation/` — pembuatan hasil baseline dan evaluasi seed lexicon.
  - `phoneme/` — inventori IPA Indonesia/Sunda dan scoring pemetaan fonetik.
  - `similarity/` — encoding kata untuk model similarity.

## Alur Program

### A) Saat Menjalankan `test_dataset.py`

Tujuan: membuat pasangan mapping (Indo → Sunda) lalu menyimpan `training_data.json`.

1. **Load kalimat dari TSV**
   - `test_dataset.py` memanggil `load_tsv_sentences(path, limit)` dari `services/data/dataset_loader.py`.
   - Fungsi ini membaca TSV dengan dukungan dua format: TSV ber-header yang punya kolom seperti `sentence` / `transcription` / `prompt`, dan TSV tanpa header seperti `utt_spk_text.tsv` di mana teks diambil dari kolom ketiga.

2. **Ekstraksi kata**
    - Kalimat diubah menjadi list kata unik dengan `extract_words(sentences)` dari `services/data/pre_processing.py`.
    - Proses pembersihan:
       - normalisasi Unicode ringan,
       - lowercase,
       - hapus tanda baca dan simbol,
       - tokenisasi kata,
       - buang kata dengan panjang ≤ 2,
       - deduplikasi dengan urutan yang stabil.

3. **Mapping banyak kata**
   - `test_dataset.py` memanggil `build_baseline_results()` di `services/evaluation/baseline_builder.py`.
   - Fungsi tersebut melakukan loop kata sumber dan memberi peringkat kandidat Sunda.

4. **Scoring & pilih kandidat terbaik (rule-based)**
   - `rank_candidates()` berada di `services/phoneme/mapper.py`.
   - Untuk scoring fonetik:
     1) Kata diubah menjadi urutan fonem menggunakan inventori `services/phoneme/ipa_id.py` dan `ipa_su.py`.
     2) Tiap fonem memiliki koordinat fitur artikulatoris.
     3) Jarak pasangan fonem dihitung dari koordinat tersebut.
     4) Ditambah penalti beda panjang (`LENGTH_PENALTY`) dan dinormalisasi:
        - `normalized_score = total_score / aligned_len`
     5) Kandidat target terbaik dipilih dengan normalized score terkecil.

5. **Filter hasil & simpan output JSON**
   - Hasil mapping di-filter oleh `NORMALIZED_SCORE_THRESHOLD`.
   - Output:
     - `results.json` — dict `{source_word: {target, score, normalized_score}}`
     - `training_data.json` — list of pairs `[{input, target, normalized_score}, ...]`

Ringkasnya:
`data/dataset_loader.py` → `evaluation/baseline_builder.py` → `phoneme/mapper.py` → simpan artefak evaluasi dan training.

### B) Saat Menjalankan `train_model.py`

Tujuan: melatih model baseline dari pasangan `training_data.json`.

1. **Load `training_data.json`**
   - `train_model.py` membaca JSON via `load_training_data()`.

2. **Encode kata menjadi vektor angka (di sinilah `encoder.py` dipakai)**
   - `train_model.py` mengimpor `encode_word()` dari `services/similarity/encoder.py`.
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
- Endpoint memanggil `map_word(source, targets)` dari `services/phoneme/mapper.py`.
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
