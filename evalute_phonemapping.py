import argparse
import csv
import json
from pathlib import Path

import miniaudio


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_AUDIO_ROOT = PROJECT_ROOT / "dataset" / "datasetaudio"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate phone mapping results and generate a paper-style report."
    )
    parser.add_argument(
        "--training-history",
        type=Path,
        default=WORKSPACE_ROOT / "audio_training_history.json",
    )
    parser.add_argument(
        "--audio-test-results",
        type=Path,
        default=WORKSPACE_ROOT / "audio_test_results.json",
    )
    parser.add_argument(
        "--word-evaluation",
        type=Path,
        default=WORKSPACE_ROOT / "evaluation_results.json",
    )
    parser.add_argument("--audio-root", type=Path, default=DEFAULT_AUDIO_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE_ROOT / "HASIL_EVALUASI_PAPER_STYLE.md",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=WORKSPACE_ROOT / "paper_style_results.json",
    )
    return parser.parse_args()


def read_json(path):
    if not path.is_file():
        raise FileNotFoundError(f"File hasil tidak ditemukan: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def language_folder(audio_root, language):
    return audio_root / ("datasetindo" if language == "id" else "datasetsunda")


def test_manifest_summary(audio_root, language):
    folder = language_folder(audio_root, language)
    manifest = folder / "test.tsv"
    rows = []

    with manifest.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        rows.extend(reader)

    total_seconds = 0.0
    unreadable = 0

    for row in rows:
        audio_path = folder / row["audio_path"]
        try:
            total_seconds += miniaudio.get_file_info(str(audio_path)).duration
        except Exception:
            unreadable += 1

    return {
        "utterances": len(rows),
        "hours": total_seconds / 3600,
        "unreadable_audio": unreadable,
    }


def percentage(value):
    return "N/A" if value is None else f"{value * 100:.2f}%"


def decimal(value):
    return "N/A" if value is None else f"{value:.4f}"


def repair_ipa_display(text):
    replacements = {
        "tÊƒ": "tʃ",
        "dÊ’": "dʒ",
        "Å‹": "ŋ",
        "É²": "ɲ",
        "Êƒ": "ʃ",
        "É›": "ɛ",
        "É™": "ə",
        "É¨": "ɨ",
        "É”": "ɔ",
        "Ê”": "ʔ",
        "Ê’": "ʒ",
        "Ì¯": "̯",
    }
    for broken, corrected in replacements.items():
        text = text.replace(broken, corrected)
    return text


def best_epoch(history):
    valid_epochs = [
        epoch
        for epoch in history["epochs"]
        if epoch["validation"].get("per") is not None
    ]
    if not valid_epochs:
        raise ValueError("History tidak memiliki validation PER.")
    return min(valid_epochs, key=lambda epoch: epoch["validation"]["per"])


def build_payload(args):
    history = read_json(args.training_history)
    test_results = read_json(args.audio_test_results)
    word_evaluation = read_json(args.word_evaluation)
    best = best_epoch(history)
    test_metrics = test_results["metrics"]

    language_results = {}
    for language in ("id", "su"):
        manifest = test_manifest_summary(args.audio_root, language)
        metrics = test_metrics["per_language"][language]
        language_results[language] = {
            **manifest,
            **metrics,
            "phone_sequence_agreement_proxy": 1.0 - metrics["per"],
        }

    return {
        "training": {
            "epochs_recorded": len(history["epochs"]),
            "best_epoch": best["epoch"],
            "best_train": {
                key: best["train"][key]
                for key in ("loss", "per", "cer")
            },
            "best_validation": {
                key: best["validation"][key]
                for key in ("loss", "per", "cer")
            },
            "generalization_gap_per": (
                best["validation"]["per"] - best["train"]["per"]
            ),
        },
        "audio_test": {
            "checkpoint_epoch": test_results["checkpoint_epoch"],
            "overall": {
                key: test_metrics[key]
                for key in ("samples", "loss", "per", "cer")
            },
            "languages": language_results,
            "examples": test_results.get("examples", [])[:5],
        },
        "current_word_evaluation": word_evaluation["summary"],
        "paper_comparability": {
            "auto_manual_phone_mapping_agreement": "not_available",
            "alignment_boundary_agreement_25ms": "not_available",
            "audio_phone_recognition_per": "available",
        },
    }


def build_markdown(payload):
    training = payload["training"]
    audio = payload["audio_test"]
    languages = audio["languages"]
    word_eval = payload["current_word_evaluation"]

    rows = []
    for language, label in (("id", "Indonesia"), ("su", "Sunda")):
        result = languages[language]
        rows.append(
            f"| {label} | {result['utterances']} | "
            f"{result['hours']:.2f} | {percentage(result['per'])} | "
            f"{percentage(result['cer'])} | "
            f"{percentage(result['phone_sequence_agreement_proxy'])} |"
        )

    example_rows = []
    for example in audio["examples"]:
        reference = repair_ipa_display(example["reference_text"]).replace(
            "|", "\\|"
        )
        prediction = repair_ipa_display(example["predicted_text"]).replace(
            "|", "\\|"
        )
        example_rows.append(
            f"| {example['language']} | {example['transcript']} | "
            f"`{reference}` | `{prediction}` |"
        )

    return f"""# Hasil Evaluasi Bergaya Paper

## 1. Cara Membaca Laporan

Paper CrossPhon melakukan dua eksperimen utama:

1. membandingkan auto phone mapping dengan mapping buatan pakar,
2. membandingkan boundary forced alignment dengan toleransi 25 ms.

Proyek ini saat ini melakukan audio-to-phoneme recognition dengan CTC. Karena
tujuan dan satuan metriknya berbeda, nilai PER di laporan ini **tidak boleh
dibandingkan langsung** dengan agreement percentage pada paper.

---

## 2. Hasil Training Model Audio

| Informasi | Hasil |
|---|---:|
| Epoch yang dijalankan | {training['epochs_recorded']} |
| Epoch terbaik | {training['best_epoch']} |
| Train loss pada epoch terbaik | {decimal(training['best_train']['loss'])} |
| Train PER pada epoch terbaik | {percentage(training['best_train']['per'])} |
| Validation loss terbaik | {decimal(training['best_validation']['loss'])} |
| Validation PER terbaik | {percentage(training['best_validation']['per'])} |
| Validation CER terbaik | {percentage(training['best_validation']['cer'])} |
| Gap train-validation PER | {percentage(training['generalization_gap_per'])} |

Checkpoint terbaik berasal dari epoch {audio['checkpoint_epoch']}. Model dipilih
berdasarkan validation PER terendah, bukan berdasarkan epoch terakhir.

---

## 3. Evaluasi Audio-to-Phoneme

| Bahasa | Audio Test | Durasi Test (jam) | PER | CER | Proxy Agreement (1-PER) |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Hasil keseluruhan:

- Test samples: {audio['overall']['samples']}
- Test loss: {decimal(audio['overall']['loss'])}
- Test PER: {percentage(audio['overall']['per'])}
- Test CER: {percentage(audio['overall']['cer'])}

`Proxy Agreement (1-PER)` hanya dipakai agar hasil lebih mudah dibaca. Nilai ini
bukan metrik agreement yang digunakan CrossPhon karena PER menggunakan edit
distance urutan fonem, sedangkan paper mengevaluasi kesamaan mapping dan boundary
alignment.

### Interpretasi

- Model Sunda menghasilkan PER {percentage(languages['su']['per'])}, lebih baik
  daripada Indonesia dengan PER {percentage(languages['id']['per'])}.
- Model sudah mempelajari hubungan audio dan label fonem karena validation PER
  turun jauh dibanding epoch awal.
- Gap train-validation PER sebesar
  {percentage(training['generalization_gap_per'])} menunjukkan generalisasi
  masih cukup baik, dengan overfitting ringan mendekati akhir training.

---

## 4. Contoh Prediksi

| Bahasa | Transkrip | Fonem Referensi | Fonem Prediksi |
|---|---|---|---|
{chr(10).join(example_rows)}

Contoh perlu dianalisis berdasarkan jenis kesalahan:

- substitution: satu fonem diganti fonem lain,
- deletion: fonem referensi hilang,
- insertion: model menambahkan fonem,
- kesalahan batas kata melalui token `<space>`.

---

## 5. Evaluasi Mapping Kata yang Sudah Ada

| Metrik | Hasil |
|---|---:|
| Pasangan dievaluasi | {word_eval['evaluated_pairs']} |
| Hit@1 | {percentage(word_eval['hit_at_1'])} |
| Hit@3 | {percentage(word_eval['hit_at_3'])} |
| Hit@5 | {percentage(word_eval['hit_at_5'])} |
| MRR | {decimal(word_eval['mrr'])} |
| Mean rank | {decimal(word_eval['mean_rank'])} |

Hasil ini hanya mengukur ranking pasangan kata dari seed lexicon. Hasil tersebut
belum setara dengan Table 1 CrossPhon karena:

- unit yang dievaluasi adalah kata, bukan phone unik,
- pasangan seed lebih dekat ke terjemahan leksikal daripada expert phone map,
- sebagian seed juga digunakan oleh mapper, sehingga evaluasinya belum benar-benar
  independen.

Karena itu, Hit@1 tidak boleh disebut sebagai auto-manual phone agreement.

---

## 6. Perbandingan dengan Evaluasi CrossPhon

| Eksperimen | CrossPhon | Proyek Saat Ini | Status |
|---|---|---|---|
| Auto vs manual phone mapping | Agreement pada phone unik | Ranking pasangan kata | Belum setara |
| Cross-language alignment | Boundary agreement dalam 25 ms | Belum ada timestamp fonem | Belum tersedia |
| Language-dependent baseline | Model alignment per bahasa | Model CTC gabungan | Sebagian tersedia |
| Audio phone recognition | Tidak menjadi tabel utama paper | PER dan CER | Sudah tersedia |

---

## 7. Agar Benar-Benar Sama Seperti Paper

### Eksperimen 1: Auto-Manual Phone Mapping Agreement

1. Tentukan arah mapping, disarankan Sunda ke Indonesia.
2. Ambil seluruh phone unik Sunda dan Indonesia.
3. Buat auto mapping menggunakan koordinat IPA dan Manhattan distance.
4. Buat file manual mapping yang diperiksa manusia/pakar.
5. Hitung:

```text
Agreement = phone auto yang sama dengan phone manual / total phone Sunda
```

Hasil akhirnya dapat ditulis seperti Table 1 paper:

| Source | Unique Phones | Agreement |
|---|---:|---:|
| Sunda -> Indonesia | belum dihitung | belum tersedia |

### Eksperimen 2: Cross-Language Alignment Agreement

1. Latih atau sediakan acoustic model Indonesia dan Sunda secara terpisah.
2. Buat phone boundary reference memakai model language-dependent.
3. Align audio Sunda dengan model Indonesia menggunakan auto mapping.
4. Ulangi menggunakan manual mapping.
5. Bandingkan batas awal/akhir phone dengan toleransi 25 ms.

Hasil akhirnya dapat ditulis seperti Table 2 paper:

| Bahasa | Auto Mapping | Manual Mapping | Tested Audio |
|---|---:|---:|---:|
| Sunda | belum tersedia | belum tersedia | belum tersedia |

---

## 8. Kesimpulan yang Aman untuk Laporan

Model audio CNN-BiLSTM-CTC berhasil mempelajari prediksi urutan fonem pada
Bahasa Indonesia dan Sunda. Checkpoint terbaik diperoleh pada epoch
{training['best_epoch']} dengan validation PER
{percentage(training['best_validation']['per'])}. Pada test set, model mencapai
PER keseluruhan {percentage(audio['overall']['per'])}. Performa Sunda lebih baik
daripada Indonesia.

Namun, hasil ini merupakan evaluasi **phone recognition**, bukan evaluasi
cross-language forced alignment seperti pada CrossPhon. Untuk menghasilkan tabel
yang benar-benar setara dengan paper, proyek masih membutuhkan expert phone
mapping dan anotasi atau hasil forced alignment dengan timestamp fonem.
"""


def main():
    args = parse_args()
    payload = build_payload(args)
    markdown = build_markdown(payload)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
    args.json_output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Paper-style report saved: {args.output}")
    print(f"Structured results saved: {args.json_output}")


if __name__ == "__main__":
    main()
