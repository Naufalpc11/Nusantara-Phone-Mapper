import csv


# Ambil teks dari dataset TSV multi-format.
def load_tsv_sentences(path, limit=200):
    """Baca TSV dan ambil kolom teks valid hingga batas data."""
    sentences = []
    text_columns = ("sentence", "transcription", "prompt")

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for i, row in enumerate(reader):
            if i >= limit:
                break

            # Pilih kolom teks pertama yang berisi konten.
            for col in text_columns:
                text = row.get(col)
                if text and text.strip():
                    sentences.append(text.strip())
                    break

    return sentences

import re


# Bersihkan token agar stabil untuk proses mapping/training.
def extract_words(sentences):
    """Ubah kalimat menjadi daftar kata unik yang sudah dibersihkan."""
    words = set()

    for s in sentences:
        for w in s.lower().split():
            # bersihin simbol
            w = re.sub(r'[^a-z]', '', w)

            if len(w) > 2:
                words.add(w)
    return list(words)