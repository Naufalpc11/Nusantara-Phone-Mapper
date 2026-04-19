import csv


# ========================
# LOAD DATASET TSV
# ========================
def load_tsv_sentences(path, limit=200):
    sentences = []
    text_columns = ("sentence", "transcription", "prompt")

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for i, row in enumerate(reader):
            if i >= limit:
                break

            for col in text_columns:
                text = row.get(col)
                if text and text.strip():
                    sentences.append(text.strip())
                    break

    return sentences


# ========================
# EXTRACT WORDS
# ========================
import re

def extract_words(sentences):
    words = set()

    for s in sentences:
        for w in s.lower().split():
            # bersihin simbol
            w = re.sub(r'[^a-z]', '', w)

            if len(w) > 2:
                words.add(w)

    return list(words)