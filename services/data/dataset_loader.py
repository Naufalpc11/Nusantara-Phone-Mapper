"""TSV dataset loading utilities."""

import csv

from services.data.pre_processing import extract_words as _extract_words


KNOWN_TEXT_COLUMNS = ("sentence", "transcription", "prompt", "text", "utterance")


def _extract_text_from_row(row, text_columns=KNOWN_TEXT_COLUMNS):
    for col in text_columns:
        text = row.get(col)
        if text and text.strip():
            return text.strip()

    return None


# Ambil teks dari dataset TSV multi-format.
def load_tsv_sentences(path, limit=200):
    """Baca TSV dan ambil kolom teks valid hingga batas data."""
    sentences = []

    with open(path, encoding="utf-8") as f:
        sample = f.readline()

        if not sample:
            return sentences

        first_row = sample.rstrip("\n\r").split("\t")
        first_row_lower = {value.strip().lower() for value in first_row}

        f.seek(0)

        # Support TSV with headers and headerless TSV such as utt_spk_text.tsv.
        if first_row_lower.intersection(KNOWN_TEXT_COLUMNS):
            reader = csv.DictReader(f, delimiter="\t")

            for i, row in enumerate(reader):
                if i >= limit:
                    break

                text = _extract_text_from_row(row)
                if text:
                    sentences.append(text)
        else:
            reader = csv.reader(f, delimiter="\t")

            for i, row in enumerate(reader):
                if i >= limit:
                    break

                if not row:
                    continue

                if len(row) >= 3:
                    text = row[2].strip()
                else:
                    text = row[-1].strip()

                if text:
                    sentences.append(text)

    return sentences

import re


def extract_words(sentences, min_len=3):
    """Compatibility wrapper untuk kode lama yang masih impor dari dataset_loader."""
    return _extract_words(sentences, min_len=min_len)
