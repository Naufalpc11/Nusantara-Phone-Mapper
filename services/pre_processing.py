import re
import unicodedata
from collections import Counter


WORD_PATTERN = re.compile(r"[a-z]+")


def normalize_text(text):
    """Normalisasi ringan agar token lebih konsisten untuk mapping."""
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = text.replace("’", "'").replace("`", "'")
    text = re.sub(r"[-_/]+", " ", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_words(text, min_len=3):
    """Ubah kalimat jadi token kata yang lebih bersih."""
    normalized_text = normalize_text(text)
    tokens = WORD_PATTERN.findall(normalized_text)

    if min_len <= 1:
        return tokens

    return [token for token in tokens if len(token) >= min_len]


def extract_words(sentences, min_len=3, max_words=None):
    """Ubah list kalimat menjadi list kata unik yang diurutkan berdasarkan frekuensi."""
    word_counts = Counter()

    for sentence in sentences:
        for token in tokenize_words(sentence, min_len=min_len):
            word_counts[token] += 1

    words = [word for word, _ in word_counts.most_common()]

    if max_words is not None:
        return words[:max_words]

    return words