import json
import re
from pathlib import Path

from services.ipa_id import word_to_phonemes
from services.ipa_su import word_to_phonemes_su


WORD_PATTERN = re.compile(r"[a-z']+")
BLANK_TOKEN = "<blank>"
UNKNOWN_TOKEN = "<unk>"
WORD_SEPARATOR = "<space>"


def transcript_to_phonemes(transcript, language):
    """Convert a transcript into atomic IPA-like tokens for CTC training."""
    words = WORD_PATTERN.findall((transcript or "").lower())
    tokens = []

    for word_index, word in enumerate(words):
        entries = (
            word_to_phonemes(word)
            if language == "id"
            else word_to_phonemes_su(word)
        )
        if word_index > 0:
            tokens.append(WORD_SEPARATOR)
        tokens.extend(entry["ipa"] for entry in entries)

    return tokens


class PhonemeVocabulary:
    def __init__(self, tokens):
        ordered_tokens = [BLANK_TOKEN, UNKNOWN_TOKEN, WORD_SEPARATOR]
        ordered_tokens.extend(
            token
            for token in sorted(set(tokens))
            if token not in ordered_tokens
        )
        self.tokens = ordered_tokens
        self.token_to_id = {
            token: token_id for token_id, token in enumerate(self.tokens)
        }

    @property
    def blank_id(self):
        return self.token_to_id[BLANK_TOKEN]

    @property
    def unknown_id(self):
        return self.token_to_id[UNKNOWN_TOKEN]

    def __len__(self):
        return len(self.tokens)

    def encode(self, tokens):
        return [
            self.token_to_id.get(token, self.unknown_id)
            for token in tokens
        ]

    def decode(self, token_ids):
        return [
            self.tokens[token_id]
            if 0 <= token_id < len(self.tokens)
            else UNKNOWN_TOKEN
            for token_id in token_ids
        ]

    def to_dict(self):
        return {"tokens": self.tokens}

    @classmethod
    def from_dict(cls, data):
        vocabulary = cls([])
        vocabulary.tokens = list(data["tokens"])
        vocabulary.token_to_id = {
            token: token_id
            for token_id, token in enumerate(vocabulary.tokens)
        }
        return vocabulary

    def save(self, path):
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path):
        return cls.from_dict(
            json.loads(Path(path).read_text(encoding="utf-8"))
        )


def greedy_ctc_decode(logits, lengths, blank_id):
    """Collapse repeated CTC predictions and remove blank tokens."""
    predictions = logits.argmax(dim=-1)
    decoded = []

    for sequence, length in zip(predictions, lengths):
        result = []
        previous = None
        for token_id in sequence[: int(length)].tolist():
            if token_id != blank_id and token_id != previous:
                result.append(token_id)
            previous = token_id
        decoded.append(result)

    return decoded


def tokens_to_text(tokens):
    words = []
    current_word = []

    for token in tokens:
        if token == WORD_SEPARATOR:
            if current_word:
                words.append("".join(current_word))
                current_word = []
        elif token not in {BLANK_TOKEN, UNKNOWN_TOKEN}:
            current_word.append(token)

    if current_word:
        words.append("".join(current_word))

    return " ".join(words)
