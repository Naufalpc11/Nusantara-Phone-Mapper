"""IPA inventory for Bahasa Indonesia."""

VOWELS_ID = {
    "a": {"ipa": "a", "type": "vowel", "coord": (6, 1, 0), "diacritic": []},
    "i": {"ipa": "i", "type": "vowel", "coord": (0, 0, 0), "diacritic": []},
    "u": {"ipa": "u", "type": "vowel", "coord": (0, 2, 1), "diacritic": []},
    "e": {"ipa": "e", "type": "vowel", "coord": (2, 0, 0), "diacritic": []},
    "é": {"ipa": "ɛ", "type": "vowel", "coord": (4, 0, 0), "diacritic": []},
    "ə": {"ipa": "ə", "type": "vowel", "coord": (3, 1, 0), "diacritic": []},
    "o": {"ipa": "o", "type": "vowel", "coord": (2, 2, 1), "diacritic": []},
}

DIPHTHONGS_ID = {
    "ai": {"ipa": "ai̯", "type": "diphthong", "coord": ((6, 1, 0), (0, 0, 0)), "diacritic": []},
    "au": {"ipa": "au̯", "type": "diphthong", "coord": ((6, 1, 0), (0, 2, 1)), "diacritic": []},
    "oi": {"ipa": "oi̯", "type": "diphthong", "coord": ((2, 2, 1), (0, 0, 0)), "diacritic": []},
}

CONSONANTS_ID = {
    "p": {"ipa": "p", "type": "consonant", "coord": (0, 0, 0), "diacritic": []},
    "b": {"ipa": "b", "type": "consonant", "coord": (0, 0, 1), "diacritic": []},
    "t": {"ipa": "t", "type": "consonant", "coord": (0, 3, 0), "diacritic": []},
    "d": {"ipa": "d", "type": "consonant", "coord": (0, 3, 1), "diacritic": []},
    "k": {"ipa": "k", "type": "consonant", "coord": (0, 7, 0), "diacritic": []},
    "g": {"ipa": "g", "type": "consonant", "coord": (0, 7, 1), "diacritic": []},
    "q": {"ipa": "ʔ", "type": "consonant", "coord": (0, 10, 0), "diacritic": []},
    "m": {"ipa": "m", "type": "consonant", "coord": (1, 0, 1), "diacritic": []},
    "n": {"ipa": "n", "type": "consonant", "coord": (1, 3, 1), "diacritic": []},
    "ny": {"ipa": "ɲ", "type": "consonant", "coord": (1, 6, 1), "diacritic": []},
    "ng": {"ipa": "ŋ", "type": "consonant", "coord": (1, 7, 1), "diacritic": []},
    "r": {"ipa": "r", "type": "consonant", "coord": (2, 3, 1), "diacritic": []},
    "f": {"ipa": "f", "type": "consonant", "coord": (4, 1, 0), "diacritic": []},
    "v": {"ipa": "v", "type": "consonant", "coord": (4, 1, 1), "diacritic": []},
    "s": {"ipa": "s", "type": "consonant", "coord": (4, 3, 0), "diacritic": []},
    "z": {"ipa": "z", "type": "consonant", "coord": (4, 3, 1), "diacritic": []},
    "sy": {"ipa": "ʃ", "type": "consonant", "coord": (4, 4, 0), "diacritic": []},
    "kh": {"ipa": "x", "type": "consonant", "coord": (4, 7, 0), "diacritic": []},
    "h": {"ipa": "h", "type": "consonant", "coord": (4, 10, 0), "diacritic": []},
    "c": {"ipa": "tʃ", "type": "cluster", "coord": ((0, 3, 0), (4, 4, 0)), "diacritic": []},
    "j": {"ipa": "dʒ", "type": "cluster", "coord": ((0, 3, 1), (4, 4, 1)), "diacritic": []},
    "w": {"ipa": "w", "type": "consonant", "coord": (6, 7, 1), "diacritic": []},
    "y": {"ipa": "j", "type": "consonant", "coord": (6, 6, 1), "diacritic": []},
    "l": {"ipa": "l", "type": "consonant", "coord": (7, 3, 1), "diacritic": []},
}

CLUSTERS_ID = {
    "pr": {"ipa": "pr", "type": "cluster", "coord": ((0, 0, 0), (2, 3, 1)), "diacritic": []},
    "br": {"ipa": "br", "type": "cluster", "coord": ((0, 0, 1), (2, 3, 1)), "diacritic": []},
    "tr": {"ipa": "tr", "type": "cluster", "coord": ((0, 3, 0), (2, 3, 1)), "diacritic": []},
    "dr": {"ipa": "dr", "type": "cluster", "coord": ((0, 3, 1), (2, 3, 1)), "diacritic": []},
    "kr": {"ipa": "kr", "type": "cluster", "coord": ((0, 7, 0), (2, 3, 1)), "diacritic": []},
    "gr": {"ipa": "gr", "type": "cluster", "coord": ((0, 7, 1), (2, 3, 1)), "diacritic": []},
    "str": {"ipa": "str", "type": "cluster", "coord": ((4, 3, 0), (0, 3, 0), (2, 3, 1)), "diacritic": []},
    "sp": {"ipa": "sp", "type": "cluster", "coord": ((4, 3, 0), (0, 0, 0)), "diacritic": []},
    "st": {"ipa": "st", "type": "cluster", "coord": ((4, 3, 0), (0, 3, 0)), "diacritic": []},
    "sk": {"ipa": "sk", "type": "cluster", "coord": ((4, 3, 0), (0, 7, 0)), "diacritic": []},
    "kl": {"ipa": "kl", "type": "cluster", "coord": ((0, 7, 0), (7, 3, 1)), "diacritic": []},
    "gl": {"ipa": "gl", "type": "cluster", "coord": ((0, 7, 1), (7, 3, 1)), "diacritic": []},
    "fl": {"ipa": "fl", "type": "cluster", "coord": ((4, 1, 0), (7, 3, 1)), "diacritic": []},
    "pl": {"ipa": "pl", "type": "cluster", "coord": ((0, 0, 0), (7, 3, 1)), "diacritic": []},
    "bl": {"ipa": "bl", "type": "cluster", "coord": ((0, 0, 1), (7, 3, 1)), "diacritic": []},
    "sw": {"ipa": "sw", "type": "cluster", "coord": ((4, 3, 0), (6, 7, 1)), "diacritic": []},
}

PHONEME_ID = {}
PHONEME_ID.update(VOWELS_ID)
PHONEME_ID.update(DIPHTHONGS_ID)
PHONEME_ID.update(CONSONANTS_ID)
PHONEME_ID.update(CLUSTERS_ID)

_PARSE_ORDER = sorted(PHONEME_ID.keys(), key=len, reverse=True)


def word_to_phonemes(word: str) -> list[dict]:
    word = word.lower().strip()
    result = []
    i = 0

    while i < len(word):
        matched = False
        for key in _PARSE_ORDER:
            if word[i : i + len(key)] == key:
                result.append({"grapheme": key, **PHONEME_ID[key]})
                i += len(key)
                matched = True
                break

        if not matched:
            result.append({
                "grapheme": word[i],
                "ipa": word[i],
                "type": "unknown",
                "coord": None,
                "diacritic": [],
            })
            i += 1

    return result


def word_to_ipa(word: str) -> str:
    return "".join(entry["ipa"] for entry in word_to_phonemes(word))


def phoneme_sequence(word: str) -> list[str]:
    return [entry["ipa"] for entry in word_to_phonemes(word)]


def sentence_to_ipa(sentence: str) -> str:
    return " ".join(word_to_ipa(word) for word in sentence.strip().split())
