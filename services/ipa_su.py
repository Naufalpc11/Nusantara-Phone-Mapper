"""IPA inventory for Bahasa Sunda."""

VOWELS_SU = {
    "a": {"ipa": "a", "type": "vowel", "coord": (6, 1, 0), "diacritic": []},
    "i": {"ipa": "i", "type": "vowel", "coord": (0, 0, 0), "diacritic": []},
    "u": {"ipa": "u", "type": "vowel", "coord": (0, 2, 1), "diacritic": []},
    "e": {"ipa": "e", "type": "vowel", "coord": (2, 0, 0), "diacritic": []},
    "é": {"ipa": "ɛ", "type": "vowel", "coord": (4, 0, 0), "diacritic": []},
    "eu": {"ipa": "ɨ", "type": "vowel", "coord": (0, 2, 0), "diacritic": []},
    "o": {"ipa": "o", "type": "vowel", "coord": (2, 2, 1), "diacritic": []},
    "ö": {"ipa": "ɔ", "type": "vowel", "coord": (4, 2, 1), "diacritic": []},
}

DIPHTHONGS_SU = {
    "ai": {"ipa": "ai̯", "type": "diphthong", "coord": ((6, 1, 0), (0, 0, 0)), "diacritic": []},
    "au": {"ipa": "au̯", "type": "diphthong", "coord": ((6, 1, 0), (0, 2, 1)), "diacritic": []},
    "ui": {"ipa": "ui̯", "type": "diphthong", "coord": ((0, 2, 1), (0, 0, 0)), "diacritic": []},
    "ieu": {"ipa": "iɨ̯", "type": "diphthong", "coord": ((0, 0, 0), (0, 2, 0)), "diacritic": []},
    "eui": {"ipa": "ɨi̯", "type": "diphthong", "coord": ((0, 2, 0), (0, 0, 0)), "diacritic": []},
    "ua": {"ipa": "ua̯", "type": "diphthong", "coord": ((0, 2, 1), (6, 1, 0)), "diacritic": []},
    "ia": {"ipa": "ia̯", "type": "diphthong", "coord": ((0, 0, 0), (6, 1, 0)), "diacritic": []},
}

NASAL_FINALS_SU = {
    "eung": {"ipa": "ɨŋ", "type": "cluster", "coord": None, "diacritic": []},
    "eun": {"ipa": "ɨn", "type": "cluster", "coord": None, "diacritic": []},
    "eum": {"ipa": "ɨm", "type": "cluster", "coord": None, "diacritic": []},
    "ing": {"ipa": "iŋ", "type": "cluster", "coord": None, "diacritic": []},
    "ang": {"ipa": "aŋ", "type": "cluster", "coord": None, "diacritic": []},
    "ong": {"ipa": "oŋ", "type": "cluster", "coord": None, "diacritic": []},
    "ung": {"ipa": "uŋ", "type": "cluster", "coord": None, "diacritic": []},
}

CONSONANTS_SU = {
    "p": {"ipa": "p", "type": "consonant", "coord": (0, 0, 0), "diacritic": []},
    "b": {"ipa": "b", "type": "consonant", "coord": (0, 0, 1), "diacritic": []},
    "t": {"ipa": "t", "type": "consonant", "coord": (0, 3, 0), "diacritic": []},
    "d": {"ipa": "d", "type": "consonant", "coord": (0, 3, 1), "diacritic": []},
    "k": {"ipa": "k", "type": "consonant", "coord": (0, 7, 0), "diacritic": []},
    "g": {"ipa": "g", "type": "consonant", "coord": (0, 7, 1), "diacritic": []},
    "'": {"ipa": "ʔ", "type": "consonant", "coord": (0, 10, 0), "diacritic": []},
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
    "h": {"ipa": "h", "type": "consonant", "coord": (4, 10, 0), "diacritic": []},
    "c": {"ipa": "tʃ", "type": "cluster", "coord": ((0, 3, 0), (4, 4, 0)), "diacritic": []},
    "j": {"ipa": "dʒ", "type": "cluster", "coord": ((0, 3, 1), (4, 4, 1)), "diacritic": []},
    "w": {"ipa": "w", "type": "consonant", "coord": (6, 7, 1), "diacritic": []},
    "y": {"ipa": "j", "type": "consonant", "coord": (6, 6, 1), "diacritic": []},
    "l": {"ipa": "l", "type": "consonant", "coord": (7, 3, 1), "diacritic": []},
}

PHONEME_SU = {}
PHONEME_SU.update(NASAL_FINALS_SU)
PHONEME_SU.update(DIPHTHONGS_SU)
PHONEME_SU.update(VOWELS_SU)
PHONEME_SU.update(CONSONANTS_SU)

_PARSE_ORDER_SU = sorted(PHONEME_SU.keys(), key=len, reverse=True)


def word_to_phonemes_su(word: str) -> list[dict]:
    word = word.lower().strip()
    result = []
    i = 0

    while i < len(word):
        matched = False
        for key in _PARSE_ORDER_SU:
            if word[i : i + len(key)] == key:
                result.append({"grapheme": key, **PHONEME_SU[key]})
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


def word_to_ipa_su(word: str) -> str:
    return "".join(entry["ipa"] for entry in word_to_phonemes_su(word))


def phoneme_sequence_su(word: str) -> list[str]:
    return [entry["ipa"] for entry in word_to_phonemes_su(word)]
