# ========================
# IPA → VECTOR
# ========================
def ipa_to_vector(phoneme):
    consonants = {
        "p": (0, 0, 0),
        "b": (0, 0, 1),
        "t": (0, 4, 0),
        "d": (0, 4, 1),
        "k": (0, 7, 0),
        "g": (0, 7, 1),

        "m": (1, 0, 1),
        "n": (1, 4, 1),

        "s": (4, 4, 0),
        "z": (4, 4, 1),
        "f": (4, 1, 0),
        "v": (4, 1, 1),

        "l": (7, 4, 1),
        "r": (6, 4, 1),
        "h": (4, 10, 0),
    }

    vowels = {
        "i": (0, 0, 0),
        "e": (2, 0, 0),
        "a": (6, 1, 0),
        "o": (2, 2, 1),
        "u": (0, 2, 1),
    }

    if phoneme in consonants:
        return consonants[phoneme]
    elif phoneme in vowels:
        return vowels[phoneme]
    else:
        return None


# ========================
# DISTANCE
# ========================
def phonetic_distance(a, b):
    manner_a, place_a, voice_a = a
    manner_b, place_b, voice_b = b

    score = 0

    if manner_a != manner_b:
        score += 5

    score += abs(place_a - place_b)

    if voice_a != voice_b:
        score += 1

    return score


# ========================
# WORD MAPPING
# ========================
from services.ipa import text_to_ipa


LENGTH_PENALTY = 6


def map_word(source_word, target_words):
    src_phonemes = text_to_ipa(source_word)

    results = []

    for target in target_words:
        tgt_phonemes = text_to_ipa(target)

        total_score = 0

        for s, t in zip(src_phonemes, tgt_phonemes):
            s_vec = ipa_to_vector(s)
            t_vec = ipa_to_vector(t)

            if not s_vec or not t_vec:
                total_score += 10
                continue

            total_score += phonetic_distance(s_vec, t_vec)

        # Penalti untuk fonem yang tidak ter-align (beda panjang kata)
        length_diff = abs(len(src_phonemes) - len(tgt_phonemes))
        total_score += length_diff * LENGTH_PENALTY

        # Skor dinormalisasi supaya fair antara kata pendek dan panjang
        aligned_len = max(len(src_phonemes), len(tgt_phonemes), 1)
        normalized_score = total_score / aligned_len

        results.append((target, total_score, normalized_score))

    if not results:
        return None

    best = min(results, key=lambda x: x[2])

    return best