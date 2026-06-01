from services.ipa_id import phoneme_sequence as phoneme_sequence_id
from services.ipa_su import phoneme_sequence_su


def ipa_to_vector(phoneme):
    """Ubah satu fonem menjadi vektor fitur artikulasi."""
    mapping = {
        "p": (0, 0, 0),
        "b": (0, 0, 1),
        "t": (0, 3, 0),
        "d": (0, 3, 1),
        "k": (0, 7, 0),
        "g": (0, 7, 1),
        "ʔ": (0, 10, 0),
        "m": (1, 0, 1),
        "n": (1, 3, 1),
        "ɲ": (1, 6, 1),
        "ŋ": (1, 7, 1),
        "r": (2, 3, 1),
        "f": (4, 1, 0),
        "v": (4, 1, 1),
        "s": (4, 3, 0),
        "z": (4, 3, 1),
        "ʃ": (4, 4, 0),
        "x": (4, 7, 0),
        "h": (4, 10, 0),
        "tʃ": (4, 4, 0),
        "dʒ": (4, 4, 1),
        "w": (6, 7, 1),
        "j": (6, 6, 1),
        "l": (7, 3, 1),
        "i": (0, 0, 0),
        "e": (2, 0, 0),
        "ɛ": (4, 0, 0),
        "ə": (3, 1, 0),
        "ɨ": (0, 2, 0),
        "a": (6, 1, 0),
        "o": (2, 2, 1),
        "ɔ": (4, 2, 1),
        "u": (0, 2, 1),
    }

    return mapping.get(phoneme)

def phonetic_distance(a, b):
    """Hitung jarak fonetik antara dua vektor fonem."""
    manner_a, place_a, voice_a = a
    manner_b, place_b, voice_b = b

    score = 0

    if manner_a != manner_b:
        score += 5

    score += abs(place_a - place_b)

    if voice_a != voice_b:
        score += 1

    return score

LENGTH_PENALTY = 6
LEXICAL_MAP_ID_SU = {
    "saya": "abdi",
    "aku": "abdi",
    "kamu": "anjeun",
    "anda": "anjeun",
    "dia": "anjeunna",
    "ia": "anjeunna",
    "kita": "urang",
    "kami": "urang",
    "mereka": "maranéhna",
    "apa": "naon",
    "siapa": "saha",
    "mana": "mana",
    "kapan": "iraha",
    "mengapa": "naha",
    "kenapa": "naha",
    "bagaimana": "kumaha",
    "berapa": "sabaraha",
    "makan": "dahar",
    "minum": "nginum",
    "pergi": "indit",
    "datang": "datang",
    "tidur": "sare",
    "bangun": "hudang",
    "mandi": "mandi",
    "beli": "meuli",
    "jual": "ngajual",
    "kerja": "gawe",
    "bekerja": "gawe",
    "bisa": "tiasa",
    "mau": "hayang",
    "ingin": "hayang",
    "lihat": "tingal",
    "melihat": "ningali",
    "dengar": "dangukeun",
    "mendengar": "ngadangukeun",
    "rumah": "imah",
    "air": "cai",
    "nasi": "sangu",
    "orang": "urang",
    "anak": "budak",
    "ibu": "ema",
    "bapak": "bapa",
    "teman": "batur",
    "sekolah": "sakola",
    "buku": "buku",
    "uang": "duit",
    "hari": "poe",
    "malam": "peuting",
    "pagi": "isuk",
    "siang": "beurang",
    "sore": "sore",
    "besar": "gede",
    "kecil": "leutik",
    "bagus": "sae",
    "baik": "alus",
    "baru": "anyar",
    "lama": "lami",
    "cantik": "geulis",
    "dan": "jeung",
    "atau": "atawa",
    "tapi": "tapi",
    "karena": "sabab",
    "jadi": "jadi",
    "sudah": "geus",
    "belum": "can",
    "tidak": "henteu",
    "bukan": "sanes",
    "jangan": "ulah",
    "ada": "aya",
    "ke": "ka",
    "dari": "ti",
    "di": "di",
    "dengan": "sareng",
    "untuk": "kanggo",
    "ini": "ieu",
    "itu": "eta",
    "yang": "anu",
}


def map_word(source_word, target_words):
    """Cari kandidat target dengan skor fonetik terendah untuk satu kata sumber."""
    word = (source_word or "").lower().strip()

    if word in LEXICAL_MAP_ID_SU:
        target = LEXICAL_MAP_ID_SU[word]
        return target, 0, 0.0

    src_phonemes = phoneme_sequence_id(word)

    results = []

    for target in target_words:
        tgt_phonemes = phoneme_sequence_su(target)

        total_score = 0

        aligned_len = max(len(src_phonemes), len(tgt_phonemes), 1)

        for s, t in zip(src_phonemes, tgt_phonemes):
            s_vec = ipa_to_vector(s)
            t_vec = ipa_to_vector(t)

            if not s_vec or not t_vec:
                total_score += 10
                continue

            total_score += phonetic_distance(s_vec, t_vec)

        length_diff = abs(len(src_phonemes) - len(tgt_phonemes))
        total_score += length_diff * LENGTH_PENALTY
        normalized_score = total_score / aligned_len

        results.append((target, total_score, normalized_score))

    if not results:
        return None

    best = min(results, key=lambda x: x[2])

    return best


class PhonemeMapper:
    """Compatibility wrapper for richer phoneme inventories."""

    def map_word(self, word_id: str) -> dict:
        lexical = LEXICAL_MAP_ID_SU.get((word_id or "").lower().strip())
        if lexical:
            target = lexical
            score = 0
            normalized_score = 0.0
        else:
            target = word_id
            score = -1
            normalized_score = -1.0

        return {
            "word_id": word_id,
            "word_su": target,
            "method": "lexical" if score == 0 else "fallback",
            "score": score,
            "normalized_score": normalized_score,
            "phoneme_pairs": [],
        }
