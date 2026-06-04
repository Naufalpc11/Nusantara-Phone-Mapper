from services.ipa_id import PHONEME_ID, phoneme_sequence as phoneme_sequence_id
from services.ipa_su import PHONEME_SU, phoneme_sequence_su


IPA_INDEX_ID = {entry["ipa"]: entry for entry in PHONEME_ID.values()}
IPA_INDEX_SU = {entry["ipa"]: entry for entry in PHONEME_SU.values()}

UNKNOWN_PENALTY = 10


def _lookup_entry(ipa_symbol, lang):
    index = IPA_INDEX_ID if lang == "id" else IPA_INDEX_SU
    return index.get(ipa_symbol)


def _coords_from_entry(entry):
    coord = entry.get("coord") if entry else None
    if coord is None:
        return []
    if coord and isinstance(coord[0], tuple):
        return list(coord)
    return [coord]


def _anchor_coord(entry):
    coords = _coords_from_entry(entry)
    if not coords:
        return None
    entry_type = entry.get("type") if entry else None
    if entry_type == "diphthong":
        return coords[0]
    if entry_type == "cluster":
        return coords[-1]
    return coords[0]


def _is_vowel(entry):
    return entry and entry.get("type") in {"vowel", "diphthong"}


def _distance_single(coord_a, coord_b, use_vowel_priority):
    diffs = [abs(coord_a[i] - coord_b[i]) for i in range(3)]
    manhattan = sum(diffs)
    hamming = sum(1 for diff in diffs if diff != 0)
    if use_vowel_priority:
        priority = tuple(diffs)
    else:
        priority = tuple(diffs)
    return manhattan, hamming, priority


def _phoneme_distance(entry_a, entry_b):
    coords_a = _coords_from_entry(entry_a)
    coords_b = _coords_from_entry(entry_b)

    if not coords_a or not coords_b:
        return UNKNOWN_PENALTY, 1, (UNKNOWN_PENALTY, UNKNOWN_PENALTY, UNKNOWN_PENALTY)

    use_vowel_priority = _is_vowel(entry_a) and _is_vowel(entry_b)

    if len(coords_a) == 1 and len(coords_b) == 1:
        return _distance_single(coords_a[0], coords_b[0], use_vowel_priority)

    if len(coords_a) == len(coords_b):
        total_manhattan = 0
        total_hamming = 0
        total_priority = [0, 0, 0]
        for coord_a, coord_b in zip(coords_a, coords_b):
            manhattan, hamming, priority = _distance_single(coord_a, coord_b, use_vowel_priority)
            total_manhattan += manhattan
            total_hamming += hamming
            for i in range(3):
                total_priority[i] += priority[i]
        return total_manhattan, total_hamming, tuple(total_priority)

    anchor_a = _anchor_coord(entry_a)
    anchor_b = _anchor_coord(entry_b)
    if anchor_a is None or anchor_b is None:
        return UNKNOWN_PENALTY, 1, (UNKNOWN_PENALTY, UNKNOWN_PENALTY, UNKNOWN_PENALTY)
    return _distance_single(anchor_a, anchor_b, use_vowel_priority)

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

    ranked = rank_candidates(word, target_words, top_k=1)
    if not ranked:
        return None

    best = ranked[0]
    return best["target"], best["score"], best["normalized_score"]


def score_word_pair(source_word, target_word, apply_lexical=True):
    """Hitung skor fonetik untuk satu pasangan kata."""
    src_word = (source_word or "").lower().strip()
    tgt_word = (target_word or "").lower().strip()

    if apply_lexical:
        lexical = LEXICAL_MAP_ID_SU.get(src_word)
        if lexical and lexical == tgt_word:
            return {
                "target": tgt_word,
                "score": 0,
                "normalized_score": 0.0,
                "hamming": 0,
                "priority": (0, 0, 0),
                "length_diff": 0,
                "src_len": len(phoneme_sequence_id(src_word)),
                "tgt_len": len(phoneme_sequence_su(tgt_word)),
            }

    src_phonemes = phoneme_sequence_id(src_word)
    tgt_phonemes = phoneme_sequence_su(tgt_word)

    total_manhattan = 0
    total_hamming = 0
    total_priority = [0, 0, 0]

    aligned_len = max(len(src_phonemes), len(tgt_phonemes), 1)

    for src_ipa, tgt_ipa in zip(src_phonemes, tgt_phonemes):
        src_entry = _lookup_entry(src_ipa, "id")
        tgt_entry = _lookup_entry(tgt_ipa, "su")
        manhattan, hamming, priority = _phoneme_distance(src_entry, tgt_entry)
        total_manhattan += manhattan
        total_hamming += hamming
        for i in range(3):
            total_priority[i] += priority[i]

    length_diff = abs(len(src_phonemes) - len(tgt_phonemes))
    total_manhattan += length_diff * LENGTH_PENALTY
    normalized_score = total_manhattan / aligned_len

    return {
        "target": tgt_word,
        "score": total_manhattan,
        "normalized_score": normalized_score,
        "hamming": total_hamming,
        "priority": tuple(total_priority),
        "length_diff": length_diff,
        "src_len": len(src_phonemes),
        "tgt_len": len(tgt_phonemes),
    }


def rank_candidates(source_word, target_words, top_k=None):
    """Urutkan kandidat target berdasarkan skor fonetik."""
    results = []

    for target in target_words:
        scored = score_word_pair(source_word, target)
        results.append(scored)

    if not results:
        return []

    results.sort(
        key=lambda x: (
            x["normalized_score"],
            x["hamming"],
            x["priority"],
            x["score"],
            x["length_diff"],
            x["target"],
        )
    )

    if top_k is not None:
        return results[:top_k]

    return results


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
