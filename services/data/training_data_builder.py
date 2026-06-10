"""Build labeled examples for the word-similarity model."""

import hashlib
import random

from services.phoneme.mapper import LEXICAL_MAP_ID_SU, score_word_pair


def _stable_split(source_word):
    digest = hashlib.md5(source_word.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 10
    if bucket < 7:
        return "train"
    if bucket < 9:
        return "val"
    return "test"


def _make_record(source_word, target_word, label, label_source, confidence, baseline_info=None):
    baseline_info = baseline_info or {}
    return {
        "input": source_word,
        "target": target_word,
        "label": int(label),
        "label_source": label_source,
        "confidence": float(confidence),
        "split": _stable_split(source_word),
        "baseline_rank": baseline_info.get("rank"),
        "baseline_score": baseline_info.get("score"),
        "baseline_normalized_score": baseline_info.get("normalized_score"),
    }


def _find_candidate_info(source_word, target_word, baseline_index):
    for candidate in baseline_index.get(source_word, {}).get("top_candidates", []):
        if candidate["target"] == target_word:
            return candidate
    return None


def build_training_examples(
    source_words,
    target_words,
    baseline_records,
    negatives_per_positive=2,
    include_shared_surface=True,
    include_pseudo_positives=False,
    pseudo_threshold=0.35,
    seed=42,
):
    """Bangun dataset pasangan berlabel untuk training model similarity."""
    rng = random.Random(seed)
    target_set = set(target_words)
    source_set = set(source_words)
    baseline_index = {record["input"]: record for record in baseline_records}
    records = []
    seen_pairs = set()

    def add_record(source_word, target_word, label, label_source, confidence, baseline_info=None):
        key = (source_word, target_word, label)
        if key in seen_pairs:
            return False
        seen_pairs.add(key)
        records.append(_make_record(source_word, target_word, label, label_source, confidence, baseline_info))
        return True

    positives = []

    for source_word, target_word in LEXICAL_MAP_ID_SU.items():
        if source_word not in source_set or target_word not in target_set:
            continue
        baseline_info = _find_candidate_info(source_word, target_word, baseline_index)
        if add_record(source_word, target_word, 1, "lexical_gold", 1.0, baseline_info):
            positives.append((source_word, target_word))

    if include_shared_surface:
        shared_words = sorted(source_set.intersection(target_set))
        for word in shared_words:
            lexical_target = LEXICAL_MAP_ID_SU.get(word)
            if lexical_target is not None and lexical_target != word:
                continue
            baseline_info = _find_candidate_info(word, word, baseline_index)
            if add_record(word, word, 1, "shared_surface_form", 0.8, baseline_info):
                positives.append((word, word))

    if include_pseudo_positives:
        for record in baseline_records:
            if not record["accepted"]:
                continue
            if record["normalized_score"] > pseudo_threshold:
                continue
            source_word = record["input"]
            target_word = record["target"]
            if LEXICAL_MAP_ID_SU.get(source_word) == target_word:
                continue
            if add_record(source_word, target_word, 1, "pseudo_positive", 0.6, record["top_candidates"][0]):
                positives.append((source_word, target_word))

    for source_word, positive_target in positives:
        baseline_candidates = baseline_index.get(source_word, {}).get("top_candidates", [])
        negatives_added = 0

        for candidate in baseline_candidates:
            if candidate["target"] == positive_target:
                continue
            if add_record(source_word, candidate["target"], 0, "hard_negative", 1.0, candidate):
                negatives_added += 1
            if negatives_added >= negatives_per_positive:
                break

        attempts = 0
        while negatives_added < negatives_per_positive and attempts < negatives_per_positive * 10 and target_words:
            attempts += 1
            random_target = rng.choice(target_words)
            if random_target == positive_target:
                continue
            if add_record(source_word, random_target, 0, "random_negative", 1.0):
                negatives_added += 1

    records.sort(key=lambda row: (row["input"], row["label"], row["target"]))

    positives_count = sum(1 for row in records if row["label"] == 1)
    negatives_count = len(records) - positives_count

    return {
        "summary": {
            "pairs": len(records),
            "positives": positives_count,
            "negatives": negatives_count,
            "sources": len({row["input"] for row in records}),
        },
        "results": records,
    }
