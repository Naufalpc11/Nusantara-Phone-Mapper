"""Build rule-based baseline mapping results."""

from services.phoneme.mapper import LEXICAL_MAP_ID_SU, rank_candidates


def _candidate_preview(candidate, rank):
    return {
        "rank": rank,
        "target": candidate["target"],
        "score": candidate["score"],
        "normalized_score": round(candidate["normalized_score"], 4),
        "hamming": candidate["hamming"],
        "length_diff": candidate["length_diff"],
    }


def build_baseline_results(source_words, target_words, limit=None, top_k=5, threshold=1.5):
    """Bangun hasil baseline lengkap untuk satu daftar kata sumber."""
    records = []
    legacy_results = {}

    total_words = len(source_words) if limit is None else min(len(source_words), limit)
    accepted_count = 0
    exact_match_count = 0
    lexical_seed_count = 0
    normalized_score_sum = 0.0

    for idx, source_word in enumerate(source_words[:total_words]):
        ranked = rank_candidates(source_word, target_words, top_k=top_k)
        if not ranked:
            continue

        best = ranked[0]
        accepted = best["normalized_score"] <= threshold
        exact_match = best["target"] == source_word
        lexical_target = LEXICAL_MAP_ID_SU.get(source_word)
        used_lexical_seed = lexical_target is not None and lexical_target == best["target"]

        top_candidates = [
            _candidate_preview(candidate, rank)
            for rank, candidate in enumerate(ranked, start=1)
        ]

        record = {
            "input": source_word,
            "target": best["target"],
            "score": best["score"],
            "normalized_score": round(best["normalized_score"], 4),
            "accepted": accepted,
            "exact_match": exact_match,
            "lexical_seed_target": lexical_target,
            "used_lexical_seed": used_lexical_seed,
            "top_candidates": top_candidates,
        }
        records.append(record)

        normalized_score_sum += best["normalized_score"]

        if accepted:
            accepted_count += 1
            legacy_results[source_word] = {
                "target": best["target"],
                "score": best["score"],
                "normalized_score": round(best["normalized_score"], 4),
            }

        if exact_match:
            exact_match_count += 1

        if used_lexical_seed:
            lexical_seed_count += 1

    avg_normalized_score = round(normalized_score_sum / len(records), 4) if records else None

    summary = {
        "inputs": len(records),
        "targets": len(target_words),
        "accepted": accepted_count,
        "exact_matches": exact_match_count,
        "lexical_seed_hits": lexical_seed_count,
        "avg_normalized_score": avg_normalized_score,
        "threshold": threshold,
        "top_k": top_k,
    }

    return {
        "summary": summary,
        "results": records,
        "legacy_results": legacy_results,
    }
