from services.phoneme_mapper import rank_candidates


def _safe_rate(numerator, denominator):
    if denominator == 0:
        return None
    return round(numerator / denominator, 4)


def evaluate_with_seed_lexicon(source_words, target_words, lexical_map, top_k=5):
    """Evaluasi baseline menggunakan seed lexicon yang dipercaya."""
    source_set = set(source_words)
    target_set = set(target_words)
    eval_items = []

    for source_word, gold_target in lexical_map.items():
        if source_word not in source_set or gold_target not in target_set:
            continue

        ranked = rank_candidates(source_word, target_words)
        gold_rank = None
        gold_score = None

        for rank, candidate in enumerate(ranked, start=1):
            if candidate["target"] == gold_target:
                gold_rank = rank
                gold_score = candidate["normalized_score"]
                break

        best_target = ranked[0]["target"] if ranked else None
        eval_items.append(
            {
                "input": source_word,
                "gold_target": gold_target,
                "predicted_target": best_target,
                "gold_rank": gold_rank,
                "gold_normalized_score": round(gold_score, 4) if gold_score is not None else None,
                "hit_at_1": gold_rank == 1,
                f"hit_at_{top_k}": gold_rank is not None and gold_rank <= top_k,
            }
        )

    total = len(eval_items)
    hit_at_1 = sum(1 for item in eval_items if item["hit_at_1"])
    hit_at_3 = sum(1 for item in eval_items if item["gold_rank"] is not None and item["gold_rank"] <= 3)
    hit_at_k = sum(1 for item in eval_items if item[f"hit_at_{top_k}"])
    reciprocal_rank_sum = sum(1.0 / item["gold_rank"] for item in eval_items if item["gold_rank"])
    mean_rank = None

    ranks = [item["gold_rank"] for item in eval_items if item["gold_rank"] is not None]
    if ranks:
        mean_rank = round(sum(ranks) / len(ranks), 4)

    return {
        "summary": {
            "evaluated_pairs": total,
            "hit_at_1": _safe_rate(hit_at_1, total),
            "hit_at_3": _safe_rate(hit_at_3, total),
            f"hit_at_{top_k}": _safe_rate(hit_at_k, total),
            "mrr": round(reciprocal_rank_sum / total, 4) if total else None,
            "mean_rank": mean_rank,
        },
        "results": eval_items,
    }
