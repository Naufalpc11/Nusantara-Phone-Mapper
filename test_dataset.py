import argparse
import json

from services.data.dataset_loader import load_tsv_sentences
from services.data.pre_processing import extract_words
from services.data.training_data_builder import build_training_examples
from services.evaluation.baseline_builder import build_baseline_results
from services.evaluation.evaluator import evaluate_with_seed_lexicon
from services.phoneme.mapper import LEXICAL_MAP_ID_SU


DEFAULT_INDO_PATH = "Nusantara-Phone-Mapper/dataset/indonesia/data/validated.tsv"
DEFAULT_SUNDA_PATH = "Nusantara-Phone-Mapper/dataset/sunda/sundatsv/utt_spk_text2.tsv"


def parse_args():
    parser = argparse.ArgumentParser(description="Build Indo-Sunda baseline, evaluation, and training data")
    parser.add_argument("--indo-path", type=str, default=DEFAULT_INDO_PATH)
    parser.add_argument("--sunda-path", type=str, default=DEFAULT_SUNDA_PATH)
    parser.add_argument("--data-limit", type=int, default=200000)
    parser.add_argument("--mapping-limit", type=int, default=200000)
    parser.add_argument("--indo-word-limit", type=int, default=200000)
    parser.add_argument("--sunda-word-limit", type=int, default=200000)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--negatives-per-positive", type=int, default=2)
    parser.add_argument("--include-pseudo-positives", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    indo_sentences = load_tsv_sentences(args.indo_path, limit=args.data_limit)
    sunda_sentences = load_tsv_sentences(args.sunda_path, limit=args.data_limit)

    indo_words = extract_words(indo_sentences, max_words=args.indo_word_limit)
    sunda_words = extract_words(sunda_sentences, max_words=args.sunda_word_limit)

    print("INDO:", indo_words[:10])
    print("SUNDA:", sunda_words[:10])
    print(f"Jumlah kata Indo: {len(indo_words)}")
    print(f"Jumlah kata Sunda: {len(sunda_words)}")

    baseline_artifacts = build_baseline_results(
        indo_words,
        sunda_words,
        limit=args.mapping_limit,
        top_k=args.top_k,
        threshold=args.threshold,
    )
    evaluation_artifacts = evaluate_with_seed_lexicon(
        indo_words,
        sunda_words,
        LEXICAL_MAP_ID_SU,
        top_k=args.top_k,
    )
    training_artifacts = build_training_examples(
        indo_words,
        sunda_words,
        baseline_artifacts["results"],
        negatives_per_positive=args.negatives_per_positive,
        include_pseudo_positives=args.include_pseudo_positives,
    )

    print("\n=== RINGKASAN BASELINE ===")
    print(json.dumps(baseline_artifacts["summary"], indent=2, ensure_ascii=False))

    print("\n=== RINGKASAN EVALUASI ===")
    print(json.dumps(evaluation_artifacts["summary"], indent=2, ensure_ascii=False))

    print("\n=== RINGKASAN TRAINING DATA ===")
    print(json.dumps(training_artifacts["summary"], indent=2, ensure_ascii=False))

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(baseline_artifacts["legacy_results"], f, indent=2, ensure_ascii=False)

    with open("mapping_results.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": baseline_artifacts["summary"],
                "results": baseline_artifacts["results"],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(evaluation_artifacts, f, indent=2, ensure_ascii=False)

    with open("training_data.json", "w", encoding="utf-8") as f:
        json.dump(training_artifacts["results"], f, indent=2, ensure_ascii=False)

    print("\nFile output:")
    print("- results.json: hasil baseline top-1 yang lolos threshold")
    print("- mapping_results.json: hasil baseline lengkap + top kandidat")
    print("- evaluation_results.json: evaluasi seed lexicon")
    print("- training_data.json: pasangan berlabel untuk model neural")


if __name__ == "__main__":
    main()
