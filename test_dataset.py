from services.dataset_loader import load_tsv_sentences, extract_words
from services.mapping_engine import map_all
import json


DATA_LIMIT = 500
MAPPING_LIMIT = 500
NORMALIZED_SCORE_THRESHOLD = 1.5


indo_path = "./dataset/indonesia/data/validated.tsv"
sunda_path = "./dataset/sunda/ss-corpus-su.tsv"

indo_sentences = load_tsv_sentences(indo_path, limit=DATA_LIMIT)
sunda_sentences = load_tsv_sentences(sunda_path, limit=DATA_LIMIT)

indo_words = extract_words(indo_sentences)
sunda_words = extract_words(sunda_sentences)

print("INDO:", indo_words[:10])
print("SUNDA:", sunda_words[:10])

results = map_all(indo_words, sunda_words, limit=MAPPING_LIMIT)

print("\n=== HASIL MAPPING ===")

filtered_results = {}

for k, v in results.items():
    if v:
        word, score, normalized_score = v

        if normalized_score <= NORMALIZED_SCORE_THRESHOLD:
            print(f"{k} -> {word} (score: {score}, norm: {normalized_score:.2f})")
            filtered_results[k] = {
                "target": word,
                "score": score,
                "normalized_score": round(normalized_score, 4),
            }

with open("results.json", "w", encoding="utf-8") as f:
    json.dump(filtered_results, f, indent=2, ensure_ascii=False)

print("\nHasil disimpan ke results.json")

training_data = []

for src, mapping in filtered_results.items():
    training_data.append({
        "input": src,
        "target": mapping["target"],
        "normalized_score": mapping["normalized_score"],
    })

with open("training_data.json", "w", encoding="utf-8") as f:
    json.dump(training_data, f, indent=2, ensure_ascii=False)

print("Training data disimpan ke training_data.json")
print(f"Jumlah training pair: {len(training_data)}")