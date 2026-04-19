<<<<<<< HEAD
from services.dataset_loader import load_tsv_sentences, extract_words
from services.mapping_engine import map_all
import json


# LOAD DATASET

indo_path = "./dataset/indonesia/data/validated.tsv"
sunda_path = "./dataset/sunda/ss-corpus-su.tsv"

indo_sentences = load_tsv_sentences(indo_path, limit=100)
sunda_sentences = load_tsv_sentences(sunda_path, limit=100)

indo_words = extract_words(indo_sentences)
sunda_words = extract_words(sunda_sentences)

print("INDO:", indo_words[:10])
print("SUNDA:", sunda_words[:10])


# MAPPING

results = map_all(indo_words, sunda_words, limit=50)

print("\n=== HASIL MAPPING ===")

filtered_results = {}

for k, v in results.items():
    if v:
        word, score, normalized_score = v

        # filter hasil bagus (berbasis skor ter-normalisasi)
        if normalized_score <= 1.5:
            print(f"{k} -> {word} (score: {score}, norm: {normalized_score:.2f})")
            filtered_results[k] = {
                "target": word,
                "score": score,
                "normalized_score": round(normalized_score, 4)
            }


# SAVE MAPPING

with open("results.json", "w", encoding="utf-8") as f:
    json.dump(filtered_results, f, indent=2, ensure_ascii=False)

print("\n✅ Hasil disimpan ke results.json")


# BUAT TRAINING DATA

training_data = []

for src, mapping in filtered_results.items():
    training_data.append({
        "input": src,
        "target": mapping["target"],
        "normalized_score": mapping["normalized_score"]
    })

# simpan training data
with open("training_data.json", "w", encoding="utf-8") as f:
    json.dump(training_data, f, indent=2, ensure_ascii=False)

=======
from services.dataset_loader import load_tsv_sentences, extract_words
from services.mapping_engine import map_all
import json

# ========================
# LOAD DATASET
# ========================
indo_path = "./dataset/indonesia/data/validated.tsv"
sunda_path = "./dataset/sunda/ss-corpus-su.tsv"

indo_sentences = load_tsv_sentences(indo_path, limit=100)
sunda_sentences = load_tsv_sentences(sunda_path, limit=100)

indo_words = extract_words(indo_sentences)
sunda_words = extract_words(sunda_sentences)

print("INDO:", indo_words[:10])
print("SUNDA:", sunda_words[:10])

# ========================
# 🔥 MAPPING
# ========================
results = map_all(indo_words, sunda_words, limit=50)

print("\n=== HASIL MAPPING ===")

filtered_results = {}

for k, v in results.items():
    if v:
        word, score = v

        # filter hasil bagus
        if score < 8:
            print(f"{k} -> {word} (score: {score})")
            filtered_results[k] = (word, score)

# ========================
# 💾 SAVE MAPPING
# ========================
with open("results.json", "w", encoding="utf-8") as f:
    json.dump(filtered_results, f, indent=2, ensure_ascii=False)

print("\n✅ Hasil disimpan ke results.json")

# ========================
# 🔥 STEP DL — BUAT TRAINING DATA
# ========================
training_data = []

for src, (tgt, score) in filtered_results.items():
    training_data.append({
        "input": src,
        "target": tgt
    })

# simpan training data
with open("training_data.json", "w", encoding="utf-8") as f:
    json.dump(training_data, f, indent=2, ensure_ascii=False)

>>>>>>> 7688b6479ef43ba9a09e9d2e459bbf63d590dfab
print("✅ Training data disimpan ke training_data.json")