from services.dataset_loader import load_tsv_sentences, extract_words

# path sesuai folder lu
indo_path = "./dataset/indonesia/data/validated.tsv"
sunda_path = "./dataset/sunda/ss-corpus-su.tsv"

indo_sentences = load_tsv_sentences(indo_path, limit=100)
sunda_sentences = load_tsv_sentences(sunda_path, limit=100)

indo_words = extract_words(indo_sentences)
sunda_words = extract_words(sunda_sentences)

print("INDO:", indo_words[:10])
print("SUNDA:", sunda_words[:10])