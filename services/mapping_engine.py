from services.phoneme_mapper import map_word


def map_all(source_words, target_words, limit=50):
    results = {}

    for i, word in enumerate(source_words):
        if i >= limit:
            break

        best = map_word(word, target_words)
        results[word] = best

    return results