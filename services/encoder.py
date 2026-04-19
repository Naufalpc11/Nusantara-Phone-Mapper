import string

chars = list(string.ascii_lowercase)
char2idx = {c: i+1 for i, c in enumerate(chars)}


def encode_word(word, max_len=10):
    """Encode kata ke vektor indeks karakter dengan panjang tetap."""
    vec = [char2idx.get(c, 0) for c in word[:max_len]]

    while len(vec) < max_len:
        vec.append(0)
    return vec