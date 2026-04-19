def text_to_ipa(text):
    # simplifikasi: huruf → IPA basic
    mapping = {
        "a": "a",
        "i": "i",
        "u": "u",
        "e": "e",
        "o": "o",

        "k": "k",
        "g": "g",
        "t": "t",
        "d": "d",
        "p": "p",
        "b": "b",

        "m": "m",
        "n": "n",

        "s": "s",
        "f": "f",
        "v": "v",
        "z": "z",

        "r": "r",
        "l": "l",
        "h": "h",
    }

    result = []

    for char in text.lower():
        if char in mapping:
            result.append(mapping[char])

    return result
