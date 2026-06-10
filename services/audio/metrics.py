"""Error-rate metrics for audio phoneme predictions."""

from services.audio.phonemes import WORD_SEPARATOR


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))

    for ref_index, ref_item in enumerate(reference, start=1):
        current = [ref_index]
        for hyp_index, hyp_item in enumerate(hypothesis, start=1):
            substitution = previous[hyp_index - 1] + (ref_item != hyp_item)
            insertion = current[hyp_index - 1] + 1
            deletion = previous[hyp_index] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current

    return previous[-1]


class ErrorRateAccumulator:
    def __init__(self):
        self.phone_errors = 0
        self.phone_total = 0
        self.character_errors = 0
        self.character_total = 0

    def update(self, reference_tokens, predicted_tokens):
        reference_phones = [
            token for token in reference_tokens if token != WORD_SEPARATOR
        ]
        predicted_phones = [
            token for token in predicted_tokens if token != WORD_SEPARATOR
        ]
        self.phone_errors += edit_distance(reference_phones, predicted_phones)
        self.phone_total += len(reference_phones)

        reference_text = "".join(reference_tokens).replace(WORD_SEPARATOR, " ")
        predicted_text = "".join(predicted_tokens).replace(WORD_SEPARATOR, " ")
        self.character_errors += edit_distance(reference_text, predicted_text)
        self.character_total += len(reference_text)

    def compute(self):
        return {
            "per": (
                self.phone_errors / self.phone_total
                if self.phone_total
                else None
            ),
            "cer": (
                self.character_errors / self.character_total
                if self.character_total
                else None
            ),
            "phone_errors": self.phone_errors,
            "phone_total": self.phone_total,
            "character_errors": self.character_errors,
            "character_total": self.character_total,
        }
