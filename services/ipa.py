"""Compatibility helpers for phoneme mapping.

Historically this module exposed a tiny character-based IPA converter.
It now delegates to the richer phoneme inventories in services/ipa_id.py
and services/ipa_su.py while keeping the old text_to_ipa() API intact.
"""

from services.ipa_id import word_to_ipa as word_to_ipa_id
from services.ipa_su import word_to_ipa_su


def text_to_ipa(text):
    """Convert text to a phoneme-like sequence for scoring.

    The output remains a list of IPA-like units so the existing scoring code
    in phoneme_mapper.py can keep operating without further changes.
    """
    text = (text or "").strip().lower()
    if not text:
        return []

    # Prefer Indonesian parsing because the mapper scores ID source words
    # against Sundanese target words. For generic scoring we keep the output
    # simple and tokenized as a list of IPA-like units.
    ipa_text = word_to_ipa_id(text)

    units = []
    i = 0
    while i < len(ipa_text):
        if ipa_text[i : i + 2] in {"tʃ", "dʒ", "ɲ", "ŋ", "ʃ", "ɛ", "ə", "ɨ", "ʔ", "x"}:
            units.append(ipa_text[i : i + 2])
            i += 2
        else:
            units.append(ipa_text[i])
            i += 1
    return units


def text_to_ipa_sunda(text):
    """Convenience helper for Sundanese phoneme extraction."""
    return word_to_ipa_su(text or "")
