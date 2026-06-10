import argparse
import json
import sys
from pathlib import Path

import torch

from services.audio.dataset import collate_audio_batch
from services.audio.features import load_audio, log_mel_spectrogram
from services.audio.metrics import ErrorRateAccumulator
from services.audio.model import AudioCTCModel
from services.audio.phonemes import (
    PhonemeVocabulary,
    greedy_ctc_decode,
    tokens_to_text,
    transcript_to_phonemes,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict a phoneme sequence from one audio file."
    )
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("audio_ctc_best.pt"),
    )
    parser.add_argument("--language", choices=("id", "su"))
    parser.add_argument("--transcript", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if not args.audio.is_file():
        raise FileNotFoundError(f"Audio tidak ditemukan: {args.audio}")
    if args.transcript and not args.language:
        raise ValueError("--language diperlukan jika --transcript diberikan.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    vocabulary = PhonemeVocabulary.from_dict(checkpoint["vocabulary"])
    model = AudioCTCModel(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    n_mels = checkpoint["model_config"]["n_mels"]
    waveform = load_audio(args.audio)
    features = log_mel_spectrogram(waveform, n_mels=n_mels).transpose(0, 1)
    batch = collate_audio_batch(
        [
            {
                "features": features,
                "target": torch.empty(0, dtype=torch.long),
                "phonemes": [],
                "transcript": args.transcript or "",
                "language": args.language or "",
                "audio_id": args.audio.stem,
                "audio_path": str(args.audio),
            }
        ]
    )

    with torch.no_grad():
        logits, output_lengths = model(
            batch["features"].to(device),
            batch["feature_lengths"].to(device),
        )

    predicted_ids = greedy_ctc_decode(
        logits.cpu(),
        output_lengths.cpu(),
        vocabulary.blank_id,
    )[0]
    predicted_tokens = vocabulary.decode(predicted_ids)
    result = {
        "audio": str(args.audio),
        "predicted_phonemes": predicted_tokens,
        "predicted_text": tokens_to_text(predicted_tokens),
    }

    if args.transcript:
        reference_tokens = transcript_to_phonemes(
            args.transcript,
            args.language,
        )
        metrics = ErrorRateAccumulator()
        metrics.update(reference_tokens, predicted_tokens)
        result["transcript"] = args.transcript
        result["reference_phonemes"] = reference_tokens
        result["reference_text"] = tokens_to_text(reference_tokens)
        result["metrics"] = metrics.compute()

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
