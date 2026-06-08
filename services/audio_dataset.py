import csv
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from services.audio_features import load_audio, log_mel_spectrogram
from services.audio_phonemes import transcript_to_phonemes


def load_audio_manifest(path, limit=None):
    manifest_path = Path(path)
    rows = []

    with manifest_path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        for row in reader:
            audio_path = manifest_path.parent / row["audio_path"]
            if not row.get("audio_path") or not row.get("transcript"):
                continue

            rows.append(
                {
                    **row,
                    "audio_path_resolved": audio_path,
                }
            )
            if limit is not None and len(rows) >= limit:
                break

    return rows


class AudioPhonemeDataset(Dataset):
    def __init__(
        self,
        rows,
        vocabulary,
        cache_dir=None,
        sample_rate=16000,
        n_mels=80,
    ):
        self.rows = rows
        self.vocabulary = vocabulary
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.rows)

    def _cache_path(self, row):
        if not self.cache_dir:
            return None
        return self.cache_dir / f"{row['language']}_{row['audio_id']}.pt"

    def _features(self, row):
        cache_path = self._cache_path(row)
        if cache_path and cache_path.is_file():
            return torch.load(cache_path, map_location="cpu", weights_only=True)

        waveform = load_audio(
            row["audio_path_resolved"],
            sample_rate=self.sample_rate,
        )
        features = log_mel_spectrogram(
            waveform,
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
        )

        if cache_path:
            torch.save(features, cache_path)
        return features

    def __getitem__(self, index):
        row = self.rows[index]
        phonemes = transcript_to_phonemes(
            row["transcript"],
            row["language"],
        )
        target = torch.tensor(
            self.vocabulary.encode(phonemes),
            dtype=torch.long,
        )

        return {
            "features": self._features(row).transpose(0, 1),
            "target": target,
            "phonemes": phonemes,
            "transcript": row["transcript"],
            "language": row["language"],
            "audio_id": row["audio_id"],
            "audio_path": str(row["audio_path_resolved"]),
        }


def collate_audio_batch(batch):
    features = [item["features"] for item in batch]
    targets = [item["target"] for item in batch]
    feature_lengths = torch.tensor(
        [feature.size(0) for feature in features],
        dtype=torch.long,
    )
    target_lengths = torch.tensor(
        [target.size(0) for target in targets],
        dtype=torch.long,
    )

    return {
        "features": pad_sequence(features, batch_first=True),
        "feature_lengths": feature_lengths,
        "targets": torch.cat(targets),
        "target_lengths": target_lengths,
        "target_sequences": targets,
        "phonemes": [item["phonemes"] for item in batch],
        "transcripts": [item["transcript"] for item in batch],
        "languages": [item["language"] for item in batch],
        "audio_ids": [item["audio_id"] for item in batch],
        "audio_paths": [item["audio_path"] for item in batch],
    }
