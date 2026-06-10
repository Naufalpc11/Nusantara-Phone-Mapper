import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from services.audio.dataset import (
    AudioPhonemeDataset,
    collate_audio_batch,
    load_audio_manifest,
)
from services.audio.metrics import ErrorRateAccumulator
from services.audio.model import AudioCTCModel
from services.audio.phonemes import (
    PhonemeVocabulary,
    greedy_ctc_decode,
    tokens_to_text,
    transcript_to_phonemes,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_AUDIO_ROOT = PROJECT_ROOT / "dataset" / "datasetaudio"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a CNN-BiLSTM-CTC audio-to-phoneme model."
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=DEFAULT_AUDIO_ROOT,
    )
    parser.add_argument(
        "--language",
        choices=("id", "su", "both"),
        default="both",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--conv-channels", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=192)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--limit-per-language", type=int)
    parser.add_argument("--checkpoint-path", type=Path, default=Path("audio_ctc_best.pt"))
    parser.add_argument(
        "--history-path",
        type=Path,
        default=Path("audio_training_history.json"),
    )
    parser.add_argument(
        "--test-results-path",
        type=Path,
        default=Path("audio_test_results.json"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("audio_feature_cache"),
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def selected_languages(language):
    return ("id", "su") if language == "both" else (language,)


def dataset_directory(audio_root, language):
    return audio_root / ("datasetindo" if language == "id" else "datasetsunda")


def load_split_rows(audio_root, languages, split, limit_per_language=None):
    rows = []
    for language in languages:
        manifest = dataset_directory(audio_root, language) / f"{split}.tsv"
        language_rows = load_audio_manifest(manifest, limit=limit_per_language)
        if not language_rows:
            raise ValueError(f"Manifest tidak menghasilkan data valid: {manifest}")
        rows.extend(language_rows)
    return rows


def build_vocabulary(rows):
    tokens = []
    for row in rows:
        tokens.extend(
            transcript_to_phonemes(row["transcript"], row["language"])
        )
    return PhonemeVocabulary(tokens)


def build_loader(
    rows,
    vocabulary,
    cache_dir,
    n_mels,
    batch_size,
    shuffle,
    num_workers,
):
    dataset = AudioPhonemeDataset(
        rows,
        vocabulary,
        cache_dir=cache_dir,
        n_mels=n_mels,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_audio_batch,
        pin_memory=torch.cuda.is_available(),
    )


def run_epoch(
    model,
    loader,
    loss_fn,
    vocabulary,
    device,
    optimizer=None,
    max_examples=20,
):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    metrics = ErrorRateAccumulator()
    language_metrics = {}
    examples = []

    for batch in loader:
        features = batch["features"].to(device)
        feature_lengths = batch["feature_lengths"].to(device)
        targets = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)

        with torch.set_grad_enabled(training):
            logits, output_lengths = model(features, feature_lengths)
            log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
            loss = loss_fn(
                log_probs,
                targets,
                output_lengths.cpu(),
                target_lengths.cpu(),
            )

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        batch_size = features.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        decoded_ids = greedy_ctc_decode(
            logits.detach().cpu(),
            output_lengths.cpu(),
            vocabulary.blank_id,
        )

        for index, predicted_ids in enumerate(decoded_ids):
            reference_tokens = batch["phonemes"][index]
            predicted_tokens = vocabulary.decode(predicted_ids)
            metrics.update(reference_tokens, predicted_tokens)
            language = batch["languages"][index]
            language_metrics.setdefault(language, ErrorRateAccumulator()).update(
                reference_tokens,
                predicted_tokens,
            )

            if len(examples) < max_examples:
                examples.append(
                    {
                        "audio_id": batch["audio_ids"][index],
                        "language": language,
                        "transcript": batch["transcripts"][index],
                        "reference_phonemes": reference_tokens,
                        "predicted_phonemes": predicted_tokens,
                        "reference_text": tokens_to_text(reference_tokens),
                        "predicted_text": tokens_to_text(predicted_tokens),
                    }
                )

    result = metrics.compute()
    result["loss"] = total_loss / total_samples if total_samples else None
    result["samples"] = total_samples
    result["per_language"] = {
        language: accumulator.compute()
        for language, accumulator in language_metrics.items()
    }
    result["examples"] = examples
    return result


def checkpoint_payload(model, vocabulary, args, epoch, validation):
    return {
        "model_state_dict": model.state_dict(),
        "model_config": model.config,
        "vocabulary": vocabulary.to_dict(),
        "training_config": {
            **vars(args),
            "audio_root": str(args.audio_root),
            "checkpoint_path": str(args.checkpoint_path),
            "history_path": str(args.history_path),
            "test_results_path": str(args.test_results_path),
            "cache_dir": str(args.cache_dir),
        },
        "epoch": epoch,
        "validation": validation,
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    for output_path in (
        args.checkpoint_path,
        args.history_path,
        args.test_results_path,
    ):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    languages = selected_languages(args.language)

    train_rows = load_split_rows(
        args.audio_root,
        languages,
        "train",
        args.limit_per_language,
    )
    validation_rows = load_split_rows(
        args.audio_root,
        languages,
        "validation",
        args.limit_per_language,
    )
    test_rows = load_split_rows(
        args.audio_root,
        languages,
        "test",
        args.limit_per_language,
    )
    vocabulary = build_vocabulary(train_rows)

    train_loader = build_loader(
        train_rows,
        vocabulary,
        args.cache_dir / "train",
        args.n_mels,
        args.batch_size,
        True,
        args.num_workers,
    )
    validation_loader = build_loader(
        validation_rows,
        vocabulary,
        args.cache_dir / "validation",
        args.n_mels,
        args.batch_size,
        False,
        args.num_workers,
    )
    test_loader = build_loader(
        test_rows,
        vocabulary,
        args.cache_dir / "test",
        args.n_mels,
        args.batch_size,
        False,
        args.num_workers,
    )

    model = AudioCTCModel(
        vocabulary_size=len(vocabulary),
        n_mels=args.n_mels,
        conv_channels=args.conv_channels,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    loss_fn = nn.CTCLoss(
        blank=vocabulary.blank_id,
        zero_infinity=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    print(
        "Start audio training | "
        f"device={device}, languages={','.join(languages)}, "
        f"train={len(train_rows)}, validation={len(validation_rows)}, "
        f"test={len(test_rows)}, vocabulary={len(vocabulary)}, "
        f"epochs={args.epochs}, batch_size={args.batch_size}"
    )

    history = {
        "config": {
            **vars(args),
            "audio_root": str(args.audio_root),
            "checkpoint_path": str(args.checkpoint_path),
            "history_path": str(args.history_path),
            "test_results_path": str(args.test_results_path),
            "cache_dir": str(args.cache_dir),
            "device": str(device),
            "languages": list(languages),
            "vocabulary_size": len(vocabulary),
        },
        "epochs": [],
    }
    best_per = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_result = run_epoch(
            model,
            train_loader,
            loss_fn,
            vocabulary,
            device,
            optimizer=optimizer,
            max_examples=0,
        )
        validation_result = run_epoch(
            model,
            validation_loader,
            loss_fn,
            vocabulary,
            device,
            max_examples=5,
        )
        history["epochs"].append(
            {
                "epoch": epoch,
                "train": {
                    key: value
                    for key, value in train_result.items()
                    if key != "examples"
                },
                "validation": validation_result,
            }
        )
        args.history_path.write_text(
            json.dumps(history, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_result['loss']:.4f} | "
            f"train_PER={train_result['per']:.4f} | "
            f"val_loss={validation_result['loss']:.4f} | "
            f"val_PER={validation_result['per']:.4f} | "
            f"val_CER={validation_result['cer']:.4f}"
        )

        validation_per = validation_result["per"]
        if validation_per is not None and validation_per < best_per:
            best_per = validation_per
            epochs_without_improvement = 0
            torch.save(
                checkpoint_payload(
                    model,
                    vocabulary,
                    args,
                    epoch,
                    validation_result,
                ),
                args.checkpoint_path,
            )
            print(f"Best checkpoint saved: {args.checkpoint_path}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(
                f"Early stopping: validation PER tidak membaik selama "
                f"{args.patience} epoch."
            )
            break

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_result = run_epoch(
        model,
        test_loader,
        loss_fn,
        vocabulary,
        device,
        max_examples=20,
    )
    test_payload = {
        "checkpoint_epoch": checkpoint["epoch"],
        "languages": list(languages),
        "metrics": {
            key: value
            for key, value in test_result.items()
            if key != "examples"
        },
        "examples": test_result["examples"],
    }
    args.test_results_path.write_text(
        json.dumps(test_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        "Test result | "
        f"loss={test_result['loss']:.4f}, "
        f"PER={test_result['per']:.4f}, "
        f"CER={test_result['cer']:.4f}"
    )
    print(f"Training history saved: {args.history_path}")
    print(f"Test results saved: {args.test_results_path}")


if __name__ == "__main__":
    main()
