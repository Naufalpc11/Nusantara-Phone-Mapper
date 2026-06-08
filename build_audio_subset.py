import argparse
import csv
import os
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "dataset"

DEFAULT_INDONESIAN_MANIFEST = DATASET_ROOT / "indonesia" / "data" / "validated.tsv"
DEFAULT_INDONESIAN_AUDIO = DATASET_ROOT / "indonesia" / "data" / "clips"
DEFAULT_SUNDANESE_MANIFEST = DATASET_ROOT / "sunda" / "sundatsv" / "utt_spk_text1.tsv"
DEFAULT_SUNDANESE_ROOT = DATASET_ROOT / "sunda"
DEFAULT_OUTPUT_ROOT = DATASET_ROOT / "datasetaudio"

MANIFEST_FIELDS = [
    "language",
    "audio_id",
    "audio_path",
    "transcript",
    "speaker_id",
    "split",
    "source_audio_path",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build balanced Indonesian and Sundanese audio subsets."
    )
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--train-count", type=int, default=4000)
    parser.add_argument("--validation-count", type=int, default=500)
    parser.add_argument("--test-count", type=int, default=500)
    parser.add_argument(
        "--indonesian-manifest",
        type=Path,
        default=DEFAULT_INDONESIAN_MANIFEST,
    )
    parser.add_argument(
        "--indonesian-audio-root",
        type=Path,
        default=DEFAULT_INDONESIAN_AUDIO,
    )
    parser.add_argument(
        "--sundanese-manifest",
        type=Path,
        default=DEFAULT_SUNDANESE_MANIFEST,
    )
    parser.add_argument(
        "--sundanese-audio-root",
        type=Path,
        default=DEFAULT_SUNDANESE_ROOT,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ganti subset lama di folder output yang sama.",
    )
    return parser.parse_args()


def validate_args(args):
    split_total = args.train_count + args.validation_count + args.test_count
    if args.count <= 0:
        raise ValueError("--count harus lebih besar dari 0.")
    if split_total != args.count:
        raise ValueError(
            "Jumlah --train-count, --validation-count, dan --test-count "
            "harus sama dengan --count."
        )

    for path in [
        args.indonesian_manifest,
        args.indonesian_audio_root,
        args.sundanese_manifest,
        args.sundanese_audio_root,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Path sumber tidak ditemukan: {path}")

    if (
        args.output_root.exists()
        and any(args.output_root.iterdir())
        and not args.overwrite
    ):
        raise FileExistsError(
            f"Folder output sudah berisi data: {args.output_root}. "
            "Gunakan --overwrite untuk mengganti subset lama secara aman."
        )


def assign_speaker_disjoint_splits(rows, train_count, validation_count, test_count):
    groups = {}

    for row in rows:
        # Missing speaker IDs are treated as different speakers.
        speaker_key = row["speaker_id"] or f"unknown:{row['audio_id']}"
        groups.setdefault(speaker_key, []).append(row)

    capacities = {
        "train": train_count,
        "validation": validation_count,
        "test": test_count,
    }
    remaining = dict(capacities)

    # Large speaker groups are placed first, then smaller groups fill the gaps.
    ordered_groups = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))

    for speaker_id, speaker_rows in ordered_groups:
        group_size = len(speaker_rows)
        available_splits = [
            split for split, capacity in remaining.items() if capacity >= group_size
        ]
        if not available_splits:
            raise RuntimeError(
                "Tidak dapat membuat split speaker-disjoint dengan jumlah yang diminta. "
                f"Speaker {speaker_id} memiliki {group_size} audio."
            )

        selected_split = max(
            available_splits,
            key=lambda split: (
                remaining[split] / capacities[split],
                remaining[split],
            ),
        )
        for row in speaker_rows:
            row["split"] = selected_split
        remaining[selected_split] -= group_size

    if any(remaining.values()):
        raise RuntimeError(
            "Pembagian speaker-disjoint tidak mencapai jumlah split yang diminta: "
            f"sisa kapasitas {remaining}."
        )


def relative_to_project(path):
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def select_indonesian_rows(manifest_path, audio_root, count):
    selected = []
    seen_audio_ids = set()

    with manifest_path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")

        for row in reader:
            filename = (row.get("path") or "").strip()
            transcript = (row.get("sentence") or "").strip()
            if not filename or not transcript:
                continue

            source_audio = audio_root / filename
            audio_id = Path(filename).stem
            if not source_audio.is_file() or audio_id in seen_audio_ids:
                continue

            selected.append(
                {
                    "audio_id": audio_id,
                    "filename": filename,
                    "transcript": transcript,
                    "speaker_id": (row.get("client_id") or "").strip(),
                    "source_audio": source_audio,
                }
            )
            seen_audio_ids.add(audio_id)

            if len(selected) == count:
                break

    if len(selected) < count:
        raise RuntimeError(
            f"Hanya menemukan {len(selected)} audio Indonesia valid dari target {count}."
        )

    return selected


def index_sundanese_audio(audio_root):
    audio_index = {}

    for directory, _, filenames in os.walk(audio_root):
        for filename in filenames:
            if not filename.lower().endswith(".flac"):
                continue

            path = Path(directory) / filename
            audio_id = path.stem
            existing = audio_index.get(audio_id)

            # Prefer files from the original ASR folders over manually copied subsets.
            if existing is None or (
                "sundatsv" in existing.parts and "sundatsv" not in path.parts
            ):
                audio_index[audio_id] = path

    return audio_index


def select_sundanese_rows(manifest_path, audio_index, count):
    selected = []
    seen_audio_ids = set()

    with manifest_path.open(encoding="utf-8", newline="") as file:
        reader = csv.reader(file, delimiter="\t")

        for row in reader:
            if len(row) < 3:
                continue

            audio_id = row[0].strip()
            speaker_id = row[1].strip()
            transcript = row[2].strip()
            source_audio = audio_index.get(audio_id)

            if (
                not audio_id
                or not transcript
                or source_audio is None
                or audio_id in seen_audio_ids
            ):
                continue

            selected.append(
                {
                    "audio_id": audio_id,
                    "filename": source_audio.name,
                    "transcript": transcript,
                    "speaker_id": speaker_id,
                    "source_audio": source_audio,
                }
            )
            seen_audio_ids.add(audio_id)

            if len(selected) == count:
                break

    if len(selected) < count:
        raise RuntimeError(
            f"Hanya menemukan {len(selected)} audio Sunda valid dari target {count}."
        )

    return selected


def write_manifest(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def build_language_subset(
    language,
    selected_rows,
    output_dir,
):
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []

    for index, item in enumerate(selected_rows):
        destination = audio_dir / item["filename"]
        shutil.copy2(item["source_audio"], destination)

        manifest_rows.append(
            {
                "language": language,
                "audio_id": item["audio_id"],
                "audio_path": f"audio/{destination.name}",
                "transcript": item["transcript"],
                "speaker_id": item["speaker_id"],
                "split": item["split"],
                "source_audio_path": relative_to_project(item["source_audio"]),
            }
        )

        if (index + 1) % 100 == 0:
            print(f"{language}: copied {index + 1}/{len(selected_rows)} audio")

    write_manifest(output_dir / "metadata.tsv", manifest_rows)

    for split in ("train", "validation", "test"):
        split_rows = [row for row in manifest_rows if row["split"] == split]
        write_manifest(output_dir / f"{split}.tsv", split_rows)

    return manifest_rows


def prepare_output_root(output_root, overwrite):
    output_root = output_root.resolve()
    dataset_root = DATASET_ROOT.resolve()

    if output_root == dataset_root or dataset_root not in output_root.parents:
        raise ValueError(
            f"Folder output harus berada di dalam folder dataset: {dataset_root}"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        return

    # Hanya folder hasil generator yang boleh dibersihkan.
    for directory_name in ("datasetindo", "datasetsunda"):
        generated_dir = (output_root / directory_name).resolve()
        if output_root not in generated_dir.parents:
            raise ValueError(f"Path output tidak aman: {generated_dir}")
        if generated_dir.exists():
            shutil.rmtree(generated_dir)


def main():
    args = parse_args()
    validate_args(args)

    print("Selecting Indonesian audio...")
    indonesian_rows = select_indonesian_rows(
        args.indonesian_manifest,
        args.indonesian_audio_root,
        args.count,
    )

    print("Indexing Sundanese audio...")
    sundanese_audio_index = index_sundanese_audio(args.sundanese_audio_root)
    print(f"Indexed {len(sundanese_audio_index)} unique Sundanese audio files.")

    print("Selecting Sundanese audio...")
    sundanese_rows = select_sundanese_rows(
        args.sundanese_manifest,
        sundanese_audio_index,
        args.count,
    )

    print("Creating speaker-disjoint splits...")
    assign_speaker_disjoint_splits(
        indonesian_rows,
        args.train_count,
        args.validation_count,
        args.test_count,
    )
    assign_speaker_disjoint_splits(
        sundanese_rows,
        args.train_count,
        args.validation_count,
        args.test_count,
    )

    prepare_output_root(args.output_root, args.overwrite)

    print("Building Indonesian subset...")
    build_language_subset(
        "id",
        indonesian_rows,
        args.output_root / "datasetindo",
    )

    print("Building Sundanese subset...")
    build_language_subset(
        "su",
        sundanese_rows,
        args.output_root / "datasetsunda",
    )

    print(f"Audio subset created at: {args.output_root}")
    print(
        f"Each language contains {args.count} audio files: "
        f"{args.train_count} train, {args.validation_count} validation, "
        f"and {args.test_count} test."
    )


if __name__ == "__main__":
    main()
