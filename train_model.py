import argparse
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from services.similarity.encoder import encode_word


def parse_args():
    """Baca konfigurasi training dari command line argument."""
    parser = argparse.ArgumentParser(description="Train Indo-Sunda similarity model")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--data-path", type=str, default="training_data.json")
    parser.add_argument("--model-path", type=str, default="model_similarity.pt")
    parser.add_argument("--negatives", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_labeled_training_data(data):
    """Muat format training data baru yang sudah memiliki label eksplisit."""
    samples = []
    labels = []
    weights = []

    for row in data:
        inp = (row.get("input") or "").lower().strip()
        tgt = (row.get("target") or "").lower().strip()
        label = row.get("label")

        if not inp or not tgt or label is None:
            continue

        confidence = float(row.get("confidence", 1.0))
        samples.append((inp, tgt))
        labels.append(float(label))
        weights.append(confidence)

    if not samples:
        raise ValueError("training_data.json tidak memiliki pasangan berlabel yang valid.")

    x_tensor = torch.tensor(
        [encode_word(inp) + encode_word(tgt) for inp, tgt in samples],
        dtype=torch.float32,
    )
    y_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    w_tensor = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)
    positive_count = sum(1 for label in labels if label == 1.0)
    return x_tensor, y_tensor, w_tensor, len(samples), positive_count


def load_training_data(path, negatives, seed):
    """Muat training data JSON lalu ubah ke tensor pasangan dan label."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        raise ValueError("training_data.json kosong. Jalankan test_dataset.py dulu.")

    if isinstance(data, list) and data and "label" in data[0]:
        return load_labeled_training_data(data)

    inputs = [(row.get("input") or "").lower().strip() for row in data]
    targets = [(row.get("target") or "").lower().strip() for row in data]
    pairs = [(inp, tgt) for inp, tgt in zip(inputs, targets) if inp and tgt]
    unique_targets = sorted({tgt for _, tgt in pairs})

    if not pairs:
        raise ValueError("training_data.json tidak memiliki pasangan input-target yang valid.")

    rng = torch.Generator().manual_seed(seed)
    samples = []
    labels = []

    for inp, tgt in pairs:
        samples.append((inp, tgt))
        labels.append(1.0)

        for _ in range(negatives):
            neg_idx = torch.randint(0, len(unique_targets), (1,), generator=rng).item()
            neg_tgt = unique_targets[neg_idx]
            if neg_tgt == tgt:
                continue
            samples.append((inp, neg_tgt))
            labels.append(0.0)

    x_tensor = torch.tensor(
        [encode_word(inp) + encode_word(tgt) for inp, tgt in samples],
        dtype=torch.float32,
    )
    y_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    w_tensor = torch.ones_like(y_tensor)
    return x_tensor, y_tensor, w_tensor, len(samples), len(pairs)


def split_dataset(dataset, val_split, seed):
    """Bagi dataset menjadi train dan validation secara reproducible."""
    if not 0 <= val_split < 1:
        raise ValueError("--val-split harus berada di rentang 0 sampai kurang dari 1.")

    total_samples = len(dataset)

    if total_samples < 2 or val_split == 0:
        return dataset, None

    val_size = max(1, int(total_samples * val_split))
    train_size = total_samples - val_size

    if train_size == 0:
        train_size = total_samples - 1
        val_size = 1

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    return train_dataset, val_dataset


def evaluate(model, loader, loss_fn):
    """Hitung loss dan akurasi sederhana pada validation set."""
    model.eval()
    running_loss = 0.0
    total_samples = 0
    correct = 0

    with torch.no_grad():
        for batch_x, batch_y, batch_w in loader:
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            weighted_loss = (loss * batch_w).mean()
            probs = torch.sigmoid(pred)
            preds = (probs >= 0.5).float()
            correct += (preds == batch_y).sum().item()
            batch_size = batch_x.size(0)
            running_loss += weighted_loss.item() * batch_size
            total_samples += batch_size

    if total_samples == 0:
        return None, None

    return running_loss / total_samples, correct / total_samples


def main():
    """Jalankan training loop similarity dan simpan artefak model."""
    args = parse_args()
    torch.manual_seed(args.seed)

    x_tensor, y_tensor, w_tensor, total_samples, total_positives = load_training_data(
        args.data_path,
        args.negatives,
        args.seed,
    )

    dataset = TensorDataset(x_tensor, y_tensor, w_tensor)
    train_dataset, val_dataset = split_dataset(dataset, args.val_split, args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = None

    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = nn.Sequential(
        nn.Linear(20, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )

    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_loss_history = []
    val_loss_history = []
    val_acc_history = []
    train_samples = len(train_dataset)
    val_samples = len(val_dataset) if val_dataset is not None else 0

    print(
        "Start training | "
        f"total_pairs={total_samples}, positives={total_positives}, train_samples={train_samples}, "
        f"val_samples={val_samples}, batch_size={args.batch_size}, epochs={args.epochs}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for batch_x, batch_y, batch_w in train_loader:
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            weighted_loss = (loss * batch_w).mean()

            optimizer.zero_grad()
            weighted_loss.backward()
            optimizer.step()

            running_loss += weighted_loss.item() * batch_x.size(0)

        train_loss = running_loss / train_samples
        val_loss, val_acc = evaluate(model, val_loader, loss_fn) if val_loader is not None else (None, None)

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            if val_loss is None:
                print(f"Epoch {epoch:03d}/{args.epochs} | train_loss={train_loss:.6f} | val_loss=N/A")
            else:
                print(
                    f"Epoch {epoch:03d}/{args.epochs} | "
                    f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
                    f"val_acc={val_acc:.4f}"
                )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": vars(args),
            "final_train_loss": train_loss_history[-1],
            "final_val_loss": val_loss_history[-1],
            "final_val_acc": val_acc_history[-1],
        },
        args.model_path,
    )

    with open("loss_history.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "train_loss": train_loss_history,
                "val_loss": val_loss_history,
                "val_acc": val_acc_history,
                "summary": {
                    "total_pairs": total_samples,
                    "train_samples": train_samples,
                    "val_samples": val_samples,
                    "final_train_loss": train_loss_history[-1],
                    "final_val_loss": val_loss_history[-1],
                    "final_val_acc": val_acc_history[-1],
                },
            },
            f,
            indent=2,
        )

    print(f"Checkpoint saved: {args.model_path}")
    print("Loss history saved: loss_history.json")
    if val_loss_history[-1] is None:
        print(f"Final train loss: {train_loss_history[-1]:.6f}")
        print("Final val loss: N/A")
    else:
        print(f"Final train loss: {train_loss_history[-1]:.6f}")
        print(f"Final val loss: {val_loss_history[-1]:.6f}")
        print(f"Final val acc: {val_acc_history[-1]:.4f}")


if __name__ == "__main__":
    main()
