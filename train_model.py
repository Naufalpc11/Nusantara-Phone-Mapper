import argparse
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from services.encoder import encode_word


# Konfigurasi hyperparameter via CLI.
def parse_args():
    """Baca konfigurasi training dari command line argument."""
    parser = argparse.ArgumentParser(description="Train baseline mapper model")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--data-path", type=str, default="training_data.json")
    parser.add_argument("--model-path", type=str, default="model_baseline.pt")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_training_data(path):
    """Muat training data JSON lalu ubah ke tensor input-target."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        raise ValueError("training_data.json kosong. Jalankan test_dataset.py dulu.")

    x_tensor = torch.tensor([encode_word(d["input"]) for d in data], dtype=torch.float32)
    y_tensor = torch.tensor([encode_word(d["target"]) for d in data], dtype=torch.float32)
    return x_tensor, y_tensor, len(data)


def main():
    """Jalankan training loop baseline dan simpan artefak model."""
    args = parse_args()
    torch.manual_seed(args.seed)

    x_tensor, y_tensor, total_samples = load_training_data(args.data_path)

    dataset = TensorDataset(x_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 10),
    )

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_history = []

    print(f"Start training | samples={total_samples}, batch_size={args.batch_size}, epochs={args.epochs}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        # Mini-batch update parameter model.
        for batch_x, batch_y in loader:
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)

        epoch_loss = running_loss / total_samples
        loss_history.append(epoch_loss)

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:03d}/{args.epochs} | loss={epoch_loss:.6f}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": vars(args),
            "final_loss": loss_history[-1],
        },
        args.model_path,
    )

    with open("loss_history.json", "w", encoding="utf-8") as f:
        json.dump(loss_history, f, indent=2)

    print(f"Checkpoint saved: {args.model_path}")
    print("Loss history saved: loss_history.json")


if __name__ == "__main__":
    main()
