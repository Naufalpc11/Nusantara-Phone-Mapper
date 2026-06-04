import argparse
import json

import torch
import torch.nn as nn

from services.encoder import encode_word


def parse_args():
    parser = argparse.ArgumentParser(description="Predict Indo-Sunda similarity")
    parser.add_argument("--model-path", type=str, default="model_similarity.pt")
    parser.add_argument("--input-id", type=str, required=True)
    parser.add_argument("--input-su", type=str, required=True)
    return parser.parse_args()


def build_model():
    return nn.Sequential(
        nn.Linear(20, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )


def main():
    args = parse_args()

    model = build_model()
    checkpoint = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    word_id = args.input_id.lower().strip()
    word_su = args.input_su.lower().strip()

    feature = encode_word(word_id) + encode_word(word_su)
    x_tensor = torch.tensor([feature], dtype=torch.float32)

    with torch.no_grad():
        logit = model(x_tensor).squeeze(1)
        prob = torch.sigmoid(logit).item()

    result = {
        "input_id": word_id,
        "input_su": word_su,
        "skor": round(prob * 100, 2),
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
