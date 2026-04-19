import json

import torch
import torch.nn as nn

from services.encoder import encode_word


with open("training_data.json", encoding="utf-8") as f:
    data = json.load(f)

if not data:
    raise ValueError("training_data.json kosong. Jalankan test_dataset.py dulu.")

X = torch.tensor([encode_word(d["input"]) for d in data], dtype=torch.float32)
Y = torch.tensor([encode_word(d["target"]) for d in data], dtype=torch.float32)

model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 10),
)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(200):
    pred = model(X)
    loss = loss_fn(pred, Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")
