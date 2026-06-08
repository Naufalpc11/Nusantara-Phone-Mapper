import torch
import torch.nn as nn


def downsample_lengths(lengths):
    lengths = (lengths + 1) // 2
    lengths = (lengths + 1) // 2
    return lengths


class AudioCTCModel(nn.Module):
    def __init__(
        self,
        vocabulary_size,
        n_mels=80,
        conv_channels=64,
        hidden_size=192,
        num_layers=2,
        dropout=0.2,
    ):
        super().__init__()
        self.config = {
            "vocabulary_size": vocabulary_size,
            "n_mels": n_mels,
            "conv_channels": conv_channels,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
        }

        self.convolution = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, conv_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(),
        )

        frequency_size = (n_mels + 1) // 2
        frequency_size = (frequency_size + 1) // 2
        self.projection = nn.Linear(conv_channels * frequency_size, hidden_size)
        self.recurrent = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, vocabulary_size)

    def forward(self, features, feature_lengths):
        # Input: [batch, time, mel]
        x = features.transpose(1, 2).unsqueeze(1)
        x = self.convolution(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(x.size(0), x.size(1), -1)
        x = self.projection(x)
        x, _ = self.recurrent(x)
        logits = self.classifier(self.dropout(x))
        return logits, downsample_lengths(feature_lengths)
