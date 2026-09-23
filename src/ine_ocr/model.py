"""Compact CRNN recognizer portable from Apple MPS to ONNX CPU."""

from __future__ import annotations

import torch
from torch import nn

from .alphabet import NUM_CLASSES


class ConvBlock(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int, stride) -> None:
        super().__init__(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=stride,
                      padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=output_channels),
            nn.SiLU(inplace=True),
        )


class MachineLineRecognizer(nn.Module):
    def __init__(self, hidden_size: int = 192, recurrent_layers: int = 2,
                 dropout: float = 0.15, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.recurrent_layers = recurrent_layers
        self.dropout = dropout
        self.num_classes = num_classes
        self.encoder = nn.Sequential(
            ConvBlock(1, 64, (2, 2)),
            ConvBlock(64, 128, (2, 2)),
            ConvBlock(128, 192, (2, 1)),
            ConvBlock(192, 256, (2, 1)),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(0, 1), bias=False),
            nn.GroupNorm(num_groups=8, num_channels=256),
            nn.SiLU(inplace=True),
        )
        self.sequence = nn.GRU(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=recurrent_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if recurrent_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)
        features = features.mean(dim=2).transpose(1, 2)
        sequence, _ = self.sequence(features)
        return self.classifier(sequence)

    def architecture(self) -> dict:
        return {
            "hidden_size": self.hidden_size,
            "recurrent_layers": self.recurrent_layers,
            "dropout": self.dropout,
            "num_classes": self.num_classes,
        }
