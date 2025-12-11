"""
Lightweight ECG->BP model used before the tuned architecture.

This keeps the original configuration (Conv + 2-layer LSTM with 128 hidden
units) so we can compare “simple” versus “tuned” runs side-by-side.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class ECG2BPSimpleConfig:
    input_channels: int = 1
    conv_channels: int = 32
    conv_kernel: int = 7
    conv_layers: int = 2
    lstm_hidden_size: int = 128
    lstm_layers: int = 2
    lstm_dropout: float = 0.1
    bidirectional: bool = False
    output_size: int = 1


class ConvFeatureExtractor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        layers = []
        channels = in_channels
        for _ in range(num_layers):
            layers.append(
                nn.Conv1d(
                    channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            channels = out_channels
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class ECG2BPSimple(nn.Module):
    def __init__(self, config: Optional[ECG2BPSimpleConfig] = None) -> None:
        super().__init__()
        self.config = config or ECG2BPSimpleConfig()
        self.feature_extractor = ConvFeatureExtractor(
            in_channels=self.config.input_channels,
            out_channels=self.config.conv_channels,
            kernel_size=self.config.conv_kernel,
            num_layers=self.config.conv_layers,
        )
        self.recurrent = nn.LSTM(
            input_size=self.config.conv_channels,
            hidden_size=self.config.lstm_hidden_size,
            num_layers=self.config.lstm_layers,
            dropout=(
                self.config.lstm_dropout if self.config.lstm_layers > 1 else 0.0
            ),
            bidirectional=self.config.bidirectional,
            batch_first=True,
        )
        lstm_out = self.config.lstm_hidden_size * (
            2 if self.config.bidirectional else 1
        )
        self.head = nn.Linear(lstm_out, self.config.output_size)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        outputs, _ = self.recurrent(features)
        return self.head(outputs)


__all__ = ["ECG2BPSimple", "ECG2BPSimpleConfig"]
