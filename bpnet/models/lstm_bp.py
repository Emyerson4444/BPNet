"""
Core neural network for BPNet.

The model takes in a sequence of ECG samples and predicts a matching sequence
of blood-pressure values.  We keep the architecture intentionally small and
documented so it is easy to tweak, debug, and explain to others.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class ECG2BPConfig:
    """Default hyperparameters"""

    input_channels: int = 1  # Single-channel ECG by default.
    conv_channels: int = 32  # Number of filters in each conv layer.
    conv_kernel: int = 7  # Wide enough to cover multiple ECG samples per beat.
    conv_layers: int = 2  # Stack a couple of layers for richer features.
    lstm_hidden_size: int = 256  # Larger hidden state captures more context.
    lstm_layers: int = 3  # Deeper LSTM to model longer temporal patterns.
    lstm_dropout: float = 0.2  # Heavier dropout for the deeper stack.
    bidirectional: bool = False  # Keep False for causal predictions unless experimenting.
    output_size: int = 1  # Predict one blood-pressure value per timestep.


class ConvFeatureExtractor(nn.Module):
    """Lightweight 1D CNN that learns local ECG morphology."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2  # Same padding keeps sequence length unchanged.
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
            layers.append(nn.BatchNorm1d(out_channels))  # Stabilizes training.
            layers.append(nn.ReLU(inplace=True))  # Non-linear activation.
            channels = out_channels
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert (batch, seq_len, channels) -> (batch, seq_len, conv_channels).

        PyTorch Conv1d expects channel dimension in the middle, hence the transpose.
        """
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class ECG2BP(nn.Module):
    """End-to-end ECG -> BP regressor."""

    def __init__(self, config: Optional[ECG2BPConfig] = None) -> None:
        super().__init__()
        self.config = config or ECG2BPConfig()
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
                self.config.lstm_dropout
                if self.config.lstm_layers > 1
                else 0.0
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
        """Use standard initializations that work well with ReLU/LSTM layers."""
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
        """
        Args:
            x: Tensor shaped (batch, seq_len, channels) containing ECG samples.

        Returns:
            Tensor shaped (batch, seq_len, 1) with predicted blood pressure.
        """
        features = self.feature_extractor(x)
        outputs, _ = self.recurrent(features)
        return self.head(outputs)


__all__ = ["ECG2BP", "ECG2BPConfig"]
