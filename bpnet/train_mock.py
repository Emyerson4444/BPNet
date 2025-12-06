"""
Synthetic training harness for ECG2BP.

Creates mock ECG/BP sequences so we can verify the model compiles and a
training loop runs even before PulseDB subsets are available locally.
"""
from __future__ import annotations

import math
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader, Dataset

from bpnet.models.lstm_bp import ECG2BP, ECG2BPConfig


class MockECGDataset(Dataset):
    """Simple sine-wave generator for toy ECG/BP pairs."""

    def __init__(self, num_segments: int, seq_len: int, device: torch.device):
        super().__init__()
        self.seq_len = seq_len
        self.device = device
        self.inputs, self.targets = self._generate(num_segments)

    def _generate(self, num_segments: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Creates noisy sine waves and a smoothed BP version of them."""
        t = torch.linspace(0, 1, self.seq_len, device=self.device)
        inputs = []
        targets = []
        for _ in range(num_segments):
            freq = torch.rand(1, device=self.device) * 5 + 1
            phase = torch.rand(1, device=self.device) * 2 * math.pi
            ecg = torch.sin(2 * math.pi * freq * t + phase)
            ecg += 0.1 * torch.randn_like(ecg)
            bp = torch.nn.functional.avg_pool1d(
                ecg.view(1, 1, -1), kernel_size=5, padding=2, stride=1
            ).view(-1)
            bp = 120 + 15 * bp
            inputs.append(ecg.unsqueeze(-1))
            targets.append(bp.unsqueeze(-1))
        return torch.stack(inputs), torch.stack(targets)

    def __len__(self) -> int:
        """Number of fake segments available."""
        return self.inputs.size(0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one synthetic (ECG, BP) pair."""
        return self.inputs[idx], self.targets[idx]


def train_on_mock(
    epochs: int = 5,
    batch_size: int = 16,
    num_segments: int = 256,
    seq_len: int = 250,
) -> None:
    """Runs a tiny training loop to ensure everything is wired correctly."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MockECGDataset(num_segments, seq_len, device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    config = ECG2BPConfig(
        input_channels=1,
        conv_channels=16,
        conv_kernel=5,
        conv_layers=2,
        lstm_hidden_size=64,
        lstm_layers=2,
        lstm_dropout=0.1,
        output_size=1,
    )
    model = ECG2BP(config).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Training ECG2BP on mock dataset with config:", asdict(config))
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for inputs, targets in loader:
            # forward, loss, backward, optimizer step.
            optimizer.zero_grad()
            preds = model(inputs.to(device))
            loss = criterion(preds, targets.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
        print(f"Epoch {epoch}: train MSE = {total_loss/len(dataset):.4f}")


if __name__ == "__main__":
    train_on_mock()
