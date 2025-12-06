"""
Training script for the ECG2BP LSTM on PulseDB subsets.

Design goals: keep the CLI simple, show clear training/validation metrics, and
store checkpoints so experiments are reproducible.
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from bpnet.data import PulseDBSequenceDataset
from bpnet.models.lstm_bp import ECG2BP, ECG2BPConfig


def parse_args() -> argparse.Namespace:
    """Collect command-line arguments for the trainer."""
    parser = argparse.ArgumentParser(description="Train ECG2BP on PulseDB subsets.")
    parser.add_argument("--train_mat", required=True, help="Path to training subset .mat file")
    parser.add_argument("--val_mat", help="Optional validation subset .mat file")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_dir", default="runs/lstm")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--resume", help="Checkpoint path to resume from")
    parser.add_argument("--skip_input_norm", action="store_true", help="Disable per-segment ECG normalization")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    # Model settings
    parser.add_argument("--conv_channels", type=int, default=32)
    parser.add_argument("--conv_kernel", type=int, default=7)
    parser.add_argument("--conv_layers", type=int, default=2)
    parser.add_argument("--lstm_hidden_size", type=int, default=128)
    parser.add_argument("--lstm_layers", type=int, default=2)
    parser.add_argument("--lstm_dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", action="store_true")
    return parser.parse_args()


def create_dataloader(
    mat_path: str,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    normalize_input: bool,
) -> DataLoader:
    """Wrap the dataset into a DataLoader with the desired batch size."""
    dataset = PulseDBSequenceDataset(mat_path=mat_path, normalize_input=normalize_input)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    max_grad_norm: float,
) -> float:
    """Train the model for a single epoch and return average loss."""
    model.train()
    total_loss = 0.0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        preds = model(inputs)
        loss = criterion(preds, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
    return total_loss / len(loader.dataset)


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate on validation data and return loss, MAE, and correlation."""
    model.eval()
    total_loss = 0.0
    preds_list = []
    targets_list = []
    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs.to(device))
            loss = criterion(outputs, targets.to(device))
            total_loss += loss.item() * inputs.size(0)
            preds_list.append(outputs.cpu().numpy())
            targets_list.append(targets.numpy())
    preds = np.concatenate(preds_list, axis=0).reshape(-1)
    targets = np.concatenate(targets_list, axis=0).reshape(-1)
    corr = float(np.corrcoef(preds, targets)[0, 1]) if np.std(preds) > 0 and np.std(targets) > 0 else math.nan
    mae = float(np.mean(np.abs(preds - targets)))
    return {
        "loss": total_loss / len(loader.dataset),
        "mae": mae,
        "corr": corr,
    }


def prepare_model(args: argparse.Namespace, device: torch.device) -> ECG2BP:
    """Instantiate ECG2BP using CLI hyperparameters."""
    config = ECG2BPConfig(
        conv_channels=args.conv_channels,
        conv_kernel=args.conv_kernel,
        conv_layers=args.conv_layers,
        lstm_hidden_size=args.lstm_hidden_size,
        lstm_layers=args.lstm_layers,
        lstm_dropout=args.lstm_dropout,
        bidirectional=args.bidirectional,
    )
    return ECG2BP(config).to(device)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist and return its Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_checkpoint(state: dict, checkpoint_dir: Path, is_best: bool = False) -> None:
    """Save a checkpoint dict and optionally copy it as best.pt."""
    epoch = state.get("epoch", 0)
    ckpt_path = checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
    torch.save(state, ckpt_path)
    if is_best:
        torch.save(state, checkpoint_dir / "best.pt")


def main() -> None:
    """Entry point tying together data loading, training, validation, and logging."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = create_dataloader(
        args.train_mat,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        normalize_input=not args.skip_input_norm,
    )
    val_loader = (
        create_dataloader(
            args.val_mat,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            normalize_input=not args.skip_input_norm,
        )
        if args.val_mat
        else None
    )

    model = prepare_model(args, device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_epoch = 1
    best_val = math.inf

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val = checkpoint.get("best_val", best_val)

    log_dir = ensure_dir(Path(args.log_dir) / dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    writer = SummaryWriter(log_dir=str(log_dir))
    checkpoint_dir = ensure_dir(args.checkpoint_dir)

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, args.max_grad_norm
        )
        writer.add_scalar("Loss/train", train_loss, epoch)

        val_metrics: Optional[Dict[str, float]] = None
        improved = False
        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, criterion, device)
            writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
            writer.add_scalar("MAE/val", val_metrics["mae"], epoch)
            if not math.isnan(val_metrics["corr"]):
                writer.add_scalar("Corr/val", val_metrics["corr"], epoch)
            improved = val_metrics["loss"] < best_val
            best_val = min(best_val, val_metrics["loss"])

        save_checkpoint(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val": best_val,
                "train_loss": train_loss,
                "val_metrics": val_metrics,
            },
            checkpoint_dir,
            is_best=improved,
        )

    writer.close()


if __name__ == "__main__":
    main()
