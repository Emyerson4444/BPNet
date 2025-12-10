"""
Command-line trainer for BPNet's ECG->BP model.

Adds richer logging/metrics so each training run captures more insight.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from bpnet.data import PulseDBSequenceDataset
from bpnet.models.lstm_bp import ECG2BP, ECG2BPConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ECG2BP on PulseDB.")
    parser.add_argument("--train_mat", required=True, help="Path to training subset .mat")
    parser.add_argument("--val_mat", help="Optional validation subset .mat")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_dir", default="runs/lstm")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--resume", help="Checkpoint path to resume from")
    parser.add_argument("--skip_input_norm", action="store_true", help="Disable per-segment ECG z-scoring")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--sample_dump_limit", type=int, default=5, help="Segments to dump for qualitative review each epoch")
    parser.add_argument("--train_fraction", type=float, default=1.0, help="Fraction of training data to use (0-1]")
    parser.add_argument("--val_fraction", type=float, default=1.0, help="Fraction of validation data to use (0-1]")
    # Model hyperparameters
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
    fraction: float = 1.0,
) -> DataLoader:
    dataset = PulseDBSequenceDataset(mat_path=mat_path, normalize_input=normalize_input)
    if fraction < 1.0:
        fraction = max(0.0, min(1.0, fraction))
        subset_size = max(1, int(len(dataset) * fraction))
        indices = np.random.permutation(len(dataset))[:subset_size]
        dataset = Subset(dataset, indices.tolist())
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
    collect_stats: bool = False,
    epoch: int = 0,
) -> Tuple[float, Optional[Dict[str, float]], Optional[np.ndarray], Optional[np.ndarray]]:
    model.train()
    total_loss = 0.0
    preds_history = [] if collect_stats else None
    targets_history = [] if collect_stats else None
    progress_iter = tqdm(loader, desc=f"Train Epoch {epoch}", leave=False, ncols=80)
    for batch_idx, (inputs, targets) in enumerate(progress_iter):
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        preds = model(inputs)
        loss = criterion(preds, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        if collect_stats:
            preds_history.append(preds.detach().cpu().numpy())
            targets_history.append(targets.detach().cpu().numpy())
        else:
            progress_iter.set_postfix({"loss": f"{loss.item():.4f}"})

    metrics = None
    all_targets = None
    all_preds = None
    if collect_stats and preds_history:
        all_preds = np.concatenate(preds_history, axis=0)
        all_targets = np.concatenate(targets_history, axis=0)
        metrics = compute_metrics(all_targets, all_preds)
        metrics["loss"] = total_loss / len(loader.dataset)
    return total_loss / len(loader.dataset), metrics, all_targets, all_preds


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> Dict[str, float]:
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
    preds = np.concatenate(preds_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)
    metrics = compute_metrics(targets, preds)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def compute_metrics(targets: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    preds_flat = preds.reshape(-1)
    targets_flat = targets.reshape(-1)
    mae = float(np.mean(np.abs(preds_flat - targets_flat)))
    rmse = float(np.sqrt(np.mean((preds_flat - targets_flat) ** 2)))
    corr = float(np.corrcoef(preds_flat, targets_flat)[0, 1]) if np.std(preds_flat) > 0 and np.std(targets_flat) > 0 else math.nan
    return {"mae": mae, "rmse": rmse, "corr": corr}


def prepare_model(args: argparse.Namespace, device: torch.device) -> ECG2BP:
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
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_checkpoint(state: dict, checkpoint_dir: Path, is_best: bool = False) -> None:
    epoch = state.get("epoch", 0)
    torch.save(state, checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt")
    if is_best:
        torch.save(state, checkpoint_dir / "best.pt")


def save_config(log_dir: Path, args: argparse.Namespace) -> None:
    with (log_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)


def save_sample_predictions(
    epoch: int,
    targets: Optional[np.ndarray],
    preds: Optional[np.ndarray],
    log_dir: Path,
    limit: int = 5,
) -> None:
    if targets is None or preds is None:
        return
    limit = min(limit, targets.shape[0])
    out_path = log_dir / f"epoch_{epoch:03d}_samples.npz"
    np.savez(out_path, target_bp=targets[:limit], pred_bp=preds[:limit])


def init_metrics_csv(log_dir: Path) -> Path:
    """Create a CSV file for human-readable metrics."""
    csv_path = log_dir / "metrics.csv"
    if not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epoch",
                    "train_loss",
                    "train_mae",
                    "train_rmse",
                    "train_corr",
                    "val_loss",
                    "val_mae",
                    "val_rmse",
                    "val_corr",
                ]
            )
    return csv_path


def append_metrics_row(
    csv_path: Path,
    epoch: int,
    train_loss: float,
    train_metrics: Optional[Dict[str, float]],
    val_metrics: Optional[Dict[str, float]],
) -> None:
    """Append a single epoch row to the CSV log."""

    def metric_value(metric_dict: Optional[Dict[str, float]], key: str) -> Optional[float]:
        if metric_dict is None:
            return None
        return metric_dict.get(key)

    row = [
        epoch,
        train_loss,
        metric_value(train_metrics, "mae"),
        metric_value(train_metrics, "rmse"),
        metric_value(train_metrics, "corr"),
        metric_value(val_metrics, "loss"),
        metric_value(val_metrics, "mae"),
        metric_value(val_metrics, "rmse"),
        metric_value(val_metrics, "corr"),
    ]
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = create_dataloader(
        args.train_mat,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        normalize_input=not args.skip_input_norm,
        fraction=args.train_fraction,
    )
    val_loader = (
        create_dataloader(
            args.val_mat,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            normalize_input=not args.skip_input_norm,
            fraction=args.val_fraction,
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
    save_config(log_dir, args)
    csv_path = init_metrics_csv(log_dir)
    checkpoint_dir = ensure_dir(args.checkpoint_dir)
    print(f"Training samples: {len(train_loader.dataset)} (fraction={args.train_fraction})")
    if val_loader is not None:
        print(f"Validation samples: {len(val_loader.dataset)} (fraction={args.val_fraction})")

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_metrics, train_targets, train_preds = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            args.max_grad_norm,
            collect_stats=True,
            epoch=epoch,
        )
        writer.add_scalar("Loss/train", train_loss, epoch)
        if train_metrics:
            writer.add_scalar("MAE/train", train_metrics["mae"], epoch)
            writer.add_scalar("RMSE/train", train_metrics["rmse"], epoch)
            if not math.isnan(train_metrics["corr"]):
                writer.add_scalar("Corr/train", train_metrics["corr"], epoch)

        val_metrics: Optional[Dict[str, float]] = None
        improved = False
        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, criterion, device)
            writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
            writer.add_scalar("MAE/val", val_metrics["mae"], epoch)
            writer.add_scalar("RMSE/val", val_metrics["rmse"], epoch)
            if not math.isnan(val_metrics["corr"]):
                writer.add_scalar("Corr/val", val_metrics["corr"], epoch)
            improved = val_metrics["loss"] < best_val
            best_val = min(best_val, val_metrics["loss"])

        save_sample_predictions(epoch, train_targets, train_preds, log_dir, limit=args.sample_dump_limit)

        append_metrics_row(csv_path, epoch, train_loss, train_metrics, val_metrics)
        if val_metrics:
            print(
                f"Epoch {epoch:03d} | train_loss={train_loss:.4f} "
                f"| val_loss={val_metrics['loss']:.4f} | val_corr={val_metrics['corr']:.3f}"
            )
        else:
            print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f}")

        save_checkpoint(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val": best_val,
                "train_loss": train_loss,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "config": asdict(model.config),
            },
            checkpoint_dir,
            is_best=improved,
        )

    writer.close()


if __name__ == "__main__":
    main()
