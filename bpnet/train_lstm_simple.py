"""
Original training script for the simple ECG->BP model.

This mirrors the earlier implementation before we added CSV logging,
schedulers, and extra knobs.  Useful for comparison experiments.
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import csv
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from bpnet.data import PulseDBSequenceDataset
from bpnet.models.lstm_bp_simple import ECG2BPSimple, ECG2BPSimpleConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the simple ECG2BP model.")
    parser.add_argument("--train_mat", required=True, help="Path to training subset (.mat or .npz)")
    parser.add_argument("--val_mat", help="Optional validation subset")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_dir", default="runs/lstm_simple")
    parser.add_argument("--checkpoint_dir", default="checkpoints/lstm_simple")
    parser.add_argument("--resume", help="Resume checkpoint path")
    parser.add_argument("--skip_input_norm", action="store_true")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--train_fraction", type=float, default=1.0)
    parser.add_argument("--val_fraction", type=float, default=1.0)
    return parser.parse_args()


def create_dataloader(
    mat_path: str,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    normalize_input: bool,
    fraction: float,
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
) -> float:
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
    return {"loss": total_loss / len(loader.dataset), "mae": mae, "corr": corr}


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_config(log_dir: Path, args: argparse.Namespace) -> None:
    config = {
        "train_mat": args.train_mat,
        "val_mat": args.val_mat,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "train_fraction": args.train_fraction,
        "val_fraction": args.val_fraction,
        "skip_input_norm": args.skip_input_norm,
        "conv_channels": 32,
        "conv_kernel": 7,
        "conv_layers": 2,
        "lstm_hidden_size": 128,
        "lstm_layers": 2,
        "lstm_dropout": 0.1,
        "log_dir": str(log_dir),
    }
    with (log_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def init_metrics_csv(log_dir: Path) -> Path:
    csv_path = log_dir / "metrics.csv"
    if not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "val_mae", "val_corr"])
    return csv_path


def append_metrics_row(
    csv_path: Path,
    epoch: int,
    train_loss: float,
    val_metrics: Optional[Dict[str, float]],
) -> None:
    val_loss = val_metrics.get("loss") if val_metrics else None
    val_mae = val_metrics.get("mae") if val_metrics else None
    val_corr = val_metrics.get("corr") if val_metrics else None
    row = [
        epoch,
        train_loss,
        val_loss,
        val_mae,
        val_corr,
    ]
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def save_sample_predictions(
    epoch: int,
    loader: Optional[DataLoader],
    model: torch.nn.Module,
    device: torch.device,
    log_dir: Path,
    limit: int = 5,
) -> None:
    if loader is None:
        return
    model.eval()
    preds = []
    targets = []
    collected = 0
    with torch.no_grad():
        for inputs, target in loader:
            inputs = inputs.to(device)
            output = model(inputs).cpu().numpy()
            preds.append(output)
            targets.append(target.numpy())
            collected += inputs.size(0)
            if collected >= limit:
                break
    if not preds:
        return
    preds_arr = np.concatenate(preds, axis=0)[:limit]
    targets_arr = np.concatenate(targets, axis=0)[:limit]
    out_path = log_dir / f"epoch_{epoch:03d}_samples.npz"
    np.savez(out_path, target_bp=targets_arr, pred_bp=preds_arr)
    model.train()


def save_checkpoint(state: dict, checkpoint_dir: Path, is_best: bool) -> None:
    epoch = state.get("epoch", 0)
    torch.save(state, checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt")
    if is_best:
        torch.save(state, checkpoint_dir / "best.pt")


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

    model = ECG2BPSimple(ECG2BPSimpleConfig()).to(device)
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
    metrics_csv = init_metrics_csv(log_dir)
    checkpoint_dir = ensure_dir(args.checkpoint_dir)
    print(f"Training samples: {len(train_loader.dataset)} (fraction={args.train_fraction})")
    if val_loader is not None:
        print(f"Validation samples: {len(val_loader.dataset)} (fraction={args.val_fraction})")

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, args.max_grad_norm)
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

        if val_metrics:
            print(
                f"Epoch {epoch:03d} | train_loss={train_loss:.4f} "
                f"| val_loss={val_metrics['loss']:.4f} | val_corr={val_metrics['corr']:.3f}"
            )
        else:
            print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f}")

        append_metrics_row(metrics_csv, epoch, train_loss, val_metrics)
        save_sample_predictions(epoch, val_loader, model, device, log_dir)

        save_checkpoint(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val": best_val,
            },
            checkpoint_dir,
            is_best=improved,
        )

    writer.close()


if __name__ == "__main__":
    main()
