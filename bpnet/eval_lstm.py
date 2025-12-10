"""
Evaluation utility for ECG2BP checkpoints.

Loads a saved model, runs it on a dataset, and prints/saves readable metrics
so we can discuss results without opening TensorBoard.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from bpnet.data import PulseDBSequenceDataset
from bpnet.models.lstm_bp import ECG2BP, ECG2BPConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an ECG2BP checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt")
    parser.add_argument("--eval_mat", required=True, help="Path to .mat/.npz subset used for evaluation")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--skip_input_norm", action="store_true", help="Disable per-segment ECG normalization")
    parser.add_argument("--output_json", help="Optional path to save metrics as JSON")
    parser.add_argument("--samples_out", help="Optional npz file with target/pred samples")
    parser.add_argument("--sample_limit", type=int, default=5, help="How many samples to dump if saving npz output")
    # Optional overrides if checkpoint lacks config info.
    parser.add_argument("--conv_channels", type=int, default=None)
    parser.add_argument("--conv_kernel", type=int, default=None)
    parser.add_argument("--conv_layers", type=int, default=None)
    parser.add_argument("--lstm_hidden_size", type=int, default=None)
    parser.add_argument("--lstm_layers", type=int, default=None)
    parser.add_argument("--lstm_dropout", type=float, default=None)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--force_config_override", action="store_true", help="Force use of CLI hyperparameters even if checkpoint stores config")
    return parser.parse_args()


def create_loader(mat_path: str, batch_size: int, num_workers: int, normalize: bool) -> DataLoader:
    dataset = PulseDBSequenceDataset(mat_path=mat_path, normalize_input=normalize)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


def compute_metrics(targets: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    preds_flat = preds.reshape(-1)
    targets_flat = targets.reshape(-1)
    mae = float(np.mean(np.abs(preds_flat - targets_flat)))
    rmse = float(np.sqrt(np.mean((preds_flat - targets_flat) ** 2)))
    corr = float(np.corrcoef(preds_flat, targets_flat)[0, 1]) if np.std(preds_flat) > 0 and np.std(targets_flat) > 0 else math.nan
    mse = float(np.mean((preds_flat - targets_flat) ** 2))
    return {"mae": mae, "rmse": rmse, "corr": corr, "mse": mse}


def build_config(args: argparse.Namespace, checkpoint: Dict) -> ECG2BPConfig:
    config_dict = checkpoint.get("config")
    if config_dict and not args.force_config_override:
        config = ECG2BPConfig(**config_dict)
    else:
        defaults = ECG2BPConfig()
        bidirectional = defaults.bidirectional
        if args.bidirectional:
            bidirectional = True
        config = ECG2BPConfig(
            conv_channels=args.conv_channels if args.conv_channels is not None else defaults.conv_channels,
            conv_kernel=args.conv_kernel if args.conv_kernel is not None else defaults.conv_kernel,
            conv_layers=args.conv_layers if args.conv_layers is not None else defaults.conv_layers,
            lstm_hidden_size=args.lstm_hidden_size if args.lstm_hidden_size is not None else defaults.lstm_hidden_size,
            lstm_layers=args.lstm_layers if args.lstm_layers is not None else defaults.lstm_layers,
            lstm_dropout=args.lstm_dropout if args.lstm_dropout is not None else defaults.lstm_dropout,
            bidirectional=bidirectional,
        )
    return config


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[Dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for inputs, target in loader:
            outputs = model(inputs.to(device))
            preds.append(outputs.cpu().numpy())
            targets.append(target.numpy())
    preds_arr = np.concatenate(preds, axis=0)
    targets_arr = np.concatenate(targets, axis=0)
    metrics = compute_metrics(targets_arr, preds_arr)
    metrics["segments"] = targets_arr.shape[0]
    return metrics, targets_arr, preds_arr


def maybe_save_metrics(metrics: Dict[str, float], path: Optional[str]) -> None:
    if not path:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)


def maybe_save_samples(targets: np.ndarray, preds: np.ndarray, path: Optional[str], limit: int) -> None:
    if not path:
        return
    limit = min(limit, targets.shape[0])
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, target_bp=targets[:limit], pred_bp=preds[:limit])


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = build_config(args, checkpoint)
    model = ECG2BP(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    loader = create_loader(
        args.eval_mat,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        normalize=not args.skip_input_norm,
    )
    metrics, targets, preds = evaluate(model, loader, device)
    print(
        "Evaluation results:"
        f" segments={metrics['segments']} | mse={metrics['mse']:.4f}"
        f" | rmse={metrics['rmse']:.4f} | mae={metrics['mae']:.4f}"
        f" | corr={metrics['corr']:.4f}"
    )
    maybe_save_metrics(metrics, args.output_json)
    maybe_save_samples(targets, preds, args.samples_out, args.sample_limit)


if __name__ == "__main__":
    main()
