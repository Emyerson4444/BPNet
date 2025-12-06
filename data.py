"""
Friendly data helpers for PulseDB.

These helpers hide the MATLAB `.mat` quirks behind a simple PyTorch-friendly
API so the rest of the codebase just thinks in numpy arrays and tensors.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from mat73 import loadmat
except ImportError as exc:  # pragma: no cover - library check
    raise ImportError(
        "The mat73 package is required to read PulseDB .mat files. "
        "Install it via `pip install mat73`."
    ) from exc


def _to_numpy(data) -> np.ndarray:
    """Converts MATLAB cell arrays/struct fields into numpy arrays."""
    array = np.asarray(data)
    if array.dtype == object:
        array = array.astype(str)
    return array


def load_subset(mat_path: str | Path) -> dict:
    """
    Loads a PulseDB subset `.mat` file produced by Generate_Subsets.m.

    Returns:
        dict containing numpy arrays for signals, SBP/DBP labels, and subjects.
    """

    mat_path = Path(mat_path)
    if not mat_path.exists():
        raise FileNotFoundError(f"Subset file not found: {mat_path}")
    data = loadmat(str(mat_path))
    if "Subset" not in data:
        raise KeyError(f"'Subset' key not in MATLAB file: {mat_path}")
    subset = data["Subset"]
    signals = np.asarray(subset["Signals"], dtype=np.float32)
    sbp = np.asarray(subset["SBP"], dtype=np.float32).reshape(-1, 1)
    dbp = np.asarray(subset["DBP"], dtype=np.float32).reshape(-1, 1)
    subjects = _to_numpy(subset["Subject"]).reshape(-1)
    return {
        "signals": signals,
        "sbp": sbp,
        "dbp": dbp,
        "subjects": subjects,
    }


@dataclass
class PulseDBSequenceDataset(Dataset):
    """
    PyTorch dataset yielding (ECG, BP) sequences from a subset file.

    Args:
        mat_path: Path to the `.mat` file.
        input_channel: Which channel to treat as ECG (0 = ECG).
        target_channel: Which channel to predict (2 = ABP).
        normalize_input: Whether to z-score each ECG segment individually.
    """

    mat_path: str
    input_channel: int = 0
    target_channel: int = 2
    normalize_input: bool = True

    def __post_init__(self) -> None:
        """Load everything once so __getitem__ stays fast."""
        subset = load_subset(self.mat_path)
        signals = subset["signals"]
        if signals.ndim != 3:
            raise ValueError(f"Expected signals with shape (N, C, T), got {signals.shape}")

        ecg = signals[:, self.input_channel, :]
        target = signals[:, self.target_channel, :]

        if self.normalize_input:
            mean = ecg.mean(axis=1, keepdims=True)
            std = ecg.std(axis=1, keepdims=True) + 1e-6
            ecg = (ecg - mean) / std

        self.inputs = torch.from_numpy(ecg[:, :, None]).float()
        self.targets = torch.from_numpy(target[:, :, None]).float()
        self._subjects: List[str] = subset["subjects"].tolist()

    def __len__(self) -> int:
        """Number of 10-second segments."""
        return self.inputs.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a single (input, target) pair as tensors."""
        return self.inputs[index], self.targets[index]

    def subject_ids(self) -> Sequence[str]:
        """Expose subject IDs for subject-wise splits."""
        return self._subjects
