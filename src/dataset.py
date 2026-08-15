"""
src/dataset.py - PyTorch Dataset for KLA AI-Based Image Restoration

Handles loading of paired NoisyLR (128x128) and GT (256x256) images
from the KLA hackathon dataset.  Supports reading directly from ZIP
archives or from extracted directories.

Arrays are loaded lazily — one sample at a time inside __getitem__() —
so the full dataset is never held in RAM simultaneously.

Verified dataset properties (from PHASE 1 analysis):
- 3200 paired training samples
- NoisyLR shape: (128, 128), dtype: float32, values may exceed [0, 1]
- GT shape: (256, 256), dtype: float32, values in [0, 1]
- 400 test NoisyLR samples (no GT)
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Constants derived from verified dataset analysis
# ---------------------------------------------------------------------------
NOISY_LR_SHAPE: Tuple[int, int] = (128, 128)
GT_SHAPE: Tuple[int, int] = (256, 256)
EXPECTED_DTYPE = np.float32

TRAIN_SAMPLES: int = 3200
TEST_SAMPLES: int = 400
TRAIN_SPLIT: int = 2880
VAL_SPLIT: int = 320
SPLIT_SEED: int = 42


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_npy_from_zip(
    zip_handle: zipfile.ZipFile,
    member_name: str,
) -> np.ndarray:
    """Load a single .npy array from an already-open ZipFile handle."""
    with zip_handle.open(member_name) as f:
        data = np.load(io.BytesIO(f.read()))
    return data


def _collect_npy_stems(names: List[str], prefix: str) -> Dict[str, str]:
    """Return {stem: full_member_name} for .npy files under *prefix* in a ZIP.

    Handles both ``prefix/ID.npy`` and ``train/prefix/ID.npy`` layouts.

    Filters out:
    - macOS metadata directories (``__MACOSX/...``)
    - macOS resource-fork files (filenames starting with ``._``)
    - Any non-.npy entries
    """
    mapping: Dict[str, str] = {}
    for name in names:
        # Normalise path separators
        normed = name.replace("\\", "/")

        # Skip macOS metadata directories
        if "__MACOSX" in normed:
            continue

        parts = normed.split("/")
        filename = parts[-1]

        # Skip macOS resource-fork files (e.g. "._001804.npy")
        if filename.startswith("._"):
            continue

        # Accept  "prefix/ID.npy"  or  "subdir/prefix/ID.npy"
        if len(parts) >= 2 and parts[-2] == prefix and filename.endswith(".npy"):
            stem = Path(filename).stem
            mapping[stem] = name
    return mapping


def _collect_npy_stems_from_dir(directory: Path, subfolder: str) -> Dict[str, str]:
    """Return {stem: full_path_str} for .npy files in directory/subfolder/."""
    folder = directory / subfolder
    if not folder.is_dir():
        raise FileNotFoundError(
            f"Expected subfolder '{subfolder}' not found in {directory}"
        )
    mapping: Dict[str, str] = {}
    for entry in sorted(folder.iterdir()):
        if entry.suffix == ".npy" and entry.is_file():
            mapping[entry.stem] = str(entry)
    return mapping


def _validate_array(
    arr: np.ndarray,
    expected_shape: Tuple[int, int],
    label: str,
    sample_id: str,
) -> None:
    """Raise informative errors if an array does not match expectations."""
    if arr.shape != expected_shape:
        raise ValueError(
            f"{label} sample '{sample_id}' has shape {arr.shape}, "
            f"expected {expected_shape}"
        )
    if arr.dtype != EXPECTED_DTYPE:
        raise ValueError(
            f"{label} sample '{sample_id}' has dtype {arr.dtype}, "
            f"expected {EXPECTED_DTYPE}"
        )
    if np.isnan(arr).any() or np.isinf(arr).any():
        raise ValueError(
            f"{label} sample '{sample_id}' contains NaN or Inf values"
        )


# ---------------------------------------------------------------------------
# Scanning helpers  (scan for member names only — no data loaded)
# ---------------------------------------------------------------------------

def _scan_paired_zip(
    zip_path: Path,
) -> Tuple[Dict[str, str], Dict[str, str], List[str]]:
    """Scan a training ZIP and return member-name mappings without loading data.

    Returns
    -------
    noisy_members : dict[str, str]
        {stem: ZIP member name} for NoisyLR files.
    gt_members : dict[str, str]
        {stem: ZIP member name} for GT files.
    paired_ids : list[str]
        Sorted stems present in both NoisyLR and GT.
    """
    if not zip_path.is_file():
        raise FileNotFoundError(f"ZIP archive not found: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        all_names = zf.namelist()

    noisy_members = _collect_npy_stems(all_names, "NoisyLR")
    gt_members = _collect_npy_stems(all_names, "GT")

    if not noisy_members:
        raise FileNotFoundError(
            f"No NoisyLR .npy files found in {zip_path}. "
            f"Expected paths like 'NoisyLR/<ID>.npy' or 'train/NoisyLR/<ID>.npy'."
        )
    if not gt_members:
        raise FileNotFoundError(
            f"No GT .npy files found in {zip_path}. "
            f"Expected paths like 'GT/<ID>.npy' or 'train/GT/<ID>.npy'."
        )

    paired_ids = sorted(set(noisy_members.keys()) & set(gt_members.keys()))
    if not paired_ids:
        raise ValueError(
            "No matching NoisyLR/GT pairs found. "
            f"NoisyLR stems (first 5): {sorted(noisy_members.keys())[:5]}, "
            f"GT stems (first 5): {sorted(gt_members.keys())[:5]}"
        )

    return noisy_members, gt_members, paired_ids


def _scan_test_zip(
    zip_path: Path,
) -> Tuple[Dict[str, str], List[str]]:
    """Scan a test ZIP and return member-name mappings without loading data.

    Returns
    -------
    noisy_members : dict[str, str]
        {stem: ZIP member name} for NoisyLR test files.
    test_ids : list[str]
        Sorted stems.
    """
    if not zip_path.is_file():
        raise FileNotFoundError(f"Test ZIP archive not found: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        all_names = zf.namelist()

    noisy_members = _collect_npy_stems(all_names, "NoisyLR")
    if not noisy_members:
        raise FileNotFoundError(
            f"No NoisyLR .npy files found in {zip_path}."
        )

    test_ids = sorted(noisy_members.keys())
    return noisy_members, test_ids


# ---------------------------------------------------------------------------
# Training / Validation Dataset  (lazy loading)
# ---------------------------------------------------------------------------

class KLARestorationDataset(Dataset):
    """PyTorch Dataset for paired NoisyLR → GT image restoration.

    Each ``__getitem__`` call returns::

        {
            "noisy":     torch.float32  [1, 128, 128],
            "gt":        torch.float32  [1, 256, 256],
            "sample_id": str,
        }

    Data is loaded **lazily** — one sample per ``__getitem__`` call.
    When backed by a ZIP archive the handle is opened once per
    OS-level process (safe for ``DataLoader`` with ``num_workers > 0``
    because each worker is a separate process and gets its own handle).

    NoisyLR values are preserved exactly as stored (may be outside [0, 1]).
    GT values are expected to lie within [0, 1] and are not modified.

    Parameters
    ----------
    sample_ids : list[str]
        Ordered list of sample filename stems (e.g. ["00001", "00002"]).
    zip_path : Path or None
        Path to the ZIP archive.  Mutually exclusive with *data_dir*.
    data_dir : Path or None
        Path to an extracted directory.  Mutually exclusive with *zip_path*.
    noisy_members : dict[str, str]
        Mapping from stem → ZIP member name **or** filesystem path for
        each NoisyLR file.
    gt_members : dict[str, str]
        Mapping from stem → ZIP member name **or** filesystem path for
        each GT file.
    validate : bool
        If True, validate array shape, dtype, NaN/Inf on every access.
    """

    def __init__(
        self,
        sample_ids: List[str],
        noisy_members: Dict[str, str],
        gt_members: Dict[str, str],
        zip_path: Optional[Path] = None,
        data_dir: Optional[Path] = None,
        validate: bool = True,
    ) -> None:
        super().__init__()

        if zip_path is None and data_dir is None:
            raise ValueError("Exactly one of zip_path or data_dir must be provided.")
        if zip_path is not None and data_dir is not None:
            raise ValueError("Provide zip_path or data_dir, not both.")

        self.sample_ids = sample_ids
        self._noisy_members = noisy_members
        self._gt_members = gt_members
        self._zip_path = zip_path
        self._data_dir = data_dir
        self._validate = validate

        # Will be lazily opened once per process (see _get_zip_handle)
        self._zip_handle: Optional[zipfile.ZipFile] = None

        # Verify that every requested ID has a mapping
        missing_noisy = [s for s in sample_ids if s not in self._noisy_members]
        missing_gt = [s for s in sample_ids if s not in self._gt_members]
        if missing_noisy:
            raise ValueError(
                f"{len(missing_noisy)} sample IDs missing NoisyLR entry "
                f"(first 5: {missing_noisy[:5]})"
            )
        if missing_gt:
            raise ValueError(
                f"{len(missing_gt)} sample IDs missing GT entry "
                f"(first 5: {missing_gt[:5]})"
            )

    # -- ZIP handle lifecycle -----------------------------------------------

    def _get_zip_handle(self) -> zipfile.ZipFile:
        """Return a lazily-opened, per-process ZipFile handle."""
        if self._zip_handle is None:
            assert self._zip_path is not None
            self._zip_handle = zipfile.ZipFile(self._zip_path, "r")
        return self._zip_handle

    def __getstate__(self) -> Dict[str, Any]:
        """Exclude the open ZipFile handle when pickling (DataLoader workers)."""
        state = self.__dict__.copy()
        state["_zip_handle"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore state; the handle will be lazily re-opened in the worker."""
        self.__dict__.update(state)

    def __del__(self) -> None:
        """Close the ZipFile handle when the dataset is garbage-collected."""
        if self._zip_handle is not None:
            try:
                self._zip_handle.close()
            except Exception:
                pass

    # -- Core interface -----------------------------------------------------

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample_id = self.sample_ids[index]

        # Lazy load from ZIP or directory
        if self._zip_path is not None:
            zf = self._get_zip_handle()
            noisy_np = _load_npy_from_zip(zf, self._noisy_members[sample_id])
            gt_np = _load_npy_from_zip(zf, self._gt_members[sample_id])
        else:
            noisy_np = np.load(self._noisy_members[sample_id])
            gt_np = np.load(self._gt_members[sample_id])

        if self._validate:
            _validate_array(noisy_np, NOISY_LR_SHAPE, "NoisyLR", sample_id)
            _validate_array(gt_np, GT_SHAPE, "GT", sample_id)

        # Convert to float32 tensors with channel dimension: [1, H, W]
        # NoisyLR values are preserved exactly (may be outside [0, 1])
        # GT values are preserved in their original [0, 1] range
        noisy_tensor = torch.from_numpy(noisy_np).unsqueeze(0)  # [1, 128, 128]
        gt_tensor = torch.from_numpy(gt_np).unsqueeze(0)        # [1, 256, 256]

        return {
            "noisy": noisy_tensor,    # float32, [1, 128, 128]
            "gt": gt_tensor,          # float32, [1, 256, 256]
            "sample_id": sample_id,   # str
        }


# ---------------------------------------------------------------------------
# Test Dataset  (NoisyLR only, no GT — lazy loading)
# ---------------------------------------------------------------------------

class KLATestDataset(Dataset):
    """PyTorch Dataset for test-time inference (NoisyLR only, no GT).

    Each ``__getitem__`` call returns::

        {
            "noisy":     torch.float32  [1, 128, 128],
            "sample_id": str,
        }

    Same lazy-loading and ZIP-handle lifecycle as ``KLARestorationDataset``.

    Parameters
    ----------
    sample_ids : list[str]
        Ordered list of sample filename stems.
    zip_path : Path or None
        Path to the test ZIP archive.
    data_dir : Path or None
        Path to an extracted test directory with NoisyLR/ subfolder.
    noisy_members : dict[str, str]
        Mapping from stem → ZIP member name or filesystem path.
    validate : bool
        If True, validate array shape, dtype, NaN/Inf on access.
    """

    def __init__(
        self,
        sample_ids: List[str],
        noisy_members: Dict[str, str],
        zip_path: Optional[Path] = None,
        data_dir: Optional[Path] = None,
        validate: bool = True,
    ) -> None:
        super().__init__()

        if zip_path is None and data_dir is None:
            raise ValueError("Exactly one of zip_path or data_dir must be provided.")
        if zip_path is not None and data_dir is not None:
            raise ValueError("Provide zip_path or data_dir, not both.")

        self.sample_ids = sample_ids
        self._noisy_members = noisy_members
        self._zip_path = zip_path
        self._data_dir = data_dir
        self._validate = validate
        self._zip_handle: Optional[zipfile.ZipFile] = None

        missing = [s for s in sample_ids if s not in self._noisy_members]
        if missing:
            raise ValueError(
                f"{len(missing)} sample IDs missing NoisyLR entry "
                f"(first 5: {missing[:5]})"
            )

    # -- ZIP handle lifecycle -----------------------------------------------

    def _get_zip_handle(self) -> zipfile.ZipFile:
        if self._zip_handle is None:
            assert self._zip_path is not None
            self._zip_handle = zipfile.ZipFile(self._zip_path, "r")
        return self._zip_handle

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_zip_handle"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)

    def __del__(self) -> None:
        if self._zip_handle is not None:
            try:
                self._zip_handle.close()
            except Exception:
                pass

    # -- Core interface -----------------------------------------------------

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample_id = self.sample_ids[index]

        if self._zip_path is not None:
            zf = self._get_zip_handle()
            noisy_np = _load_npy_from_zip(zf, self._noisy_members[sample_id])
        else:
            noisy_np = np.load(self._noisy_members[sample_id])

        if self._validate:
            _validate_array(noisy_np, NOISY_LR_SHAPE, "NoisyLR", sample_id)

        noisy_tensor = torch.from_numpy(noisy_np).unsqueeze(0)  # [1, 128, 128]

        return {
            "noisy": noisy_tensor,    # float32, [1, 128, 128]
            "sample_id": sample_id,   # str
        }


# ---------------------------------------------------------------------------
# Public factory functions
# ---------------------------------------------------------------------------

def create_train_val_datasets(
    data_source: Union[str, Path],
    seed: int = SPLIT_SEED,
    train_ratio: float = 0.9,
    validate: bool = True,
) -> Tuple[KLARestorationDataset, KLARestorationDataset]:
    """Create reproducible training and validation datasets.

    Only the ZIP member listing (or directory listing) is performed up
    front.  Actual ``.npy`` data is loaded lazily inside ``__getitem__``.

    Parameters
    ----------
    data_source : str or Path
        Path to either:
        - A ZIP archive (e.g. ``train.zip``) containing NoisyLR/ and GT/ folders
        - An extracted directory with NoisyLR/ and GT/ sub-folders
    seed : int
        Random seed for the train/validation split (default: 42).
    train_ratio : float
        Fraction of data used for training (default: 0.9 → 2880 train, 320 val).
    validate : bool
        If True, validate array shapes and dtypes on access.

    Returns
    -------
    train_dataset : KLARestorationDataset
    val_dataset : KLARestorationDataset
    """
    source = Path(data_source)

    zip_path: Optional[Path] = None
    data_dir: Optional[Path] = None

    if source.is_file() and source.suffix.lower() == ".zip":
        # Scan ZIP for member names only (no data loaded)
        noisy_members, gt_members, all_ids = _scan_paired_zip(source)
        zip_path = source
    elif source.is_dir():
        # Scan directory for file paths only (no data loaded)
        noisy_members = _collect_npy_stems_from_dir(source, "NoisyLR")
        gt_members = _collect_npy_stems_from_dir(source, "GT")
        all_ids = sorted(set(noisy_members.keys()) & set(gt_members.keys()))
        if not all_ids:
            raise ValueError(
                f"No matching NoisyLR/GT pairs found in {source}."
            )
        data_dir = source
    else:
        raise FileNotFoundError(
            f"Data source not found or unsupported: {source}. "
            f"Provide a .zip file or an extracted directory."
        )

    print(f"[dataset] Found {len(all_ids)} paired samples in {source}")

    # Reproducible shuffle and split
    rng = np.random.RandomState(seed)
    shuffled_ids = np.array(all_ids)
    rng.shuffle(shuffled_ids)

    split_idx = int(len(shuffled_ids) * train_ratio)
    train_ids = sorted(shuffled_ids[:split_idx].tolist())
    val_ids = sorted(shuffled_ids[split_idx:].tolist())

    print(
        f"[dataset] Split (seed={seed}): "
        f"{len(train_ids)} train, {len(val_ids)} val"
    )

    train_dataset = KLARestorationDataset(
        sample_ids=train_ids,
        noisy_members=noisy_members,
        gt_members=gt_members,
        zip_path=zip_path,
        data_dir=data_dir,
        validate=validate,
    )
    val_dataset = KLARestorationDataset(
        sample_ids=val_ids,
        noisy_members=noisy_members,
        gt_members=gt_members,
        zip_path=zip_path,
        data_dir=data_dir,
        validate=validate,
    )

    return train_dataset, val_dataset


def create_test_dataset(
    data_source: Union[str, Path],
    validate: bool = True,
) -> KLATestDataset:
    """Create a test dataset for inference (NoisyLR only, no GT).

    Only the ZIP member listing (or directory listing) is performed up
    front.  Actual ``.npy`` data is loaded lazily inside ``__getitem__``.

    Parameters
    ----------
    data_source : str or Path
        Path to either:
        - A ZIP archive (e.g. ``Test_NoisyLR.zip``) containing NoisyLR/ folder
        - An extracted directory with a NoisyLR/ sub-folder
    validate : bool
        If True, validate array shapes and dtypes on access.

    Returns
    -------
    test_dataset : KLATestDataset
    """
    source = Path(data_source)

    zip_path: Optional[Path] = None
    data_dir: Optional[Path] = None

    if source.is_file() and source.suffix.lower() == ".zip":
        noisy_members, test_ids = _scan_test_zip(source)
        zip_path = source
    elif source.is_dir():
        noisy_members = _collect_npy_stems_from_dir(source, "NoisyLR")
        test_ids = sorted(noisy_members.keys())
        if not test_ids:
            raise FileNotFoundError(
                f"No NoisyLR .npy files found in {source}/NoisyLR/"
            )
        data_dir = source
    else:
        raise FileNotFoundError(
            f"Test data source not found or unsupported: {source}. "
            f"Provide a .zip file or an extracted directory."
        )

    print(f"[dataset] Found {len(test_ids)} test samples in {source}")

    return KLATestDataset(
        sample_ids=test_ids,
        noisy_members=noisy_members,
        zip_path=zip_path,
        data_dir=data_dir,
        validate=validate,
    )
