"""
Dataset classes for FAseg training and fine-tuning.

Classes
-------
LimbArrayDataset
    Synthetic data loader with noise injection, TOF filtering, and
    domain randomisation (gap / speckle augmentation).
RealLimbDataset
    Real manually-segmented data loader for fine-tuning (float32,
    stride-6 downsampled, deterministic data/label pairing).
RealLimbDataset2
    Real manually-segmented data loader for fine-tuning (int16,
    pre-cropped, random source X per label Y).
"""

from __future__ import annotations

import os
import random
import re
from typing import Optional

import numpy as np
import torch
from scipy.ndimage import zoom
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# File-sorting helpers
# ---------------------------------------------------------------------------

def _extract_number(f: str) -> int:
    """Extract integer suffix from ``label.N`` style filenames."""
    return int(f.split(".")[-1])


def _extract_model_src(filename: str) -> tuple[int, int]:
    """Extract ``(model, src)`` from ``test{model}_src_{src}`` filenames."""
    m = re.match(r"test(\d+)_src_(\d+)", filename)
    if m:
        return int(m.group(1)), int(m.group(2))
    return (float("inf"), float("inf"))


def _extract_model_src2(filename: str) -> tuple[int, int]:
    """Extract ``(model, src)`` from ``dat{model}_src_{src}`` filenames."""
    m = re.match(r"dat(\d+)_src_(\d+)", filename)
    if m:
        return int(m.group(1)), int(m.group(2))
    return (float("inf"), float("inf"))


# ===================================================================
# LimbArrayDataset — synthetic pretraining
# ===================================================================

class LimbArrayDataset(Dataset):
    """Synthetic TOF ultrasound dataset with noise and augmentations.

    Each sample consists of a 2-D ``(num_receiver, net_nt)`` signal image
    and a binary mask of the same shape indicating the wavefront boundary.

    Parameters
    ----------
    data_dir : str
        Directory of ``.dat`` simulation files (float32).
    noise_dir : str
        Directory of ``.int16`` noise slice files.
    label_dir : str
        Directory of ``label.N`` binary label files.
    data_scale : float
        Multiplicative scale factor applied to the simulated signal.
    noise_scale : float
        Multiplicative scale factor applied to the noise.
    load_noise : bool
        If ``False``, noise is zeroed out (for ablation).
    noise_dt : float
        Original noise sampling interval (seconds).
    sim_dt : float | None
        Simulation sampling interval; defaults to *noise_dt*.
    sim_nt : int | None
        Simulation time length (samples); defaults to 7001.
    net_nt : int | None
        Time length input to the network; defaults to *sim_nt*.
    normalize_need : bool
        If ``True``, divide by the per-sample max absolute value.
    shift_enabled : bool
        If ``True``, circularly shift rows to place the active emitter
        at row 0.
    crop_enabled : bool
        If ``True``, apply ``crop_rows × crop_cols`` cropping.
    crop_rows : tuple[int, int]
        Row crop range ``(r0, r1)``.
    crop_cols : tuple[int, int]
        Column crop range ``(c0, c1)``.
    interval : int
        Emitter down-sampling interval; every *interval*-th emitter is used.
    augment_flip : bool
        If ``True``, double the dataset size via vertical flipping.
    further_scaling : float
        Optional additional clipping & rescaling after normalisation.
    TOF_filt_enabled : bool
        If ``True``, apply TOF Gaussian attenuation filter (requires
        ``shift_enabled=True``).
    filt_strength : float
        Attenuation factor at the TOF centre (centre → 1/filt_strength).
    filt_max_ratio : float
        Maximum fraction of the allowable filter window to use (randomised).
    gap_prob : float
        Probability of applying speckle noise and gap artifacts.
    gap_max_ratio : float
        Maximum gap attenuation ratio.
    gap_max_grid : tuple[int, int]
        ``(min, max)`` half-width (rows) of the gap band.
    filt_max_steps_path : str | None
        Path to ``filt_max_steps_vec.csv`` (calibration data for filter
        window bounds).  Required when ``TOF_filt_enabled=True``.

    Notes
    -----
    The constructor parameters mirror the original ``final_stage_data_reading.py``
    with one addition: **filt_max_steps_path** replaces the hardcoded absolute
    path in the legacy code.
    """

    def __init__(
        self,
        data_dir: str,
        noise_dir: str,
        label_dir: str,
        data_scale: float = 1.0,
        noise_scale: float = 1.0,
        load_noise: bool = True,
        noise_dt: float = 4e-8,
        sim_dt: Optional[float] = None,
        sim_nt: Optional[int] = None,
        net_nt: Optional[int] = None,
        normalize_need: bool = True,
        shift_enabled: bool = False,
        crop_enabled: bool = False,
        crop_rows: tuple[int, int] = (64, 448),
        crop_cols: tuple[int, int] = (300, 684),
        interval: int = 32,
        augment_flip: bool = True,
        further_scaling: float = 1.0,
        TOF_filt_enabled: bool = False,
        filt_strength: float = 32.0,
        filt_max_ratio: float = 0.5,
        gap_prob: float = 0.0,
        gap_max_ratio: float = 0.8,
        gap_max_grid: tuple[int, int] = (3, 25),
        filt_max_steps_path: Optional[str] = None,
    ):
        # ---- Scan & sort data files ----
        self.data_files = sorted(
            [os.path.join(data_dir, f) for f in os.listdir(data_dir)
             if f.startswith("dat")],
            key=lambda f: _extract_model_src2(os.path.basename(f)),
        )
        assert self.data_files, f"No data files found in {data_dir}"

        self.noise_files = sorted(
            [os.path.join(noise_dir, f) for f in os.listdir(noise_dir)
             if f.startswith("test")],
            key=lambda f: _extract_model_src(os.path.basename(f)),
        )
        if load_noise:
            assert self.noise_files, f"No noise files found in {noise_dir}"

        self.label_files = sorted(
            [os.path.join(label_dir, f) for f in os.listdir(label_dir)
             if f.startswith("label.")],
            key=_extract_number,
        )
        assert self.label_files, f"No label files found in {label_dir}"

        # ---- Parameters ----
        self.data_scale = data_scale
        self.noise_scale = noise_scale
        self.load_noise = load_noise
        self.noise_dt = noise_dt
        self.sim_dt = sim_dt if sim_dt is not None else noise_dt
        self.normalize_need = normalize_need
        self.shift_enabled = shift_enabled
        self.crop_enabled = crop_enabled
        self.crop_rows = crop_rows
        self.crop_cols = crop_cols
        self.interval = interval
        self.augment_flip = augment_flip
        self.further_scaling = further_scaling

        # Domain randomisation
        self.randomization_prob = gap_prob
        self.gap_max_ratio = gap_max_ratio
        self.gap_grid_range = gap_max_grid

        # Fixed geometry
        self.num_receiver = 512
        self.num_emitter = 256

        # TOF filter
        self.TOF_filt_enabled = TOF_filt_enabled
        self.filt_strength = filt_strength
        self.filt_max_ratio = filt_max_ratio
        if TOF_filt_enabled:
            if filt_max_steps_path is None:
                # Default: look for CSV next to this module
                filt_max_steps_path = os.path.join(
                    os.path.dirname(__file__), "filt_max_steps_vec.csv"
                )
            self.filt_max_tsteps = np.loadtxt(filt_max_steps_path, delimiter=",")
            assert self.filt_max_tsteps.shape[0] == self.num_receiver, (
                f"filt_max_tsteps size mismatch: expected {self.num_receiver}, "
                f"got {self.filt_max_tsteps.shape[0]}"
            )
        else:
            self.filt_max_tsteps = None

        # Time lengths
        self.num_time = sim_nt if sim_nt is not None else 7001
        self.net_nt = net_nt if net_nt is not None else self.num_time

        # Dataset size
        self.samples_per_file = self.num_emitter // self.interval
        self.base_length = len(self.data_files)
        self.length = self.base_length * (2 if self.augment_flip else 1)

        # Noise file pairing cache
        self.noise_map: dict[tuple[int, int], str] = {}

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def _load_sim(self, filepath: str) -> np.ndarray:
        """Load a single ``.dat`` file → ``(num_receiver, net_nt)``."""
        data = np.fromfile(filepath, dtype=np.float32)
        data = data.reshape(self.num_receiver, self.num_time)
        return np.array(data[:, : self.net_nt])

    def _load_noise(self, filepath: str) -> np.ndarray:
        """Load and interpolate a noise file to match ``sim_dt``."""
        if not self.load_noise:
            return np.zeros((self.num_receiver, self.net_nt), dtype=np.float32)

        arr = np.fromfile(filepath, dtype=np.int16)
        noise = arr.reshape((self.num_receiver, -1)).astype(np.float32)
        N = noise.shape[1]

        # How many network time steps can the noise cover?
        M = int((self.noise_dt * N) // self.sim_dt)
        M = min(M, self.net_nt)

        # Interpolate from noise_dt → sim_dt
        t_old = np.arange(N) * self.noise_dt
        t_new = np.arange(M) * self.sim_dt
        interp_part = np.vstack([
            np.interp(t_new, t_old, noise[ch, :])
            for ch in range(self.num_receiver)
        ])

        # Pad by repeating the tail of the interpolated segment
        pad = self.net_nt - M
        if pad > 0:
            assert pad <= M, (
                f"Noise interpolation length M={M} too small to pad={pad}"
            )
            pad_part = interp_part[:, -pad:]
            out = np.hstack([interp_part, pad_part])
        else:
            out = interp_part

        return out[:, : self.net_nt]

    def _load_label_column(
        self, filepath: str, emitter_idx: int
    ) -> np.ndarray:
        """Read boundary times for one emitter → ``(num_receiver,)`` in ms."""
        m = np.memmap(
            filepath, dtype=np.float32, mode="r",
            shape=(self.num_emitter, self.num_receiver),
        )
        return np.array(m[emitter_idx, :]) * 1e-3

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Determine flip
        do_flip = self.augment_flip and idx >= self.base_length
        idx0 = idx - self.base_length if do_flip else idx

        file_idx = idx0 // self.samples_per_file
        emitter_idx = (idx0 % self.samples_per_file) * self.interval

        # ---- Noise file (random pairing, cached) ----
        key = (file_idx, emitter_idx)
        if key not in self.noise_map:
            self.noise_map[key] = random.choice(self.noise_files)
        noise_file = self.noise_map[key]

        # ---- Load simulation & noise ----
        # NOTE: data_files is indexed by idx0 (one file per emitter position,
        # 16800 files total), while label_files is indexed by file_idx (one
        # file per model containing all 256 emitter boundaries, 2100 files).
        # The ratio is samples_per_file (8) = num_emitter (256) / interval (32).
        sim_vol = self._load_sim(self.data_files[idx0])
        img = sim_vol * self.data_scale
        noise2d = self._load_noise(noise_file) * self.noise_scale

        # ---- Label → mask ----
        boundary_times = self._load_label_column(
            self.label_files[file_idx], emitter_idx
        )
        boundary_idx = (boundary_times / self.sim_dt).astype(int)
        boundary_idx = np.clip(boundary_idx, 0, self.net_nt - 1)

        t = np.arange(self.net_nt)[None, :]               # (1, net_nt)
        b = boundary_idx[:, None]                          # (R, 1)
        mask = (t >= b).astype(np.float32)

        # ---- Emitter shift ----
        if self.shift_enabled:
            shift = emitter_idx * 2
            img = np.vstack([img[shift:], img[:shift]])
            mask = np.vstack([mask[shift:], mask[:shift]])

        # ---- TOF filter + domain randomisation ----
        if self.TOF_filt_enabled and self.shift_enabled:
            self._apply_augmentations(img, mask)

        # ---- Add noise ----
        noisy = img + noise2d

        # ---- Flip along receiver axis ----
        if do_flip:
            noisy = noisy[::-1, :].copy()
            mask = mask[::-1, :].copy()

        # ---- Crop ----
        if self.crop_enabled and self.shift_enabled:
            r0, r1 = self.crop_rows
            c0, c1 = self.crop_cols
            noisy = noisy[r0:r1, c0:c1]
            mask = mask[r0:r1, c0:c1]
        else:
            noisy = noisy[:, 100:]
            mask = mask[:, 100:]

        noisy = np.clip(noisy, -8200, 8200)
        mask = np.clip(mask, -8200, 8200)

        # ---- To tensor ----
        img_t = torch.from_numpy(noisy.astype(np.float32)).unsqueeze(0)
        mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)

        # ---- Normalise ----
        if self.normalize_need:
            max_val = img_t.abs().max()
            if max_val > 0:
                img_t = img_t / max_val
                if not np.isclose(self.further_scaling, np.float32(1.0)):
                    img_t = torch.clip(
                        img_t, -self.further_scaling, self.further_scaling
                    )
                    img_t = img_t / self.further_scaling

        return img_t, mask_t

    # ------------------------------------------------------------------
    # Augmentation (kept inline for exact behavioural fidelity)
    # ------------------------------------------------------------------

    def _apply_augmentations(self, img: np.ndarray, mask: np.ndarray) -> None:
        """TOF filter + optional speckle & gap artifacts (in-place)."""
        # Compute TOF positions
        center_list = mask.argmax(axis=1)
        max_direct_t_grid = (self.filt_max_tsteps / self.sim_dt).astype(int)
        max_filt_gridlens = np.maximum(max_direct_t_grid - center_list, 0)

        # Row range for filtering
        if self.crop_enabled:
            r0, r1 = self.crop_rows
            R = range(r0, r1)
        else:
            R = range(img.shape[0])

        win_ratio = random.uniform(0.5, self.filt_max_ratio)

        for r in R:
            center = center_list[r]
            filt_len = int(max_filt_gridlens[r] * win_ratio)
            if filt_len < 3:
                continue

            left = max(center - filt_len, 0)
            right = min(center + filt_len + 1, self.net_nt)
            win_length = right - left
            if win_length < 3:
                continue

            # Inverted Gaussian window
            mu = win_length // 2
            sigma = win_length * 0.2
            x = np.arange(win_length)
            window = 1.0 - np.exp(-0.5 * ((x - mu) / sigma) ** 2)
            scaling_value = 1.0 / self.filt_strength
            filt_curve = scaling_value + (1.0 - scaling_value) * window
            img[r, left:right] *= filt_curve

        # ---- Speckle noise ----
        if self.randomization_prob > 0:
            if random.random() < self.randomization_prob:
                speckle_width = 8
                speckle_sigma = 5 / 2
                for r in range(img.shape[0]):
                    center = center_list[r]
                    left = max(center, 0)
                    right = min(center + speckle_width // 2, img.shape[1])
                    noise = np.random.normal(0, speckle_sigma, size=(right - left,))
                    img[r, left:right] += noise

        # ---- Gap artifact ----
            if random.random() < self.randomization_prob:
                gap_center = random.randint(228, 284)
                gap_lens = random.randint(
                    self.gap_grid_range[0], self.gap_grid_range[1]
                )
                gap_ratio = random.uniform(0.3, self.gap_max_ratio)
                for r in range((gap_center - gap_lens), (gap_center + gap_lens)):
                    center = center_list[r]
                    filt_len = int(max_filt_gridlens[r] * gap_ratio)
                    left = max(center - filt_len, 0)
                    right = min(center + filt_len + 1, self.net_nt)
                    img[r, left:right] *= 0.05


# ===================================================================
# RealLimbDataset — real (in-vivo/ex-vivo) data for fine-tuning
# ===================================================================

class RealLimbDataset(Dataset):
    """Real manually-segmented limb data — float32, stride-6 downsampled.

    Used for fine-tuning on ``manual_seg_data`` (LPP thigh).

    Each ``.bin`` file is reshaped to ``(512, 4955)``, subsampled by
    stride 6 along the time axis, cropped to ``(384, 384)``, and
    per-sample max-abs normalised + clipped to ``[-1, 1]``.

    Parameters
    ----------
    data_dir : str
        Directory containing ``sliceX_srcY.bin`` and ``srcY_mask.npy`` files.
    flip_enabled : bool
        If ``True``, apply random vertical flips (50% probability).
    """

    def __init__(self, data_dir: str, flip_enabled: bool = False):
        self.data_dir = data_dir
        self.flip_enabled = flip_enabled

        self.data_files = sorted(
            [f for f in os.listdir(data_dir) if f.endswith(".bin")]
        )
        self.label_files = sorted(
            [f for f in os.listdir(data_dir) if f.endswith("_mask.npy")]
        )
        assert len(self.data_files) == len(self.label_files), (
            "Mismatch between .bin and _mask.npy file counts"
        )

    def __len__(self) -> int:
        return len(self.data_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        data_path = os.path.join(self.data_dir, self.data_files[idx])
        label_path = os.path.join(self.data_dir, self.label_files[idx])

        # Load & downsample raw data
        raw = np.fromfile(data_path, dtype=np.float32).reshape((512, 4955))
        raw = raw[:, ::6]                              # (512, 826)

        # Crop to ROI
        sub = raw[64:448, 300:684]                     # (384, 384)

        # Normalise & clip
        vmax = np.max(np.abs(sub))
        if vmax > 0:
            sub = sub / vmax
        sub = np.clip(sub, -1.0, 1.0)

        # Load & process mask
        mask = np.load(label_path)                     # (512, 620)
        mask = np.transpose(mask)                      # (620, 512)
        target_height = 4955 // 6                      # 826
        scale_factor = target_height / mask.shape[1]
        mask_resized = zoom(mask, (1, scale_factor), order=0)  # nearest
        mask = mask_resized[64:448, 300:684]           # (384, 384)

        # Random flip
        if self.flip_enabled and np.random.rand() < 0.5:
            sub = np.flip(sub, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()

        img = torch.from_numpy(sub).unsqueeze(0).float()
        mask = torch.from_numpy(mask).long()
        return img, mask


# ===================================================================
# RealLimbDataset2 — pre-cropped int16 variant
# ===================================================================

class RealLimbDataset2(Dataset):
    """Real manually-segmented limb data — int16, pre-cropped.

    Used for fine-tuning on ``manual_seg_beef``.

    Each ``.bin`` file is already ``(384, 384)`` int16.  Labels are
    randomly paired with source X positions (1–64) per label Y.

    Parameters
    ----------
    data_dir : str
        Directory containing ``sliceX_srcY.bin`` and ``srcY_mask.npy`` files.
    flip_enabled : bool
        If ``True``, apply random vertical flips (50% probability).
    min_srcY : int | None
        Lower bound (inclusive) for source Y filtering.
    max_srcY : int | None
        Upper bound (inclusive) for source Y filtering.
    """

    def __init__(
        self,
        data_dir: str,
        flip_enabled: bool = False,
        min_srcY: Optional[int] = None,
        max_srcY: Optional[int] = None,
    ):
        self.data_dir = data_dir
        self.flip_enabled = flip_enabled

        # Collect all label files and parse srcY
        self.label_files = sorted(
            [f for f in os.listdir(data_dir) if f.endswith("_mask.npy")]
        )
        self.src_ids: list[int] = []
        for f in self.label_files:
            try:
                y_str = f.replace("src", "").replace("_mask.npy", "")
                y = int(y_str)
                if (min_srcY is None or y >= min_srcY) and \
                   (max_srcY is None or y <= max_srcY):
                    self.src_ids.append(y)
            except ValueError:
                continue

        # Random X pairing per Y (fixed at construction)
        self.XY_pairs: list[tuple[int, int]] = []
        for y in self.src_ids:
            # Normal version: one random cross-section per source (needs all 64 slices).
            # x = random.randint(1, 64)
            # Limited-data version: only slice 32 is shipped, so fix x = 32.
            x = 32
            self.XY_pairs.append((x, y))

    def __len__(self) -> int:
        return len(self.XY_pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.XY_pairs[idx]

        data_path = os.path.join(self.data_dir, f"slice{x}_src{y}.bin")
        label_path = os.path.join(self.data_dir, f"src{y}_mask.npy")

        # Load pre-cropped signal
        sub = np.fromfile(data_path, dtype=np.int16).reshape((384, 384))
        vmax = np.max(np.abs(sub))
        if vmax > 0:
            sub = sub / vmax
        sub = np.clip(sub, -1.0, 1.0)

        # Load & process mask
        mask = np.load(label_path)                     # (512, 620)
        mask = np.transpose(mask)                      # (620, 512)
        target_height = 4955 // 6                      # 826
        scale_factor = target_height / mask.shape[1]
        mask_resized = zoom(mask, (1, scale_factor), order=0)
        mask = mask_resized[0:384, 300:684]            # (384, 384)

        # Random flip
        if self.flip_enabled and np.random.rand() < 0.5:
            sub = np.flip(sub, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()

        img = torch.from_numpy(sub).unsqueeze(0).float()
        mask = torch.from_numpy(mask).long()
        return img, mask
