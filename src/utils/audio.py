# src/utils/audio.py
from typing import Tuple, Optional

import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt
import torch


def load_wav(path: str) -> Tuple[np.ndarray, int]:
    """Load a WAV file as mono float32.

    Returns:
        (audio, sample_rate)
        audio shape: [T]
    """
    data, sr = sf.read(path)
    if data.ndim == 2:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    return data, int(sr)


def _butter_highpass(cutoff: float, sr: int, order: int = 4):
    nyq = 0.5 * sr
    norm = cutoff / nyq
    b, a = butter(order, norm, btype="highpass", analog=False)
    return b, a


def highpass_20k(x: np.ndarray, sr: int, order: int = 4) -> np.ndarray:
    """Apply 20 kHz high-pass filter as in the paper.

    If sr <= 40 kHz (Nyquist < 20 kHz), return unchanged.
    """
    cutoff = 20000.0
    if sr <= 2 * cutoff or x.size == 0:
        return x.astype(np.float32)

    b, a = _butter_highpass(cutoff, sr, order=order)
    y = filtfilt(b, a, x).astype(np.float32)
    return y


def resample_linear(x: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Simple linear resampling using numpy.interp."""
    if orig_sr == target_sr:
        return x.astype(np.float32)
    if x.size == 0:
        return x.astype(np.float32)

    duration = x.shape[0] / float(orig_sr)
    t_orig = np.linspace(0.0, duration, num=x.shape[0], endpoint=False)
    target_len = int(round(duration * target_sr))
    t_new = np.linspace(0.0, duration, num=target_len, endpoint=False)
    y = np.interp(t_new, t_orig, x).astype(np.float32)
    return y


def pad_or_trim(x: np.ndarray, target_len: int) -> np.ndarray:
    """Pad with zeros or trim to exact target_len."""
    cur_len = x.shape[0]
    if cur_len == target_len:
        return x.astype(np.float32)
    if cur_len > target_len:
        return x[:target_len].astype(np.float32)
    pad_width = target_len - cur_len
    return np.pad(x, (0, pad_width), mode="constant", constant_values=0.0).astype(
        np.float32
    )


def load_waveform_1d(
    path: str,
    target_sr: Optional[int] = None,
    target_len: Optional[int] = None,
    apply_highpass: bool = True,
) -> torch.Tensor:
    """Load waveform and prepare [1, L] tensor for Conv1d.

    Steps:
      1) read wav (mono, float32)
      2) optional 20 kHz high-pass
      3) optional resample to target_sr
      4) optional pad/trim to target_len
      5) intensity normalization (divide by max |x|, if > 0)

    Returns:
        torch.Tensor shape [1, L]
    """
    x, sr = load_wav(path)

    if apply_highpass:
        x = highpass_20k(x, sr)

    if target_sr is not None and sr != target_sr:
        x = resample_linear(x, sr, target_sr)
        sr = target_sr

    if target_len is not None:
        x = pad_or_trim(x, target_len)

    x = x.astype(np.float32)

    # === 논문 스타일 intensity normalization ===
    # 각 벡터를 max(|x|)로 나눔 (max_abs==0이면 그대로 둠)
    if x.size > 0:
        max_abs = np.max(np.abs(x))
        if max_abs > 0.0:
            x = x / max_abs

    t = torch.from_numpy(x).unsqueeze(0)  # [1, L]
    return t
