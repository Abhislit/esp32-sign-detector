"""
mhi.py - Motion History Image computation, training-side reference.

THIS FILE IS THE SOURCE OF TRUTH for MHI semantics. It is a deliberate port of
src/mhi.c, not a reimplementation. Every constant below must equal its twin in
src/app_config.h.

Train/serve preprocessing skew is the number one way an on-device vision model
fails silently: the model trains at 94% on the PC and returns garbage on the
device, and nothing in the logs says why. tools/test_mhi_parity.c compiles the
C implementation natively and diffs its output against this one, so a drift
here is caught in a second rather than in a week of debugging.

Run `make parity` (or tools/run_parity.sh) after touching either side.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# Must match src/app_config.h exactly.
# --------------------------------------------------------------------------
MHI_W = 64
MHI_H = 64
MHI_PIXELS = MHI_W * MHI_H
MHI_SEQ_LEN = 12
MHI_MOTION_THRESHOLD = 26
MHI_DECAY_NUM = 225          # decay = 225/256 = 0.879
MHI_NORMALIZE = 1

# Motion trigger. Must match MOTION_TRIGGER_PIXELS in src/app_config.h.
# Kept low on purpose: a slow, deliberate swipe moves only ~64 px per frame,
# so a "5% of frame" threshold would mean careful recordings never trigger.
MOTION_TRIGGER_PIXELS = 40


def rgb565_luma(p: np.ndarray) -> np.ndarray:
    """Vectorised equivalent of mhi_rgb565_luma() in src/mhi.c.

    Input is uint16 RGB565. Weights are BT.601 scaled to sum to 256, and the
    5:6:5 channels are re-expanded to 8 bits first, exactly as the C code does.
    """
    r = (p >> 11) & 0x1F
    g = (p >> 5) & 0x3F
    b = p & 0x1F
    return ((r * 8 * 77 + g * 4 * 150 + b * 8 * 29) >> 8).astype(np.uint8)


def mhi_step(prev: np.ndarray, mhi: np.ndarray, cur: np.ndarray,
             motion_threshold: int = MHI_MOTION_THRESHOLD,
             decay_num: int = MHI_DECAY_NUM) -> tuple[np.ndarray, np.ndarray, int]:
    """One MHI update.

    Mirrors mhi_step() in src/mhi.c: decay the history, then let any pixel that
    moved past the threshold jump to full white.

    Returns (new_prev, new_mhi, motion_energy).
    """
    diff = np.abs(cur.astype(np.int16) - prev.astype(np.int16))
    mask = diff > motion_threshold

    # uint8 wraparound-free decay, matching the C shift-and-multiply exactly.
    decayed = (mhi.astype(np.uint32) * decay_num) >> 8
    out = np.where(mask, np.uint8(255), decayed.astype(np.uint8))

    energy = int(mask.sum())
    return cur.copy(), out, energy


def normalize(mhi: np.ndarray) -> tuple[np.ndarray, int]:
    """Scale an MHI to full 0..255. Mirrors mhi_normalize() in src/mhi.c.

    Returns (normalised, observed_max).
    """
    peak = int(mhi.max()) if mhi.size else 0
    if peak == 0:
        return np.zeros_like(mhi), 0
    # Integer floor division, same as the C operator.
    return ((mhi.astype(np.uint32) * 255) // peak).astype(np.uint8), peak


def sequence_to_mhi(seq: np.ndarray,
                    motion_threshold: int = MHI_MOTION_THRESHOLD,
                    decay_num: int = MHI_DECAY_NUM,
                    do_normalize: bool = MHI_NORMALIZE) -> np.ndarray:
    """Turn a recorded grayscale sequence into the single MHI the model sees.

    `seq` has shape (T, MHI_H, MHI_W), dtype uint8 - this is exactly the byte
    blob /api/record returns.

    Note the first frame only primes the history, so a T-frame sequence yields
    a T-1-frame MHI evolution. The final MHI is returned, which is what
    mhi_step_normalized() produces on the device at the last captured frame.
    """
    if seq.ndim != 3 or seq.shape[1] != MHI_H or seq.shape[2] != MHI_W:
        raise ValueError(
            f"expected (T, {MHI_H}, {MHI_W}) uint8, got {seq.shape} {seq.dtype}"
        )

    prev = seq[0].copy()
    mhi = np.zeros((MHI_H, MHI_W), dtype=np.uint8)

    for t in range(1, seq.shape[0]):
        prev, mhi, _ = mhi_step(prev, mhi, seq[t], motion_threshold, decay_num)

    if do_normalize:
        mhi, _ = normalize(mhi)
    return mhi


def mhi_energy(seq: np.ndarray,
               motion_threshold: int = MHI_MOTION_THRESHOLD) -> np.ndarray:
    """Per-frame motion pixel counts, for filtering junk recordings."""
    if seq.shape[0] < 2:
        return np.zeros(0, dtype=np.int32)
    diffs = np.abs(seq[1:].astype(np.int16) - seq[:-1].astype(np.int16))
    return (diffs > motion_threshold).sum(axis=(1, 2)).astype(np.int32)


def mhi_to_png_bytes(mhi: np.ndarray) -> bytes:
    """Render an MHI as a PNG, for eyeballing a dataset."""
    try:
        from io import BytesIO

        from PIL import Image
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Pillow is required for mhi_to_png_bytes()") from exc

    buf = BytesIO()
    Image.fromarray(mhi, mode="L").save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------------------
# Model input conversion
#
# The canonical MHI on disk and in the golden vector is uint8 0..255. TFLite
# wants the same real-valued signal in 0..1, and for an int8 model it wants
# that quantised. Getting this wrong is silent: a truncated `astype(np.int8)`
# on a 0..1 float yields only {0, 1} and the model still returns a confident,
# meaningless class.
#
# The quantised mapping is the standard asymmetric one, real = (q - zp) * scale,
# so q = real / scale + zp. src/infer.cpp uses the identical formula.
# --------------------------------------------------------------------------

def to_input_tensor(mhi_uint8: np.ndarray, dtype, scale: float = 1.0,
                    zero_point: int = 0) -> np.ndarray:
    """Canonical uint8 MHI -> a TFLite input tensor of the given dtype."""
    x = mhi_uint8.astype(np.float32) / 255.0
    if np.issubdtype(np.dtype(dtype), np.floating):
        return x.astype(np.float32)
    q = np.round(x / float(scale) + float(zero_point))
    lo, hi = (np.iinfo(dtype).min, np.iinfo(dtype).max)
    return q.clip(lo, hi).astype(dtype)


def dequantize_output(raw: np.ndarray, dtype, scale: float,
                      zero_point: int) -> np.ndarray:
    """TFLite output tensor -> real-valued probabilities."""
    if np.issubdtype(np.dtype(dtype), np.floating):
        return raw.astype(np.float64)
    return (raw.astype(np.float64) - float(zero_point)) * float(scale)


def describe() -> dict:
    """The preprocessing config, embedded into the dataset for provenance.

    train.py writes this next to the model so a mismatch is obvious later.
    """
    return {
        "mhi_w": MHI_W,
        "mhi_h": MHI_H,
        "seq_len": MHI_SEQ_LEN,
        "motion_threshold": MHI_MOTION_THRESHOLD,
        "decay_num": MHI_DECAY_NUM,
        "normalize": MHI_NORMALIZE,
        "decay": MHI_DECAY_NUM / 256.0,
    }


def describe_capture() -> dict:
    """Firmware-side capture constants that pc/ also needs to agree on."""
    return {"trigger_pixels": MOTION_TRIGGER_PIXELS}


if __name__ == "__main__":
    import json

    print(json.dumps(describe(), indent=2))
