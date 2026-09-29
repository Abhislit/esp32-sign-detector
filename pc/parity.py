#!/usr/bin/env python3
"""
parity.py - prove pc/mhi.py and src/mhi.c compute identical Motion History
Images. Run via tools/run_parity.sh, or directly once a host build exists.

Exits non-zero on any byte mismatch, so it can gate a commit.
"""

from __future__ import annotations

import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pc"))

import mhi  # noqa: E402

WORK = ROOT / "build" / "parity"
C_BIN = WORK / "test_mhi_parity"
INPUT = WORK / "input.bin"
C_OUT = WORK / "c_out.bin"


def synth_sequence(n_frames: int, seed: int = 7) -> np.ndarray:
    """A bright square sweeping across a dark field, plus mild sensor noise.

    Representative of a hand moving through frame, and it exercises both the
    mask branch and the decay branch, which pure noise also does but less
    visibly. Deterministic given (n_frames, seed).
    """
    rng = np.random.default_rng(seed)
    seq = np.zeros((n_frames, mhi.MHI_H, mhi.MHI_W), dtype=np.uint8)
    seq += 20  # dim background

    side = 18
    for t in range(n_frames):
        cx = 12 + int((mhi.MHI_W - 24 - side) * t / max(1, n_frames - 1))
        cy = mhi.MHI_H // 2
        y0, y1 = cy - side // 2, cy + side // 2
        x0, x1 = cx, cx + side
        seq[t, max(0, y0):y1, max(0, x0):x1] = 210  # bright moving "hand"
        # Sensor noise so the threshold boundary gets exercised.
        seq[t] = np.clip(seq[t].astype(np.int16)
                         + rng.integers(-6, 7, (mhi.MHI_H, mhi.MHI_W)),
                         0, 255).astype(np.uint8)
    return seq


def write_input(seq: np.ndarray) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    with INPUT.open("wb") as f:
        f.write(struct.pack("<H", seq.shape[0]))
        f.write(seq.tobytes())


def read_c_output() -> tuple[int, np.ndarray, np.ndarray, list[int]]:
    raw = C_OUT.read_bytes()
    peak = raw[0]
    shape = (mhi.MHI_H, mhi.MHI_W)
    mhi_raw = np.frombuffer(raw[1:1 + mhi.MHI_PIXELS],
                            dtype=np.uint8).reshape(shape)
    mhi_norm = np.frombuffer(raw[1 + mhi.MHI_PIXELS:1 + 2 * mhi.MHI_PIXELS],
                             dtype=np.uint8).reshape(shape)
    n = (len(raw) - 1 - 2 * mhi.MHI_PIXELS) // 4
    energies = list(struct.unpack(f"<{n}I", raw[1 + 2 * mhi.MHI_PIXELS:]))
    return peak, mhi_raw, mhi_norm, energies


def compare(name: str, a: np.ndarray, b: np.ndarray) -> int:
    if a.shape != b.shape:
        print(f"  FAIL {name}: shape {a.shape} != {b.shape}")
        return 1
    diff = np.abs(a.astype(np.int32) - b.astype(np.int32))
    n_bad = int((diff > 0).sum())
    if n_bad:
        worst = int(diff.max())
        print(f"  FAIL {name}: {n_bad}/{a.size} pixels differ, max delta {worst}")
        return 1
    print(f"  ok   {name}: {a.size} pixels identical")
    return 0


def check_quantization() -> int:
    """The int8 input mapping must be q = real/scale + zero_point.

    This is the bug class that produced a confident-but-meaningless class: a
    plain `astype(np.int8)` on a 0..1 float yields only {0, 1}, and the model
    still returns a normal-looking answer. src/infer.cpp uses the same formula,
    so getting it right here keeps the firmware correct too.
    """
    print("\n[quantisation helpers]")
    fails = 0

    scale, zp = 1.0 / 255.0, -128
    u8 = np.arange(256, dtype=np.uint8)
    q = mhi.to_input_tensor(u8, np.int8, scale, zp)
    # real = u8/255, so q = u8 - 128 exactly.
    want = u8.astype(np.int16) + zp
    if not np.array_equal(q.astype(np.int16), want):
        print("  FAIL int8 mapping is not q = real/scale + zero_point")
        fails += 1
    else:
        print("  ok   int8 mapping: byte b -> q = b - 128, full range used")
    if len(np.unique(q)) != 256:
        print(f"  FAIL only {len(np.unique(q))} distinct codes from 256 inputs")
        fails += 1
    else:
        print("  ok   all 256 input levels map to distinct codes")

    f = mhi.to_input_tensor(u8, np.float32)
    if abs(float(f[255]) - 1.0) > 1e-6 or float(f[0]) != 0.0:
        print(f"  FAIL float32 mapping wrong: f[0]={f[0]} f[255]={f[255]}")
        fails += 1
    else:
        print("  ok   float32 mapping: 0..255 -> 0.0..1.0")

    raw = np.array([-128, -64, 0, 64, 127], dtype=np.int8)
    back = mhi.dequantize_output(raw, np.int8, 1.0 / 255.0, -128)
    want_back = (np.array([-128, -64, 0, 64, 127], dtype=np.float64) + 128) / 255.0
    if not np.allclose(back, want_back, atol=1e-6):
        print(f"  FAIL dequantise: got {back} want {want_back}")
        fails += 1
    else:
        print(f"  ok   dequantise maps -128..127 -> "
              f"{back[0]:.3f}..{back[-1]:.3f}")

    # Demonstrate the failure mode this helper exists to prevent: casting a
    # 0..1 float straight to int8 keeps almost no information, and the model
    # still returns a confident class. The correct mapping preserves all 256.
    naive = (np.array([0.0, 0.25, 0.5, 0.75, 1.0]) * 1.0).astype(np.int8)
    correct = mhi.to_input_tensor(np.array([0, 64, 128, 191, 255], np.uint8),
                                  np.int8, scale, zp)
    if len(np.unique(naive)) >= len(np.unique(correct)):
        print(f"  FAIL naive cast kept {len(np.unique(naive))} codes, "
              f"correct keeps {len(np.unique(correct))}; "
              "the test no longer demonstrates the failure mode")
        fails += 1
    else:
        print(f"  ok   naive float->int8 keeps {len(np.unique(naive))}/5 levels, "
              f"correct mapping keeps {len(np.unique(correct))}")

    return fails


def main() -> int:
    if not C_BIN.exists():
        print(f"error: {C_BIN} not built. Run tools/run_parity.sh", file=sys.stderr)
        return 2

    failures = 0
    cases = [
        ("default params", synth_sequence(12), mhi.MHI_MOTION_THRESHOLD,
         mhi.MHI_DECAY_NUM, mhi.MHI_NORMALIZE),
        ("low threshold", synth_sequence(12, seed=11), 4, mhi.MHI_DECAY_NUM, 1),
        ("high threshold", synth_sequence(12, seed=11), 90, mhi.MHI_DECAY_NUM, 1),
        ("fast decay", synth_sequence(16, seed=3), mhi.MHI_MOTION_THRESHOLD, 90, 1),
        ("slow decay", synth_sequence(16, seed=3), mhi.MHI_MOTION_THRESHOLD, 250, 1),
        ("no normalize", synth_sequence(12, seed=5), mhi.MHI_MOTION_THRESHOLD,
         mhi.MHI_DECAY_NUM, 0),
        ("long sequence", synth_sequence(30, seed=9), mhi.MHI_MOTION_THRESHOLD,
         mhi.MHI_DECAY_NUM, 1),
    ]

    for name, seq, thresh, decay, do_norm in cases:
        print(f"\n[{name}]  T={seq.shape[0]} threshold={thresh} "
              f"decay={decay}/256 normalize={do_norm}")
        write_input(seq)

        proc = subprocess.run(
            [str(C_BIN), str(INPUT), str(C_OUT),
             str(thresh), str(decay), str(do_norm)],
            capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"  FAIL C harness exited {proc.returncode}: {proc.stderr}")
            failures += 1
            continue
        print(f"  {proc.stderr.strip()}")

        # --- Python reference, same params -----------------------------
        prev = seq[0].copy()
        acc = np.zeros((mhi.MHI_H, mhi.MHI_W), dtype=np.uint8)
        energies = [0]
        for t in range(1, seq.shape[0]):
            prev, acc, e = mhi.mhi_step(prev, acc, seq[t], thresh, decay)
            energies.append(e)

        if do_norm:
            py_norm, py_peak = mhi.normalize(acc)
        else:
            # No normalisation stage, so there is no peak to report either.
            py_norm, py_peak = acc.copy(), 0

        c_peak, c_raw, c_norm, c_energy = read_c_output()

        failures += compare("raw mhi", c_raw, acc)
        failures += compare("normalised", c_norm, py_norm)

        if c_peak != py_peak:
            print(f"  FAIL peak: C={c_peak} py={py_peak}")
            failures += 1
        else:
            print(f"  ok   peak: {c_peak}")

        if c_energy != energies:
            bad = [i for i, (x, y) in enumerate(zip(c_energy, energies)) if x != y]
            print(f"  FAIL motion_energy at frames {bad}")
            failures += 1
        else:
            print(f"  ok   motion_energy: {len(energies)} frames identical")

    failures += check_quantization()

    print("\n" + "=" * 60)
    if failures:
        print(f"PARITY FAILED: {failures} mismatch(es)")
        print("src/mhi.c and pc/mhi.py have diverged. Do not train a model "
              "until this passes - the device would silently misclassify.")
        return 1
    print("PARITY OK: C and Python produce bit-identical MHIs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
