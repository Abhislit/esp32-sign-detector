#!/usr/bin/env python3
"""
verify_export.py - close the loop between pc/train.py's output and the firmware.

Reads the C headers train.py generated straight out of src/ - the model blob
and the golden input are recovered by parsing the hex arrays, not by trusting
the in-memory objects the training run used - then runs the recovered TFLite
model on the recovered golden input and checks it still produces the recorded
expectation.

If this passes, a device that reports a selftest PASS is telling you something
real. If it fails, the export step corrupted something, and the device would
have blamed the hardware.

    python3 tools/verify_export.py
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(ROOT / "pc"))

import mhi  # noqa: E402

HEX = re.compile(r"0x([0-9a-fA-F]{2})")


def parse_blob(path: Path, symbol: str) -> bytes:
    """Recover a byte array from a generated header.

    Accepts both `name[] = {` (xxd's form) and `name[4096] = {`.
    """
    text = path.read_text()
    m = re.search(rf"{symbol}\s*(?:\[[^\]]*\])?\s*=\s*\{{(.*?)\}}", text, re.S)
    if not m:
        raise SystemExit(f"error: {symbol}[] not found in {path.name}")
    return bytes(int(h, 16) for h in HEX.findall(m.group(1)))


def parse_defines(path: Path) -> dict:
    text = path.read_text()
    out = {}
    for name in ("GOLDEN_EXPECTED_CLASS", "GOLDEN_EXPECTED_SCORE_MILLI",
                 "GOLDEN_INPUT_W", "GOLDEN_INPUT_H"):
        m = re.search(rf"#define\s+{name}\s+(\d+)", text)
        if m:
            out[name] = int(m.group(1))
    return out


def main() -> int:
    for f in ("model.h", "labels.h", "golden_vector.h"):
        if not (SRC / f).exists():
            sys.exit(f"error: src/{f} not found. Run: python3 pc/train.py "
                     f"--data data/raw")

    try:
        import tensorflow as tf
    except ImportError:
        sys.exit("error: tensorflow not available in this interpreter.\n"
                 "       Use the training venv:  source .venv/bin/activate")

    blob = parse_blob(SRC / "model.h", "gesture_model_data")
    golden = parse_blob(SRC / "golden_vector.h", "golden_input")
    d = parse_defines(SRC / "golden_vector.h")

    print(f"  model.h        {len(blob)} bytes")
    print(f"  golden_vector  {len(golden)} px, expects class "
          f"{d.get('GOLDEN_EXPECTED_CLASS')} at "
          f"{d.get('GOLDEN_EXPECTED_SCORE_MILLI') / 1000:.3f}")

    fails = 0

    if blob[4:8] != b"TFL3":
        print("  FAIL model.h does not contain a TFLite flatbuffer")
        fails += 1

    m = re.search(r"#define\s+GESTURE_LABELS\s*\{(.*?)\}",
                  (SRC / "labels.h").read_text(), re.S)
    if not m:
        print("  FAIL GESTURE_LABELS missing from labels.h")
        return 1
    labels = re.findall(r'"([^"]*)"', m.group(1))
    print(f"  labels.h       {len(labels)} classes: {labels}")

    cfg = (SRC / "app_config.h").read_text()
    mc = re.search(r"#define\s+NUM_GESTURES\s+(\d+)", cfg)
    if mc:
        n_cfg = int(mc.group(1))
        if n_cfg != len(labels):
            print(f"  FAIL NUM_GESTURES is {n_cfg} in app_config.h but the "
                  f"model has {len(labels)} outputs. The firmware would read "
                  f"past the output tensor.")
            fails += 1
        else:
            print(f"  ok             NUM_GESTURES ({n_cfg}) matches the model")

    want_w = d.get("GOLDEN_INPUT_W")
    if want_w and want_w * d.get("GOLDEN_INPUT_H", 0) != len(golden):
        print(f"  FAIL golden_input is {len(golden)} px but the header claims "
              f"{want_w}x{d.get('GOLDEN_INPUT_H')}")
        fails += 1

    # --- run the recovered blob on the recovered input -------------------
    interp = tf.lite.Interpreter(model_content=blob)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    x = np.frombuffer(golden, dtype=np.uint8).reshape(want_w, d.get("GOLDEN_INPUT_H", want_w))
    # Use the shared conversion so this cannot drift from train.py or the
    # firmware. A naive `astype(int8)` on a 0..1 float yields {0,1} and the
    # model still returns a confident, meaningless class.
    q = mhi.to_input_tensor(x, inp["dtype"], *inp["quantization"])
    interp.set_tensor(inp["index"], q.reshape(1, q.shape[0], q.shape[1], 1))
    interp.invoke()
    probs = mhi.dequantize_output(interp.get_tensor(out["index"])[0],
                                  out["dtype"], *out["quantization"])

    got = int(probs.argmax())
    score = float(probs[got])
    want = d.get("GOLDEN_EXPECTED_CLASS", -1)
    want_score = d.get("GOLDEN_EXPECTED_SCORE_MILLI", 0) / 1000.0

    print(f"  replay         class={got} ({labels[got] if got < len(labels) else '?'})"
          f" score={score:.4f}")
    print(f"  recorded       class={want} "
          f"({labels[want] if 0 <= want < len(labels) else '?'})"
          f" score={want_score:.4f}")

    if got != want:
        print("  FAIL the embedded model does not reproduce the recorded class.\n"
              "        The golden vector is a broken canary, so /api/selftest "
              "would mislead you.")
        fails += 1
    else:
        print("  ok             golden vector reproduces from the embedded blob")

    if abs(score - want_score) > 0.02:
        print(f"  WARN score drifted {abs(score - want_score):.4f}; the device "
              "compares within 0.05 so it will still pass")
    else:
        print("  ok             score within tolerance")

    print()
    if fails:
        print(f"export verification FAILED ({fails} problem(s))")
        return 1
    print("export verification passed - flash and check /api/selftest")
    return 0


if __name__ == "__main__":
    sys.exit(main())
