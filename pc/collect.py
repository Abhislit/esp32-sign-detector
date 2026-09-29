#!/usr/bin/env python3
"""
collect.py - drive the device's /api/record endpoint to build a dataset.

The device arms a label, waits for a motion trigger, then returns MHI_SEQ_LEN
consecutive 64x64 grayscale frames as raw bytes. That raw sequence is saved
verbatim: the MHI is derived on the PC by pc/mhi.py, so preprocessing stays
in one place and the recordings remain reusable if the MHI constants change.

Typical run:

    python3 pc/collect.py --url http://192.168.4.1 --reps 150

Prompts you for each repetition so you can set up the next gesture. Use
--no-prompt for unattended batch capture.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pc"))

import mhi  # noqa: E402

DEFAULT_URL = "http://192.168.4.1"


def get(url: str, timeout: float):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, dict(r.headers), r.read()


def get_json(url: str, timeout: float):
    _, _, body = get(url, timeout)
    return json.loads(body.decode())


def check_device(url: str) -> dict:
    try:
        info = get_json(f"{url}/api/info", timeout=5)
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        sys.exit(
            f"error: cannot reach {url}/api/info ({exc}).\n"
            "       Check the AP join, the IP, and that the device is running."
        )

    psram = info.get("psram_bytes", 0)
    if psram == 0:
        sys.exit(
            "error: device reports no PSRAM. Stop here.\n"
            "       Recording allocates 49KB + 49KB in PSRAM. See M0 in README.md"
        )

    dev_mhi = info.get("mhi", {})
    for key, pc_val in mhi.describe().items():
        dev_val = dev_mhi.get(key)
        if dev_val is not None and dev_val != pc_val:
            sys.exit(
                f"error: MHI mismatch on '{key}': device={dev_val} pc={pc_val}\n"
                "       src/app_config.h and pc/mhi.py have diverged.\n"
                "       Run tools/run_parity.sh and fix before collecting."
            )

    print("device ok:")
    print(f"  psram        {psram} bytes")
    print(f"  model        {info['inference']['available']}")
    print(f"  mhi          {dev_mhi}")
    print(f"  trigger_px   {info.get('trigger_pixels')}")
    return info


def collect_one(url: str, label: int, timeout: float) -> bytes:
    """Returns the raw sample bytes, or raises."""
    _, headers, body = get(f"{url}/api/record?label={label}", timeout=timeout)
    ctype = headers.get("Content-Type", "")
    if "octet-stream" not in ctype:
        raise RuntimeError(body.decode(errors="replace")[:200])
    return body


def save_sample(out_dir: Path, label: int, index: int, raw: bytes) -> Path:
    d = out_dir / f"label{label}"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{index:05d}.bin"
    path.write_bytes(raw)
    return path


def preview(path: Path, label: int, preview_dir: Path) -> None:
    """Save the derived MHI as a PNG so a recording can be sanity-checked."""
    try:
        seq = np.frombuffer(path.read_bytes(), dtype=np.uint8)
        seq = seq.reshape(mhi.MHI_SEQ_LEN, mhi.MHI_H, mhi.MHI_W)
        png = mhi.mhi_to_png_bytes(mhi.sequence_to_mhi(seq))
    except (RuntimeError, ValueError):
        return  # Pillow absent or bad blob; preview is best-effort
    d = preview_dir / f"label{label}"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{path.stem}.png").write_bytes(png)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=DEFAULT_URL, help="device base URL")
    ap.add_argument("--out", default=str(ROOT / "data" / "raw"),
                    help="output directory")
    ap.add_argument("--reps", type=int, default=100,
                    help="repetitions per label")
    ap.add_argument("--labels", type=int, default=None,
                    help="override label count (default: ask the device)")
    ap.add_argument("--start-label", type=int, default=0,
                    help="resume from this label index")
    ap.add_argument("--timeout", type=float, default=30.0,
                    help="seconds to wait for a gesture before giving up")
    ap.add_argument("--no-prompt", action="store_true",
                    help="do not press Enter between reps")
    ap.add_argument("--preview", action="store_true",
                    help="also write MHI PNGs for spot-checking")
    args = ap.parse_args()

    url = args.url.rstrip("/")
    info = check_device(url)

    try:
        labels = get_json(f"{url}/api/labels", timeout=5)
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        labels = [str(i) for i in range(args.labels or 6)]

    n_labels = args.labels if args.labels is not None else len(labels)
    out_dir = Path(args.out)
    preview_dir = out_dir.parent / "preview"
    sample_bytes = mhi.MHI_SEQ_LEN * mhi.MHI_PIXELS

    print(f"\nlabels ({n_labels}): {labels}")
    print(f"target: {args.reps} reps each, {sample_bytes} bytes per sample")
    print(f"out:   {out_dir}\n")

    if not args.no_prompt:
        print("Performing a gesture repeatedly looks better than doing it once "
              "slowly.\nVary speed and starting position across reps, but keep "
              "the camera fixed.\n")

    total = ok = 0
    fails = 0
    for li in range(args.start_label, n_labels):
        name = labels[li] if li < len(labels) else str(li)
        for rep in range(args.reps):
            total += 1
            if not args.no_prompt:
                try:
                    input(f"\n  [{name}] rep {rep + 1}/{args.reps}  "
                          f"(press Enter, then perform the gesture) > ")
                except EOFError:
                    args.no_prompt = True

            try:
                raw = collect_one(url, li, args.timeout)
            except urllib.error.HTTPError as exc:
                body = exc.read().decode(errors="replace")
                print(f"    device error: {body[:120]}")
                fails += 1
                continue
            except (RuntimeError, urllib.error.URLError, OSError) as exc:
                msg = getattr(exc, "reason", exc)
                print(f"    failed: {str(msg)[:120]}")
                fails += 1
                continue

            if len(raw) != sample_bytes:
                print(f"    wrong size {len(raw)} (want {sample_bytes}), skipped")
                fails += 1
                continue

            path = save_sample(out_dir, li, rep, raw)
            if args.preview:
                preview(path, li, preview_dir)
            ok += 1
            print(f"    saved {path.name} ({len(raw)} B)")

    print(f"\ndone: {ok}/{total} samples, {fails} failures -> {out_dir}")
    if ok == 0:
        print("\nNothing collected. Common causes:")
        print("  - motion never reached MOTION_TRIGGER_PIXELS; check the")
        print("    'motion px' readout on the dashboard while waving")
        print("  - lighting too flat, raise MHI_MOTION_THRESHOLD in "
              "src/app_config.h")
        print("  - hand outside the field of view")
    print("\nnext: python3 pc/train.py --data", out_dir)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
