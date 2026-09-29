#!/usr/bin/env python3
"""
mock_device.py - a stand-in for the ESP32-CAM, for developing the PC tooling.

Implements the same HTTP surface as the firmware, backed by synthetic
gestures, so pc/collect.py, pc/check_device.py and pc/train.py can be
exercised before any hardware exists. The synthetic gestures are deliberately
distinguishable (left swipe, right swipe, clockwise circle, anticlockwise
circle, vertical, random) so a full train-then-infer round trip is meaningful
rather than just exercising the plumbing.

    python3 tools/mock_device.py --port 8099 &
    python3 pc/check_device.py --url http://127.0.0.1:8099
    python3 pc/collect.py   --url http://127.0.0.1:8099 --reps 40 --no-prompt

It is a development aid, not a simulator: it does not model camera noise,
lighting change, or PSRAM behaviour.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pc"))

import mhi  # noqa: E402

W = H = mhi.MHI_W
PIX = mhi.MHI_PIXELS
SEQ = mhi.MHI_SEQ_LEN
SAMPLE_BYTES = SEQ * PIX

LABELS = ["swipe_left", "swipe_right", "circle_cw", "circle_ccw",
          "vertical", "unknown"]

_has_model = (ROOT / "src" / "model.h").exists()
_has_golden = (ROOT / "src" / "golden_vector.h").exists()

_state_lock = threading.Lock()
_state = {"frames": 0, "motion_energy": 0, "fps": 15000}
_recent_mhi = np.zeros((H, W), dtype=np.uint8)


def synth_sequence(label: int, seed: int | None = None) -> np.ndarray:
    """A bright square following a label-specific trajectory."""
    rng = np.random.default_rng(seed if seed is not None else rng_seed(label))
    seq = np.zeros((SEQ, H, W), dtype=np.uint8)
    seq += 22

    side = 16
    margin = 14
    span = W - 2 * margin - side
    cy = H // 2

    for t in range(SEQ):
        p = t / (SEQ - 1)
        if label == 0:      # left to right
            x, y = margin + span * p, cy
        elif label == 1:    # right to left
            x, y = margin + span * (1 - p), cy
        elif label == 2:    # clockwise circle
            a = 2 * math.pi * p
            x = margin + span / 2 + (span / 2) * math.cos(a)
            y = cy + 18 * math.sin(a)
        elif label == 3:    # anticlockwise circle
            a = -2 * math.pi * p
            x = margin + span / 2 + (span / 2) * math.cos(a)
            y = cy + 18 * math.sin(a)
        elif label == 4:    # vertical
            x = margin + span / 2
            y = margin + (H - 2 * margin - side) * p
        else:               # unknown: jitter in place
            x = margin + span * p
            y = cy + rng.integers(-8, 9)

        xi, yi = int(x), int(y)
        y0, y1 = max(0, yi - side // 2), min(H, yi + side // 2)
        x0, x1 = max(0, xi - side // 2), min(W, xi + side // 2)
        seq[t, y0:y1, x0:x1] = 205
        seq[t] = np.clip(seq[t].astype(np.int16)
                         + rng.integers(-5, 6, (H, W)), 0, 255).astype(np.uint8)
    return seq


def rng_seed(label: int) -> int:
    with _state_lock:
        _state["frames"] += 1
        return label * 1000 + _state["frames"]


def bmp_encode(gray: np.ndarray) -> bytes:
    offbits = 14 + 40 + 1024
    total = offbits + gray.size
    out = bytearray(offbits + gray.size)
    out[0:2] = b"BM"
    struct.pack_into("<I", out, 2, total)
    struct.pack_into("<I", out, 10, offbits)
    struct.pack_into("<I", out, 14, 40)
    struct.pack_into("<i", out, 18, W)
    struct.pack_into("<i", out, 22, H)
    struct.pack_into("<H", out, 26, 1)
    struct.pack_into("<H", out, 28, 8)
    struct.pack_into("<I", out, 34, gray.size)
    for i in range(256):
        out[54 + i * 4: 58 + i * 4] = bytes((i, i, i, 0))
    for row in range(H):
        src = gray[H - 1 - row].tobytes()
        out[offbits + row * W: offbits + (row + 1) * W] = src
    return bytes(out)


# A 1x1 JPEG is not valid for a browser but is a valid HTTP body; check_device
# only checks that bytes came back. Real size numbers come from the device.
TINY_JPEG = bytes.fromhex(
    "ffd8ffe000104a46494600010100000100010000ffdb004300"
    + "08060607060508070707090909080a0c140d0c0b0b0c191213"
    + "0f141d1a1f1e1d1a1c1c20242e2720222c231c1c2837292c303134"
    + "34341f27393d38323c2e333432ffc0000b080001000101011100"
    + "ffc40014000100000000000000000000000000000009ffda000801"
    + "0100000000d2cfffd9"
)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):  # quieter output
        pass

    def _send(self, ctype: str, body: bytes, extra: dict | None = None):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        global _recent_mhi
        path, _, query = self.path.partition("?")

        if path == "/":
            self._send("text/html", b"<html><body>mock device</body></html>")
            return

        if path == "/api/info":
            body = {
                "mcu_cores": 2, "flash_mb": 4,
                "psram_bytes": 4 * 1024 * 1024,
                "heap_internal_free_total": [180000, 280000],
                "heap_psram_free_total": [2800000, 3000000],
                "mhi": mhi.describe(),
                "inference": {
                    "available": _has_model, "arena_bytes": 98304,
                    "classes": len(LABELS), "last_ms": 37 if _has_model else 0,
                },
                "vote": {"window": 4, "min_wins": 3,
                         "min_score": 0.70, "cooldown": 8},
                "trigger_pixels": mhi.MOTION_TRIGGER_PIXELS,
            }
            self._send("application/json", json.dumps(body).encode())
            return

        if path == "/api/labels":
            self._send("application/json", json.dumps(LABELS).encode())
            return

        if path == "/api/state":
            with _state_lock:
                s = dict(_state)
            body = {
                "label": LABELS[0], "last_class": 0, "leader_class": 0,
                "leader_score": 0.87, "infer_ms": 37 if _has_model else 0,
                "frames": s["frames"], "fps": s["fps"],
                "motion_energy": s["motion_energy"],
                "model": "ready" if _has_model else "absent",
                "scores": [0.9, 0.05, 0.02, 0.01, 0.01, 0.01],
            }
            self._send("application/json", json.dumps(body).encode())
            return

        if path == "/api/mhi.bmp":
            self._send("image/bmp", bmp_encode(_recent_mhi),
                       {"Cache-Control": "no-store"})
            return

        if path == "/api/frame.jpg":
            self._send("image/jpeg", TINY_JPEG)
            return

        if path == "/api/selftest":
            if not (_has_model and _has_golden):
                self._send("application/json", json.dumps({
                    "ran": False, "reason": "no model"}).encode())
                return
            self._send("application/json", json.dumps({
                "ran": True, "pass": True, "class_match": True,
                "score_match": True, "got_class": 0,
                "got_label": LABELS[0], "want_class": 0,
                "want_label": LABELS[0], "got_score": 0.91,
                "want_score": 0.90, "infer_ms": 37,
            }).encode())
            return

        if path == "/api/record":
            q = dict(kv.split("=", 1) for kv in query.split("&") if "=" in kv)
            label = int(q.get("label", 0))
            seq = synth_sequence(label)
            derived = mhi.sequence_to_mhi(seq)
            _recent_mhi = derived
            with _state_lock:
                _state["motion_energy"] = int(mhi.mhi_energy(seq).max())
            hdr = (f"label={label} bytes={SAMPLE_BYTES} seq={SEQ} w={W} h={H} "
                   f"threshold={mhi.MHI_MOTION_THRESHOLD} "
                   f"decay={mhi.MHI_DECAY_NUM}")
            self._send("application/octet-stream", bytes(seq.tobytes()),
                       {"X-Sample-Header": hdr})
            return

        self.send_error(404)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"mock ESP32-CAM on http://{a.host}:{a.port}")
    print(f"  labels: {LABELS}")
    print(f"  model present: {_has_model}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
