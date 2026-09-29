#!/usr/bin/env python3
"""
check_device.py - milestone M0. Verify the hardware and the firmware/PC config
agree, BEFORE spending a day collecting data.

Three things kill this project early and all three are checkable in seconds:

  1. No (or wrong-size) PSRAM. The AI-Thinker datasheet claims 4MB; many cheap
     clones ship with 0MB or 2MB. Without it the camera cannot allocate a
     framebuffer and the dataset buffers cannot be allocated at all.
  2. MHI constants diverged between src/app_config.h and pc/mhi.py. The model
     would train on one preprocessing and run on another, and nothing would
     report an error.
  3. esp-nn missing, which is ~10x slower inference and will look like "the
     chip is just weak".

    python3 pc/check_device.py --url http://192.168.4.1
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pc"))

import mhi  # noqa: E402

OK, WARN, BAD = "PASS", "WARN", "FAIL"
_results: list[tuple[str, str, str]] = []


def check(level: str, name: str, detail: str) -> None:
    _results.append((level, name, detail))
    print(f"  [{level}] {name}: {detail}")


def fetch(url: str, timeout: float = 6.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://192.168.4.1")
    args = ap.parse_args()
    url = args.url.rstrip("/")

    print(f"probing {url}\n")

    # --- reachability --------------------------------------------------
    try:
        info = json.loads(fetch(f"{url}/api/info").decode())
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        print(f"  [{BAD}] cannot reach {url}/api/info: {exc}\n")
        print("  Checklist:")
        print("    - PC and board on the same network (AP mode: 192.168.4.1)")
        print("    - 5V supply, not 3V3. The camera and WiFi together brown "
              "out USB power")
        print("    - serial monitor at 115200 for the boot log and IP")
        print("    - if in STA mode, pass --url http://<assigned-ip>")
        return 2

    # --- PSRAM ---------------------------------------------------------
    psram = int(info.get("psram_bytes", 0))
    if psram == 0:
        check(BAD, "psram", "0 bytes. The camera will not allocate framebuffers "
                            "and dataset capture is impossible.")
    elif psram < 2 * 1024 * 1024:
        check(WARN, "psram", f"{psram} bytes. Below the 4MB the AI-Thinker "
                             "datasheet claims. QVGA will be tight.")
    else:
        check(OK, "psram", f"{psram} bytes ({psram / 1024 / 1024:.1f} MB)")

    # --- memory headroom -----------------------------------------------
    free_i, total_i = info.get("heap_internal_free_total", [0, 0])
    if total_i:
        pct = 100.0 * free_i / total_i if total_i else 0
        detail = f"{free_i} / {total_i} free ({pct:.0f}%)"
        if pct <= 25:
            detail += ". Below ~25% the model has no room to allocate its arena."
        check(OK if pct > 25 else (WARN if pct > 10 else BAD),
              "internal heap", detail)

    free_p, total_p = info.get("heap_psram_free_total", [0, 0])
    if total_p:
        pct = 100.0 * free_p / total_p if total_p else 0
        check(OK, "psram heap", f"{free_p} / {total_p} free ({pct:.0f}%)")

    # --- MHI config parity ---------------------------------------------
    dev = info.get("mhi", {})
    pc_cfg = mhi.describe()
    mismatches = [k for k, v in pc_cfg.items() if k in dev and dev[k] != v]
    for k in mismatches:
        check(BAD, f"mhi.{k}", f"device={dev[k]} pc={pc_cfg[k]}")
    if not mismatches:
        check(OK, "mhi config", "device and pc/mhi.py agree on "
                                f"{len(pc_cfg)} parameters")

    for k, v in mhi.describe_capture().items():
        dev_v = info.get(k)
        if dev_v is None:
            continue
        if dev_v != v:
            check(BAD, k, f"device={dev_v} pc={v}. Edit src/app_config.h or "
                          "pc/mhi.py so they match.")
        else:
            check(OK, k, f"{dev_v}")

    # --- trigger reachability -------------------------------------------
    try:
        state = json.loads(fetch(f"{url}/api/state").decode())
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        state = {}
    trig = int(info.get("trigger_pixels", 0))
    energy = int(state.get("motion_energy", 0))
    if energy > 0:
        ratio = energy / trig if trig else 0
        level = OK if ratio >= 2 else WARN
        check(level, "motion trigger", f"currently {energy} px moving, "
                                       f"threshold {trig} px")
    else:
        check(WARN, "motion trigger", "0 px moving at rest. Wave a hand in "
                                      "front of the camera and re-read /api/state")
    if trig and trig < 20:
        check(WARN, "trigger sensitivity", f"threshold {trig} px is very low; "
                                           "the recorder may trigger on "
                                           "lighting changes or sensor noise")
    elif trig and trig > 150:
        check(WARN, "trigger sensitivity", f"threshold {trig} px is high. A "
                                           "slow gesture moves only ~64 px per "
                                           "frame, so recordings will time out")

    # --- camera ----------------------------------------------------------
    try:
        jpg = fetch(f"{url}/api/frame.jpg")
        check(OK if len(jpg) > 500 else WARN, "camera",
              f"jpeg {len(jpg)} bytes"
              + ("" if len(jpg) > 500 else " - suspiciously small, is the "
                                        "camera actually streaming?"))
    except (urllib.error.URLError, OSError) as exc:
        check(BAD, "camera", f"frame fetch failed: {exc}")

    try:
        bmp = fetch(f"{url}/api/mhi.bmp")
        expect = 14 + 40 + 1024 + mhi.MHI_PIXELS
        check(OK if len(bmp) == expect else BAD, "mhi endpoint",
              f"{len(bmp)} bytes (expected {expect})")
    except (urllib.error.URLError, OSError) as exc:
        check(BAD, "mhi endpoint", f"fetch failed: {exc}")

    # --- model ------------------------------------------------------------
    if info.get("inference", {}).get("available"):
        check(OK, "model", "loaded")
        try:
            st = json.loads(fetch(f"{url}/api/selftest").decode())
            if not st.get("ran"):
                check(WARN, "selftest", st.get("reason", "did not run"))
            elif st.get("pass"):
                check(OK, "selftest", f"class {st['got_class']} "
                                      f"({st['got_label']}) matches the golden "
                                      f"vector, {st['infer_ms']} ms")
            else:
                check(BAD, "selftest",
                      f"device said {st['got_class']} ({st['got_label']}) "
                      f"score {st['got_score']}, golden vector says "
                      f"{st['want_class']} ({st['want_label']}) "
                      f"{st['want_score']}")
                if info["inference"].get("last_ms", 0) > 400:
                    check(WARN, "inference speed",
                          f"{info['inference']['last_ms']} ms per invoke. "
                          "Expected tens of ms with esp-nn; several seconds "
                          "suggests esp-nn is not linked in.")
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            check(WARN, "selftest", f"could not run: {exc}")
    else:
        check(WARN, "model", "none loaded - expected before pc/train.py. "
                             "Camera and MHI pipeline still work without it.")

    # --- verdict ----------------------------------------------------------
    bad = [r for r in _results if r[0] == BAD]
    warn = [r for r in _results if r[0] == WARN]
    print(f"\n{'=' * 60}")
    print(f"{len(_results)} checks: "
          f"{len(_results) - len(bad) - len(warn)} pass, "
          f"{len(warn)} warn, {len(bad)} fail")

    if bad:
        print("\nBLOCKING:")
        for _, name, detail in bad:
            print(f"  - {name}: {detail}")
        print("\nFix these before collecting data. See docs/TROUBLESHOOTING.md")
        return 1

    if warn:
        print("\nNon-blocking, but worth knowing:")
        for _, name, detail in warn:
            print(f"  - {name}: {detail}")

    print("\nReady for pc/collect.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
