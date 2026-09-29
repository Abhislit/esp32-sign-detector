# Troubleshooting

Ordered by how often each one actually happens, and by how much time it costs
to diagnose wrongly.

---

## Before anything else: is it the hardware?

```bash
make check URL=http://192.168.4.1
```

Ninety seconds, and it rules out the three most expensive failures: missing
PSRAM, a camera that is not actually streaming, and firmware/PC config drift.

---

## Firmware will not flash or boot

**`Failed to connect to ESP32`**
- The board is not in bootloader mode. `IO0` must be grounded, and you need a
  *reset* while it is grounded.
- Wiring: check TX/RX are crossed, and that grounds are common.
- Try a lower `upload_speed` in `platformio.ini` (already set to 460800).

**`Failed to write to target flash`**
- Power. Use a real 5V/2A supply, not a USB port.
- A long or thin jumper to `GND` is a common cause of exactly this symptom.

**Boots, then dies immediately**
- `IO0` is still grounded to `GND`. Remove it.

**Boots and reboots in a loop**
- Brownout. Check the 5V rail under load (camera plus WiFi is not light).
- `Guru Meditation` with a task trace usually means a watchdog or a bad
  pointer. `CONFIG_ESP_TASK_WDT_PANIC` is set to `n` in `sdkconfig.defaults`
  so the task watchdog logs instead of panicking.

---

## PSRAM

**`/api/info` reports `"psram_bytes": 0`**

The board has no usable PSRAM. **This is the most common hardware surprise** —
the AI-Thinker datasheet claims 4 MB and many cheap clones ship with 0 or 2 MB.

The firmware falls back to QQVGA 160x120 with a single framebuffer in internal
RAM. It will run, but accuracy will be poor and dataset capture allocates in
PSRAM so it will not work at all.

Options:
- Verify with a PSRAM test sketch. If the board genuinely has none, buy one
  that does — ESP32-CAM boards listing "PSRAM" in the product title are usually
  genuine, unmarked cheap clones usually are not.
- Consider an ESP32-S3, which has PSRAM standard and far more compute.

**PSRAM detected but the camera still will not initialise**
- `CONFIG_SPIRAM` and `CONFIG_ESP32_SPIRAM_SUPPORT` must both be set; they are
  in `sdkconfig.defaults`.
- A corrupted `sdkconfig` overrides `sdkconfig.defaults`. Delete `sdkconfig`
  and rebuild.

---

## The MHI is blank or full of noise

Look at the **motion history image** panel on the dashboard and the
`motion px` readout while waving.

| Symptom | Cause | Fix |
|---|---|---|
| MHI stays black | threshold too high, or too little light | lower `MHI_MOTION_THRESHOLD`; add light |
| MHI fully white, constant | threshold too low, or sensor noise / auto-exposure hunting | raise `MHI_MOTION_THRESHOLD`; improve lighting |
| Flickering speckle even at rest | auto-exposure or auto-gain chasing | more even lighting; accept a slightly higher threshold |
| Motion visible but weak | hand is small in frame | move the camera closer, or crop to the region of interest |

**The MHI fills up whenever the exposure changes**
The driver runs the OV2640's auto-exposure, and every AE adjustment changes
every pixel, which reads as full-frame motion. Either use steadier lighting or
fix the exposure. `camera_init()` deliberately does *not* poke AE registers
because that code is camera-agnostic and fragile; it logs a comment explaining
the trade-off. This is the single biggest source of background motion.

---

## Recording always times out

`collect.py` reports `timeout: no motion detected`.

The trigger never fired, so `motion px` never reached
`MOTION_TRIGGER_PIXELS`.

Watch `motion px` on the dashboard while performing the gesture **at the speed
you intend to use**, and compare against `/api/info`'s `trigger_pixels`.

**This bites specifically with slow gestures.** A slow, deliberate swipe moves
only about 64 pixels per frame. The default trigger is 40, which catches it. If
you raised it to 200 on the intuition that "5% of the frame sounds right", slow
gestures will never trigger and every sample will time out.

---

## Recording succeeds but training drops samples

Run:

```bash
make inspect
```

It reports per-class counts, peak motion energy, and pairwise class distances.

**`every sample below --min-energy`** — the filter is too aggressive for how
you perform that gesture. This is the same slow-gesture problem, in the
training step instead of the capture step. Re-run with `--min-energy 20`, or
collect with more lighting contrast.

**`labels A and B look nearly identical`** — either the two gestures really are
too similar at 64x64, or one was mislabelled during collection. Use
`--compare A B` to print both mean MHIs as ASCII art and judge.

**A class is missing entirely** — `train.py` now exits with an explicit error
rather than silently training on the classes that survived. Use
`pc/inspect_data.py` to find out why.

---

## Training accuracy is high, device accuracy is terrible

This is the classic failure mode, and it is worth working through in order.

### 1. Run the golden-vector self-test first

```bash
curl http://192.168.4.1/api/selftest
```

**If this fails, stop.** The model is being fed something other than what it
was trained on, and every other diagnosis is downstream of that.

`infer.cpp` prints the same result at boot. Re-check on the PC side too:

```bash
make verify
```

### 2. If the self-test passes but live recognition fails

The preprocessing is correct, so the problem is the *data*, not the pipeline:

- **Overfitting.** `train.py` prints train/test accuracy and warns if the gap
  exceeds 20 points. With fewer than ~100 samples per class this is expected.
  Collect more — this is usually the real answer.
- **Lighting or position drift.** The model saw one lighting condition. Train
  accuracy is high because the training MHIs all look alike.
- **Performance drift.** You trained on the gestures you performed carefully,
  and are now performing them casually. Collect at the speed you actually use.
- **A gesture pair that is not separable.** `make inspect` reports pairwise
  distances; below about 12 is genuinely hard at 64x64.

### 3. If the self-test fails

| Cause | Check |
|---|---|
| int8 quantisation collapsed | re-export without `--quantize`; see below |
| Preprocessing drift | `make test` — C and Python MHIs must be bit-identical |
| Model/input mismatch | `make verify` checks `NUM_GESTURES` and the golden vector |
| Wrong model flashed | `make verify` then re-flash |

---

## int8 quantisation returns one class for everything

The model always predicts the same label, regardless of input. This is a known
failure mode, not your bug:
[espressif/esp-tflite-micro#108](https://github.com/espressif/esp-tflite-micro/issues/108).

In order:

1. **Go back to float32** and confirm accuracy is good. This is the
   recommended path; Espressif's own reference gesture model ships float32.
2. If you must have int8, use `ReLU` or `ReLU6`, never `ELU` — some TFLM
   kernels mishandle it.
3. Ensure the representative dataset is drawn from **real MHIs**.
   `export_tflite` samples 256 real training images; random noise calibrates
   badly and is a common cause.
4. Verify with the golden vector. A quantisation collapse shows up there
   immediately and unambiguously, which is precisely why the self-test exists.

---

## Inference is very slow

Read `infer_ms` from `/api/state`.

If it is **hundreds of milliseconds to seconds**, esp-nn is probably not
linked in. On this class of chip esp-nn is worth roughly **10x**
(Espressif's 250 KB int8 person-detection model: 4084 ms without, 380 ms
with). Verify the dependency resolved in `idf_component.yml`:

```yaml
dependencies:
  espressif/esp-tflite-micro: "^1.3.5"
```

Then, if you need more speed:
- Export with `--quantize`, but see the section above first.
- Reduce `--width` in `train.py` (16 → 12 or 8) and retrain.
- Lower the camera framesize; the 64x64 MHI does not benefit from QVGA.
- Move the tensor arena out of internal RAM only as a last resort — it saves
  RAM but is slower.

If it is **tens of milliseconds**, you are where you expected to be.

---

## `AllocateTensors` failed

```
AllocateTensors failed - arena too small? Current: 98304 bytes
```

Raise `TENSOR_ARENA_SIZE` in `src/app_config.h`. The printed number is the
current size. Check free internal heap at `/api/info` first — the arena is in
internal SRAM, and the camera buffers have already taken PSRAM.

---

## The dashboard will not load

- Confirm you are on the same network. In AP mode join `SignDetect` /
  `signdetect` and use `http://192.168.4.1`.
- In STA mode, watch the serial log for `got IP ...`.
- `/api/info` is the lightest endpoint to test connectivity with.
- The board is hot after long sessions and the thermal throttling can drop
  WiFi. Check `/api/info` still responds.

---

## `make test` fails

**MHI parity fails** — `src/mhi.c` and `pc/mhi.py` have diverged. Do not train
a model until this passes; the device would misclassify with nothing in the
logs to explain it. Compare `src/app_config.h` against the constants at the top
of `pc/mhi.py`, and check the `mhi_prime` / `mhi_step` sequencing matches on
both sides.

**Vote tests fail** — `vote.c` changed. The assertions in
`tools/test_vote.c` describe intended behaviour; treat a failure as a real
regression, not a stale test.

---

## TensorFlow will not install

```
error: tensorflow is not installed
Your system Python is 3.14.x. TensorFlow does not support that version yet.
```

TF 2.21 supports Python 3.10–3.13. Do not install it system-wide:

```bash
make venv
source .venv/bin/activate
```

This uses `uv` to fetch a 3.12 interpreter. If you would rather use
`python3.12 -m venv .venv`, install `tensorflow-cpu numpy pillow
opencv-python-headless` into it manually — the names the code needs are the
same.

---

## Still stuck

1. `make check` — capture the full output.
2. `curl /api/info` and `curl /api/selftest`.
3. `make test` — must be green.
4. The serial log at 115200, from boot.
5. `make inspect` — if a dataset exists.

Those five together distinguish "the hardware is wrong", "the firmware is
wrong", "the preprocessing is wrong", and "the data is wrong", which are the
four things that can be wrong.
