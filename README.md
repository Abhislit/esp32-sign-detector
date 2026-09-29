# ESP32-CAM Dynamic Sign-Gesture Detector

On-device recognition of dynamic hand gestures (motion signs) on an AI-Thinker
**ESP32-CAM**, with no PC involved at inference time.

The camera runs a tiny int8/float32 CNN on a **Motion History Image** - the
gesture's temporal structure encoded directly into a 64x64 image - so a single
inference classifies a whole gesture rather than a single frame.

```
OV2640 -> Motion History Image -> tiny CNN -> windowed max-vote -> sign
240MHz        64x64 gray        ~7k params    stabilises output
```

```
+---------------------------+
|   OV2640  QVGA 320x240    |
+------------+--------------+
             | box-average luma downscale
             v
+---------------------------+
|   motion mask  64x64      |  |I_t - I_{t-1}| > 26
+------------+--------------+
             |
             v
+---------------------------+
|   MHI  max(0.88*H, mask)  |  recent motion bright, old motion fades
+------------+--------------+
             |  one image = one gesture
             v
+---------------------------+
|   gesture CNN  ~7k params |
+------------+--------------+
             |  flicker between frames
             v
+---------------------------+
|   windowed max-vote       |  3 of 4 frames, score >= 0.70
+---------------------------+
             v
          "swipe_left"
```

---

## Status

| Milestone | State |
|---|---|
| M0 hardware verification | tool ready (`make check`) |
| M1 toolchain | ready (`make venv`) |
| M2 camera + MHI, no ML | firmware complete |
| M3 data collection | tool ready (`make collect`) |
| M4 training + export | tool ready, verified end to end on synthetic data |
| M5 on-device inference | firmware complete, **needs hardware to verify** |
| M6 temporal voting | firmware complete, **needs hardware to verify** |

Everything testable without a board is tested: the MHI C and Python
implementations are proven bit-identical, the int8 quantisation mapping is
verified, the vote decoder has 19 unit assertions, both target-only translation
units are syntax-checked in every header combination, and the export path
replays the golden vector correctly. What remains unverified is anything that
needs the chip itself: the PlatformIO build, the esp32-camera integration, and
the TFLM interpreter at runtime. See [Hardware required](#hardware-required).

---

## Hardware required

- **AI-Thinker ESP32-CAM** with an **OV2640** module. ~$6.
- **USB-TTL adapter** — the board has no USB port. Without this you cannot
  flash it. This is the item most likely to be missing.
- **5V / 2A supply.** Camera plus WiFi will brown out a weak USB port.
- A second device (laptop) on the same network, or a phone on the board's AP.

### Wiring for flashing

| USB-TTL | ESP32-CAM |
|---------|-----------|
| `5V` | `5V` (or `VUSB`) |
| `GND` | `GND` |
| `TX` | `U0R` |
| `RX` | `U0T` |
| `IO0` | `GND` — **only while flashing** |

Ground the `IO0` pin to enter the serial bootloader, remove it before pressing
reset, or the board will boot into the flashing stub instead of the firmware.

### PSRAM is the first thing to check

The AI-Thinker datasheet claims 4 MB of PSRAM. **Many cheap clones ship with
0 MB or 2 MB.** Without PSRAM the camera cannot allocate a framebuffer and the
dataset buffers cannot be allocated at all, so the project cannot work.

`make check` reports this in the first second. Do not collect any data until
it passes.

---

## Quick start

```bash
make venv      # Python 3.12 + TensorFlow. Your system Python 3.14 has no TF wheels.
make test      # host tests: MHI parity + vote decoder, no hardware needed
make build     # compile firmware (downloads the ESP-IDF toolchain)
make upload    # flash
make check     # M0 verification: PSRAM, camera, config parity
```

Then open the dashboard at `http://192.168.4.1` (join AP `SignDetect` /
`signdetect`) and follow the milestone sequence below.

---

## The five milestones

Do these in order. Each one is cheap and each one rules out a whole class of
failure before you spend a day on it.

### M0 — Verify the hardware

```bash
make check URL=http://192.168.4.1
```

Confirms PSRAM size, camera frames, memory headroom, and that the MHI
constants in firmware match `pc/mhi.py`. Stops with a clear error if PSRAM is
missing. **Do not skip this.**

### M1 — Toolchain

```bash
make venv
```

TensorFlow ships no wheels for Python 3.14, which is the system interpreter on
current Ubuntu. `make venv` uses `uv` to fetch a 3.12 build. Everything on the
PC side runs in that venv:

```bash
source .venv/bin/activate
```

The firmware side needs no separate ESP-IDF install — PlatformIO fetches the
toolchain on first build.

### M2 — Camera and MHI, before any model

```bash
make build && make upload && make check
```

Open the dashboard. The **motion history image** panel should light up as you
move a hand, and `motion px` should rise.

This stage exists to find lighting, framing, and threshold problems while they
are still free to fix. The firmware builds and runs without a model —
`infer.cpp` compiles to a stub when `src/model.h` is absent, and `/api/info`
reports `"available": false`.

Tune two constants in `src/app_config.h` from what you see here:

| Constant | Meaning | Symptom if wrong |
|---|---|---|
| `MHI_MOTION_THRESHOLD` | per-pixel luma change counted as motion | too low: MHI fills with noise. too high: hand never registers |
| `MOTION_TRIGGER_PIXELS` | pixels changed before recording arms | too high: slow gestures never trigger, every sample times out |

`MOTION_TRIGGER_PIXELS` is deliberately low (40). A slow, deliberate swipe
moves only about **64 pixels** per frame, so the intuitive "5% of the frame"
choice (~200) would mean careful recordings never trigger at all.

If you change either, re-run `make test` and re-sync `pc/mhi.py` — `make check`
will flag the drift.

### M3 — Record a dataset

Start with **4–6 motion gestures plus an `unknown` class.** Real sign-language
*words* need consistent hand position, speed and scale in frame, and are not a
reasonable first target on a 240 MHz dual-core. The `unknown` class is not
optional: without it the model is forced to label every movement as one of your
signs, which is how you get a device that fires constantly.

```bash
make collect URL=http://192.168.4.1 REPS=150
```

Aim for **150+ repetitions per class**, varying speed and starting position.
Consistent-looking data comes from repeating the motion, not from slowing down
between reps.

Before training, always look at what you recorded:

```bash
make inspect
```

This catches the usual problems: a class whose every sample was filtered out,
mislabelled classes that look identical, and footage that is too dark or blown
out. It also prints the mean MHI of any two classes as ASCII art, which is the
fastest way to see whether two gestures are actually distinguishable.

### M4 — Train

```bash
make train
```

Writes `src/model.h`, `src/labels.h`, `src/golden_vector.h`, and
`data/model_manifest.json`.

If you use a different number of classes, update `NUM_GESTURES` in
`src/app_config.h` — the firmware will read past the end of the output tensor
otherwise. `make verify` checks this for you.

```bash
make verify
```

Recovers the model blob and golden input from the generated C headers and
replays them, confirming the export step did not corrupt anything. This is what
makes the device-side self-test trustworthy.

**Start in float32.** Do not add `--quantize` until float32 accuracy is proven.
Quantised TFLite models on this pipeline tend to collapse to a single class
([espressif/esp-tflite-micro#108](https://github.com/espressif/esp-tflite-micro/issues/108)),
and Espressif's own reference gesture model ships float32 for the same reason.

### M5 — Inference and self-test

```bash
make build && make upload
curl http://192.168.4.1/api/selftest
```

The boot log prints the same check:

```
I (1234) sign: selftest PASS  class=1(swipe_right) score=0.997 expected 1(swipe_right) 0.997 [38ms]
```

The device is fed a real recorded MHI whose answer was computed on the PC. A
`PASS` proves the whole path — capture, luma downscale, MHI, quantisation,
interpreter — agrees with the PC. A `FAIL` means the pipeline is broken, not
that the model is bad, and that distinction is worth minutes of debugging time.

If it fails, see [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).

### M6 — Live recognition

Dashboard shows the per-class scores and the leading class. Signs are emitted
on the serial log when the vote passes:

```
I (48210) sign: SIGN swipe_left (0.914)
```

---

## Project layout

```
src/
  app_config.h      every tunable constant, in one place
  mhi.c/.h          luma downscale, motion mask, MHI, normalise
  infer.cpp/.h        TFLM interpreter; compiles to a stub without a model
  vote.c/.h         windowed max-vote temporal decoder
  main.c            camera, WiFi, HTTP server, capture task, recording
  model.h           generated by pc/train.py
  labels.h          generated by pc/train.py
  golden_vector.h   generated by pc/train.py

pc/
  mhi.py            MHI reference implementation (source of truth)
  check_device.py   M0 hardware + config verification
  collect.py        dataset recording client
  inspect_data.py   dataset sanity check
  train.py          training, TFLite export, C header generation
  parity.py         proves mhi.py and mhi.c agree

tools/
  run_tests.sh      host test suite
  run_parity.sh     MHI parity only
  test_mhi_parity.c host harness that runs the firmware's MHI code
  test_vote.c       vote decoder unit tests
  verify_export.py  replays the exported headers
  mock_device.py    fake ESP32-CAM for PC-side development
  esp_stubs/        minimal ESP-IDF and TFLM headers for host syntax checks

docs/
  ARCHITECTURE.md   why this design, and the alternatives
  TROUBLESHOOTING.md
```

Note that `src/infer.cpp` is C++, not C: `tflite::MicroInterpreter` is a class
and the TFLite Micro API has no C equivalent, so it must be a `.cpp`
translation unit. Espressif's own reference example is `app_model.cpp` for the
same reason.

---

## Testing

```bash
make test
```

Four suites, none needing hardware:

**MHI parity.** Compiles `src/mhi.c` natively, feeds it the same bytes
`pc/mhi.py` gets, and compares pixel for pixel across seven parameter
combinations.

This is the cheapest insurance in the project. Train/serve preprocessing skew
is the single most common way an on-device vision model fails: it trains at
94% on the PC, returns garbage on the device, and logs no error anywhere. The
suite also caught a genuine bug during development — the firmware's first
`mhi_step` discarded a frame as a primer, so the device and the training
pipeline were offset by one frame.

**Quantisation helpers.** Asserts the int8 input mapping is the standard
`q = real/scale + zero_point`, and demonstrates the failure it prevents. This
caught three wrong implementations at once, all of which produced a confident
but meaningless class rather than an error.

**Vote decoder.** 19 assertions covering window fill, confidence floor,
minority flicker, cooldown, ring wrap, and reset.

**Firmware syntax.** `src/main.c` and `src/infer.cpp` are the two files that
only compile on the target, so they are syntax-checked on the host against
minimal ESP-IDF and TFLM stub headers (`tools/esp_stubs/`) in all five
combinations of generated headers that can legitimately exist — including
`model.h` present with `labels.h` missing, which is what a partial training
run leaves behind. This found three real bugs: `infer.c` used C++-only TFLM
types and could not compile as C at all, the model symbol name did not match
what `train.py` emits, and `infer_available()` was defined twice.

```bash
make mock        # fake device on :8099
make check URL=http://127.0.0.1:8099
make collect URL=http://127.0.0.1:8099 REPS=40
```

`tools/mock_device.py` implements the same HTTP API with synthetic gestures, so
the entire PC pipeline can be developed before hardware exists. It does not
model camera noise, lighting change, or PSRAM behaviour.

---

## Choosing gestures that will actually work

Measured with `pc/inspect_data.py` on the synthetic mock dataset, which is
indicative of real behaviour:

| Gesture pair | MHI distance | Verdict |
|---|---|---|
| circle clockwise vs anticlockwise | 31.4 | comfortably separable |
| swipe vs circle | 49.9 | very separable |
| vertical vs swipe | 26.9 | separable |
| swipe left vs swipe right | **11.2** | marginal |

Curved and circular gestures separate well. Straight swipes in opposite
directions are the weakest case, because a uniform-speed translation produces
a fairly solid motion band and the direction gradient is subtle.

Prefer gestures that differ in **shape** over gestures that differ only in
**direction**. Two circular gestures, or a circle and a vertical, beat two
opposite swipes. `pc/inspect_data.py --compare A B` prints the mean MHI of any
two classes so you can check before recording a full dataset.

---

## HTTP API

| Endpoint | Purpose |
|---|---|
| `/` | dashboard |
| `/api/info` | PSRAM, heap, MHI config, inference state |
| `/api/state` | current class, scores, fps, motion energy |
| `/api/mhi.bmp` | current MHI as a grayscale BMP |
| `/api/frame.jpg` | camera frame as JPEG |
| `/api/stream` | MJPEG stream |
| `/api/labels` | class names |
| `/api/record?label=N` | arm and capture one sample; returns the raw sequence |
| `/api/selftest` | golden-vector check against the model |

---

## Configuration

Everything tunable is in `src/app_config.h`:

| Constant | Default | Notes |
|---|---|---|
| `MHI_W`, `MHI_H` | 64 | model input size |
| `MHI_SEQ_LEN` | 12 | frames per recorded sample |
| `MHI_MOTION_THRESHOLD` | 26 | per-pixel luma change counted as motion |
| `MHI_DECAY_NUM` | 225 | MHI decay, /256. Larger = longer memory |
| `MHI_NORMALIZE` | 1 | normalise MHI by its own peak |
| `MOTION_TRIGGER_PIXELS` | 40 | changed pixels before recording arms |
| `NUM_GESTURES` | 6 | **must match the trained model** |
| `TENSOR_ARENA_SIZE` | 96 KB | TFLM arena, internal RAM |
| `VOTE_WINDOW` | 4 | frames in the vote |
| `VOTE_MIN_WINS` | 3 | wins needed to emit |
| `VOTE_MIN_SCORE` | 0.70 | confidence floor |
| `VOTE_COOLDOWN` | 8 | frames before the same sign can re-fire |

After changing any MHI constant, update the matching value in `pc/mhi.py` and
run `make test`. `make check` reports divergence on the device.

Wi-Fi credentials are build flags, not source edits — see the comment at the
bottom of `platformio.ini`.

---

## Known limits

- **Not firmware-tested.** No ESP32-CAM was available during development, so
  the PlatformIO build itself, the esp32-camera integration, and the TFLM
  interpreter at runtime are unverified. The sources are syntax-checked and
  the algorithms and tooling are tested, so expect to fix build and bring-up
  issues at M1/M2 — not to find a broken pipeline.
- **One user, one camera position, fixed lighting.** The augmentation makes it
  tolerant of hand position and speed, not of a moved camera or a different
  room.
- **Straight opposite swipes are marginal** at 64x64 (see above).
- **Thermal.** Continuous camera plus WiFi makes the board hot. Check
  `/api/info` for headroom in a long session.
- **Not words.** 4–6 motion signs is the realistic target. See
  [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for what would change to
  attempt real sign-language words.
