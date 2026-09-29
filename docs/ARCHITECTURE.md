# Architecture

Why this design, and what the alternatives would have cost.

---

## The core problem

Dynamic sign language is a *temporal* problem: the meaning is in how the hand
moves, not where it is. A 240 MHz dual-core ESP32 with 520 KB of SRAM and a
weak OV2640 cannot afford a per-frame vision model plus a sequence model on
top. Two obvious approaches both fail on budget.

### Approach A — per-frame CNN + temporal model

Landmark a hand per frame, classify each frame, then run an LSTM/TCN over the
sequence.

- Landmark extraction needs a model ~10 MB. Nowhere near the ESP32's 4 MB
  flash, and the landmark network is far heavier than a gesture classifier.
- Even with landmarks, you still owe a temporal model.
- Accurate only with good hand detection, which is its own research problem.

### Approach B — a big frame classifier with video input

- A model big enough to consume 12+ frames will not fit the compute budget.
- Espressif's own 250 KB int8 person-detection model takes **380 ms** on an
  ESP32 with esp-nn (4084 ms without). That sets the ceiling on how much you
  can afford per invoke.

### Approach C — hand landmarks offloaded to a PC

Stream frames to a laptop, run MediaPipe, send landmarks back, classify
on-device. This is a legitimate and much more capable design.

It is not what this project does, because it removes the property that makes an
embedded gesture detector interesting: no network, no PC, inference where the
sensor is. It is also worth knowing that the ESP32-CAM still has to do the
capture, so the board does not go away.

---

## The chosen design: Motion History Image

**Encode the temporal dimension into the image, then use a single spatial CNN.**

A Motion History Image is a 64x64 image where each pixel records *when* it last
moved:

```
H(x, y, t) = max( decay * H(x, y, t-1),  255 if |I_t - I_{t-1}| > threshold )
```

Recent motion is white, old motion fades toward black. The decay gradient is a
clock. One image contains the entire gesture.

### Why this works

1. **The temporal model is free.** There is no recurrent layer. A plain 2D CNN
   sees the whole gesture in one invoke.

2. **Direction survives.** A motion-energy image — a binary mask of "did this
   move at all" accumulated over time — throws direction away, so left and
   right swipes become identical. The MHI keeps the gradient, so they remain
   distinguishable. This is the entire reason for using an MHI over the simpler
   alternative.

3. **It is cheap.** A 64x64 buffer is 4 KB. Per frame: one 16x16 luma
   downscale, then 4096 subtract/abs-compare/multiply operations. On a
   240 MHz core that is well under a millisecond, and it runs on every
   captured frame.

4. **It inverts the usual data requirement.** The expensive part (a vision
   model) is small because motion already discards most of the image. The
   expensive part becomes *collecting data*, which is honest: on a board this
   constrained, your own gesture data is the thing you do not have.

### The honest cost

The MHI discards appearance. It knows *where* and *when* something moved, not
what it looked like. So:

- Two signs with similar trajectories but different handshapes are hard to
  separate. This is why the vocabulary starts at 4–6 and why curved gestures
  are preferred over opposite swipes.
- A gesture performed from a very different starting position produces a
  shifted MHI. Handled with translation augmentation, not architecturally.
- Sign-language *words* involve handshape changes that the MHI will only see
  weakly. See "Going further" below.

---

## Why the MHI is computed on the PC during training

The recorded samples are **raw 12-frame grayscale sequences**, not MHIs.

The MHI is then derived on the PC by `pc/mhi.py`, which is a literal port of
`src/mhi.c`.

This is deliberate. Deriving the MHI offline means:

- The dataset survives changes to the MHI constants. Tune the decay or
  threshold, retrain, and reuse every recording you ever made.
- Preprocessing exists in one place, so there is far less room for the training
  and inference pipelines to disagree.
- It is far cheaper to iterate — no reflash per parameter change.

The cost is that the two implementations must stay in sync, which is a
guarantee made by proof rather than by discipline:

```bash
make test    # compiles src/mhi.c natively, diffs against pc/mhi.py
```

This is not theoretical. The parity suite caught a real bug during
development: the firmware's first `mhi_step` consumed a frame as a primer,
while the PC implementation used the recorded frame 0 as the reference. The two
were offset by one frame, which would have produced a model that trained at
>95% and misclassified on the device with nothing in the logs. The fix was an
explicit `mhi_prime()` on both sides.

---

## Model

```
Input 64x64x1
  Conv2D(16, 3x3, stride 2) + ReLU   -> 32x32x16     147k MAC
  MaxPool2D(2)                       -> 16x16x16
  Conv2D(32, 3x3)          + ReLU    -> 16x16x32     1.18M MAC
  MaxPool2D(2)                       ->  8x8x32
  AveragePooling2D(8)                ->  1x1x32      (global average pool)
  Flatten                           -> 32
  Dropout(0.3)
  Dense(64) + ReLU
  Dense(6, softmax)
```

About **7,300 parameters**, ~1.3 MMAC per invoke, **~34 KB** as a float32
TFLite model. Reference point: Espressif's 250 KB int8 person-detection model
takes 380 ms on an ESP32 with esp-nn. This model is roughly 20x smaller, so tens
of milliseconds is the expected range. **Measure it** — `/api/state` reports
`infer_ms` and the boot log prints it.

`AveragePooling2D(8)` is written out rather than using
`GlobalAveragePooling2D` because it lowers cleanly to TFLite's
`AveragePool2D`, which is registered in the firmware's op resolver, whereas
GAP can lower to a `MEAN` op that is not.

### float32 first

int8 quantisation is available via `--quantize` and `infer.cpp` handles both
input types. It is not the default, for a specific reason: quantised models on
this kind of pipeline tend to collapse to always predicting one class, and the
canonical report is
[espressif/esp-tflite-micro#108](https://github.com/espressif/esp-tflite-micro/issues/108).
Espressif's own reference gesture model also ships float32.

Prove accuracy in float32 first. If the measured `infer_ms` is too slow, then
try int8 — and use the golden-vector self-test to tell a genuine speedup from
a quantisation collapse.

---

## Temporal voting

The CNN runs every captured frame, so one gesture produces many overlapping
predictions and a raw argmax flickers. `vote.c` holds the last `VOTE_WINDOW`
(4) results and emits a sign only when one class wins at least `VOTE_MIN_WINS`
(3) of them and clears `VOTE_MIN_SCORE` (0.70). A `VOTE_COOLDOWN` of 8 frames
prevents immediate re-firing, while still allowing a deliberate repeat.

This mirrors the approach in
[Espressif's April 2026 gesture-recognition post](https://developer.espressif.com/blog/2026/04/gesture-recognition-based-on-tflite/):
motion-triggered capture, a fixed window, a max-score vote, and a mandatory
`unknown` class.

The `unknown` class is not optional. Without it the softmax is forced to place
every movement into one of your signs, so the device fires constantly on
ordinary hand movement. The Espressif reference uses the same trick.

---

## The `unknown` class and negative examples

Training data should include:

- Deliberate random hand movement
- Hands at rest that were recorded by a false trigger
- Gestures performed *slowly* or *fast* beyond the training distribution

A single `unknown` class index handles all of these. It does not need to be
clean — it needs to cover whatever the model will actually see in the world.

---

## Task layout

```
core 1, prio 5   capture loop
core 0           WiFi stack, HTTP server
```

`esp_camera_fb_get()` blocks until the next frame, so the **camera is the
clock**. The MHI is stepped on every captured frame, which keeps its temporal
decay correct. Inference runs *after* the MHI, so if inference is slower than
the camera, frames are simply missed — the MHI still advances one step per
captured frame and remains coherent.

The alternative, running inference on its own schedule, would decouple the two
and silently corrupt the temporal encoding.

The dataset ring buffer lives in PSRAM. The TFLM arena is deliberately in
internal RAM: the camera buffers are already in PSRAM, which leaves about
400 KB of internal SRAM, and TFLM tensor access is heavily random, so a PSRAM
arena would roughly double inference time for a model this size.

---

## Going further

### To attempt real sign-language words

Words involve handshape transitions, not just trajectories, and the MHI sees
handshape changes only weakly. Options, in increasing order of effort:

1. **Larger input.** 96x96 or 128x128 preserves fine handshape detail. Costs
   roughly 2–4x the compute. Fits PSRAM; the flash budget is not the
   constraint.
2. **Two-channel input.** Feed the MHI and the current downscaled frame as two
   channels, so the model gets trajectory *and* handshape. Cheap and
   architectural — the rest of the pipeline is unchanged.
3. **A real temporal head.** Stack 3–4 MHI frames as channels instead of one,
   giving the CNN a short temporal window. Still a single invoke.
4. **Landmarks.** The accurate answer, and the one that needs a much faster
   chip.

### If accuracy plateaus below ~90%

The honest answer is usually a faster chip, not more tuning. An **ESP32-S3**
has vector instructions and gives roughly 6x the throughput of a plain ESP32 on
convolution, plus more PSRAM for a larger model. The C in this project is
written against the ESP-IDF camera and TFLM APIs, both of which the S3 supports,
so it is a board swap plus a model retrain.

That is worth knowing at M5 rather than discovering it after a week of tuning
on the wrong hardware.
