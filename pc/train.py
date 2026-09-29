#!/usr/bin/env python3
"""
train.py - build a dataset from recorded sequences, train the gesture CNN,
export it for TensorFlow Lite Micro, and emit the C headers.

    python3 pc/train.py --data data/raw
    python3 pc/train.py --data data/raw --quantize        # int8, later
    python3 pc/train.py --data data/raw --selftest-only   # re-verify export

Pipeline:
  1. read data/raw/label<N>/*.bin  (raw grayscale sequences from /api/record)
  2. derive one MHI per sequence via pc/mhi.py   <- the source of truth
  3. drop junk samples (no motion)
  4. stratified train/val/test split
  5. train a small CNN, with augmentation
  6. export TFLite (float32 by default), write src/model.h + src/labels.h
  7. write src/golden_vector.h so the device can self-check on boot

Why float32 first: int8 quantised TFLite models on this pipeline tend to
collapse to a single class (espressif/esp-tflite-micro#108). Prove accuracy in
float32, then try --quantize. Espressif's own reference gesture model ships
float32 too.

Time-reversal is deliberately NOT used as augmentation: for an MHI, reversing a
sequence turns a left swipe into a right swipe. Augmenting with it would teach
the model to ignore exactly the signal the architecture exists to capture.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pc"))

import mhi  # noqa: E402

try:
    import cv2
except ImportError:  # rotation augmentation degrades to shift-only
    cv2 = None


# ======================================================================
# Data
# ======================================================================

def load_dataset(data_dir: Path, labels: list[str], min_energy: int,
                 verbose: bool = True):
    """Returns (X, y) where X is (N, 64, 64, 1) float32 in 0..1."""
    xs, ys, dropped = [], [], 0

    for li, name in enumerate(labels):
        d = data_dir / f"label{li}"
        if not d.is_dir():
            print(f"  label {li} ({name}): no directory, skipping")
            continue

        files = sorted(d.glob("*.bin"))
        kept = 0
        for f in files:
            raw = f.read_bytes()
            if len(raw) != mhi.MHI_SEQ_LEN * mhi.MHI_PIXELS:
                dropped += 1
                continue
            seq = np.frombuffer(raw, dtype=np.uint8).reshape(
                mhi.MHI_SEQ_LEN, mhi.MHI_H, mhi.MHI_W)

            # Reject recordings where the operator never moved: the MHI would
            # be blank and the label meaningless.
            energy = mhi.mhi_energy(seq)
            if energy.size == 0 or int(energy.max()) < min_energy:
                dropped += 1
                continue

            img = mhi.sequence_to_mhi(seq)
            xs.append(img.astype(np.float32) / 255.0)
            ys.append(li)
            kept += 1

        if verbose:
            print(f"  label {li} ({name}): {kept}/{len(files)} usable")

    if not xs:
        raise SystemExit(
            "no usable samples. Run pc/collect.py first.\n"
            "If samples exist but were all dropped, lower --min-energy."
        )

    X = np.stack(xs)[..., None]
    y = np.asarray(ys, dtype=np.int32)

    counts = Counter(y.tolist())
    print(f"\ndataset: {X.shape[0]} samples, {X.shape[1]}x{X.shape[2]}, "
          f"{X.shape[3]} channel, {dropped} dropped")
    print("  per class:", {labels[k]: v for k, v in sorted(counts.items())})

    # A class that vanished entirely is the dangerous case: it usually means
    # the motion filter is too aggressive for how that gesture is performed,
    # not that you forgot to record it.
    missing = [labels[i] for i in range(len(labels)) if counts.get(i, 0) == 0]
    if missing:
        print(f"\n  ERROR: no usable samples for {missing}.")
        print("  These were recorded but every sample fell below --min-energy "
              f"({min_energy}).")
        print("  A slow, deliberate gesture moves only a small number of "
              "pixels per frame and gets filtered out.")
        print(f"  Re-run with --min-energy 20, or check the recordings with:")
        print("    python3 pc/inspect_data.py --data", data_dir)
        raise SystemExit(2)

    rarest = min(counts.get(i, 0) for i in range(len(labels)))
    if rarest < 40:
        print(f"\n  WARNING: the smallest class has only {rarest} samples. "
              "Under ~100 the model will overfit and the device result will "
              "not generalise. Collect more before trusting it.")
    return X, y


def stratified_split(y: np.ndarray, seed: int = 42):
    """Deterministic 70/15/15 split, stratified by class."""
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        a, b = int(0.70 * n), int(0.85 * n)
        tr += idx[:a].tolist()
        va += idx[a:b].tolist()
        te += idx[b:].tolist()
    rng.shuffle(tr); rng.shuffle(va); rng.shuffle(te)
    return (np.array(tr), np.array(va), np.array(te))


# ======================================================================
# Augmentation
# ======================================================================

def augment(X: np.ndarray, rng: np.random.Generator, shift: int = 7,
            rot_deg: float = 12.0, gain: float = 0.25, erase: int = 9):
    """Shifts, small rotations, contrast jitter, and occlusion.

    Shift is the highest-value augmentation here: with a bolted-down camera the
    single biggest source of train/test disagreement is the hand not landing
    on exactly the same pixels twice.
    """
    n = X.shape[0]
    out = np.empty_like(X)

    for i in range(n):
        img = X[i, :, :, 0]
        angle = rng.uniform(-rot_deg, rot_deg)
        if cv2 is not None and abs(angle) > 0.5:
            h = img.shape[0]
            m = cv2.getRotationMatrix2D((h / 2, h / 2), angle, 1.0)
            img = cv2.warpAffine(img, m, (h, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        elif cv2 is None and abs(angle) > 0.5:
            angle = 0.0

        dx, dy = rng.integers(-shift, shift + 1, 2)
        img = np.roll(img, (int(dy), int(dx)), axis=(0, 1))
        # np.roll wraps; kill the wrapped edges so they read as "no motion".
        if dy > 0:
            img[:dy, :] = 0
        elif dy < 0:
            img[dy:, :] = 0
        if dx > 0:
            img[:, :dx] = 0
        elif dx < 0:
            img[:, dx:] = 0

        g = 1.0 + rng.uniform(-gain, gain)
        b = rng.uniform(-0.08, 0.08)
        img = np.clip(img * g + b, 0.0, 1.0)

        # Occlusion: a dark patch, so the model tolerates a partially hidden hand.
        if erase > 0 and rng.random() < 0.4:
            eh = rng.integers(erase // 2, erase + 1)
            ex = rng.integers(0, max(1, img.shape[0] - eh))
            ey = rng.integers(0, max(1, img.shape[1] - eh))
            img[ex:ex + eh, ey:ey + eh] = 0.0

        out[i, :, :, 0] = img
    return out


# ======================================================================
# Model
# ======================================================================

def build_model(n_classes: int, width: int):
    import tensorflow as tf
    from tensorflow import keras

    w = width
    m = keras.Sequential([
        keras.layers.Input(shape=(mhi.MHI_H, mhi.MHI_W, 1)),

        keras.layers.Conv2D(w, 3, strides=2, padding="same",
                            activation="relu"),
        keras.layers.MaxPooling2D(2),

        keras.layers.Conv2D(w * 2, 3, padding="same", activation="relu"),
        keras.layers.MaxPooling2D(2),

        # Explicit AveragePool2D instead of GlobalAveragePooling2D: it maps
        # cleanly onto TFLite's AveragePool2D, which is registered in the
        # firmware, whereas GAP may lower to a MEAN op that is not.
        keras.layers.AveragePooling2D(8),
        keras.layers.Flatten(),

        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(n_classes, activation="softmax", name="scores"),
    ], name="gesture_cnn")
    return m


def count_params(m) -> int:
    return int(sum(int(np.prod(w.shape)) for w in m.trainable_weights))


# ======================================================================
# Metrics
# ======================================================================

def report(name: str, y_true, y_pred, labels: list[str]):
    acc = float((y_true == y_pred).mean())
    print(f"\n{name}: {acc * 100:.1f}%  ({int((y_true == y_pred).sum())}"
          f"/{len(y_true)})")

    print(f"  confusion (row=actual, col=predicted)")
    print("            " + "".join(f"{i:>5}" for i in range(len(labels))))
    for i, lab in enumerate(labels):
        row = [int(((y_true == i) & (y_pred == j)).sum())
               for j in range(len(labels))]
        print(f"    {i} {lab[:8]:<8}" + "".join(f"{v:>5}" for v in row))
    return acc


# ======================================================================
# Export
# ======================================================================

def export_tflite(model, X_sample: np.ndarray, quantize: bool,
                  out_path: Path):
    import tensorflow as tf

    if not quantize:
        conv = tf.lite.TFLiteConverter.from_keras_model(model)
        conv.optimizations = []
        conv.inference_input_type = tf.float32
        conv.inference_output_type = tf.float32
        return conv.convert()

    def rep_gen():
        # A representative set drawn from the real MHI distribution. Random
        # noise makes poor calibration data and is a common cause of the
        # always-one-class int8 failure.
        for i in range(0, min(256, X_sample.shape[0]), 1):
            yield [X_sample[i:i + 1].astype(np.float32)]

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep_gen
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


def write_model_h(blob: bytes, out: Path):
    """Emit a C array. Uses xxd so the file is byte-identical to what the
    README's manual steps would produce."""
    tmp = out.with_suffix(".tflite")
    tmp.write_bytes(blob)
    h = subprocess.run(["xxd", "-i", "-n", "gesture_model_data", str(tmp)],
                       capture_output=True, text=True, check=True).stdout
    # xxd emits `unsigned char name[]` and `unsigned int name_len`. The TFLM
    # schema API wants a const pointer, and the length belongs in flash rather
    # than consuming a byte of DRAM for a compile-time constant.
    h = h.replace("unsigned char gesture_model_data[",
                  "static const unsigned char gesture_model_data[")
    h = h.replace("unsigned int gesture_model_data_len",
                  "static const unsigned int gesture_model_data_len")
    guard = ("/* Generated by pc/train.py. Do not edit by hand.\n"
             "   Source: the TFLite blob in build/model.tflite. */\n"
             "#pragma once\n\n")
    out.write_text(guard + h + "\n")
    tmp.unlink()
    return len(blob)


def write_labels_h(labels: list[str], out: Path):
    body = ", ".join(f'"{l}"' for l in labels)
    out.write_text(
        "/* Generated by pc/train.py. Do not edit by hand. */\n"
        "#pragma once\n\n"
        "/* Must match NUM_GESTURES in src/app_config.h. */\n"
        f"#define GESTURE_LABELS {{ {body} }}\n"
    )


def write_golden_h(x: np.ndarray, exp_class: int, exp_score: float,
                   out: Path):
    """One fixed input plus the expected top-1, for an on-device self-test.

    Feeding the device an input whose PC-side answer is already known turns
    'the model misclassifies' into 'the pipeline is broken', which is the
    difference between minutes and days of debugging.
    """
    data = (np.clip(x[0, :, :, 0], 0, 1) * 255).round().astype(np.uint8).ravel()
    # Chunk the list of values, not the joined string: slicing a joined string
    # cuts hex tokens in half.
    toks = [f"0x{v:02x}" for v in data]
    per = 12
    lines = ["    " + ", ".join(toks[i:i + per])
             for i in range(0, len(toks), per)]
    out.write_text(
        "/* Generated by pc/train.py. Do not edit by hand. */\n"
        "#pragma once\n\n"
        "#include <stdint.h>\n\n"
        f"#define GOLDEN_INPUT_W {mhi.MHI_W}\n"
        f"#define GOLDEN_INPUT_H {mhi.MHI_H}\n"
        f"#define GOLDEN_EXPECTED_CLASS {exp_class}\n"
        f"#define GOLDEN_EXPECTED_SCORE_MILLI {int(round(exp_score * 1000))}\n\n"
        "/* A real recorded MHI, and what the TFLite interpreter returned for "
        "it on the PC. */\n"
        f"static const uint8_t golden_input[{mhi.MHI_PIXELS}] = {{\n"
        + ",\n".join(lines) + "\n};\n"
    )


# ======================================================================
# Main
# ======================================================================

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--labels", nargs="*", default=None,
                    help="label names, in index order (default: auto-detect)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--width", type=int, default=16,
                    help="base channel count; 24-32 if accuracy is short")
    ap.add_argument("--min-energy", type=int, default=25,
                    help="drop samples whose peak per-frame motion is below "
                         "this. Keep it low: a slow gesture legitimately moves "
                         "only ~50-100 px of 4096")
    ap.add_argument("--aug-copies", type=int, default=8,
                    help="augmented copies of the training set (0 disables)")
    ap.add_argument("--aug-shift", type=int, default=7,
                    help="max random translation in pixels")
    ap.add_argument("--aug-rot", type=float, default=12.0,
                    help="max random rotation in degrees")
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--quantize", action="store_true",
                    help="export int8 (only after float32 accuracy is proven)")
    ap.add_argument("--out-model", default=str(ROOT / "src" / "model.h"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    try:
        import tensorflow as tf
    except ImportError:
        sys.exit(
            "error: tensorflow is not installed.\n"
            f"Your system Python is {sys.version.split()[0]}. TensorFlow does "
            "not support that version yet.\n"
            "  uv venv --python 3.12 .venv\n"
            "  source .venv/bin/activate\n"
            "  pip install tensorflow numpy pillow\n"
            "  python3 pc/train.py --data data/raw"
        )
    from tensorflow import keras

    keras.utils.set_random_seed(args.seed)
    data_dir = Path(args.data)

    if not data_dir.is_dir():
        sys.exit(f"error: {data_dir} not found. Run pc/collect.py first.")

    if args.labels:
        labels = list(args.labels)
    else:
        n = len([d for d in data_dir.glob("label*") if d.is_dir()])
        labels = [f"sign{i}" for i in range(n)] or ["sign0"]

    print("=" * 64)
    print("loading dataset")
    print("=" * 64)
    X, y = load_dataset(data_dir, labels, args.min_energy)

    tr, va, te = stratified_split(y, seed=args.seed)
    Xtr, ytr = X[tr], y[tr]
    Xva, yva = X[va], y[va]
    Xte, yte = X[te], y[te]
    print(f"  split: train={len(tr)} val={len(va)} test={len(te)}")

    model = build_model(len(labels), args.width)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # Augmentation is materialised up front rather than done online: with a
    # dataset this small the extra samples are the difference between a model
    # that generalises across two days of recording and one that memorised the
    # first hundred.
    rng = np.random.default_rng(args.seed)
    if args.aug_copies > 0:
        print(f"\naugmenting training set: {args.aug_copies} copies "
              f"(shift <= {args.aug_shift}px, rot <= {args.aug_rot}deg)")
        Xtr_aug = np.concatenate(
            [Xtr] + [augment(Xtr, rng, shift=args.aug_shift,
                             rot_deg=args.aug_rot) for _ in range(args.aug_copies)],
            axis=0)
        ytr_aug = np.concatenate([ytr] * (args.aug_copies + 1), axis=0)
        print(f"  training set {len(Xtr)} -> {len(Xtr_aug)}")
    else:
        Xtr_aug, ytr_aug = Xtr, ytr

    print(f"\ntraining {args.epochs} epochs...")
    hist = model.fit(
        Xtr_aug, ytr_aug,
        validation_data=(Xva, yva),
        epochs=args.epochs,
        batch_size=args.batch,
        verbose=2,
        callbacks=[
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                              patience=6, min_lr=1e-5),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                         restore_best_weights=True),
        ],
    )

    def predict(Xq):
        return model.predict(Xq, verbose=0).argmax(axis=1)

    print("\n" + "=" * 64)
    print("results")
    print("=" * 64)
    a_tr = report("train", ytr, predict(Xtr), labels)
    a_va = report("val  ", yva, predict(Xva), labels)
    a_te = report("test ", yte, predict(Xte), labels)

    gap = a_tr - a_te
    if gap > 0.20:
        print(f"\n  WARNING: train-test gap is {gap * 100:.0f} points - heavy "
              "overfit. Collect more samples or lower --width before "
              "flashing. A model this overfit will not work on the device.")

    print("\n" + "=" * 64)
    print("export")
    print("=" * 64)
    try:
        blob = export_tflite(model, X, args.quantize, Path(args.out_model))
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"error: TFLite conversion failed:\n{exc}\n"
                 "If this mentions unsupported ops, simplify build_model().")

    nbytes = write_model_h(blob, Path(args.out_model))
    write_labels_h(labels, ROOT / "src" / "labels.h")
    print(f"  src/model.h   {nbytes} bytes "
          f"({'int8' if args.quantize else 'float32'})")
    print(f"  src/labels.h  {len(labels)} classes: {labels}")

    # --- golden vector from the TFLite interpreter, not Keras ------------
    # Index into the test split itself, not into the full dataset, and pick the
    # most confidently classified sample: a golden vector that sits on a
    # boundary would be a poor on-device canary.
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_content=blob)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    test_i = int(np.argmax(model.predict(Xte, verbose=0).max(axis=1)))
    in_scale, in_zp = inp["quantization"]
    interp.set_tensor(inp["index"],
                      mhi.to_input_tensor(
                          (Xte[test_i, :, :, 0] * 255.0).round().astype(np.uint8),
                          inp["dtype"], in_scale, in_zp).reshape(1, mhi.MHI_H,
                                                                 mhi.MHI_W, 1))
    interp.invoke()
    probs = mhi.dequantize_output(interp.get_tensor(out["index"])[0],
                                  out["dtype"], *out["quantization"])
    g_class = int(probs.argmax())
    g_score = float(probs[g_class])
    write_golden_h(Xte[test_i:test_i + 1], g_class, g_score,
                   ROOT / "src" / "golden_vector.h")
    print(f"  src/golden_vector.h  class={g_class} ({labels[g_class]}) "
          f"score={g_score:.3f}")

    manifest = {
        "labels": labels,
        "mhi": mhi.describe(),
        "samples": int(X.shape[0]),
        "split": {"train": len(tr), "val": len(va), "test": len(te)},
        "width": args.width,
        "params": count_params(model),
        "model_bytes": nbytes,
        "quantized": bool(args.quantize),
        "accuracy": {"train": a_tr, "val": a_va, "test": a_te},
        "epochs_run": len(hist.history["loss"]),
        "final_val_loss": float(hist.history["val_loss"][-1]),
        "golden": {"class": g_class, "score": g_score},
        "tensorflow": tf.__version__,
        "python": sys.version.split()[0],
    }
    (ROOT / "data" / "model_manifest.json").write_text(
        json.dumps(manifest, indent=2))
    print("  data/model_manifest.json")

    print("\n" + "=" * 64)
    print("next")
    print("=" * 64)
    if len(labels) != 6:
        print(f"  !! NUM_GESTURES is 6 in src/app_config.h but this model has "
              f"{len(labels)} classes. Edit it, or the firmware will read "
              "past the output tensor.")
    print("  1. check src/app_config.h NUM_GESTURES matches")
    print("  2. pio run -t upload")
    print("  3. open the dashboard, GET /api/selftest")
    if args.quantize:
        print("\n  int8 exported. If /api/selftest disagrees with the golden")
        print("  vector, that is the known int8 collapse "
              "(espressif/esp-tflite-micro#108). Re-run without --quantize.")
    else:
        print("\n  float32 exported, which is the recommended starting point.")
        print("  Try --quantize later only if inference is too slow.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
