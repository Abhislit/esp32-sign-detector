/*
 * app_config.h - compile-time tunables for the sign detector.
 *
 * Every value here MUST be mirrored in pc/mhi.py (MODULE DEFAULTS) and in
 * pc/train.py. A mismatch between firmware and training preprocessing is the
 * single most common way an on-device vision model silently fails. tools/
 * contains a parity test that fails loudly when C and Python disagree.
 */
#pragma once

/* ---- Network -------------------------------------------------------------- */
#ifndef APP_WIFI_SSID
#define APP_WIFI_SSID ""
#endif
#ifndef APP_WIFI_PASS
#define APP_WIFI_PASS ""
#endif
#define APP_WIFI_SSID_MAX 32
#define APP_WIFI_PASS_MAX 64
#define APP_AP_SSID "SignDetect"
#define APP_AP_PASS "signdetect"

/* ---- Motion History Image ------------------------------------------------- */
#define MHI_W 64
#define MHI_H 64
#define MHI_PIXELS (MHI_W * MHI_H)

/*
 * Number of consecutive 64x64 grayscale frames captured per recorded sample.
 * This is the RAW SEQUENCE length sent to the PC. The PC derives the MHI from
 * it, so changing this changes the dataset format.
 */
#define MHI_SEQ_LEN 12

/*
 * |I_t - I_{t-1}| per-pixel threshold above which a pixel counts as moving.
 * Raise it if the room lighting produces constant spurious motion.
 */
#define MHI_MOTION_THRESHOLD 26

/*
 * MHI decay numerator over 256. 225/256 = 0.879.
 * Larger = longer temporal memory (sees more of the gesture), but smears
 * consecutive gestures together. 0.88 means the history fades over roughly
 * MHI_SEQ_LEN frames.
 */
#define MHI_DECAY_NUM 225

/*
 * Normalise the MHI by its own max before inference. Handles exposure drift
 * between sessions. Set to 0 to keep absolute motion magnitude, which lets the
 * model reject no-motion frames but makes it lighting-sensitive.
 */
#define MHI_NORMALIZE 1

/* ---- Motion trigger ------------------------------------------------------- */
/*
 * A frame counts as "motion started" when at least this many of the 64x64
 * pixels changed. 40/4096 = ~1% of the frame.
 *
 * This is deliberately LOW. Measured against the capture path: a slow,
 * deliberate swipe moves only ~64 pixels between frames, so a threshold near
 * 200 (the intuitive "5% of the frame" choice) would mean a careful,
 * well-lit recording simply never triggers and every sample times out. Lower
 * is safer than higher here - the cost of a false trigger is one wasted
 * recording, the cost of a high one is a session where nothing captures.
 *
 * Tune from the dashboard: watch "motion px" while waving at the speed you
 * intend to perform the signs.
 */
#define MOTION_TRIGGER_PIXELS 40

/* ---- Model ---------------------------------------------------------------- */
/* Tensor arena. The gesture CNN needs ~60KB; this is deliberate headroom. */
#define TENSOR_ARENA_SIZE (96 * 1024)

/* Number of classes. Must match the trained model's output layer exactly. */
#ifndef NUM_GESTURES
#define NUM_GESTURES 6
#endif

/* ---- Temporal voting ------------------------------------------------------ */
/*
 * Sliding window length. Must be <= MHI_SEQ_LEN, otherwise the vote spans
 * more frames than the model can see.
 */
#define VOTE_WINDOW 4
/* A class must win at least this many of the VOTE_WINDOW slots to be emitted. */
#define VOTE_MIN_WINS 3
/* Minimum model score (0..1) for the winning class to be emitted. */
#define VOTE_MIN_SCORE 0.70f
/* Frames to ignore after emitting, to avoid repeat-firing the same sign. */
#define VOTE_COOLDOWN 8

/* ---- Dataset recording ---------------------------------------------------- */
/* Each captured sample is returned as one HTTP body of this many bytes. */
#define RECORD_SAMPLE_BYTES (MHI_SEQ_LEN * MHI_PIXELS)
