/*
 * infer.h - TensorFlow Lite Micro gesture classifier.
 *
 * Compiles to a no-op stub when src/model.h is absent, so the firmware builds
 * and runs the camera + MHI pipeline before any model has been trained
 * (milestone M2 in docs/ARCHITECTURE.md).
 */
#pragma once

#include <stdint.h>
#include <stdbool.h>

bool infer_init(void);

/* True once a real model is loaded and AllocateTensors() succeeded. */
bool infer_available(void);

/* Run inference on a 64x64 uint8 MHI. Returns class id, or -1 on failure. */
int infer_run(const uint8_t *mhi, float *out_score);

/* Softmax-normalised scores for every class. Returns class count, or 0. */
int infer_scores(const uint8_t *mhi, float *scores, int max_classes);

/* Milliseconds taken by the most recent infer_run(). */
uint32_t infer_last_ms(void);

/* Name of class `id`, or "?" if out of range. */
const char *infer_label(int id);

/*
 * On-device self-test. Runs the model against a real recorded MHI whose
 * answer was computed on the PC (src/golden_vector.h) and reports whether they
 * agree.
 *
 * This exists because "the model misclassifies" and "the pipeline is broken"
 * look identical from the dashboard, and telling them apart is the difference
 * between a minute and a week. A disagreement here means preprocessing or
 * quantisation is wrong, not that the model is bad.
 */
typedef struct {
    bool ran;         /* a golden vector and a model were both present */
    bool class_ok;
    bool score_ok;    /* within 0.05 of the expected score */
    int got_class;
    int want_class;
    float got_score;
    float want_score;
} infer_selftest_t;

bool infer_selftest(infer_selftest_t *out);
