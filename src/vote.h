/*
 * vote.h - temporal decoder over per-frame model outputs.
 *
 * The CNN is run every capture tick, so one gesture produces many overlapping
 * predictions. A raw argmax flickers between classes. This holds a short
 * window of recent (class, score) pairs and only emits a sign when one class
 * wins a majority of the window AND clears a confidence floor. A cooldown
 * prevents the same sign re-firing.
 */
#pragma once

#include <stdint.h>
#include <stdbool.h>
#include "app_config.h"

#define VOTE_MAX_CLASSES 16

typedef struct {
    uint8_t window;
    uint8_t min_wins;
    float min_score;
    uint8_t cooldown;

    uint8_t cls[VOTE_MAX_CLASSES];
    float score[VOTE_MAX_CLASSES];
    uint8_t head;   /* next write index */
    uint8_t count;  /* valid entries, saturates at `window` */

    uint8_t cooldown_left;
    int8_t last_emitted; /* -1 = nothing emitted yet */
} vote_ctx_t;

void vote_init(vote_ctx_t *ctx);

/* Clear the window and cooldown, keeping configured thresholds. */
void vote_reset(vote_ctx_t *ctx);

/* Push one frame result. Returns true (and fills out_cls and out_score) when a
 * sign should be emitted this tick. */
bool vote_push(vote_ctx_t *ctx, int class_id, float score,
               int *out_cls, float *out_score);

/* Class currently leading the window, for the dashboard. -1 if window empty. */
int vote_leader(const vote_ctx_t *ctx);
