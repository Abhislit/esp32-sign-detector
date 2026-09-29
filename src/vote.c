/*
 * vote.c - windowed max-vote temporal decoder. See vote.h.
 */
#include "vote.h"

#include <string.h>

void vote_init(vote_ctx_t *ctx)
{
    memset(ctx, 0, sizeof(*ctx));
    ctx->window = VOTE_WINDOW;
    ctx->min_wins = VOTE_MIN_WINS;
    ctx->min_score = VOTE_MIN_SCORE;
    ctx->cooldown = VOTE_COOLDOWN;
    ctx->last_emitted = -1;
}

void vote_reset(vote_ctx_t *ctx)
{
    const uint8_t w = ctx->window;
    const uint8_t mw = ctx->min_wins;
    const float ms = ctx->min_score;
    const uint8_t cd = ctx->cooldown;
    memset(ctx, 0, sizeof(*ctx));
    ctx->window = w;
    ctx->min_wins = mw;
    ctx->min_score = ms;
    ctx->cooldown = cd;
    ctx->last_emitted = -1;
}

bool vote_push(vote_ctx_t *ctx, int class_id, float score,
               int *out_cls, float *out_score)
{
    if (out_cls != NULL) {
        *out_cls = -1;
    }
    if (out_score != NULL) {
        *out_score = 0.0f;
    }

    if (class_id < 0 || class_id >= VOTE_MAX_CLASSES) {
        return false;
    }

    /* Sliding window: overwrite the oldest slot once full. */
    ctx->cls[ctx->head] = (uint8_t) class_id;
    ctx->score[ctx->head] = score;
    ctx->head = (uint8_t) ((ctx->head + 1u) % (ctx->window ? ctx->window : 1u));
    if (ctx->count < ctx->window) {
        ctx->count++;
    }

    if (ctx->cooldown_left > 0) {
        ctx->cooldown_left--;
    }

    /* Not enough evidence yet. */
    if (ctx->count < ctx->window) {
        return false;
    }

    /* Tally wins per class, tracking the best score seen for each. */
    uint8_t wins[VOTE_MAX_CLASSES];
    float best[VOTE_MAX_CLASSES];
    memset(wins, 0, sizeof(wins));
    memset(best, 0, sizeof(best));

    for (uint8_t i = 0; i < ctx->window; i++) {
        const uint8_t c = ctx->cls[i];
        wins[c]++;
        if (ctx->score[i] > best[c]) {
            best[c] = ctx->score[i];
        }
    }

    int leader = -1;
    uint8_t leader_wins = 0;
    for (int c = 0; c < VOTE_MAX_CLASSES; c++) {
        if (wins[c] > leader_wins) {
            leader_wins = wins[c];
            leader = c;
        }
    }

    if (leader < 0 || leader_wins < ctx->min_wins) {
        return false;
    }
    if (best[leader] < ctx->min_score) {
        return false;
    }
    if (ctx->cooldown_left > 0) {
        return false;
    }
    if (leader == ctx->last_emitted) {
        /* Same sign as last time: suppress, but let the cooldown expire so a
         * deliberate repeat is still possible. */
        return false;
    }

    ctx->last_emitted = (int8_t) leader;
    ctx->cooldown_left = ctx->cooldown;
    if (out_cls != NULL) {
        *out_cls = leader;
    }
    if (out_score != NULL) {
        *out_score = best[leader];
    }
    return true;
}

int vote_leader(const vote_ctx_t *ctx)
{
    if (ctx->count == 0) {
        return -1;
    }
    uint8_t wins[VOTE_MAX_CLASSES];
    memset(wins, 0, sizeof(wins));
    for (uint8_t i = 0; i < ctx->count; i++) {
        wins[ctx->cls[i]]++;
    }
    int leader = -1;
    uint8_t leader_wins = 0;
    for (int c = 0; c < VOTE_MAX_CLASSES; c++) {
        if (wins[c] > leader_wins) {
            leader_wins = wins[c];
            leader = c;
        }
    }
    return leader;
}
