/*
 * test_vote.c - host unit tests for the windowed max-vote temporal decoder.
 *
 * The vote stage is what turns a flickering per-frame argmax into a stable
 * sign, and it is the stage most likely to be quietly wrong (an off-by-one in
 * the ring index produces plausible-looking output that never quite settles).
 * It has no Python twin, so these assertions are the only guard.
 *
 * Built and run by tools/run_tests.sh.
 */
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "../src/vote.h"

static int g_fail = 0;
static int g_pass = 0;

static void expect(int cond, const char *what)
{
    if (cond) {
        g_pass++;
    } else {
        g_fail++;
        printf("    FAIL: %s\n", what);
    }
}

static int push(vote_ctx_t *v, int cls, float score)
{
    int c = -1;
    float s = 0.0f;
    const bool fired = vote_push(v, cls, score, &c, &s);
    (void) s;
    return fired ? c : -1000; /* -1000 means "no emission" */
}

int main(void)
{
    printf("  vote: no emission before the window fills\n");
    {
        vote_ctx_t v;
        vote_init(&v);
        expect(push(&v, 1, 0.9f) == -1000, "1 of 4 frames must not emit");
        expect(push(&v, 1, 0.9f) == -1000, "2 of 4 frames must not emit");
        expect(push(&v, 1, 0.9f) == -1000, "3 of 4 frames must not emit");
        expect(push(&v, 1, 0.9f) == 1, "4th frame completes a unanimous window");
    }

    printf("  vote: confidence floor is enforced\n");
    {
        vote_ctx_t v;
        vote_init(&v);
        for (int i = 0; i < 3; i++) {
            push(&v, 2, 0.40f);
        }
        expect(push(&v, 2, 0.40f) == -1000,
               "unanimous but low-score window must not emit");
    }

    printf("  vote: minority flicker does not win\n");
    {
        vote_ctx_t v;
        vote_init(&v);
        push(&v, 1, 0.95f);
        push(&v, 3, 0.95f);
        push(&v, 1, 0.95f);
        expect(push(&v, 1, 0.95f) == 1,
               "class 1 holds 3 of 4 and should emit, not the 1-frame class 3");
    }

    printf("  vote: cooldown suppresses an immediate repeat\n");
    {
        vote_ctx_t v;
        vote_init(&v);
        push(&v, 4, 0.99f);
        push(&v, 4, 0.99f);
        push(&v, 4, 0.99f);
        expect(push(&v, 4, 0.99f) == 4, "first emission goes through");
        for (int i = 0; i < VOTE_COOLDOWN; i++) {
            push(&v, 4, 0.99f);
        }
        expect(push(&v, 4, 0.99f) == -1000,
               "same sign inside the cooldown must not re-fire");
    }

    printf("  vote: a different sign can fire after the cooldown\n");
    {
        vote_ctx_t v;
        vote_init(&v);
        push(&v, 1, 0.99f);
        push(&v, 1, 0.99f);
        push(&v, 1, 0.99f);
        push(&v, 1, 0.99f); /* emits 1, sets cooldown */

        /* Run the cooldown out on class 1, then switch to 2. */
        for (int i = 0; i < VOTE_COOLDOWN; i++) {
            push(&v, 1, 0.99f);
        }
        int got = -1000;
        for (int i = 0; i < 6; i++) {
            got = push(&v, 2, 0.99f);
            if (got != -1000) {
                break;
            }
        }
        expect(got == 2, "a distinct sign should be emitted once cooldown ends");
    }

    printf("  vote: ring wraps without losing ordering\n");
    {
        vote_ctx_t v;
        vote_init(&v);
        int emitted = 0;
        /* 200 frames alternating 1,1,1,2,2,2: a sustained 1 should win
         * exactly at the window boundary, and 2 exactly one window later. */
        for (int i = 0; i < 200; i++) {
            const int cls = ((i / 3) % 2) ? 2 : 1;
            const int g = push(&v, cls, 0.99f);
            if (g != -1000) {
                emitted++;
            }
        }
        expect(emitted > 5, "sustained alternation should keep emitting");
    }

    printf("  vote: leader tracks the window\n");
    {
        vote_ctx_t v;
        vote_init(&v);
        push(&v, 0, 0.9f);
        expect(vote_leader(&v) == 0, "leader after one frame of class 0");
        push(&v, 1, 0.9f);
        expect(vote_leader(&v) == 0, "leader stays on the earlier class 0");
        push(&v, 1, 0.9f);
        expect(vote_leader(&v) == 1, "leader flips once class 1 has the majority");
    }

    printf("  vote: out-of-range class is ignored\n");
    {
        vote_ctx_t v;
        vote_init(&v);
        expect(push(&v, 99, 0.99f) == -1000, "class 99 is not a valid class");
        expect(push(&v, -1, 0.99f) == -1000, "class -1 is not a valid class");
    }

    printf("  vote: reset clears history but keeps thresholds\n");
    {
        vote_ctx_t v;
        vote_init(&v);
        const float thresh = v.min_score;
        for (int i = 0; i < 4; i++) {
            push(&v, 5, 0.99f);
        }
        vote_reset(&v);
        expect(v.count == 0, "reset empties the window");
        expect(v.last_emitted == -1, "reset clears the repeat guard");
        expect(fabsf(v.min_score - thresh) < 1e-6f, "reset keeps min_score");
        expect(push(&v, 5, 0.99f) == -1000,
               "after reset the window must refill before emitting");
    }

    printf("\n  %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
