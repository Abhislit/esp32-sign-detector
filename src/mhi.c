/*
 * mhi.c - Motion History Image preprocessing. See mhi.h for the rationale.
 *
 * This file is compiled twice: once for the ESP32, and once natively by
 * tools/test_mhi_parity.c so its output can be diffed against pc/mhi.py.
 * Keep it free of ESP-IDF headers and of any floating point so the host build
 * and the device build are bit-identical.
 */
#include "mhi.h"

#include <string.h>

/* BT.601 luma weights scaled to 256. Sum is 256, so no clamp is needed
 * for 5:6:5 input (worst case yields 250). */
uint8_t mhi_rgb565_luma(uint16_t p)
{
    const uint32_t r = (uint32_t) ((p >> 11) & 0x1Fu);
    const uint32_t g = (uint32_t) ((p >> 5) & 0x3Fu);
    const uint32_t b = (uint32_t) (p & 0x1Fu);
    /* Re-expand 5:6:5 to 8:8:8 then weight. */
    return (uint8_t) ((r * 8u * 77u + g * 4u * 150u + b * 8u * 29u) >> 8);
}

void mhi_init(mhi_ctx_t *ctx)
{
    memset(ctx, 0, sizeof(*ctx));
    ctx->motion_threshold = MHI_MOTION_THRESHOLD;
    ctx->decay_num = MHI_DECAY_NUM;
    ctx->primed = false;
}

void mhi_reset(mhi_ctx_t *ctx)
{
    memset(ctx->prev, 0, sizeof(ctx->prev));
    memset(ctx->mhi, 0, sizeof(ctx->mhi));
    ctx->motion_energy = 0;
    ctx->primed = false;
}

void mhi_prime(mhi_ctx_t *ctx, const uint8_t *frame)
{
    memcpy(ctx->prev, frame, MHI_PIXELS);
    memset(ctx->mhi, 0, MHI_PIXELS);
    ctx->motion_energy = 0;
    ctx->primed = true;
}

bool mhi_step(mhi_ctx_t *ctx, const uint8_t *cur, uint8_t *out_mhi)
{
    if (!ctx->primed) {
        /* Defensive: mhi_prime() was skipped. Treat this frame as the primer
         * rather than reporting motion against a garbage prev buffer. */
        mhi_prime(ctx, cur);
        if (out_mhi != NULL) {
            memset(out_mhi, 0, MHI_PIXELS);
        }
        return false;
    }

    uint32_t energy = 0;
    for (int i = 0; i < MHI_PIXELS; i++) {
        int d = (int) cur[i] - (int) ctx->prev[i];
        if (d < 0) {
            d = -d;
        }
        /* Decay, then let fresh motion overwrite. Matches pc/mhi.py. */
        uint8_t v = (uint8_t) (((uint32_t) ctx->mhi[i] * ctx->decay_num) >> 8);
        if (d > (int) ctx->motion_threshold) {
            v = 255u;
            energy++;
        }
        ctx->mhi[i] = v;
        if (out_mhi != NULL) {
            out_mhi[i] = v;
        }
    }

    memcpy(ctx->prev, cur, MHI_PIXELS);
    ctx->motion_energy = energy;
    return true;
}

uint8_t mhi_normalize(const uint8_t *mhi, uint8_t *out)
{
    uint8_t mx = 0;
    for (int i = 0; i < MHI_PIXELS; i++) {
        if (mhi[i] > mx) {
            mx = mhi[i];
        }
    }
    if (mx == 0) {
        memset(out, 0, MHI_PIXELS);
        return 0;
    }
    for (int i = 0; i < MHI_PIXELS; i++) {
        out[i] = (uint8_t) (((uint32_t) mhi[i] * 255u) / mx);
    }
    return mx;
}

void mhi_step_normalized(mhi_ctx_t *ctx, const uint8_t *cur, uint8_t *out)
{
    uint8_t raw[MHI_PIXELS];
    mhi_step(ctx, cur, raw);
#if MHI_NORMALIZE
    mhi_normalize(raw, out);
#else
    memcpy(out, raw, MHI_PIXELS);
#endif
}
void mhi_luma_downscale(const uint8_t *rgb565, uint16_t w, uint16_t h,
                        uint8_t *out, uint16_t out_w, uint16_t out_h)
{
    for (uint16_t oy = 0; oy < out_h; oy++) {
        const uint16_t y0 = (uint16_t) ((uint32_t) oy * h / out_h);
        uint16_t y1 = (uint16_t) ((uint32_t) (oy + 1u) * h / out_h);
        if (y1 <= y0) {
            y1 = (uint16_t) (y0 + 1u);
        }
        for (uint16_t ox = 0; ox < out_w; ox++) {
            const uint16_t x0 = (uint16_t) ((uint32_t) ox * w / out_w);
            uint16_t x1 = (uint16_t) ((uint32_t) (ox + 1u) * w / out_w);
            if (x1 <= x0) {
                x1 = (uint16_t) (x0 + 1u);
            }

            uint32_t acc = 0;
            uint32_t n = 0;
            for (uint16_t y = y0; y < y1; y++) {
                const uint8_t *row = rgb565 + (size_t) y * w * 2u;
                for (uint16_t x = x0; x < x1; x++) {
                    uint16_t px;
                    memcpy(&px, row + (size_t) x * 2u, sizeof(px));
                    acc += mhi_rgb565_luma(px);
                    n++;
                }
            }
            out[(size_t) oy * out_w + ox] = (uint8_t) (n ? (acc / n) : 0);
        }
    }
}
