/*
 * mhi.h - Motion History Image preprocessing.
 *
 * A single small CNN sees an entire dynamic gesture at once, because the
 * temporal dimension is encoded INTO the image: recent motion is bright,
 * old motion fades toward black. Swipe-left and swipe-right produce different
 * images because the gradient direction differs, which a plain motion-energy
 * image would throw away.
 *
 * Pipeline per frame:
 *   RGB565 frame -> box-average luma downscale -> 64x64 gray
 *                -> motion mask |I_t - I_{t-1}| > threshold
 *                -> MHI = max(MHI * decay, mask ? 255 : 0)
 *                -> optional max-normalise
 *
 * The identical algorithm lives in pc/mhi.py. Keep the two in lockstep.
 */
#pragma once

#include <stdint.h>
#include <stdbool.h>
#include "app_config.h"

typedef struct {
    uint8_t prev[MHI_PIXELS];
    uint8_t mhi[MHI_PIXELS];
    uint8_t motion_threshold;
    uint8_t decay_num;
    bool primed;
    uint32_t motion_energy; /* pixels above threshold in the newest frame */
} mhi_ctx_t;

/* One grayscale 64x64 frame. */
typedef uint8_t mhi_frame_t[MHI_PIXELS];

/* A full recorded sample: MHI_SEQ_LEN consecutive grayscale frames. */
typedef uint8_t mhi_sequence_t[MHI_SEQ_LEN * MHI_PIXELS];

void mhi_init(mhi_ctx_t *ctx);

/* Clear history and re-prime on the next step. */
void mhi_reset(mhi_ctx_t *ctx);

/*
 * Seed the history with the first frame. MUST be called once before the first
 * mhi_step(); there is no previous frame to difference against otherwise.
 *
 * The split exists so the firmware and the PC-side implementation agree on
 * which frame is the reference. A recorded sequence's frame 0 was already the
 * device's `prev` when frame 1 was captured, so on the PC the whole sequence
 * participates. tools/run_parity.sh enforces this.
 */
void mhi_prime(mhi_ctx_t *ctx, const uint8_t *frame);

/* Step the MHI. `cur` is 64x64 luma, `out_mhi` receives the 64x64 MHI.
 * Returns false (and zeroes out_mhi) if the context was never primed. */
bool mhi_step(mhi_ctx_t *ctx, const uint8_t *cur, uint8_t *out_mhi);

/* Scale an MHI to full 0..255 range. Returns the observed max. */
uint8_t mhi_normalize(const uint8_t *mhi, uint8_t *out);

/* Convenience: step then normalise, matching what inference sees. */
void mhi_step_normalized(mhi_ctx_t *ctx, const uint8_t *cur, uint8_t *out);

/*
 * Box-average downscale of an RGB565 framebuffer to `out_w` x `out_h` luma.
 * The framebuffer is little-endian RGB565 as produced by esp_camera.
 */
void mhi_luma_downscale(const uint8_t *rgb565, uint16_t w, uint16_t h,
                        uint8_t *out, uint16_t out_w, uint16_t out_h);

/* Single-pixel RGB565 -> luma. Exposed so the host parity test can use it. */
uint8_t mhi_rgb565_luma(uint16_t p);
