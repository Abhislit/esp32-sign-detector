/*
 * test_mhi_parity.c - native host harness proving src/mhi.c and pc/mhi.py agree.
 *
 * Built for the host, not the ESP32. The flow is:
 *
 *   pc/parity.py      synthesises a sequence, writes input.bin
 *   this program      reads input.bin, runs mhi.c, writes c_out.bin
 *   pc/parity.py      runs mhi.py, compares byte for byte
 *
 * Feeding both sides the same bytes from a file (rather than each generating
 * its own input) keeps input synthesis out of the comparison, so any mismatch
 * is unambiguously a preprocessing difference.
 *
 * usage: test_mhi_parity <input.bin> <output.bin> <threshold> <decay_num> <normalize>
 *
 * Build and run via tools/run_parity.sh.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "../src/mhi.h"

#define MAX_FRAMES 64

static uint8_t frames[MAX_FRAMES][MHI_PIXELS];
static uint8_t mhi_out[MHI_PIXELS];
static uint8_t norm_out[MHI_PIXELS];
static uint32_t energy_out[MAX_FRAMES];

int main(int argc, char **argv)
{
    if (argc < 6) {
        fprintf(stderr,
                "usage: %s <input.bin> <output.bin> <threshold> <decay> <normalize>\n",
                argv[0]);
        return 2;
    }

    const int threshold = atoi(argv[3]);
    const int decay = atoi(argv[4]);
    const int do_norm = atoi(argv[5]);

    if (threshold < 0 || threshold > 255 || decay < 1 || decay > 255) {
        fprintf(stderr, "threshold/decay out of range\n");
        return 2;
    }

    FILE *f = fopen(argv[1], "rb");
    if (f == NULL) {
        fprintf(stderr, "cannot open %s\n", argv[1]);
        return 2;
    }

    /* Layout: uint16 n_frames, then n_frames * MHI_PIXELS bytes. */
    uint16_t n = 0;
    if (fread(&n, sizeof(n), 1, f) != 1) {
        fprintf(stderr, "cannot read frame count\n");
        fclose(f);
        return 2;
    }
    if (n == 0 || n > MAX_FRAMES) {
        fprintf(stderr, "bad frame count %u\n", (unsigned) n);
        fclose(f);
        return 2;
    }
    for (uint16_t i = 0; i < n; i++) {
        if (fread(frames[i], 1, MHI_PIXELS, f) != MHI_PIXELS) {
            fprintf(stderr, "short read on frame %u\n", (unsigned) i);
            fclose(f);
            return 2;
        }
    }
    fclose(f);

    /* Run the exact device code path, with the parameters under test. */
    mhi_ctx_t ctx;
    mhi_init(&ctx);
    ctx.motion_threshold = (uint8_t) threshold;
    ctx.decay_num = (uint8_t) decay;

    /* Frame 0 is the primer, exactly as the capture task does it. The whole
     * sequence then contributes, matching pc/mhi.py. */
    mhi_prime(&ctx, frames[0]);

    energy_out[0] = 0; /* priming frame produces no motion */
    for (uint16_t i = 1; i < n; i++) {
        mhi_step(&ctx, frames[i], mhi_out);
        energy_out[i] = ctx.motion_energy;
    }

    /* Mirror mhi_step_normalized()'s output stage under test. */
    uint8_t peak = 0;
    if (do_norm) {
        peak = mhi_normalize(mhi_out, norm_out);
    } else {
        memcpy(norm_out, mhi_out, MHI_PIXELS);
    }

    FILE *o = fopen(argv[2], "wb");
    if (o == NULL) {
        fprintf(stderr, "cannot write %s\n", argv[2]);
        return 2;
    }
    fwrite(&peak, sizeof(peak), 1, o);
    fwrite(mhi_out, 1, MHI_PIXELS, o);
    fwrite(norm_out, 1, MHI_PIXELS, o);
    for (uint16_t i = 0; i < n; i++) {
        fwrite(&energy_out[i], sizeof(uint32_t), 1, o);
    }
    fclose(o);

    fprintf(stderr,
            "C: T=%u threshold=%d decay=%d/%d normalize=%d peak=%u energy=%u\n",
            (unsigned) n, threshold, decay, 256, do_norm, (unsigned) peak,
            (unsigned) (n ? energy_out[n - 1] : 0));
    return 0;
}
