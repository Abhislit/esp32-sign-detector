/*
 * infer.c - TFLM gesture classifier.
 *
 * Float32 is the default target: Espressif's own reference gesture model ships
 * float32, and int8 on this pipeline is where projects tend to silently
 * collapse to "always class 0" (espressif/esp-tflite-micro#108). The code
 * handles int8 too so the optimisation can be tried later, but only after
 * accuracy is proven in float32.
 *
 * Requires esp-nn to be enabled, which is a ~10x difference on this chip.
 * Verify with ESP_LOGI at boot that it is active.
 */
#include "infer.h"
#include "app_config.h"

#include <string.h>
#include <math.h>
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"

#if __has_include("model.h")
#define HAVE_MODEL 1
#include "model.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#if __has_include("golden_vector.h")
#define HAVE_GOLDEN 1
#include "golden_vector.h"
#endif
#else
#define HAVE_MODEL 0
#endif

#if __has_include("labels.h")
#define HAVE_LABELS 1
#include "labels.h"
#else
#define HAVE_LABELS 0
#endif

static const char *TAG = "infer";

static uint32_t s_last_ms = 0;
static bool s_ready = false;

/* Defined unconditionally: infer_label() needs it whenever labels.h is absent,
 * which includes the legitimate case of a model without a labels.h. */
static const char *s_fallback_label = "class";

#if HAVE_LABELS
static const char *const s_labels[] = GESTURE_LABELS;
#define LABEL_COUNT ((int) (sizeof(s_labels) / sizeof(s_labels[0])))
#endif

#if HAVE_MODEL

/*
 * Arena lives in internal RAM deliberately. The camera framebuffers are in
 * PSRAM, which leaves roughly 400KB of internal SRAM here, and TFLM tensor
 * access is heavily random - PSRAM would roughly double inference time for
 * this model size.
 */
static uint8_t s_arena[TENSOR_ARENA_SIZE];

static tflite::MicroInterpreter *s_interp = NULL;
static TfLiteTensor *s_input = NULL;
static TfLiteTensor *s_output = NULL;

static float s_probs[NUM_GESTURES];

bool infer_init(void)
{
    if (s_ready) {
        return true;
    }

    /*
     * Symbol name must match what pc/train.py's write_model_h() emits via
     * `xxd -i -n gesture_model_data`. tools/verify_export.py reads the same
     * name, so a rename here breaks the export check too.
     */
    const tflite::Model *model = tflite::GetModel(gesture_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "model schema %d != supported %d - regenerate with pc/train.py",
                 (int) model->version(), (int) TFLITE_SCHEMA_VERSION);
        return false;
    }

    static tflite::MicroMutableOpResolver<10> resolver;
    if (resolver.AddConv2D() != kTfLiteOk ||
        resolver.AddDepthwiseConv2D() != kTfLiteOk ||
        resolver.AddMaxPool2D() != kTfLiteOk ||
        resolver.AddAveragePool2D() != kTfLiteOk ||
        resolver.AddFullyConnected() != kTfLiteOk ||
        resolver.AddSoftmax() != kTfLiteOk ||
        resolver.AddReshape() != kTfLiteOk ||
        resolver.AddRelu() != kTfLiteOk ||
        resolver.AddQuantize() != kTfLiteOk ||
        resolver.AddDequantize() != kTfLiteOk) {
        ESP_LOGE(TAG, "op resolver setup failed");
        return false;
    }

    static tflite::MicroInterpreter interp(model, resolver, s_arena, TENSOR_ARENA_SIZE);
    s_interp = &interp;

    if (s_interp->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors failed - arena too small? "
                      "Current: %d bytes, model needs more", TENSOR_ARENA_SIZE);
        return false;
    }

    s_input = s_interp->input(0);
    s_output = s_interp->output(0);
    if (s_input == NULL || s_output == NULL) {
        ESP_LOGE(TAG, "input/output tensor missing");
        return false;
    }

    if (s_input->dims->data[1] != MHI_H || s_input->dims->data[2] != MHI_W) {
        ESP_LOGW(TAG, "model input is not %dx%d, retrain with pc/train.py",
                 MHI_W, MHI_H);
    }

    ESP_LOGI(TAG, "model loaded. input type %d, output type %d",
             (int) s_input->type, (int) s_output->type);
    s_ready = true;
    return true;
}

static bool fill_input(const uint8_t *mhi)
{
    switch (s_input->type) {
    case kTfLiteFloat32: {
        float *dst = s_input->data.f;
        for (int i = 0; i < MHI_PIXELS; i++) {
            dst[i] = (float) mhi[i] / 255.0f;
        }
        return true;
    }
    case kTfLiteInt8: {
        /*
         * Standard asymmetric quantisation: real = (q - zero_point) * scale,
         * therefore q = real / scale + zero_point.
         *
         * Getting this backwards is silent rather than loud: the model still
         * returns a confident class, just a meaningless one. This is the
         * device-side twin of pc/mhi.py:to_input_tensor().
         */
        const float scale = s_input->params.scale;
        const int32_t zero = s_input->params.zero_point;
        if (scale <= 0.0f) {
            ESP_LOGE(TAG, "int8 input has non-positive scale %f", (double) scale);
            return false;
        }
        int8_t *dst = s_input->data.i8;
        for (int i = 0; i < MHI_PIXELS; i++) {
            const float real = (float) mhi[i] / 255.0f;
            float v = nearbyintf(real / scale) + (float) zero;
            if (v < -128.0f) {
                v = -128.0f;
            } else if (v > 127.0f) {
                v = 127.0f;
            }
            dst[i] = (int8_t) v;
        }
        return true;
    }
    default:
        ESP_LOGE(TAG, "unsupported input type %d", (int) s_input->type);
        return false;
    }
}

int infer_scores(const uint8_t *mhi, float *scores, int max_classes)
{
    if (!s_ready || !fill_input(mhi)) {
        return 0;
    }

    const int64_t t0 = esp_timer_get_time();
    if (s_interp->Invoke() != kTfLiteOk) {
        ESP_LOGE(TAG, "Invoke failed");
        return 0;
    }
    s_last_ms = (uint32_t) ((esp_timer_get_time() - t0) / 1000);

    int n = (int) s_output->dims->data[s_output->dims->size - 1];
    if (n > max_classes) {
        n = max_classes;
    }

    for (int i = 0; i < n; i++) {
        float v = 0.0f;
        switch (s_output->type) {
        case kTfLiteFloat32:
            v = s_output->data.f[i];
            break;
        case kTfLiteInt8: {
            int8_t q = s_output->data.i8[i];
            v = ((float) q - (float) s_output->params.zero_point) *
                s_output->params.scale;
            break;
        }
        case kTfLiteUInt8: {
            uint8_t q = s_output->data.ui8[i];
            v = ((float) q - (float) s_output->params.zero_point) *
                s_output->params.scale;
            break;
        }
        default:
            break;
        }
        s_probs[i] = v;
        scores[i] = v;
    }

    return n;
}

int infer_run(const uint8_t *mhi, float *out_score)
{
    float scores[NUM_GESTURES];
    const int n = infer_scores(mhi, scores, NUM_GESTURES);
    if (n <= 0) {
        return -1;
    }
    int best = 0;
    for (int i = 1; i < n; i++) {
        if (scores[i] > scores[best]) {
            best = i;
        }
    }
    if (out_score != NULL) {
        *out_score = scores[best];
    }
    return best;
}

bool infer_selftest(infer_selftest_t *out)
{
    memset(out, 0, sizeof(*out));
#if !HAVE_MODEL || !HAVE_GOLDEN
    return false;
#else
    if (!s_ready) {
        return false;
    }
    out->ran = true;
    out->want_class = GOLDEN_EXPECTED_CLASS;
    out->want_score = (float) GOLDEN_EXPECTED_SCORE_MILLI / 1000.0f;

    out->got_class = infer_run(golden_input, &out->got_score);
    out->class_ok = (out->got_class == out->want_class);
    /* Float accumulation differs slightly between TFLite and TFLM, so the
     * score is compared loosely; the class is the signal that matters. */
    const float d = out->got_score - out->want_score;
    out->score_ok = (d < 0.05f && d > -0.05f);
    return true;
#endif
}

#else /* !HAVE_MODEL - build without a trained model */

bool infer_init(void)
{
    ESP_LOGW(TAG, "no src/model.h found - inference disabled. "
                  "Run pc/train.py then copy model.h into src/");
    return false;
}

int infer_run(const uint8_t *mhi, float *out_score)
{
    (void) mhi;
    if (out_score != NULL) {
        *out_score = 0.0f;
    }
    return -1;
}

int infer_scores(const uint8_t *mhi, float *scores, int max_classes)
{
    (void) mhi;
    (void) scores;
    (void) max_classes;
    return 0;
}

#endif /* HAVE_MODEL */

/*
 * Defined once, for every build. The no-model branch above deliberately does
 * not define it, since s_ready can never become true there.
 */
bool infer_available(void)
{
    return s_ready;
}

const char *infer_label(int id)
{
    (void) id;
#if HAVE_LABELS
    if (id >= 0 && id < LABEL_COUNT) {
        return s_labels[id];
    }
    return "?";
#else
    return s_fallback_label;
#endif
}

uint32_t infer_last_ms(void)
{
    return s_last_ms;
}
