/*
 * tflm_stubs.h - minimal stand-ins for the TensorFlow Lite Micro headers.
 *
 * Lets src/infer.cpp be syntax-checked on the host with `g++ -fsyntax-only`,
 * which covers the branch that otherwise only compiles on the ESP32 with the
 * real esp-tflite-micro component. No logic, never linked.
 *
 * Note this is C++: tflite::MicroInterpreter is a class and the TFLM API has
 * no C equivalent. src/infer.cpp must therefore be a .cpp translation unit -
 * a .c file will not compile, which is why Espressif's own reference example
 * is app_model.cpp.
 */
#pragma once

#include <stddef.h>
#include <stdint.h>

typedef enum {
    kTfLiteOk = 0,
    kTfLiteError = 1,
} TfLiteStatus;

typedef enum {
    kTfLiteNoType = 0,
    kTfLiteFloat32 = 1,
    kTfLiteInt32 = 2,
    kTfLiteUInt8 = 3,
    kTfLiteInt64 = 4,
    kTfLiteString = 5,
    kTfLiteBool = 6,
    kTfLiteInt16 = 7,
    kTfLiteComplex64 = 8,
    kTfLiteInt8 = 9,
    kTfLiteFloat16 = 10,
} TfLiteType;

/* Real TFLM: `int size; int* data;` - flat, not a nested dimension struct. */
typedef struct {
    int size;
    int *data;
} TfLiteDims;

typedef struct {
    float scale;
    int32_t zero_point;
} TfLiteQuantizationParams;

typedef struct {
    /* Real TFLM exposes a named `data` union, which is what infer.cpp uses. */
    union Data {
        int8_t *i8;
        uint8_t *ui8;
        float *f;
        int32_t *i32;
    } data;
    TfLiteDims *dims;
    TfLiteType type;
    TfLiteQuantizationParams params;
} TfLiteTensor;

namespace tflite {

class Model {
public:
    unsigned version() const;
};

/* Pull the model bytes out of the generated C array. */
const Model *GetModel(const void *model_data);

class MicroOpResolver {
public:
    virtual ~MicroOpResolver() = default;
};

class MicroInterpreter {
public:
    MicroInterpreter(const Model *model, const MicroOpResolver &op_resolver,
                     uint8_t *tensor_arena, size_t tensor_arena_size);
    TfLiteStatus AllocateTensors();
    TfLiteStatus Invoke();
    TfLiteTensor *input(int index);
    TfLiteTensor *output(int index);
};

/*
 * The real class is a template whose N bounds the op-count assert. A stub only
 * needs the Add* methods infer.cpp calls; they all return kTfLiteOk so the
 * control flow is representative.
 */
template <unsigned N>
class MicroMutableOpResolver : public MicroOpResolver {
public:
    TfLiteStatus AddConv2D() { return kTfLiteOk; }
    TfLiteStatus AddDepthwiseConv2D() { return kTfLiteOk; }
    TfLiteStatus AddMaxPool2D() { return kTfLiteOk; }
    TfLiteStatus AddAveragePool2D() { return kTfLiteOk; }
    TfLiteStatus AddFullyConnected() { return kTfLiteOk; }
    TfLiteStatus AddSoftmax() { return kTfLiteOk; }
    TfLiteStatus AddReshape() { return kTfLiteOk; }
    TfLiteStatus AddRelu() { return kTfLiteOk; }
    TfLiteStatus AddQuantize() { return kTfLiteOk; }
    TfLiteStatus AddDequantize() { return kTfLiteOk; }
};

}  // namespace tflite
