// Derived from MLX (Apple Inc., MIT) quantized.h, the bits==8 affine path.
// See NOTICE and THIRD_PARTY_LICENSES/.
//
// Affine-Q8 group-64 dot for the router: 256-column blocks, eight contiguous
// values per lane, scale * quant + sum * bias, one final simd_sum.
inline float q8_dot(device const bfloat* x, device const uint* w, device const bfloat* scales,
                    device const bfloat* biases, uint k_size, uint lane) {
    float result = 0.0f;
    device const uchar* bytes = reinterpret_cast<device const uchar*>(w);
    for (uint k = 0; k < k_size; k += 256u) {
        const uint x0 = k + lane * 8u;
        float sum = 0.0f;
        float quant = 0.0f;
        for (uint i = 0; i < 8u; ++i) {
            const float value = float(x[x0 + i]);
            sum += value;
            quant += value * float(bytes[x0 + i]);
        }
        const uint group = (k >> 6u) + (lane >> 3u);
        result += float(scales[group]) * quant + sum * float(biases[group]);
    }
    return simd_sum(result);
}

// Multi-row q8 dot: weight bytes load once per block; rows accumulate
// independently with q8_dot's exact per-row order.
inline void q8_dot_ms(device const bfloat* x0, ulong x_stride, uint row_count,
                      device const uint* w, device const bfloat* scales,
                      device const bfloat* biases, uint k_size, uint lane,
                      thread float* out) {
    float acc[kQ4BatchRowsMax];
    for (uint r = 0u; r < kQ4BatchRowsMax; ++r) {
        acc[r] = 0.0f;
    }
    device const uchar* bytes = reinterpret_cast<device const uchar*>(w);
    for (uint k = 0; k < k_size; k += 256u) {
        const uint x_off = k + lane * 8u;
        uchar wb[8];
        for (uint i = 0; i < 8u; ++i) {
            wb[i] = bytes[x_off + i];
        }
        const uint group = (k >> 6u) + (lane >> 3u);
        const float scale = float(scales[group]);
        const float bias = float(biases[group]);
        for (uint r = 0u; r < row_count; ++r) {
            device const bfloat* x = x0 + ulong(r) * x_stride;
            float sum = 0.0f;
            float quant = 0.0f;
            for (uint i = 0; i < 8u; ++i) {
                const float value = float(x[x_off + i]);
                sum += value;
                quant += value * float(wb[i]);
            }
            acc[r] += scale * quant + sum * bias;
        }
    }
    for (uint r = 0u; r < row_count; ++r) {
        out[r] = simd_sum(acc[r]);
    }
}
