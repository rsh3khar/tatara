// Packed affine-Q4 group-64 dot: the M4-vectorized variant with two packed
// word loads and unorm4x8 unpacking per lane's 16 values. Chunk, group, and
// reduction order match q4_dot; the float4 dot-product interleave is the
// admitted arithmetic contract for the expert kernels.
inline float q4_dot_packed(device const bfloat* x, device const uint* w,
                           device const bfloat* scales, device const bfloat* biases, uint k_size,
                           uint lane) {
    float result = 0.0f;
    for (uint k = 0; k < k_size; k += 512u) {
        const uint word0 = (k >> 3u) + lane * 2u;
        const uint group = (k >> 6u) + (lane >> 2u);
        const float scale_value = float(scales[group]);
        const float bias_value = float(biases[group]);
        const uint x0 = k + lane * 16u;
        float sum = 0.0f;
        float quant = 0.0f;
        for (uint wi = 0; wi < 2u; ++wi) {
            const uint word = w[word0 + wi];
            const uint i = wi * 8u;
            const float4 even = float4(float(x[x0 + i + 0u]), float(x[x0 + i + 2u]),
                                       float(x[x0 + i + 4u]), float(x[x0 + i + 6u]));
            const float4 odd = float4(float(x[x0 + i + 1u]), float(x[x0 + i + 3u]),
                                      float(x[x0 + i + 5u]), float(x[x0 + i + 7u]));
            const float4 quant_even = unpack_unorm4x8_to_float(word & 0x0f0f0f0fu) * 255.0f;
            const float4 quant_odd = unpack_unorm4x8_to_float((word >> 4u) & 0x0f0f0f0fu) * 255.0f;
            sum += dot(even + odd, float4(1.0f));
            quant += dot(even, quant_even) + dot(odd, quant_odd);
        }
        result += scale_value * quant + sum * bias_value;
    }
    return simd_sum(result);
}

// Multi-row packed dot: weight words and affine constants load once per
// 512-column block; up to kQ4BatchRowsMax rows (arbitrary per-row input
// element offsets) accumulate independently with q4_dot_packed's exact
// per-row arithmetic order.
inline void q4_dot_packed_ms(device const bfloat* x_base,
                             thread const uint* row_offsets, uint row_count,
                             device const uint* w, device const bfloat* scales,
                             device const bfloat* biases, uint k_size, uint lane,
                             thread float* out) {
    float acc[kQ4BatchRowsMax];
    for (uint r = 0u; r < kQ4BatchRowsMax; ++r) {
        acc[r] = 0.0f;
    }
    for (uint k = 0; k < k_size; k += 512u) {
        const uint word0 = (k >> 3u) + lane * 2u;
        const uint group = (k >> 6u) + (lane >> 2u);
        const float scale_value = float(scales[group]);
        const float bias_value = float(biases[group]);
        const uint x0 = k + lane * 16u;
        uint words_cache[2];
        words_cache[0] = w[word0];
        words_cache[1] = w[word0 + 1u];
        for (uint r = 0u; r < row_count; ++r) {
            device const bfloat* x = x_base + row_offsets[r];
            float sum = 0.0f;
            float quant = 0.0f;
            for (uint wi = 0; wi < 2u; ++wi) {
                const uint word = words_cache[wi];
                const uint i = wi * 8u;
                const float4 even = float4(float(x[x0 + i + 0u]), float(x[x0 + i + 2u]),
                                           float(x[x0 + i + 4u]), float(x[x0 + i + 6u]));
                const float4 odd = float4(float(x[x0 + i + 1u]), float(x[x0 + i + 3u]),
                                          float(x[x0 + i + 5u]), float(x[x0 + i + 7u]));
                const float4 quant_even = unpack_unorm4x8_to_float(word & 0x0f0f0f0fu) * 255.0f;
                const float4 quant_odd =
                    unpack_unorm4x8_to_float((word >> 4u) & 0x0f0f0f0fu) * 255.0f;
                sum += dot(even + odd, float4(1.0f));
                quant += dot(even, quant_even) + dot(odd, quant_odd);
            }
            acc[r] += scale_value * quant + sum * bias_value;
        }
    }
    for (uint r = 0u; r < row_count; ++r) {
        out[r] = simd_sum(acc[r]);
    }
}
